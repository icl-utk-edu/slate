// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"

namespace slate {

namespace impl {

//------------------------------------------------------------------------------
/// Distributed parallel LQ factorization.
/// Generic implementation for any target.
/// Panel and lookahead computed on host using Host OpenMP task.
///
/// ColMajor layout is assumed
///
/// @ingroup gelqf_impl
///
template <Target target, typename scalar_t>
void gelqf(
    Matrix<scalar_t>& A,
    TriangularFactors<scalar_t>& T,
    Options const& opts )
{
    using BcastList = typename Matrix<scalar_t>::BcastList;
    using lapack::device_info_int;
    using blas::real;

    // Constants
    const int priority_0 = 0;
    const int priority_1 = 1;
    // Assumes column major
    const Layout layout = Layout::ColMajor;

    // Options
    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );
    int64_t ib = get_option<int64_t>( opts, Option::InnerBlocking, 16 );
    int64_t max_panel_threads  = std::max(omp_get_max_threads()/2, 1);
    max_panel_threads = get_option<int64_t>( opts, Option::MaxPanelThreads,
                                             max_panel_threads );

    int64_t A_mt = A.mt();
    int64_t A_nt = A.nt();
    int64_t A_min_mtnt = std::min(A_mt, A_nt);

    // Make Tlocal have fixed, square nb-by-nb tiles,
    // and Treduce have fixed, rectangular ib-by-nb tiles.
    // Otherwise, edge tiles are the wrong size: mb-by-nb instead of nb-by-mb.
    int64_t nb = A.tileNb(0);
    T.clear();
    T.push_back(A.emptyLike(nb, nb));
    T.push_back(A.emptyLike(ib, nb));
    auto Tlocal  = T[0];
    auto Treduce = T[1];
    auto TlocalT = A.emptyLike(nb, nb, Op::ConjTrans);

    // workspace
    auto W = A.emptyLike();

    int64_t num_devices  = A.num_devices();
    int     panel_device = -1;
    size_t  work_size    = 0;
    lapack::Queue* queue;

    std::vector< scalar_t* > dwork_array( num_devices, nullptr );

    if (target == Target::Devices) {
        const int64_t batch_size_default = 0; // use default batch size
        int num_queues = 3 + lookahead;
        A.allocateBatchArrays( batch_size_default, num_queues );
        A.reserveDeviceWorkspace();
        W.allocateBatchArrays( batch_size_default, num_queues );
        // todo: this is demanding too much device workspace memory
        // only one tile-row of matrix W per MPI process is going to be used,
        // but W with size of whole A is being allocated
        // thus limiting the matrix size that can be processed
        // For now, allocate workspace tiles 1-by-1.
        //W.reserveDeviceWorkspace();

        // Find largest panel size and device for copying to
        // contiguous memory within internal geqrf routine
        int64_t nlocal = 0;
        int64_t first_panel_seen = -1;
        // Assume transpose on panel-evaluation
        for (int64_t i = 0; i < A.mt(); ++i) {
            for (int64_t j = i; j < A.nt(); ++j) {
                if (A.tileIsLocal( i, j )) {
                    if (first_panel_seen < 0) {
                        first_panel_seen = i;
                    }
                    if (first_panel_seen == i) {
                        if (panel_device < 0) {
                            panel_device = A.tileDevice( i, j );
                        }
                        nlocal += A.tileNb( j );
                    }
                }
            }
            if (first_panel_seen >= 0) {
                break;
            }
        }

        if (panel_device >= 0) {

            queue = A.compute_queue( panel_device );

            int64_t mb       = A.tileMb(0);
            size_t  size_tau = (size_t) std::min( nlocal, mb );
            size_t  size_A   = (size_t) blas::max( 1, nlocal ) * mb;
            size_t  hsize, dsize;

            lapack::geqrf_work_size_bytes( nlocal, mb, dwork_array[0], nlocal,
                                           &dsize, &hsize, *queue );

            work_size = size_A + size_tau + ceildiv( dsize, sizeof( scalar_t ) )
                        + ceildiv( sizeof( device_info_int ), sizeof( scalar_t ) );

            for (int64_t dev = 0; dev < num_devices; ++dev) {
                queue = A.comm_queue( dev );
                dwork_array[ dev ] = blas::device_malloc<scalar_t>( work_size, *queue );
            }
        }
    }

    // Workspace for transposed panels needs one column of tiles.
    auto AT = A.emptyLike(0, 0, Op::ConjTrans);

    // LQ tracks dependencies by block-row.
    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > block_vector(A_mt);
    uint8_t* block = block_vector.data();
    SLATE_UNUSED( block ); // Used only by OpenMP

    // set min number for omp nested active parallel regions
    slate::OmpSetMaxActiveLevels set_active_levels( MinOmpActiveLevels );

    #pragma omp parallel
    #pragma omp master
    {
        for (int64_t k = 0; k < A_min_mtnt; ++k) {
            auto  A_panel =       A.sub(k, k, k, A_nt-1);
            auto Tl_panel =  Tlocal.sub(k, k, k, A_nt-1);
            auto Tr_panel = Treduce.sub(k, k, k, A_nt-1);
            // Transposed panels.
            auto  AT_panel =      AT.sub(k, A_nt-1, k, k);
            auto TlT_panel = TlocalT.sub(k, A_nt-1, k, k);

            std::vector< int64_t > first_indices
                            = internal::gelqf_compute_first_indices(A_panel, k);

            // panel, high priority
            #pragma omp task depend(inout:block[k]) priority(1)
            {
                //--------------------
                // Instead of doing LQ of panel, we do QR of transpose( panel ),
                // so that the panel is computed in column-major for much
                // better cache efficiency.
                for (int64_t j = 0; j < A_panel.nt(); ++j) {
                    if (A_panel.tileIsLocal(0, j)) {
                        // Needed if origin is device
                        A_panel.tileGetForReading( 0, j, LayoutConvert( layout ) );
                        AT_panel.tileInsert( j, 0 );
                        AT_panel.tileModified( j, 0, HostNum );
                        tile::deepConjTranspose( A_panel(0, j), AT_panel(j, 0) );
                    }
                }

                // local panel factorization
                internal::geqrf<target>(
                    std::move(AT_panel), std::move(TlT_panel),
                    dwork_array, work_size, ib,
                    max_panel_threads, priority_1 );

                // Find first local tile, which is triangular factor (T in I - VTV^H),
                // and copy it to Tlocal.
                for (int64_t i = 0; i < TlT_panel.mt(); ++i) {
                    if (Tl_panel.tileIsLocal(0, i)) {
                        // Device GEQRF updates tiles on device. Without getting
                        // the tile for reading on Host, it is not updated
                        TlT_panel.tileGetForReading( i, 0, LayoutConvert( layout ) );
                        Tl_panel.tileInsert(0, i);
                        Tl_panel.tileModified( 0, i, HostNum );
                        tile::gecopy( TlT_panel(i, 0), Tl_panel(0, i) );
                        break;
                    }
                }
                // Copy result back.
                for (int64_t j = 0; j < A_panel.nt(); ++j) {
                    if (A_panel.tileIsLocal(0, j)) {
                        // Same as above for TlT
                        AT_panel.tileGetForReading( j, 0, LayoutConvert( layout ) );
                        A_panel.tileGetForWriting( 0, j, LayoutConvert( layout ) );
                        tile::deepConjTranspose( AT_panel(j, 0), A_panel(0, j) );
                        AT_panel.tileErase(j, 0, AllDevices);
                    }
                }
                //--------------------

                // triangle-triangle reductions
                // ttlqt handles tile transfers internally
                internal::ttlqt<Target::HostTask>(
                                std::move(A_panel),
                                std::move(Tr_panel) );

                // if a trailing matrix exists
                if (k < A_mt-1) {

                    // bcast V down col for trailing matrix update
                    if (k < A_nt) {
                        BcastList bcast_list_V;
                        for (int64_t j = k; j < A_nt; ++j) {
                            // send A(k, j) down col A(k+1:mt-1, j)
                            bcast_list_V.push_back({k, j, {A.sub(k+1, A_mt-1, j, j)}});
                        }
                        A.template listBcast<target>( bcast_list_V, layout );
                    }

                    // bcast Tlocal down col for trailing matrix update
                    if (first_indices.size() > 0) {
                        BcastList bcast_list_T;
                        for (int64_t col : first_indices) {
                            bcast_list_T.push_back({k, col, {Tlocal.sub(k+1, A_mt-1, col, col)}});
                        }
                        Tlocal.template listBcast<target>( bcast_list_T, layout );
                    }

                    // bcast Treduce down col for trailing matrix update
                    if (first_indices.size() > 1) {
                        BcastList bcast_list_T;
                        for (int64_t col : first_indices) {
                            if (col > k) // exclude the first col of this panel that has no Treduce tile
                                bcast_list_T.push_back({k, col, {Treduce.sub(k+1, A_mt-1, col, col)}});
                        }
                        Treduce.template listBcast(bcast_list_T, layout);
                    }
                }
            }

            // update lookahead row(s) on CPU, high priority
            for (int64_t i = k+1; i < (k+1+lookahead) && i < A_mt; ++i) {
                auto A_trail_i = A.sub(i, i, k, A_nt-1);

                #pragma omp task depend(in:block[k]) \
                                 depend(inout:block[i]) \
                                 priority(1)
                {
                    // Apply local reflectors
                    int queue_ik1 = i-k+1;
                    internal::unmlq<target>(
                                    Side::Right, Op::ConjTrans,
                                    std::move(A_panel),
                                    std::move(Tl_panel),
                                    std::move(A_trail_i),
                                    W.sub(i, i, k, A_nt-1),
                                    priority_1, queue_ik1 );

                    // Apply triangle-triangle reduction reflectors
                    // ttmlq handles the tile broadcasting internally
                    int tag_i = i;
                    internal::ttmlq<Target::HostTask>(
                                    Side::Right, Op::ConjTrans,
                                    std::move(A_panel),
                                    std::move(Tr_panel),
                                    std::move(A_trail_i),
                                    tag_i );
                }
            }

            // update trailing submatrix, normal priority
            if (k+1+lookahead < A_mt) {
                int64_t i = k+1+lookahead;
                auto A_trail_i = A.sub(i, A_mt-1, k, A_nt-1);

                #pragma omp task depend(in:block[k]) \
                                 depend(inout:block[k+1+lookahead]) \
                                 depend(inout:block[A_mt-1])
                {
                    // Apply local reflectors
                    int queue_ik1 = i-k+1;
                    internal::unmlq<target>(
                                    Side::Right, Op::ConjTrans,
                                    std::move(A_panel),
                                    std::move(Tl_panel),
                                    std::move(A_trail_i),
                                    W.sub(i, A_mt-1, k, A_nt-1),
                                    priority_0, queue_ik1 );

                    // Apply triangle-triangle reduction reflectors
                    // ttmlq handles the tile broadcasting internally
                    int tag_i = i;
                    internal::ttmlq<Target::HostTask>(
                                    Side::Right, Op::ConjTrans,
                                    std::move(A_panel),
                                    std::move(Tr_panel),
                                    std::move(A_trail_i),
                                    tag_i );
                }
            }

            #pragma omp task depend(inout:block[k])
            {
                // Release the whole row, not just the panel
                for (int64_t j = 0; j < A_nt; ++j) {
                    if (A.tileIsLocal(k, j)) {
                        A.tileUpdateOrigin(k, j);
                        A.releaseLocalWorkspaceTile(k, j);
                    }
                    else {
                        A.releaseRemoteWorkspaceTile(k, j);
                    }
                }

                for (int64_t j : first_indices) {
                    if (Tlocal.tileIsLocal( k, j )) {
                        // Tlocal and Treduce have the same process distribution
                        Tlocal.tileUpdateOrigin( k, j );
                        Tlocal.releaseLocalWorkspaceTile( k, j );
                        if (j != k) {
                            // j == k is the root of the reduction tree
                            // Treduce( k, k ) isn't allocated
                            Treduce.tileUpdateOrigin( k, j );
                            Treduce.releaseLocalWorkspaceTile( k, j );
                        }
                    }
                    else {
                        Tlocal.releaseRemoteWorkspaceTile( k, j );
                        Treduce.releaseRemoteWorkspaceTile( k, j );
                    }
                }
            }
        }

        #pragma omp taskwait
        A.tileUpdateAllOrigin();
    }

    if (target == Target::Devices) {
        for (int64_t dev = 0; dev < num_devices; ++dev) {
            queue = A.comm_queue( dev );
            blas::device_free( dwork_array[ dev ], *queue );
            dwork_array[ dev ] = nullptr;
        }
    }

    A.releaseWorkspace();
}

} // namespace impl

//------------------------------------------------------------------------------
/// Distributed parallel LQ factorization.
///
/// Computes a LQ factorization of an m-by-n matrix $A$.
/// The factorization has the form
/// \[
///     A = LQ,
/// \]
/// where $Q$ is a matrix with orthonormal columns and $L$ is lower triangular
/// (or lower trapezoidal if m > n).
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the m-by-n matrix $A$.
///     On exit, the elements on and below the diagonal of the array contain
///     the m-by-min(m,n) lower trapezoidal matrix $L$ (lower triangular
///     if m <= n); the elements above the diagonal represent the unitary
///     matrix $Q$ as a product of elementary reflectors.
///
/// @param[out] T
///     On exit, triangular matrices of the block reflectors.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::Lookahead:
///       Number of panels to overlap with matrix updates.
///       lookahead >= 0. Default 1.
///     - Option::InnerBlocking:
///       Inner blocking to use for panel. Default 16.
///     - Option::MaxPanelThreads:
///       Number of threads to use for panel. Default omp_get_max_threads()/2.
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  nested OpenMP parallel for loop on CPU host.
///       - HostBatch: batched BLAS on CPU host.
///       - Devices:   batched BLAS on GPU device.
///
/// @ingroup gelqf_computational
///
template <typename scalar_t>
void gelqf(
    Matrix<scalar_t>& A,
    TriangularFactors<scalar_t>& T,
    Options const& opts )
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            impl::gelqf<Target::HostTask>( A, T, opts );
            break;

        case Target::HostNest:
            impl::gelqf<Target::HostNest>( A, T, opts );
            break;

        case Target::HostBatch:
            impl::gelqf<Target::HostBatch>( A, T, opts );
            break;

        case Target::Devices:
            impl::gelqf<Target::Devices>( A, T, opts );
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gelqf<float>(
    Matrix<float>& A,
    TriangularFactors<float>& T,
    Options const& opts);

template
void gelqf<double>(
    Matrix<double>& A,
    TriangularFactors<double>& T,
    Options const& opts);

template
void gelqf< std::complex<float> >(
    Matrix< std::complex<float> >& A,
    TriangularFactors< std::complex<float> >& T,
    Options const& opts);

template
void gelqf< std::complex<double> >(
    Matrix< std::complex<double> >& A,
    TriangularFactors< std::complex<double> >& T,
    Options const& opts);

} // namespace slate
