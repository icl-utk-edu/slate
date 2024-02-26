// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

namespace slate {

namespace impl {

//------------------------------------------------------------------------------
/// Distributed parallel CALU factorization.
/// Generic implementation for any target.
/// Panel and lookahead computed on host using Host OpenMP task.
/// @ingroup gesv_impl
///
template <Target target, typename scalar_t>
int64_t getrf_tntpiv(
    Matrix<scalar_t>& A, Pivots& pivots,
    Options const& opts)
{
    using BcastList = typename Matrix<scalar_t>::BcastList;
    using BcastListTag = typename Matrix<scalar_t>::BcastListTag;
    using lapack::device_info_int;
    using lapack::device_pivot_int;

    // Constants
    const scalar_t one = 1.0;
    const int priority_0 = 0;
    const int priority_1 = 1;
    const int queue_0 = 0;
    const int queue_1 = 1;
    const int queue_2 = 2;

    // Options
    int64_t lookahead = get_option<Option::Lookahead>( opts, 1 );
    int64_t ib = get_option<Option::InnerBlocking>( opts, 16 );
    int64_t max_panel_threads  = std::max( omp_get_max_threads()/2, 1 );
    max_panel_threads = get_option<Option::MaxPanelThreads>(
                                                      opts, max_panel_threads );

    // Host can use Col/RowMajor for row swapping,
    // RowMajor is slightly more efficient.
    // Layout host_layout = Layout::RowMajor;
    // Layout target_layout = Layout::RowMajor;
    // todo: RowMajor causes issues with tileLayoutReset() when A origin is
    //       ScaLAPACK
    Layout host_layout = Layout::ColMajor;
    Layout target_layout = Layout::ColMajor;
    // GPU Devices use RowMajor for efficient row swapping.
    if (target == Target::Devices)
        target_layout = Layout::RowMajor;

    int64_t info = 0;
    int64_t A_nt = A.nt();
    int64_t A_mt = A.mt();
    int64_t min_mt_nt = std::min(A.mt(), A.nt());
    pivots.resize(min_mt_nt);

    // setting up dummy variables for case the when target == host
    int64_t num_devices  = A.num_devices();
    int     panel_device = -1;
    size_t  dwork_bytes  = 0;

    std::vector< char* > dwork_array( num_devices, nullptr );

    if (target == Target::Devices) {
        const int64_t batch_size_default = 0;
        int num_queues = 3 + lookahead;
        A.allocateBatchArrays( batch_size_default, num_queues );
        A.reserveDeviceWorkspace();

        int64_t mlocal = 0;
        int64_t first_panel_seen = -1;
        for (int64_t j = 0; j < A.nt(); ++j) {
            for (int64_t i = j; i < A.mt(); ++i) {
                if (A.tileIsLocal(i, j)) {
                    if (first_panel_seen < 0) {
                        first_panel_seen = j;
                    }
                    if (first_panel_seen == j) {
                        if (panel_device < 0) {
                            panel_device = A.tileDevice(i, j);
                        }
                        mlocal += A.tileMb(i);
                        // Asserting 1-D distribution for device
                        assert( panel_device == A.tileDevice(i, j) );
                    }
                }
            }
            if (first_panel_seen >= 0) {
                break;
            }
        }

        if (panel_device >= 0) {
            lapack::Queue* comm_queue = A.comm_queue(panel_device);

            int64_t nb           = A.tileNb(0);
            int64_t diag_len     = blas::min( A.tileMb(0), nb );
            size_t  size_A_bytes = blas::max( 1, mlocal ) * nb * sizeof( scalar_t );
            size_t  ipiv_bytes   = diag_len * sizeof( device_pivot_int );

            // Find size of the workspace needed
            scalar_t* dA = (scalar_t*) dwork_array[ 0 ];
            size_t hsize, dsize;
            lapack::getrf_work_size_bytes( mlocal, nb, dA, mlocal,
                                           &dsize, &hsize, *comm_queue );

            // Pad arrays to 8-byte boundaries.
            dsize        = roundup( dsize,        size_t( 8 ) );
            size_A_bytes = roundup( size_A_bytes, size_t( 8 ) );
            ipiv_bytes   = roundup( ipiv_bytes,   size_t( 8 ) );

            // Size of dA, dwork, dipiv, and dinfo in bytes.
            dwork_bytes = dsize + size_A_bytes + ipiv_bytes
                        + sizeof( device_info_int );

            for (int64_t dev = 0; dev < num_devices; ++dev) {
                lapack::Queue* queue = A.comm_queue( dev );
                dwork_array[ dev ]
                    = blas::device_malloc<char>( dwork_bytes, *queue );
            }
        }
    }

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > column_vector(A_nt);
    uint8_t* column = column_vector.data();
    SLATE_UNUSED( column ); // Used only by OpenMP
    std::vector< uint8_t > diag_vector(A_nt);
    uint8_t* diag = diag_vector.data();
    SLATE_UNUSED( diag ); // Used only by OpenMP

    // Running two listBcastMT's simultaneously can hang due to task ordering
    // This dependency avoids that
    uint8_t listBcastMT_token;
    SLATE_UNUSED(listBcastMT_token); // Only used by OpenMP

    // workspace
    auto Awork = A.emptyLike();

    // set min number for omp nested active parallel regions
    slate::OmpSetMaxActiveLevels set_active_levels( MinOmpActiveLevels );

    #pragma omp parallel
    #pragma omp master
    {
        int64_t kk = 0;
        for (int64_t k = 0; k < min_mt_nt; ++k) {

            int64_t diag_len = std::min(A.tileMb(k), A.tileNb(k));
            pivots.at(k).resize(diag_len);

            // panel, high priority
            #pragma omp task depend(inout:column[k]) \
                             depend(inout:diag[k]) \
                             priority(1)
            {
                auto Apanel = Awork.sub( k, A_mt-1, k, k );
                Apanel.insertLocalTiles();

                // Factor A(k:mt-1, k) using tournament pivoting to get
                // pivots and diagonal tile, Akk in workspace Apanel.
                int64_t iinfo;
                internal::getrf_tntpiv_panel<target>(
                    A.sub(k, A_mt-1, k, k), std::move(Apanel),
                    dwork_array, dwork_bytes, diag_len, ib,
                    pivots.at(k), max_panel_threads, priority_1, &iinfo );
                if (info == 0 && iinfo > 0) {
                    info = kk + iinfo;
                }

                // Root broadcasts the pivot to all ranks.
                // todo: Panel ranks send the pivots to the right.
                {
                    trace::Block trace_block("MPI_Bcast");

                    MPI_Bcast(pivots.at(k).data(),
                              sizeof(Pivot)*pivots.at(k).size(),
                              MPI_BYTE, A.tileRank(k, k), A.mpiComm());
                }

                // swap rows in A(k+1:A_mt-1, k)
                int tag_k = k;
                internal::permuteRows<target>(
                    Direction::Forward, A.sub(k, A_mt-1, k, k),
                    pivots.at(k), target_layout, priority_1, tag_k, queue_0 );

                // Copy factored diagonal tile into place.
                internal::copy<Target::HostTask>(
                    Apanel.sub( 0, 0, 0, 0 ), A.sub( k, k, k, k ));

                // Broadcast Akk tile down column to ranks owning A_k+1:mt,k
                // and across row to ranks owning A_k,k+1:nt.
                BcastList bcast_list_A;
                bcast_list_A.push_back({k, k, {A.sub(k+1, A_mt-1, k, k),
                                               A.sub(k, k, k+1, A_nt-1)}});

                A.template listBcast<target>(
                    bcast_list_A, host_layout, tag_k );

                Apanel.clear();
            }

            // Finish computing panel using trsm below diagonal tile.
            // A_k+1:mt,k = A_k+1:mt,k * Tkk^{-1}
            #pragma omp task depend(in:diag[k]) \
                             depend(inout:column[k]) \
                             priority(1)
            {
                auto Akk = A.sub(k, k, k, k);
                auto Tkk = TriangularMatrix<scalar_t>(
                    Uplo::Upper, Diag::NonUnit, Akk);

                internal::trsm<target>(
                    Side::Right,
                    one, std::move(Tkk),
                         A.sub( k+1, A_mt-1, k, k ),
                    priority_1, target_layout, queue_0 );
            }

            #pragma omp task depend(inout:column[k]) \
                             depend(inout:listBcastMT_token) \
                             priority(1)
            {
                BcastListTag bcast_list;
                // bcast the tiles of the panel to the right hand side
                for (int64_t i = k+1; i < A_mt; ++i) {
                    // send A(i, k) across row A(i, k+1:nt-1)
                    const int64_t tag = i;
                    bcast_list.push_back({i, k, {A.sub(i, i, k+1, A_nt-1)}, tag});
                }
                A.template listBcastMT<target>(
                    bcast_list, target_layout );
            }

            // update lookahead column(s), high priority
            for (int64_t j = k+1; j < k+1+lookahead && j < A_nt; ++j) {
                #pragma omp task depend(in:diag[k]) \
                                 depend(inout:column[j]) \
                                 priority(1)
                {
                    // swap rows in A(k:mt-1, j)
                    int tag_j = j + A_mt;
                    int queue_jk1 = j-(k+1)+3;
                    internal::permuteRows<target>(
                        Direction::Forward, A.sub(k, A_mt-1, j, j), pivots.at(k),
                        target_layout, priority_1, tag_j, queue_jk1 );

                    auto Akk = A.sub(k, k, k, k);
                    auto Tkk = TriangularMatrix<scalar_t>(
                        Uplo::Lower, Diag::Unit, Akk);

                    // solve A(k, k) A(k, j) = A(k, j)
                    internal::trsm<target>(
                        Side::Left,
                        one, std::move( Tkk ), A.sub( k, k, j, j ),
                        priority_1, target_layout, queue_jk1 );

                    // send A(k, j) across column A(k+1:mt-1, j)
                    // todo: trsm still operates in ColMajor
                    A.tileBcast(
                        k, j, A.sub( k+1, A_mt-1, j, j ),
                        target_layout, tag_j );
                }

                #pragma omp task depend(in:column[k]) \
                                 depend(inout:column[j]) \
                                 priority(1)
                {
                    int queue_jk1 = j-(k+1)+3;
                    // A(k+1:mt-1, j) -= A(k+1:mt-1, k) * A(k, j)
                    internal::gemm<target>(
                        -one, A.sub( k+1, A_mt-1, k, k ),
                              A.sub( k, k, j, j ),
                        one,  A.sub( k+1, A_mt-1, j, j ),
                        host_layout, priority_1, queue_jk1 );
                }
            }

            // update trailing submatrix, normal priority
            if (k+1+lookahead < A_nt) {
                #pragma omp task depend(in:diag[k]) \
                                 depend(inout:column[k+1+lookahead]) \
                                 depend(inout:column[A_nt-1])
                {
                    // swap rows in A(k:mt-1, kl+1:nt-1)
                    int tag_kl1 = k+1+lookahead + A_mt;
                    internal::permuteRows<target>(
                        Direction::Forward, A.sub(k, A_mt-1, k+1+lookahead, A_nt-1),
                        pivots.at(k), target_layout, priority_0, tag_kl1, queue_1 );

                    auto Akk = A.sub(k, k, k, k);
                    auto Tkk =
                        TriangularMatrix<scalar_t>(Uplo::Lower, Diag::Unit, Akk);

                    // solve A(k, k) A(k, kl+1:nt-1) = A(k, kl+1:nt-1)
                    internal::trsm<target>(
                        Side::Left,
                        one, std::move( Tkk ),
                             A.sub( k, k, k+1+lookahead, A_nt-1 ),
                        priority_0, target_layout, queue_1 );
                }

                #pragma omp task depend(inout:column[k+1+lookahead]) \
                                 depend(inout:column[A_nt-1]) \
                                 depend(inout:listBcastMT_token)
                {
                    // send A(k, kl+1:A_nt-1) across A(k+1:mt-1, kl+1:nt-1)
                    BcastListTag bcast_list;
                    for (int64_t j = k+1+lookahead; j < A_nt; ++j) {
                        // send A(k, j) across column A(k+1:mt-1, j)
                        // tag must be distinct from sending left panel
                        const int64_t tag = j + A_mt;
                        bcast_list.push_back({k, j, {A.sub(k+1, A_mt-1, j, j)}, tag});
                    }
                    A.template listBcastMT<target>(
                        bcast_list, target_layout);

                }

                #pragma omp task depend(in:column[k]) \
                                 depend(inout:column[k+1+lookahead]) \
                                 depend(inout:column[A_nt-1])
                {
                    // A(k+1:mt-1, kl+1:nt-1) -= A(k+1:mt-1, k) * A(k, kl+1:nt-1)
                    internal::gemm<target>(
                        -one, A.sub( k+1, A_mt-1, k, k ),
                              A.sub( k, k, k+1+lookahead, A_nt-1 ),
                        one,  A.sub( k+1, A_mt-1, k+1+lookahead, A_nt-1 ),
                        host_layout, priority_0, queue_1 );
                }
            }
            // pivot to the left
            if (k > 0) {
                #pragma omp task depend(in:diag[k]) \
                                 depend(inout:column[0]) \
                                 depend(inout:column[k-1])
                {
                    // swap rows in A(k:mt-1, 0:k-1)
                    int tag = A_mt * 2; // permuteRows uses tag:tag+k-1 as tags
                    if (A.origin() == Target::Devices && target == Target::Devices) {
                        internal::permuteRows<Target::Devices>(
                            Direction::Forward, A.sub(k, A_mt-1, 0, k-1), pivots.at(k),
                            target_layout, priority_0, tag, queue_2 );
                    }
                    else {
                        internal::permuteRows<Target::HostTask>(
                            Direction::Forward, A.sub(k, A_mt-1, 0, k-1), pivots.at(k),
                            host_layout, priority_0, tag, queue_2 );
                    }
                }
            }

            #pragma omp task depend(inout:column[k])
            {
                auto left_panel = A.sub( k, A_mt-1, k, k );
                auto top_panel = A.sub( k, k, k+1, A_nt-1 );

                // Erase remote tiles on all devices including host
                left_panel.releaseRemoteWorkspace();
                top_panel.releaseRemoteWorkspace();

                // Update the origin tiles before their
                // workspace copies on devices are erased.
                left_panel.tileUpdateAllOrigin();
                top_panel.tileUpdateAllOrigin();

                // Erase local workspace on devices.
                left_panel.releaseLocalWorkspace();
                top_panel.releaseLocalWorkspace();
            }
            kk += A.tileNb( k );
        }
        #pragma omp taskwait

        A.tileLayoutReset();
    }
    A.clearWorkspace();
    if (target == Target::Devices) {
        for (int64_t dev = 0; dev < num_devices; ++dev) {
            blas::Queue* queue = A.comm_queue( dev );
            blas::device_free( dwork_array[dev], *queue );
            dwork_array[dev] = nullptr;
        }
    }

    internal::reduce_info( &info, A.mpiComm() );
    return info;
}

} // namespace impl

//------------------------------------------------------------------------------
/// Distributed parallel LU factorization.
///
/// Computes an LU factorization of a general m-by-n matrix $A$
/// using partial pivoting with row interchanges.
///
/// The factorization has the form
/// \[
///     A = P L U
/// \]
/// where $P$ is a permutation matrix, $L$ is lower triangular with unit
/// diagonal elements (lower trapezoidal if m > n), and $U$ is upper
/// triangular (upper trapezoidal if m < n).
///
/// This is the right-looking Level 3 BLAS version of the algorithm.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the matrix $A$ to be factored.
///     On exit, the factors $L$ and $U$ from the factorization $A = P L U$;
///     the unit diagonal elements of $L$ are not stored.
///
/// @param[out] pivots
///     The pivot indices that define the permutation matrix $P$.
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
/// @return 0: successful exit
/// @return i > 0: $U(i,i)$ is exactly zero, where $i$ is a 1-based index.
///         The factorization has been completed, but the factor $U$ is exactly
///         singular, and division by zero will occur if it is used
///         to solve a system of equations.
///
/// @ingroup gesv_computational
///
template <typename scalar_t>
int64_t getrf_tntpiv(
    Matrix<scalar_t>& A, Pivots& pivots,
    Options const& opts)
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            return impl::getrf_tntpiv<Target::HostTask>( A, pivots, opts );

        case Target::HostNest:
            return impl::getrf_tntpiv<Target::HostNest>( A, pivots, opts );

        case Target::HostBatch:
            return impl::getrf_tntpiv<Target::HostBatch>( A, pivots, opts );

        case Target::Devices:
            return impl::getrf_tntpiv<Target::Devices>( A, pivots, opts );
    }
    return -2;  // shouldn't happen
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
int64_t getrf_tntpiv<float>(
    Matrix<float>& A, Pivots& pivots,
    Options const& opts);

template
int64_t getrf_tntpiv<double>(
    Matrix<double>& A, Pivots& pivots,
    Options const& opts);

template
int64_t getrf_tntpiv< std::complex<float> >(
    Matrix< std::complex<float> >& A, Pivots& pivots,
    Options const& opts);

template
int64_t getrf_tntpiv< std::complex<double> >(
    Matrix< std::complex<double> >& A, Pivots& pivots,
    Options const& opts);

} // namespace slate
