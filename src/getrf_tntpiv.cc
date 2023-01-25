// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
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
void getrf_tntpiv(
    Matrix<scalar_t>& A, Pivots& pivots,
    Options const& opts)
{
    using BcastList = typename Matrix<scalar_t>::BcastList;
    using BcastListTag = typename Matrix<scalar_t>::BcastListTag;
    using lapack::device_info_int;
    using lapack::device_pivot_int;


    const scalar_t one = 1.0;

    // Options
    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );
    int64_t ib = get_option<int64_t>( opts, Option::InnerBlocking, 16 );
    int64_t max_panel_threads  = std::max( omp_get_max_threads()/2, 1 );
    max_panel_threads = get_option<int64_t>( opts, Option::MaxPanelThreads,
                                             max_panel_threads );

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

    const int priority_one = 1;
    const int priority_zero = 0;
    int64_t A_nt = A.nt();
    int64_t A_mt = A.mt();
    int64_t min_mt_nt = std::min(A.mt(), A.nt());
    int life_factor_one = 1;
    const int queue_0 = 0;
    const int queue_1 = 1;
    const int64_t batch_size_zero = 0;
    const int num_queues = 2 + lookahead;
    bool is_shared = target == Target::Devices && lookahead > 0;
    pivots.resize(min_mt_nt);

    // setting up dummy variables for case the when target == host
    int64_t num_devices  = A.num_devices();
    int     panel_device = -1;
    size_t  work_size    = 0;

    std::vector< scalar_t* > dwork_array( num_devices, nullptr );

    if (target == Target::Devices) {
        A.allocateBatchArrays(batch_size_zero, num_queues);
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

            int64_t nb       = A.tileNb(0);
            int64_t size_A   = blas::max( 1, mlocal ) * nb;
            int64_t diag_len = blas::min( A.tileMb(0), nb );
            size_t  hsize, dsize;

            // Find size of the workspace needed
            lapack::getrf_work_size_bytes( mlocal, nb, dwork_array[0], mlocal,
                                           &dsize, &hsize, *comm_queue );

            // Size of dA, dipiv, dwork and dinfo
            work_size = size_A + diag_len + ceildiv(dsize, sizeof(scalar_t))
                        + ceildiv(sizeof(device_info_int), sizeof(scalar_t));

            for (int64_t dev = 0; dev < num_devices; ++dev) {
                lapack::Queue* queue = A.comm_queue( dev );
                dwork_array[dev] = blas::device_malloc<scalar_t>(work_size, *queue);
            }
        }
    }

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > column_vector(A_nt);
    uint8_t* column = column_vector.data();

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
        for (int64_t k = 0; k < min_mt_nt; ++k) {

            int64_t diag_len = std::min(A.tileMb(k), A.tileNb(k));
            pivots.at(k).resize(diag_len);

            // panel, high priority
            #pragma omp task depend(inout:column[k]) \
                             priority(priority_one)
            {
                auto Apanel = Awork.sub( k, A_mt-1, k, k );
                Apanel.insertLocalTiles();

                // Factor A(k:mt-1, k) using tournament pivoting to get
                // pivots and diagonal tile, Akk in workspace Apanel.
                internal::getrf_tntpiv_panel<target>(
                    A.sub(k, A_mt-1, k, k), std::move(Apanel),
                    dwork_array, work_size, diag_len, ib,
                    pivots.at(k), max_panel_threads, priority_one);

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
                    pivots.at(k), target_layout, priority_one, tag_k, queue_0);

                // Copy factored diagonal tile into place.
                internal::copy<Target::HostTask>(
                    Apanel.sub( 0, 0, 0, 0 ), A.sub( k, k, k, k ));

                // Broadcast Akk tile down column to ranks owning A_k+1:mt,k
                // and across row to ranks owning A_k,k+1:nt.
                BcastList bcast_list_A;
                bcast_list_A.push_back({k, k, {A.sub(k+1, A_mt-1, k, k),
                                               A.sub(k, k, k+1, A_nt-1)}});

                A.template listBcast<target>(
                    bcast_list_A, host_layout, tag_k, life_factor_one, is_shared);

                Apanel.clear();
            }

            // Finish computing panel using trsm below diagonal tile.
            // A_k+1:mt,k = A_k+1:mt,k * Tkk^{-1}
            #pragma omp task depend(inout:column[k]) \
                             depend(inout:listBcastMT_token) \
                             priority(priority_one)
            {
                auto Akk = A.sub(k, k, k, k);
                auto Tkk = TriangularMatrix<scalar_t>(
                    Uplo::Upper, Diag::NonUnit, Akk);

                internal::trsm<target>(
                    Side::Right,
                    one, std::move(Tkk),
                         A.sub( k+1, A_mt-1, k, k ),
                    priority_one, Layout::ColMajor, queue_0);

                BcastListTag bcast_list;
                // bcast the tiles of the panel to the right hand side
                for (int64_t i = k+1; i < A_mt; ++i) {
                    // send A(i, k) across row A(i, k+1:nt-1)
                    const int64_t tag = i;
                    bcast_list.push_back({i, k, {A.sub(i, i, k+1, A_nt-1)}, tag});
                }
                A.template listBcastMT<target>(
                    bcast_list, Layout::ColMajor, life_factor_one, is_shared);
            }

            // update lookahead column(s), high priority
            for (int64_t j = k+1; j < k+1+lookahead && j < A_nt; ++j) {
                #pragma omp task depend(in:column[k]) \
                                 depend(inout:column[j]) \
                                 priority(priority_one)
                {
                    // swap rows in A(k:mt-1, j)
                    int tag_j = j;
                    internal::permuteRows<target>(
                        Direction::Forward, A.sub(k, A_mt-1, j, j), pivots.at(k),
                        target_layout, priority_one, tag_j, j-k+1);

                    auto Akk = A.sub(k, k, k, k);
                    auto Tkk = TriangularMatrix<scalar_t>(
                        Uplo::Lower, Diag::Unit, Akk);

                    // solve A(k, k) A(k, j) = A(k, j)
                    internal::trsm<target>(
                        Side::Left,
                        one, std::move( Tkk ), A.sub( k, k, j, j ),
                        priority_one, Layout::ColMajor, j-k+1);

                    // send A(k, j) across column A(k+1:mt-1, j)
                    // todo: trsm still operates in ColMajor
                    A.tileBcast(
                        k, j, A.sub( k+1, A_mt-1, j, j ),
                        Layout::ColMajor, tag_j );

                    // A(k+1:mt-1, j) -= A(k+1:mt-1, k) * A(k, j)
                    internal::gemm<target>(
                        -one, A.sub( k+1, A_mt-1, k, k ),
                              A.sub( k, k, j, j ),
                        one,  A.sub( k+1, A_mt-1, j, j ),
                        host_layout, priority_one, j-k+1);
                }
            }

            // pivot to the left
            if (k > 0) {
                #pragma omp task depend(in:column[k]) \
                                 depend(inout:column[0]) \
                                 depend(inout:column[k-1])
                {
                    // swap rows in A(k:mt-1, 0:k-1)
                    int tag = 1 + k + A_mt * 2;
                    internal::permuteRows<Target::HostTask>(
                        Direction::Forward, A.sub(k, A_mt-1, 0, k-1), pivots.at(k),
                        host_layout, priority_zero, tag, queue_0);
                }
            }

            // update trailing submatrix, normal priority
            if (k+1+lookahead < A_nt) {
                #pragma omp task depend(in:column[k]) \
                                 depend(inout:column[k+1+lookahead]) \
                                 depend(inout:listBcastMT_token) \
                                 depend(inout:column[A_nt-1])
                {
                    // swap rows in A(k:mt-1, kl+1:nt-1)
                    int tag_kl1 = k+1+lookahead;
                    internal::permuteRows<target>(
                        Direction::Forward, A.sub(k, A_mt-1, k+1+lookahead, A_nt-1),
                        pivots.at(k), target_layout, priority_zero, tag_kl1, queue_1);

                    auto Akk = A.sub(k, k, k, k);
                    auto Tkk =
                        TriangularMatrix<scalar_t>(Uplo::Lower, Diag::Unit, Akk);

                    // solve A(k, k) A(k, kl+1:nt-1) = A(k, kl+1:nt-1)
                    internal::trsm<target>(
                        Side::Left,
                        one, std::move( Tkk ),
                             A.sub( k, k, k+1+lookahead, A_nt-1 ),
                        priority_zero, Layout::ColMajor, queue_1);

                    // send A(k, kl+1:A_nt-1) across A(k+1:mt-1, kl+1:nt-1)
                    BcastListTag bcast_list;
                    for (int64_t j = k+1+lookahead; j < A_nt; ++j) {
                        // send A(k, j) across column A(k+1:mt-1, j)
                        // tag must be distinct from sending left panel
                        const int64_t tag = j + A_mt;
                        bcast_list.push_back({k, j, {A.sub(k+1, A_mt-1, j, j)}, tag});
                    }
                    A.template listBcastMT<target>(
                        bcast_list, Layout::ColMajor);

                    // A(k+1:mt-1, kl+1:nt-1) -= A(k+1:mt-1, k) * A(k, kl+1:nt-1)
                    internal::gemm<target>(
                        -one, A.sub( k+1, A_mt-1, k, k ),
                              A.sub( k, k, k+1+lookahead, A_nt-1 ),
                        one,  A.sub( k+1, A_mt-1, k+1+lookahead, A_nt-1 ),
                        host_layout, priority_zero, queue_1);
                }
            }
            if (is_shared) {
                #pragma omp task depend(inout:column[k])
                {
                    for (int64_t i = k+1; i < A_mt; ++i) {
                        if (A.tileIsLocal(i, k)) {
                            A.tileUpdateOrigin(i, k);

                            std::set<int> dev_set;
                            A.sub(i, i, k+1, A_nt-1).getLocalDevices(&dev_set);

                            for (auto device : dev_set) {
                                A.tileUnsetHold(i, k, device);
                                A.tileRelease(i, k, device);
                            }
                        }
                    }
                }
            }
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
/// TODO: return value
/// @retval 0 successful exit
/// @retval >0 for return value = $i$, $U(i,i)$ is exactly zero. The
///         factorization has been completed, but the factor $U$ is exactly
///         singular, and division by zero will occur if it is used
///         to solve a system of equations.
///
/// @ingroup gesv_computational
///
template <typename scalar_t>
void getrf_tntpiv(
    Matrix<scalar_t>& A, Pivots& pivots,
    Options const& opts)
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            impl::getrf_tntpiv<Target::HostTask>( A, pivots, opts );
            break;

        case Target::HostNest:
            impl::getrf_tntpiv<Target::HostNest>( A, pivots, opts );
            break;

        case Target::HostBatch:
            impl::getrf_tntpiv<Target::HostBatch>( A, pivots, opts );
            break;

        case Target::Devices:
            impl::getrf_tntpiv<Target::Devices>( A, pivots, opts );
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void getrf_tntpiv<float>(
    Matrix<float>& A, Pivots& pivots,
    Options const& opts);

template
void getrf_tntpiv<double>(
    Matrix<double>& A, Pivots& pivots,
    Options const& opts);

template
void getrf_tntpiv< std::complex<float> >(
    Matrix< std::complex<float> >& A, Pivots& pivots,
    Options const& opts);

template
void getrf_tntpiv< std::complex<double> >(
    Matrix< std::complex<double> >& A, Pivots& pivots,
    Options const& opts);

} // namespace slate
