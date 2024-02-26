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
/// Distributed parallel LU factorization.
/// Generic implementation for any target.
/// Panel and lookahead computed on host using Host OpenMP task.
/// @ingroup gesv_impl
///
template <Target target, typename scalar_t>
int64_t getrf(
    Matrix<scalar_t>& A, Pivots& pivots,
    Options const& opts )
{
    using real_t = blas::real_type<scalar_t>;
    using BcastList = typename Matrix<scalar_t>::BcastList;

    // Constants
    const scalar_t one = 1.0;
    const int priority_0 = 0;
    const int priority_1 = 1;
    const int queue_0 = 0;
    const int queue_1 = 1;

    // Options
    real_t pivot_threshold = get_option<Option::PivotThreshold>( opts, 1.0 );
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

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > column_vector(A_nt);
    uint8_t* column = column_vector.data();
    SLATE_UNUSED( column ); // Used only by OpenMP

    // Communication of the jth tile column uses the MPI tag j
    // So, the data dependencies protect the corresponding MPI tags

    if (target == Target::Devices) {
        const int64_t batch_size_default = 0;
        int num_queues = 2 + lookahead;
        A.allocateBatchArrays( batch_size_default, num_queues );
        A.reserveDeviceWorkspace();
    }

    // set min number for omp nested active parallel regions
    slate::OmpSetMaxActiveLevels set_active_levels( MinOmpActiveLevels );

    #pragma omp parallel
    #pragma omp master
    {
        int64_t kk = 0;  // column index (not block-column)
        for (int64_t k = 0; k < min_mt_nt; ++k) {

            int64_t diag_len = std::min(A.tileMb(k), A.tileNb(k));
            pivots.at(k).resize(diag_len);

            // panel, high priority
            #pragma omp task depend(inout:column[k]) priority(1)
            {
                // factor A(k:mt-1, k)
                int64_t iinfo;
                internal::getrf_panel<Target::HostTask>(
                    A.sub(k, A_mt-1, k, k), diag_len, ib, pivots.at(k),
                    pivot_threshold, max_panel_threads, priority_1, k, &iinfo );
                if (info == 0 && iinfo > 0)
                    info = kk + iinfo;

                BcastList bcast_list_A;
                int tag_k = k;
                for (int64_t i = k; i < A_mt; ++i) {
                    // send A(i, k) across row A(i, k+1:nt-1)
                    bcast_list_A.push_back({i, k, {A.sub(i, i, k+1, A_nt-1)}});
                }
                A.template listBcast<target>(
                    bcast_list_A, target_layout, tag_k );

                // Root broadcasts the pivot to all ranks.
                // todo: Panel ranks send the pivots to the right.
                {
                    trace::Block trace_block("MPI_Bcast");

                    MPI_Bcast(pivots.at(k).data(),
                              sizeof(Pivot)*pivots.at(k).size(),
                              MPI_BYTE, A.tileRank(k, k), A.mpiComm());
                }
            }
            // update lookahead column(s), high priority
            for (int64_t j = k+1; j < k+1+lookahead && j < A_nt; ++j) {
                #pragma omp task depend(in:column[k]) \
                                 depend(inout:column[j]) priority(1)
                {
                    // swap rows in A(k:mt-1, j)
                    int tag_j = j;
                    int queue_jk1 = j-k+1;
                    internal::permuteRows<target>(
                        Direction::Forward, A.sub(k, A_mt-1, j, j), pivots.at(k),
                        target_layout, priority_1, tag_j, queue_jk1 );

                    auto Akk = A.sub(k, k, k, k);
                    auto Tkk =
                        TriangularMatrix<scalar_t>(Uplo::Lower, Diag::Unit, Akk);

                    // solve A(k, k) A(k, j) = A(k, j)
                    internal::trsm<target>(
                        Side::Left,
                        one, std::move( Tkk ), A.sub(k, k, j, j),
                        priority_1, target_layout, queue_jk1 );

                    // send A(k, j) across column A(k+1:mt-1, j)
                    // todo: trsm still operates in ColMajor
                    A.tileBcast(k, j, A.sub(k+1, A_mt-1, j, j), Layout::ColMajor, tag_j);

                    // A(k+1:mt-1, j) -= A(k+1:mt-1, k) * A(k, j)
                    internal::gemm<target>(
                        -one, A.sub(k+1, A_mt-1, k, k),
                              A.sub(k, k, j, j),
                        one,  A.sub(k+1, A_mt-1, j, j),
                        target_layout, priority_1, queue_jk1 );
                }
            }
            // pivot to the left
            if (k > 0) {
                #pragma omp task depend(in:column[k]) \
                                 depend(inout:column[0]) \
                                 depend(inout:column[k-1])
                {
                    // swap rows in A(k:mt-1, 0:k-1)
                    const int tag_0 = 0;
                    if (A.origin() == Target::Devices && target == Target::Devices) {
                        internal::permuteRows<Target::Devices>(
                            Direction::Forward, A.sub(k, A_mt-1, 0, k-1), pivots.at(k),
                            target_layout, priority_0, tag_0, queue_0 );
                    }
                    else {
                        internal::permuteRows<Target::HostTask>(
                            Direction::Forward, A.sub(k, A_mt-1, 0, k-1), pivots.at(k),
                            host_layout, priority_0, tag_0, queue_0 );
                    }
                }
            }
            // update trailing submatrix, normal priority
            if (k+1+lookahead < A_nt) {
                #pragma omp task depend(in:column[k]) \
                                 depend(inout:column[k+1+lookahead]) \
                                 depend(inout:column[A_nt-1])
                {
                    // swap rows in A(k:mt-1, kl+1:nt-1)
                    int tag_kl1 = k+1+lookahead;
                    // todo: target
                    internal::permuteRows<target>(
                        Direction::Forward, A.sub(k, A_mt-1, k+1+lookahead, A_nt-1),
                        pivots.at(k), target_layout, priority_0, tag_kl1, queue_1 );

                    auto Akk = A.sub(k, k, k, k);
                    auto Tkk =
                        TriangularMatrix<scalar_t>(Uplo::Lower, Diag::Unit, Akk);

                    // solve A(k, k) A(k, kl+1:nt-1) = A(k, kl+1:nt-1)
                    // todo: target
                    internal::trsm<target>(
                        Side::Left,
                        one, std::move( Tkk ),
                             A.sub(k, k, k+1+lookahead, A_nt-1),
                        priority_0, target_layout, queue_1 );

                    // send A(k, kl+1:A_nt-1) across A(k+1:mt-1, kl+1:nt-1)
                    BcastList bcast_list_A;
                    for (int64_t j = k+1+lookahead; j < A_nt; ++j) {
                        // send A(k, j) across column A(k+1:mt-1, j)
                        bcast_list_A.push_back({k, j, {A.sub(k+1, A_mt-1, j, j)}});
                    }
                    A.template listBcast<target>(
                        bcast_list_A, target_layout, tag_kl1);

                    // A(k+1:mt-1, kl+1:nt-1) -= A(k+1:mt-1, k) * A(k, kl+1:nt-1)
                    internal::gemm<target>(
                        -one, A.sub(k+1, A_mt-1, k, k),
                              A.sub(k, k, k+1+lookahead, A_nt-1),
                        one,  A.sub(k+1, A_mt-1, k+1+lookahead, A_nt-1),
                        target_layout, priority_0, queue_1 );
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
/// Complexity (in real):
/// - $\approx m n^{2} - \frac{1}{3} n^{3}$ flops;
/// - $\approx \frac{2}{3} n^{3}$ flops for $m = n$.
/// .
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
///
///     - Option::Lookahead:
///       Number of panels to overlap with matrix updates.
///       lookahead >= 0. Default 1.
///
///     - Option::InnerBlocking:
///       Inner blocking to use for panel. Default 16.
///
///     - Option::MaxPanelThreads:
///       Number of threads to use for panel. Default omp_get_max_threads()/2.
///
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  nested OpenMP parallel for loop on CPU host.
///       - HostBatch: batched BLAS on CPU host.
///       - Devices:   batched BLAS on GPU device.
///
///     - Option::PivotThreshold:
///       Strictness of the pivot selection.  Between 0 and 1 with 1 giving
///       partial pivoting and 0 giving no pivoting.  Default 1.
///
///     - Option::MethodLU:
///       Algorithm for LU factorization.
///       - MethodLU::PartialPiv: partial pivoting [default].
///       - MethodLU::CALU: communication avoiding (tournament pivoting).
///       - MethodLU::NoPiv: no pivoting.
///         Note pivots vector is currently ignored for NoPiv.
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
int64_t getrf(
    Matrix<scalar_t>& A, Pivots& pivots,
    Options const& opts )
{
    Method method = get_option<Option::MethodLU>( opts, MethodLU::PartialPiv );

    // todo: info for tntpiv, nopiv
    if (method == MethodLU::CALU) {
        return getrf_tntpiv( A, pivots, opts );
    }
    else if (method == MethodLU::NoPiv) {
        // todo: fill in pivots vector?
        return getrf_nopiv( A, opts );
    }
    else if (method == MethodLU::PartialPiv) {
        Target target = get_option<Option::Target>( opts, Target::HostTask );

        switch (target) {
            case Target::Host:
            case Target::HostTask:
                return impl::getrf<Target::HostTask>( A, pivots, opts );

            case Target::HostNest:
                return impl::getrf<Target::HostNest>( A, pivots, opts );

            case Target::HostBatch:
                return impl::getrf<Target::HostBatch>( A, pivots, opts );

            case Target::Devices:
                return impl::getrf<Target::Devices>( A, pivots, opts );
        }
    }
    else {
        throw Exception( "unknown value for MethodLU" );
    }
    return -3;  // shouldn't happen
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
int64_t getrf<float>(
    Matrix<float>& A, Pivots& pivots,
    Options const& opts);

template
int64_t getrf<double>(
    Matrix<double>& A, Pivots& pivots,
    Options const& opts);

template
int64_t getrf< std::complex<float> >(
    Matrix< std::complex<float> >& A, Pivots& pivots,
    Options const& opts);

template
int64_t getrf< std::complex<double> >(
    Matrix< std::complex<double> >& A, Pivots& pivots,
    Options const& opts);

} // namespace slate
