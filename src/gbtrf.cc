// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/BandMatrix.hh"
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

namespace slate {

namespace impl {

//------------------------------------------------------------------------------
/// Distributed parallel band LU factorization.
/// Generic implementation for any target.
/// Panel and lookahead computed on host using Host OpenMP task.
///
/// Warning: ColMajor layout is assumed
///
template <Target target, typename scalar_t>
int64_t gbtrf(
    BandMatrix<scalar_t>& A, Pivots& pivots,
    Options const& opts )
{
    using real_t = blas::real_type<scalar_t>;
    using BcastList = typename BandMatrix<scalar_t>::BcastList;

    // Constants
    const scalar_t zero = 0.0;
    const scalar_t one = 1.0;
    const int priority_0 = 0;
    const int priority_1 = 1;
    const int tag_0 = 0;
    // Assumes column major
    const Layout layout = Layout::ColMajor;

    // Options
    real_t pivot_threshold
        = get_option<double>( opts, Option::PivotThreshold, 1.0 );
    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );
    int64_t ib = get_option<int64_t>( opts, Option::InnerBlocking, 16 );
    int64_t max_panel_threads  = std::max(omp_get_max_threads()/2, 1);
    max_panel_threads = get_option<int64_t>( opts, Option::MaxPanelThreads,
                                             max_panel_threads );

    int64_t info = 0;
    int64_t A_nt = A.nt();
    int64_t A_mt = A.mt();
    int64_t min_mt_nt = std::min(A.mt(), A.nt());
    pivots.resize(min_mt_nt);

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > column_vector(A_nt);
    uint8_t* column = column_vector.data();
    SLATE_UNUSED( column ); // Used only by OpenMP

    int64_t kl = A.lowerBandwidth();
    int64_t ku = A.upperBandwidth();

    // todo: initially, assume fixed size, square tiles for simplicity
    int64_t klt = ceildiv( kl, A.tileNb(0) );
    int64_t kut = ceildiv( ku, A.tileNb(0) );
    int64_t ku2t = ceildiv( (ku + kl), A.tileNb(0) );

    // Insert & zero potential fill above upper bandwidth
    A.upperBandwidth(kl + ku);
    //printf( "kl %lld, ku %lld, kl+ku %lld, A.lowerBW %lld, A.upperBW %lld\n",
    //        kl, ku, kl + ku, A.lowerBandwidth(), A.upperBandwidth() );
    for (int64_t i = 0; i < min_mt_nt; ++i) {
        for (int64_t j = i + 1 + kut; j < std::min(i + 1 + ku2t, A.nt()); ++j) {
            if (A.tileIsLocal(i, j)) {
                // todo: device?
                A.tileInsert(i, j);
                auto T = A(i, j);
                lapack::laset(lapack::MatrixType::General, T.mb(), T.nb(),
                              zero, zero, T.data(), T.stride());
                A.tileModified(i, j);
            }
        }
    }

    // set min number for omp nested active parallel regions
    slate::OmpSetMaxActiveLevels set_active_levels( MinOmpActiveLevels );

    #pragma omp parallel
    #pragma omp master
    {
        for (int64_t k = 0; k < min_mt_nt; ++k) {

            int64_t diag_len = std::min(A.tileMb(k), A.tileNb(k));
            pivots.at(k).resize(diag_len);

            // A( k:i_end-1, k ) is the panel
            // A( k, k+1:j_end-1 ) is the trsm
            // A( k+1:i_end-1, k+1:j_end-1 ) is the gemm
            // Compared to getrf, i_end replaces A_mt, j_end replaces A_nt.
            // "end" in the usual C++ sense of element after the last element.
            int64_t i_end = std::min(k + klt + 1, A_mt);
            int64_t j_end = std::min(k + ku2t + 1, A_nt);

            // panel, high priority
            #pragma omp task depend(inout:column[k]) priority(1)
            {
                // factor A(k:mt-1, k)
                internal::getrf_panel<Target::HostTask>(
                    A.sub(k, i_end-1, k, k), diag_len, ib, pivots.at(k),
                    pivot_threshold, max_panel_threads, priority_1, tag_0, &info );

                BcastList bcast_list_A;
                int tag_k = k;
                for (int64_t i = k; i < i_end; ++i) {
                    // send A(i, k) across row A(i, k+1:nt-1)
                    bcast_list_A.push_back({i, k, {A.sub(i, i, k+1, j_end-1)}});
                }
                A.template listBcast<>( bcast_list_A, layout, tag_k );

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
            for (int64_t j = k+1; j < k+1+lookahead && j < j_end; ++j) {
                #pragma omp task depend(in:column[k]) \
                                 depend(inout:column[j]) priority(1)
                {
                    // swap rows in A(k:mt-1, j)
                    int tag_j = j;
                    internal::permuteRows<Target::HostTask>(
                        Direction::Forward, A.sub(k, i_end-1, j, j), pivots.at(k),
                        layout, priority_1, tag_j );

                    auto Akk = A.sub(k, k, k, k);
                    auto Tkk =
                        TriangularMatrix<scalar_t>(Uplo::Lower, Diag::Unit, Akk);

                    // solve A(k, k) A(k, j) = A(k, j)
                    internal::trsm<Target::HostTask>(
                        Side::Left,
                        one, std::move( Tkk ), A.sub(k, k, j, j),
                        priority_1, layout );

                    // send A(k, j) across column A(k+1:mt-1, j)
                    A.tileBcast(k, j, A.sub(k+1, i_end-1, j, j), layout, tag_j);

                    // A(k+1:mt-1, j) -= A(k+1:mt-1, k) * A(k, j)
                    internal::gemm<Target::HostTask>(
                        -one, A.sub(k+1, i_end-1, k, k),
                              A.sub(k, k, j, j),
                        one,  A.sub(k+1, i_end-1, j, j),
                        layout, priority_1 );
                }
            }
            // Update trailing submatrix, normal priority.
            // Depends on the whole range k+1+lookahead : A_nt-1,
            // not just to j_end-1, as the dependencies daisy chain on A_nt-1.
            if (k+1+lookahead < j_end) {
                #pragma omp task depend(in:column[k]) \
                                 depend(inout:column[k+1+lookahead]) \
                                 depend(inout:column[A_nt-1])
                {
                    // swap rows in A(k:mt-1, kl+1:nt-1)
                    int tag_kl1 = k+1+lookahead;
                    internal::permuteRows<Target::HostTask>(
                        Direction::Forward, A.sub(k, i_end-1, k+1+lookahead, j_end-1),
                        pivots.at(k), layout, priority_0, tag_kl1 );

                    auto Akk = A.sub(k, k, k, k);
                    auto Tkk =
                        TriangularMatrix<scalar_t>(Uplo::Lower, Diag::Unit, Akk);

                    // solve A(k, k) A(k, kl+1:nt-1) = A(k, kl+1:nt-1)
                    internal::trsm<Target::HostTask>(
                        Side::Left,
                        one, std::move( Tkk ),
                             A.sub(k, k, k+1+lookahead, j_end-1),
                        priority_0, layout );

                    // send A(k, kl+1:j_end-1) across A(k+1:mt-1, kl+1:nt-1)
                    BcastList bcast_list_A;
                    for (int64_t j = k+1+lookahead; j < j_end; ++j) {
                        // send A(k, j) across column A(k+1:mt-1, j)
                        bcast_list_A.push_back({k, j, {A.sub(k+1, i_end-1, j, j)}});
                    }
                    A.template listBcast<>( bcast_list_A, layout, tag_kl1 );

                    // A(k+1:mt-1, kl+1:nt-1) -= A(k+1:mt-1, k) * A(k, kl+1:nt-1)
                    internal::gemm<Target::HostTask>(
                        -one, A.sub(k+1, i_end-1, k, k),
                              A.sub(k, k, k+1+lookahead, j_end-1),
                        one,  A.sub(k+1, i_end-1, k+1+lookahead, j_end-1),
                        layout, priority_0 );
                }
            }

            #pragma omp task depend(inout:column[k])
            {
                auto left_panel = A.sub( k, i_end-1, k, k );
                auto top_panel = A.sub( k, k, k+1, j_end-1 );

                // Erase remote tiles on all devices, including host
                left_panel.releaseRemoteWorkspace();
                top_panel.releaseRemoteWorkspace();

                // Update the origin tiles before their
                // workspace copies on devices are erased.
                left_panel.tileUpdateAllOrigin();
                top_panel.tileUpdateAllOrigin();

                // Erase local workspace on devices
                left_panel.releaseLocalWorkspace();
                top_panel.releaseLocalWorkspace();
            }
        }

        #pragma omp taskwait
        A.tileUpdateAllOrigin();
    }
    // Band LU does NOT pivot to the left of the panel, since it would
    // introduce fill in the lower triangle. Instead, pivoting is done
    // during the solve (gbtrs).

    A.releaseWorkspace();

    internal::reduce_info( &info, A.mpiComm() );
    return info;
}

} // namespace impl

//------------------------------------------------------------------------------
/// Distributed parallel band LU factorization.
///
/// Computes an LU factorization of a general band m-by-n matrix $A$
/// using partial pivoting with row interchanges.
///
/// The factorization has the form
/// \[
///     A = L U
/// \]
/// where $L$ is a product of permutation and unit lower triangular matrices,
/// and $U$ is upper triangular.
///
/// This is the right-looking Level 3 BLAS version of the algorithm.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the band matrix $A$ to be factored.
///     Tiles outside the bandwidth do not need to exist.
///     For tiles that are partially outside the bandwidth,
///     data outside the bandwidth should be explicitly set to zero.
///     On exit, the factors $L$ and $U$ from the factorization $A = L U$;
///     the unit diagonal elements of $L$ are not stored.
///     The upper bandwidth is increased to accommodate fill-in of $U$.
///
/// @param[out] pivots
///     The pivot indices that define the permutations.
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
///    - Option::PivotThreshold:
///      Strictness of the pivot selection.  Between 0 and 1 with 1 giving
///      partial pivoting and 0 giving no pivoting.  Default 1.
///
/// @return 0: successful exit
/// @return i > 0: $U(i,i)$ is exactly zero, where $i$ is a 1-based index.
///         The factorization has been completed, but the factor $U$ is exactly
///         singular, and division by zero will occur if it is used
///         to solve a system of equations.
///
/// @ingroup gbsv_computational
///
template <typename scalar_t>
int64_t gbtrf(
    BandMatrix<scalar_t>& A, Pivots& pivots,
    Options const& opts )
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            return impl::gbtrf<Target::HostTask>( A, pivots, opts );

        case Target::HostNest:
            return impl::gbtrf<Target::HostNest>( A, pivots, opts );

        case Target::HostBatch:
            return impl::gbtrf<Target::HostBatch>( A, pivots, opts );

        case Target::Devices:
            return impl::gbtrf<Target::Devices>( A, pivots, opts );
    }
    return -3;  // shouldn't happen
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
int64_t gbtrf<float>(
    BandMatrix<float>& A, Pivots& pivots,
    Options const& opts);

template
int64_t gbtrf<double>(
    BandMatrix<double>& A, Pivots& pivots,
    Options const& opts);

template
int64_t gbtrf< std::complex<float> >(
    BandMatrix< std::complex<float> >& A, Pivots& pivots,
    Options const& opts);

template
int64_t gbtrf< std::complex<double> >(
    BandMatrix< std::complex<double> >& A, Pivots& pivots,
    Options const& opts);

} // namespace slate
