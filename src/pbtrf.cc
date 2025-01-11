// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/HermitianBandMatrix.hh"
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

namespace slate {

namespace impl {

//------------------------------------------------------------------------------
/// Distributed parallel band Cholesky factorization.
/// Generic implementation for any target.
/// Panel and lookahead computed on host using Host OpenMP task.
///
/// Warning: ColMajor layout is assumed
///
template <Target target, typename scalar_t>
int64_t pbtrf(
    HermitianBandMatrix<scalar_t> A,
    Options const& opts )
{
    using real_t = blas::real_type<scalar_t>;
    using BcastList = typename HermitianBandMatrix<scalar_t>::BcastList;

    const scalar_t one = 1.0;
    const real_t r_one = 1.0;
    const int priority_0 = 0;
    const int priority_1 = 1;
    const int queue_0 = 0;

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    // Options
    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );

    // if upper, change to lower
    if (A.uplo() == Uplo::Upper)
        A = conj_transpose( A );

    int64_t info = 0;
    int64_t A_nt = A.nt();

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > column_vector(A_nt);
    uint8_t* column = column_vector.data();
    SLATE_UNUSED( column ); // Used only by OpenMP

    int64_t kd = A.bandwidth();

    // todo: initially, assume fixed size, square tiles for simplicity
    int64_t kdt = ceildiv( kd, A.tileNb(0) );

    #pragma omp parallel
    #pragma omp master
    {
        int64_t kk = 0;  // column index (not block-column)
        for (int64_t k = 0; k < A_nt; ++k) {

            int64_t ij_end = std::min(k + kdt + 1, A_nt);

            // panel, high priority
            #pragma omp task depend(inout:column[k]) priority(1) \
                shared( info )
            {
                // factor A(k, k)
                int64_t iinfo = internal::potrf<Target::HostTask>( A.sub( k, k ), 1 );
                if (iinfo != 0 && info == 0)
                    info = kk + iinfo;

                // send A(k, k) down col A( k+1:ij_end-1, k )
                if (k+1 < ij_end)
                    A.tileBcast(k, k, A.sub(k+1, ij_end-1, k, k), layout);

                // A(k+1:ij_end-1, k) * A(k, k)^{-H}
                if (k+1 < ij_end) {
                    auto Akk = A.sub(k, k);
                    auto Tkk = TriangularMatrix< scalar_t >(Diag::NonUnit, Akk);
                    internal::trsm<Target::HostTask>(
                        Side::Right,
                        one, conj_transpose( Tkk ),
                        A.sub(k+1, ij_end-1, k, k),
                        priority_1, layout );
                }

                BcastList bcast_list_A;
                for (int64_t i = k+1; i < ij_end; ++i) {
                    // send A(i, k) across row A(i, k+1:i) and
                    // down col A(i:ij_end-1, i).
                    bcast_list_A.push_back({i, k, {A.sub(i, i, k+1, i),
                                                   A.sub(i, ij_end-1, i, i)}});
                }
                A.template listBcast<>( bcast_list_A, layout );
            }

            // update trailing submatrix, normal priority
            if (k+1+lookahead < ij_end) {
                #pragma omp task depend(in:column[k]) \
                                 depend(inout:column[k+1+lookahead]) \
                                 depend(inout:column[A_nt-1])
                {
                    internal::herk<Target::HostTask>(
                        -r_one, A.sub(k+1+lookahead, ij_end-1, k, k),
                        r_one,  A.sub(k+1+lookahead, ij_end-1),
                        priority_0, queue_0, layout );
                }
            }

            // update lookahead column(s), normal priority
            for (int64_t j = k+1; j < k+1+lookahead && j < ij_end; ++j) {
                #pragma omp task depend(in:column[k]) \
                                 depend(inout:column[j])
                {
                    internal::herk<Target::HostTask>(
                        -r_one, A.sub(j, j, k, k),
                        r_one,  A.sub(j, j),
                        priority_0, queue_0, layout );

                    if (j+1 <= A_nt-1) {
                        auto Ajk = A.sub(j, j, k, k);
                        internal::gemm<Target::HostTask>(
                            -one, A.sub(j+1, ij_end-1, k, k),
                                  conj_transpose( Ajk ),
                            one,  A.sub(j+1, ij_end-1, j, j),
                            layout, priority_1 );
                    }
                }
            }

            #pragma omp task depend(inout:column[k])
            {
                auto panel = A.sub( k, ij_end-1, k, k );

                // Erase remote tiles on all devices, including host
                panel.releaseRemoteWorkspace();

                // Update the origin tiles before their
                // workspace copies on devices are erased.
                panel.tileUpdateAllOrigin();

                // Erase local workspace on devices
                panel.releaseLocalWorkspace();
            }

            kk += A.tileNb( k );
        }
    }

    // Debug::checkTilesLives(A);
    // Debug::printTilesLives(A);
    A.tileUpdateAllOrigin();
    A.releaseWorkspace();

    // Debug::printTilesMaps(A);

    internal::reduce_info( &info, A.mpiComm() );
    return info;
}

} // namespace impl

//------------------------------------------------------------------------------
/// Distributed parallel band Cholesky factorization.
///
/// Computes the Cholesky factorization of a Hermitian positive definite band
/// matrix $A$.
///
/// The factorization has the form
/// \[
///     A = L L^H,
/// \]
/// if $A$ is stored lower, where $L$ is a lower triangular band matrix, or
/// \[
///     A = U^H U,
/// \]
/// if $A$ is stored upper, where $U$ is an upper triangular band matrix.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the Hermitian band matrix $A$ to be factored.
///     Tiles outside the bandwidth do not need to exist.
///     For tiles that are partially outside the bandwidth,
///     data outside the bandwidth should be explicitly set to zero.
///     On exit, the factor $L$ or $U$ from the factorization
///     $A = L L^H$ or $A = U^H U$.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::Lookahead:
///       Number of panels to overlap with matrix updates.
///       lookahead >= 0. Default 1.
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  nested OpenMP parallel for loop on CPU host.
///       - HostBatch: batched BLAS on CPU host.
///       - Devices:   batched BLAS on GPU device.
///
/// @return 0: successful exit
/// @return i > 0: the leading minor of order $i$ of $A$ is not
///         positive definite, so the factorization could not
///         be completed.
///
/// @ingroup pbsv_computational
///
template <typename scalar_t>
int64_t pbtrf(
    HermitianBandMatrix<scalar_t>& A,
    Options const& opts )
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            return impl::pbtrf<Target::HostTask>( A, opts );

        case Target::HostNest:
            return impl::pbtrf<Target::HostNest>( A, opts );

        case Target::HostBatch:
            return impl::pbtrf<Target::HostBatch>( A, opts );

        case Target::Devices:
            return impl::pbtrf<Target::Devices>( A, opts );
    }
    return -2;  // shouldn't happen
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
int64_t pbtrf<float>(
    HermitianBandMatrix<float>& A,
    Options const& opts);

template
int64_t pbtrf<double>(
    HermitianBandMatrix<double>& A,
    Options const& opts);

template
int64_t pbtrf< std::complex<float> >(
    HermitianBandMatrix< std::complex<float> >& A,
    Options const& opts);

template
int64_t pbtrf< std::complex<double> >(
    HermitianBandMatrix< std::complex<double> >& A,
    Options const& opts);

} // namespace slate
