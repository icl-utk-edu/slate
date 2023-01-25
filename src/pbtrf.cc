// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/HermitianBandMatrix.hh"
#include "slate/Tile_blas.hh"
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
void pbtrf(
    HermitianBandMatrix<scalar_t> A,
    Options const& opts )
{
    using real_t = blas::real_type<scalar_t>;
    using BcastList = typename HermitianBandMatrix<scalar_t>::BcastList;

    const scalar_t one = 1.0;
    const real_t r_one = 1.0;

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    // Options
    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );

    // if upper, change to lower
    if (A.uplo() == Uplo::Upper)
        A = conjTranspose(A);

    int64_t A_nt = A.nt();

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > column_vector(A_nt);
    uint8_t* column = column_vector.data();

    int64_t kd = A.bandwidth();

    // todo: initially, assume fixed size, square tiles for simplicity
    int64_t kdt = ceildiv( kd, A.tileNb(0) );

    #pragma omp parallel
    #pragma omp master
    for (int64_t k = 0; k < A_nt; ++k) {

        int64_t ij_end = std::min(k + kdt + 1, A_nt);

        // panel, high priority
        #pragma omp task depend(inout:column[k]) priority(1)
        {
            // factor A(k, k)
            internal::potrf<Target::HostTask>(A.sub(k, k), 1);

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
                    A.sub(k+1, ij_end-1, k, k), 1);
            }

            BcastList bcast_list_A;
            for (int64_t i = k+1; i < ij_end; ++i) {
                // send A(i, k) across row A(i, k+1:i) and
                // down col A(i:ij_end-1, i).
                bcast_list_A.push_back({i, k, {A.sub(i, i, k+1, i),
                                               A.sub(i, ij_end-1, i, i)}});
            }
            A.template listBcast(bcast_list_A, layout);
        }

        // update trailing submatrix, normal priority
        if (k+1+lookahead < ij_end) {
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[k+1+lookahead]) \
                             depend(inout:column[A_nt-1])
            {
                internal::herk<Target::HostTask>(
                    -r_one, A.sub(k+1+lookahead, ij_end-1, k, k),
                    r_one,  A.sub(k+1+lookahead, ij_end-1) );
            }
        }

        // update lookahead column(s), normal priority
        for (int64_t j = k+1; j < k+1+lookahead && j < ij_end; ++j) {
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[j])
            {
                internal::herk<Target::HostTask>(
                    -r_one, A.sub(j, j, k, k),
                    r_one,  A.sub(j, j) );

                if (j+1 <= A_nt-1) {
                    auto Ajk = A.sub(j, j, k, k);
                    internal::gemm<Target::HostTask>(
                        -one, A.sub(j+1, ij_end-1, k, k),
                              conj_transpose( Ajk ),
                        one,  A.sub(j+1, ij_end-1, j, j), layout );
                }
            }
        }
    }

    // Debug::checkTilesLives(A);
    // Debug::printTilesLives(A);
    A.tileUpdateAllOrigin();
    A.releaseWorkspace();

    // Debug::printTilesMaps(A);
}

} // namespace impl

//------------------------------------------------------------------------------
/// Distributed parallel band Cholesky factorization.
///
/// Computes the Cholesky factorization of a hermitian positive definite band
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
///     On entry, the hermitian band matrix $A$ to be factored.
///     Tiles outside the bandwidth do not need to exist.
///     For tiles that are partially outside the bandwidth,
///     data outside the bandwidth should be explicitly set to zero.
///     On exit, the factors $L$ and $U$ from the factorization $A = L U$;
///     the unit diagonal elements of $L$ are not stored.
///     The upper bandwidth is increased to accomodate fill-in of $U$.
///
/// @param[out] pivots
///     The pivot indices that define the permutations.
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
/// @ingroup pbsv_computational
///
template <typename scalar_t>
void pbtrf(
    HermitianBandMatrix<scalar_t>& A,
    Options const& opts )
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            impl::pbtrf<Target::HostTask>( A, opts );
            break;

        case Target::HostNest:
            impl::pbtrf<Target::HostNest>( A, opts );
            break;

        case Target::HostBatch:
            impl::pbtrf<Target::HostBatch>( A, opts );
            break;

        case Target::Devices:
            impl::pbtrf<Target::Devices>( A, opts );
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void pbtrf<float>(
    HermitianBandMatrix<float>& A,
    Options const& opts);

template
void pbtrf<double>(
    HermitianBandMatrix<double>& A,
    Options const& opts);

template
void pbtrf< std::complex<float> >(
    HermitianBandMatrix< std::complex<float> >& A,
    Options const& opts);

template
void pbtrf< std::complex<double> >(
    HermitianBandMatrix< std::complex<double> >& A,
    Options const& opts);

} // namespace slate
