// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "internal/internal.hh"

#include <list>
#include <tuple>

namespace slate {

namespace impl {

//------------------------------------------------------------------------------
/// @internal
/// Generic implementation for any target that uses either gemmA or gemmC
/// to compute the product A^H * A.
///
/// @ingroup geqrf_specialization
///
template <Target target, typename scalar_t>
void cholqr(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& R,
    Options const& opts )
{
    const scalar_t one  = 1.0;
    const scalar_t zero = 0.0;
    auto AH = conjTranspose( A );
    HermitianMatrix<scalar_t> R_hermitian( Uplo::Upper, R );
    auto U = TriangularMatrix<scalar_t>( Diag::NonUnit, R_hermitian );

    Method method = get_option(
        opts, Option::MethodCholQR, MethodCholQR::GemmC );

    // No call to select_algo when the method is auto.
    // Instead, we replace it by the default value above.
    if (method == MethodCholQR::Auto)
        method = MethodCholQR::GemmC;

    // Compute R = AH * A.
    switch (method) {
        case MethodCholQR::GemmA:
            gemmA( one,  AH, A, zero, R, opts );
            break;
        case MethodCholQR::GemmC:
            gemmC( one,  AH, A, zero, R, opts );
            break;
        default:
            slate_error( "CholQR unknown method" );
    }

    // Compute L * L^t = chol(R).
    potrf( R_hermitian, opts );

    // Compute Q = A * U^{-1}.
    trsm( Side::Right, one, U, A, opts );
}

//------------------------------------------------------------------------------
/// @internal
/// Generic implementation for any target that uses herkC
/// to compute the product A^H * A.
///
/// @ingroup geqrf_specialization
///
template <Target target, typename scalar_t>
void cholqr(
    Matrix<scalar_t>& A,
    HermitianMatrix<scalar_t>& R,
    Options const& opts )
{
    slate_assert( R.uplo() == Uplo::Upper );

    scalar_t s_one = 1.0;
    blas::real_type<scalar_t> one  = 1.0;
    blas::real_type<scalar_t> zero = 0.0;
    auto AH = conjTranspose( A );
    auto U = TriangularMatrix<scalar_t>( Diag::NonUnit, R );

    // Compute R = AH * A.
    herk( one, AH, zero, R, opts );

    // Compute Ut * U = chol(R).
    potrf( R, opts );

    // Compute Q = A * U^{-1}.
    trsm( Side::Right, s_one, U, A, opts );
}

} // namespace impl

//------------------------------------------------------------------------------
///
/// Select the requested function to compute A^H * A.
///
///
template <Target target, typename scalar_t>
void cholqr(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& R,
    Options const& opts )
{
    Method method = get_option(
        opts, Option::MethodCholQR, MethodCholQR::Auto );

    if (method == MethodCholQR::Auto)
        method = MethodCholQR::select_algo( A, R, opts );

    switch (method) {
        case MethodCholQR::HerkC: {
            HermitianMatrix H( Uplo::Upper, R );
            impl::cholqr<target>( A, H, opts );
            break;
        }
        case MethodCholQR::GemmA:
            /* Fallthrough */
        case MethodCholQR::GemmC:{
            Options opts2 = opts;
            opts2[ Option::MethodCholQR ] = method;
            impl::cholqr<target>( A, R, opts2 );
            break;
        }
        default:
            slate_error( "CholQR unknown method" );
            break;
    }
}

//------------------------------------------------------------------------------
/// Distributed parallel Cholesky QR factorization.
///
/// Computes a QR factorization of an m-by-n matrix $A$.
/// The factorization has the form
/// \[
///     A = QR,
/// \]
/// where $Q$ is a matrix with orthonormal columns and $R$ is upper triangular
/// (or upper trapezoidal if m < n).
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the m-by-n matrix $A$.
///     On exit, the Q matrix of the same size as A.
///
/// @param[out] R
///     On exit, the R matrix of size n x n where the upper
///     triangular part contains the Cholesky factor.
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
///     - Option::MethodCholQR:
///       Select the algorithm used to computed A^H * A:
///       - Auto:
///       - GemmA:
///       - GemmC:
///       - HerkC:
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  nested OpenMP parallel for loop on CPU host.
///       - HostBatch: batched BLAS on CPU host.
///       - Devices:   batched BLAS on GPU device.
///
/// @ingroup geqrf_computational
///
template <typename scalar_t>
void cholqr(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& R,
    Options const& opts )
{
    int64_t m = A.m();
    int64_t n = A.n();
    if (m < n) {
        slate_error( "Cholesky QR requires m >= n" );
    }

    Target target = get_option( opts, Option::Target, Target::HostTask );

    // Test whether to call hemmA instead of hemm
    switch (target) {
        case Target::Host:
          /* Fall through */
        case Target::HostTask:
            cholqr<Target::HostTask>(A, R, opts);
            break;
        case Target::HostNest:
            cholqr<Target::HostNest>(A, R, opts);
            break;
        case Target::HostBatch:
            cholqr<Target::HostBatch>(A, R, opts);
            break;
        case Target::Devices:
            cholqr<Target::Devices>(A, R, opts);
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void cholqr<float>(
    Matrix<float>& A,
    Matrix<float>& R,
    Options const& opts);

template
void cholqr<double>(
    Matrix<double>& A,
    Matrix<double>& R,
    Options const& opts);

template
void cholqr< std::complex<float> >(
    Matrix< std::complex<float> >& A,
    Matrix< std::complex<float> >& R,
    Options const& opts);

template
void cholqr< std::complex<double> >(
    Matrix< std::complex<double> >& A,
    Matrix< std::complex<double> >& R,
    Options const& opts);

} // namespace slate

