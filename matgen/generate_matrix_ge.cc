// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "slate/generate_matrix.hh"
#include "random.hh"
#include "generate_matrix_utils.hh"
#include "generate_type_geev.hh"
#include "generate_type_heev.hh"
#include "generate_type_rand.hh"
#include "generate_sigma.hh"
#include "generate_type_svd.hh"

#include <exception>
#include <string>
#include <vector>
#include <limits>
#include <complex>
#include <chrono>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace slate {

//------------------------------------------------------------------------------
/// Generates an m-by-n general-storage test matrix.
/// Handles Matrix class.
/// @see generate_matrix
/// @ingroup generate_matrix
///
template <typename scalar_t>
void generate_matrix(
    MatgenParams& params,
    slate::Matrix<scalar_t>& A,
    std::vector< blas::real_type<scalar_t> >& Sigma,
    slate::Options const& opts)
{
    using entry_type = std::function< scalar_t (int64_t, int64_t) >;
    using real_t = blas::real_type<scalar_t>;

    // Constants
    const real_t nan = std::numeric_limits<real_t>::quiet_NaN();
    const real_t r_zero = 0;
    const real_t r_one  = 1;
    const scalar_t zero = 0;
    const scalar_t one  = 1;
    const real_t pi = 3.1415926535897932385;

    // ----------
    // set Sigma to unknown (nan)
    lapack::laset( lapack::MatrixType::General, Sigma.size(), 1,
                   nan, nan, Sigma.data(), Sigma.size() );

    char msg[ 256 ];
    TestMatrixType type;
    TestMatrixDist dist;
    real_t cond;
    real_t condD;
    real_t sigma_max;
    bool dominant;
    int64_t zero_col;
    decode_matrix<scalar_t>(
        params, A, type, dist, cond, condD, sigma_max, dominant, zero_col );

    int64_t seed = configure_seed(A.mpiComm(), params.seed);

    int64_t n = A.n();
    int64_t m = A.m();
    int64_t nt = A.nt();
    int64_t mt = A.mt();

    // ----- generate matrix
    switch (type) {
        case TestMatrixType::zeros:
            set(zero, zero, A);
            lapack::laset( lapack::MatrixType::General, Sigma.size(), 1,
                           r_zero, r_zero, Sigma.data(), Sigma.size() );
            break;

        case TestMatrixType::ones:
            set(one, one, A);
            lapack::laset( lapack::MatrixType::General, Sigma.size(), 1,
                           r_zero, r_one, Sigma.data(), Sigma.size() );
            if (Sigma.size() >= 1) {
                Sigma[0] = sqrt(m*n);
            }
            break;

        case TestMatrixType::identity:
            set(zero, one, A);
            lapack::laset( lapack::MatrixType::General, Sigma.size(), 1,
                           r_one, r_one, Sigma.data(), Sigma.size() );
            break;

        case TestMatrixType::ij: {
            // Scale so j*s < 1.
            real_t s = 1 / pow( 10, ceil( log10( n ) ) );
            entry_type ij_entry = [s]( int64_t i, int64_t j ) {
                return i + j * s;
            };
            set( ij_entry, A, opts );
            break;
        }

        case TestMatrixType::jordan: {
            // Jordan matrix: diag and superdiag are ones.
            entry_type jordan_entry = []( int64_t i, int64_t j ) {
                return (i == j || i + 1 == j ? 1.0 : 0.0);
            };
            set( jordan_entry, A, opts );
            break;
        }

        case TestMatrixType::jordanT: {
            // transposed Jordan matrix: diag and subdiag are ones.
            entry_type jordan_entry = []( int64_t i, int64_t j ) {
                return (i == j || i - 1 == j ? 1.0 : 0.0);
            };
            set( jordan_entry, A, opts );
            break;
        }

        case TestMatrixType::chebspec: {
            const int64_t max_mn = std::max(m, n);
            entry_type chebspec_entry = [ max_mn, one, pi ]( int64_t i, int64_t j ) {
                scalar_t x_i = std::cos( pi * ( i + 1 ) / max_mn );
                scalar_t x_j = std::cos( pi * ( j + 1 ) / max_mn );

                if (j != i) {
                    scalar_t c_i = i == max_mn - 1 ? 2 : 1;
                    scalar_t c_j = j == max_mn - 1 ? 2 : 1;
                    scalar_t sgn = ( j + i ) % 2 == 0 ? 1 : -1; // (-1)^(i+j)

                    return sgn * c_i / ( c_j * (x_j - x_i ) );
                }
                else if (j + 1 == max_mn) {
                    return scalar_t( 2 * max_mn * max_mn + 1 ) / scalar_t(-6.0);
                }
                else {
                    return scalar_t(-0.5) * x_i / ( one - x_i * x_i );
                }
            };
            set( chebspec_entry, A, opts );
            break;
        }

        // circulant matrix for the vector 1:n
        case TestMatrixType::circul: {
            const int64_t max_mn = std::max(n, m);
            entry_type circul_entry = [max_mn]( int64_t i, int64_t j ) {
                auto diff = j - i;
                return diff + (diff < 0 ? max_mn : 0) + 1;
            };
            set( circul_entry, A, opts );
            break;
        }

        case TestMatrixType::fiedler: {
            entry_type fiedler_entry=[]( int64_t i, int64_t j ) {
                return std::abs(j - i);
            };
            set( fiedler_entry, A, opts );
            break;
        }

        case TestMatrixType::gfpp: {
            int64_t n_1 = A.n() - 1;
            entry_type gfpp_entry = [n_1](  int64_t i, int64_t j) {
                if (j == n_1) { // last column
                    return 1.0;
                }
                else if (i > j) { // below the diagonal
                    return -1.0;
                }
                else if (i == j) { // diagonal
                    return 0.5;
                }
                else { // above the diagonal
                    return 0.0;
                }
            };
            set( gfpp_entry, A, opts );
            break;
        }

        case TestMatrixType::kms: {
            const double rho = 0.5;
            entry_type kms_entry = [rho]( int64_t i, int64_t j ) {
                return std::pow( rho, std::abs(j - i));
            };
            set( kms_entry, A, opts );
            break;
        }

        case TestMatrixType::orthog: {
            const int64_t max_mn = std::max(n, m);
            const scalar_t outer_const = sqrt(scalar_t(2)/scalar_t(max_mn+1));
            const scalar_t inner_const = pi/scalar_t(max_mn+1);

            entry_type orthog_entry = [ outer_const, inner_const ]( int64_t i, int64_t j) {
                scalar_t a = scalar_t(i+1) * scalar_t(j+1) * inner_const;
                return outer_const * sin(a);
            };
            set( orthog_entry, A, opts );
            break;
        }

        case TestMatrixType::riemann: {
            entry_type riemann_entry = []( int64_t i, int64_t j ) {
                auto B_i = i + 2;
                auto B_j = j + 2;
                if (B_j % B_i == 0) {
                    return B_j - 1;
                }
                else {
                    return int64_t( -1 );
                }
            };
            set( riemann_entry, A, opts );
            break;
        }

        case TestMatrixType::ris: {
            const int64_t max_mn = std::max(n, m);
            entry_type ris_entry = [max_mn]( int64_t i, int64_t j ) {
                // n-(j + 1)-(i + 1)+1.5 = n-j-i-0.5
                return 0.5 / ( max_mn - j - i - 0.5);
            };
            set( ris_entry, A, opts );
            break;
        }

        case TestMatrixType::zielkeNS: {
            const int64_t max_mn = std::max(n, m);
            const scalar_t a = 0.0;
            entry_type zielkeNS_entry = [ max_mn, a, one ]( int64_t i, int64_t j ) {
                if (j < i) {
                    return a + one;
                }
                else if (j+1 == max_mn && i == 0) {
                    return a - one;
                }
                else {
                    return a;
                }
            };
            set( zielkeNS_entry, A, opts );
            break;
        }

        case TestMatrixType::rand:
        case TestMatrixType::rands:
        case TestMatrixType::randn:
        case TestMatrixType::randb:
        case TestMatrixType::randr:
            generate_rand( A, type, dominant, sigma_max, seed, opts );
            break;

        case TestMatrixType::diag: {
            generate_sigma( params, dist, false, cond, sigma_max, A, Sigma, seed );
            break;
        }

        case TestMatrixType::svd: {
            generate_svd( params, dist, cond, condD, sigma_max, A, Sigma, seed, opts );
            break;
        }

        case TestMatrixType::poev: {
            generate_heev( params, dist, false, cond, condD, sigma_max, A, Sigma, seed, opts );
            break;
        }

        case TestMatrixType::heev: {
            generate_heev( params, dist, true, cond, condD, sigma_max, A, Sigma, seed, opts );
            break;
        }

        case TestMatrixType::geev: {
            generate_geev( params, dist, cond, sigma_max, A, Sigma, seed, opts );
            break;
        }

        case TestMatrixType::geevx: {
            generate_geevx( params, dist, cond, sigma_max, A, Sigma, seed, opts );
            break;
        }

        case TestMatrixType::minij: {
            entry_type minij_entry = []( int64_t i, int64_t j ) {
                return std::min(i + 1, j + 1);
            };
            set( minij_entry, A, opts );
            break;
        }

        case TestMatrixType::hilb: {
            entry_type hilb_entry = []( int64_t i, int64_t j ) {
                return 1.0 / (i + j + 1);
            };
            set( hilb_entry, A, opts );
            break;
        }

         case TestMatrixType::frank: {
            const int64_t max_mn = std::max(n, m);
            entry_type frank_entry = [max_mn]( int64_t i, int64_t j ) {
                if ((i - j) > 1) {
                    return int64_t( 0 );
                }
                else if ((i - j) == 1) {
                    return max_mn - j - 1;
                }
                else {
                    return max_mn - j;
                }
            };
            set( frank_entry, A, opts );
            break;
        }

        case TestMatrixType::lehmer: {
            entry_type lehmer_entry = []( int64_t i, int64_t j ) {
                return double (std::min(i, j) + 1) / (std::max(i, j) + 1);
            };
            set( lehmer_entry, A, opts );
            break;
        }

        case TestMatrixType::lotkin: {
            entry_type lotkin_entry = []( int64_t i, int64_t j ) {
                return (i == 0 ? 1.0 : (1.0 / (i + j + 1)));
            };
            set( lotkin_entry, A, opts );
            break;
        }

        case TestMatrixType::redheff: {
            entry_type redheff_entry = []( int64_t i, int64_t j ) {
                return ((j+1) % (i+1) == 0 || j == 0 ? 1 : 0);
            };
            set( redheff_entry, A, opts );
            break;
        }

        case TestMatrixType::triw: {
            entry_type triw_entry = []( int64_t i, int64_t j ) {
                if (i == j) {
                    return 1;
                }
                else if (i > j) {
                    return 0;
                }
                else {
                    return -1;
                }
            };
            set( triw_entry, A, opts );
            break;
        }

        case TestMatrixType::pei: {
            entry_type pei_entry = []( int64_t i, int64_t j ) {
                return (i == j ? 2 : 1);
            };
            set( pei_entry, A, opts);
            break;
        }

        case TestMatrixType::tridiag: {
            entry_type tridiag_entry = []( int64_t i, int64_t j ) {
                if (i == j) {
                    return 2;
                }
                else if (std::abs(i - j) == 1) {
                    return -1;
                }
                else {
                    return 0;
                }
            };
            set( tridiag_entry, A, opts );
            break;
        }
        
        case TestMatrixType::toeppen: {
            entry_type toeppen_entry = []( int64_t i, int64_t j ) {
                if (std::abs(j - i) == 1) {
                    return int ((j - i) * 10);
                }
                else if (std::abs(i - j) == 2) {
                    return 1;
                }
                else {
                    return 0;
                }
            };
            set( toeppen_entry, A, opts );
            break;
        }

        case TestMatrixType::parter: {
            entry_type parter_entry = []( int64_t i, int64_t j ) {
                return 1 / (i - j + 0.5);
            };
            set( parter_entry, A, opts );
            break;
        }

        case TestMatrixType::moler: {
            entry_type moler_entry = []( int64_t i, int64_t j ) {
                return (i == j ? i + 1 : std::min(i, j) - 1);
            };
            set( moler_entry, A, opts );
            break;
        }
    }

    // rand types have already been made diagonally dominant.
    if (dominant
        && ! (type == TestMatrixType::rand
              || type == TestMatrixType::rands
              || type == TestMatrixType::randn
              || type == TestMatrixType::randb)) {
        // make diagonally dominant; strict unless diagonal has zeros
        snprintf( msg, sizeof( msg ), "in '%s': dominant not yet implemented",
                  params.kind.c_str() );
        throw std::runtime_error( msg );
    }

    if (zero_col >= 0) {
        // Set col A( :, zero_col ) = 0.
        #pragma omp parallel
        #pragma omp master
        {
            int64_t j_global = 0;
            for (int64_t j = 0; j < nt; ++j) {
                int64_t jj = zero_col - j_global;
                if (0 <= jj && jj < A.tileNb( j )) {
                    for (int64_t i = 0; i < mt; ++i) {
                        if (A.tileIsLocal( i, j )) {
                            #pragma omp task slate_omp_default_none shared( A ) \
                                firstprivate( i, j, jj, zero )
                            {
                                A.tileGetForWriting( i, j, LayoutConvert::ColMajor );
                                auto Aij = A( i, j );
                                lapack::laset(
                                    lapack::MatrixType::General,
                                    Aij.mb(), 1, zero, zero,
                                    &Aij.at( 0, jj ), Aij.stride() );
                            }
                        }
                    }
                }
                j_global += A.tileNb( j );
            }
        }
    }

    A.tileUpdateAllOrigin();
}

//------------------------------------------------------------------------------
/// Overload without Sigma.
/// @see generate_matrix()
/// @ingroup generate_matrix
///
template <typename scalar_t>
void generate_matrix(
    MatgenParams& params,
    slate::Matrix<scalar_t>& A,
    slate::Options const& opts)
{
    using real_t = blas::real_type<scalar_t>;
    std::vector<real_t> dummy;
    generate_matrix( params, A, dummy, opts );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void generate_matrix(
    MatgenParams& params,
    slate::Matrix<float>& A,
    slate::Options const& opts);

template
void generate_matrix(
    MatgenParams& params,
    slate::Matrix<double>& A,
    slate::Options const& opts);

template
void generate_matrix(
    MatgenParams& params,
    slate::Matrix< std::complex<float> >& A,
    slate::Options const& opts);

template
void generate_matrix(
    MatgenParams& params,
    slate::Matrix< std::complex<double> >& A,
    slate::Options const& opts);

} // namespace slate
