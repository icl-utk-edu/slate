// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"
#include "print_matrix.hh"

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

#include "matrix_params.hh"
#include "matrix_generator.hh"

// -----------------------------------------------------------------------------
const int64_t idist_rand  = 1;
const int64_t idist_rands = 2;
const int64_t idist_randn = 3;

enum class TestMatrixType {
    rand      = 1,  // maps to larnv idist
    rands     = 2,  // maps to larnv idist
    randn     = 3,  // maps to larnv idist
    randb,
    zero,
    one,
    identity,
    ij,
    jordan,
    circul,
    fiedler,
    gfpp,
    orthog,
    riemann,
    ris,
    diag,
    svd,
    poev,
    heev,
    geev,
    geevx,
};

enum class TestMatrixDist {
    rand      = 1,  // maps to larnv idist
    rands     = 2,  // maps to larnv idist
    randn     = 3,  // maps to larnv idist
    arith,
    geo,
    cluster0,
    cluster1,
    rarith,
    rgeo,
    rcluster0,
    rcluster1,
    logrand,
    specified,
    none,
};

// -----------------------------------------------------------------------------
// ANSI color codes
using testsweeper::ansi_esc;
using testsweeper::ansi_red;
using testsweeper::ansi_bold;
using testsweeper::ansi_normal;

// -----------------------------------------------------------------------------
/// Splits a string by any of the delimiters.
/// Adjacent delimiters will give empty tokens.
/// See https://stackoverflow.com/questions/53849
/// @ingroup util
std::vector< std::string >
    split( const std::string& str, const std::string& delims );

std::vector< std::string >
    split( const std::string& str, const std::string& delims )
{
    size_t npos = std::string::npos;
    std::vector< std::string > tokens;
    size_t start = (str.size() > 0 ? 0 : npos);
    while (start != npos) {
        size_t end = str.find_first_of( delims, start );
        tokens.push_back( str.substr( start, end - start ));
        start = (end == npos ? npos : end + 1);
    }
    return tokens;
}

// =============================================================================
namespace slate {

// -----------------------------------------------------------------------------
/// Generates Sigma vector of singular or eigenvalues, according to distribution.
///
/// Internal function, called from generate_matrix().
///
/// @ingroup generate_matrix
///
template <typename matrix_type>
void generate_sigma(
    MatrixParams& params,
    TestMatrixDist dist, bool rand_sign,
    blas::real_type<typename matrix_type::value_type> cond,
    blas::real_type<typename matrix_type::value_type> sigma_max,
    matrix_type& A,
    std::vector< blas::real_type<typename matrix_type::value_type> >& Sigma,
    int64_t iseed[4] )
{
    using scalar_t = typename matrix_type::value_type;
    using real_t = blas::real_type<scalar_t>;

    // Constants
    const scalar_t zero = 0.0;

    // locals
    int64_t min_mn = std::min( A.m(), A.n() );

    // Ensure Sigma is allocated
    if (Sigma.size() == 0) {
        Sigma.resize(min_mn);
    }
    assert( min_mn == int64_t(Sigma.size()) );

    switch (dist) {
        case TestMatrixDist::arith:
            for (int64_t i = 0; i < min_mn; ++i) {
                Sigma[i] = 1 - i / real_t(min_mn - 1) * (1 - 1/cond);
            }
            break;

        case TestMatrixDist::rarith:
            for (int64_t i = 0; i < min_mn; ++i) {
                Sigma[i] = 1 - (min_mn - 1 - i) / real_t(min_mn - 1) * (1 - 1/cond);
            }
            break;

        case TestMatrixDist::geo:
            for (int64_t i = 0; i < min_mn; ++i) {
                Sigma[i] = pow( cond, -i / real_t(min_mn - 1) );
            }
            break;

        case TestMatrixDist::rgeo:
            for (int64_t i = 0; i < min_mn; ++i) {
                Sigma[i] = pow( cond, -(min_mn - 1 - i) / real_t(min_mn - 1) );
            }
            break;

        case TestMatrixDist::cluster0:
            Sigma[0] = 1;
            for (int64_t i = 1; i < min_mn; ++i) {
                Sigma[i] = 1/cond;
            }
            break;

        case TestMatrixDist::rcluster0:
            for (int64_t i = 0; i < min_mn-1; ++i) {
                Sigma[i] = 1/cond;
            }
            Sigma[min_mn-1] = 1;
            break;

        case TestMatrixDist::cluster1:
            for (int64_t i = 0; i < min_mn-1; ++i) {
                Sigma[i] = 1;
            }
            Sigma[min_mn-1] = 1/cond;
            break;

        case TestMatrixDist::rcluster1:
            Sigma[0] = 1/cond;
            for (int64_t i = 1; i < min_mn; ++i) {
                Sigma[i] = 1;
            }
            break;

        case TestMatrixDist::logrand: {
            real_t range = log( 1/cond );
            lapack::larnv( idist_rand, iseed, Sigma.size(), Sigma.data() );
            for (int64_t i = 0; i < min_mn; ++i) {
                Sigma[i] = exp( Sigma[i] * range );
            }
            // make cond exact
            if (min_mn >= 2) {
                Sigma[0] = 1;
                Sigma[1] = 1/cond;
            }
            break;
        }

        case TestMatrixDist::randn:
        case TestMatrixDist::rands:
        case TestMatrixDist::rand: {
            int64_t idist = (int64_t) dist;
            lapack::larnv( idist, iseed, Sigma.size(), Sigma.data() );
            break;
        }

        case TestMatrixDist::specified:
            // user-specified Sigma values; don't modify
            sigma_max = 1;
            rand_sign = false;
            break;

        case TestMatrixDist::none:
            assert( false );
            break;
    }

    if (sigma_max != 1) {
        blas::scal( Sigma.size(), sigma_max, Sigma.data(), 1 );
    }

    if (rand_sign) {
        // apply random signs
        for (int64_t i = 0; i < min_mn; ++i) {
            float rand;
            lapack::larnv( idist_rand, iseed, 1, &rand );
            if (rand > 0.5) {
                Sigma[i] = -Sigma[i];
            }
        }
    }

    // copy Sigma => A
    int64_t min_mt_nt = std::min(A.mt(), A.nt());
    set(zero, zero, A);
    int64_t S_index = 0;
    #pragma omp parallel for
    for (int64_t i = 0; i < min_mt_nt; ++i) {
        if (A.tileIsLocal(i, i)) {
            A.tileGetForWriting( i, i, LayoutConvert::ColMajor );
            auto T = A(i, i);
            for (int ii = 0; ii < A.tileNb(i); ++ii) {
                T.at(ii, ii) = Sigma[S_index + ii];
            }
        }
        S_index += A.tileNb(i);
    }

    A.tileUpdateAllOrigin();
}


// -----------------------------------------------------------------------------
/// Given matrix A with singular values such that sum(sigma_i^2) = n,
/// returns A with columns of unit norm, with the same condition number.
/// see: Davies and Higham, 2000, Numerically stable generation of correlation
/// matrices and their factors.
///
/// Internal function, called from generate_matrix().
///
/// @ingroup generate_matrix
//template <typename scalar_t>
//void generate_correlation_factor( slate::Matrix<scalar_t>& A )
//{
//    //const scalar_t eps = std::numeric_limits<scalar_t>::epsilon();
//
//    std::vector<scalar_t> x( A.n );
//    for (int64_t j = 0; j < A.n; ++j) {
//        x[j] = blas::dot( A.m, A(0,j), 1, A(0,j), 1 );
//    }
//
//    for (int64_t i = 0; i < A.n; ++i) {
//        for (int64_t j = 0; j < A.n; ++j) {
//            if ((x[i] < 1 && 1 < x[j]) || (x[i] > 1 && 1 > x[j])) {
//                scalar_t xij, d, t, c, s;
//                xij = blas::dot( A.m, A(0,i), 1, A(0,j), 1 );
//                d = sqrt( xij*xij - (x[i] - 1)*(x[j] - 1) );
//                t = (xij + std::copysign( d, xij )) / (x[j] - 1);
//                c = 1 / sqrt(1 + t*t);
//                s = c*t;
//                blas::rot( A.m, A(0,i), 1, A(0,j), 1, c, -s );
//                x[i] = blas::dot( A.m, A(0,i), 1, A(0,i), 1 );
//                //if (x[i] - 1 > 30*eps) {
//                //    printf( "i %d, x[i] %.6f, x[i] - 1 %.6e, 30*eps %.6e\n",
//                //            i, x[i], x[i] - 1, 30*eps );
//                //}
//                //assert( x[i] - 1 < 30*eps );
//                x[i] = 1;
//                x[j] = blas::dot( A.m, A(0,j), 1, A(0,j), 1 );
//                break;
//            }
//        }
//    }
//}
//
//
// -----------------------------------------------------------------------------
// specialization to complex
// can't use Higham's algorithm in complex
//template<>
//void generate_correlation_factor( slate::Matrix<std::complex<float>>& A )
//{
//    assert( false );
//}
//
//template<>
//void generate_correlation_factor( slate::Matrix<std::complex<double>>& A )
//{
//    assert( false );
//}


// -----------------------------------------------------------------------------
/// Generates matrix using SVD, $A = U Sigma V^H$.
///
/// Internal function, called from generate_matrix().
///
/// @ingroup generate_matrix
template <typename scalar_t>
void generate_svd(
    MatrixParams& params,
    TestMatrixDist dist,
    blas::real_type<scalar_t> cond,
    blas::real_type<scalar_t> condD,
    blas::real_type<scalar_t> sigma_max,
    slate::Matrix<scalar_t>& A,
    std::vector< blas::real_type<scalar_t> >& Sigma,
    int64_t iseed[4] )
{
    using real_t = blas::real_type<scalar_t>;
    assert( A.m() >= A.n() );

    // locals
    int64_t n = A.n();
    int64_t mt = A.mt();
    int64_t nt = A.nt();
    int64_t min_mt_nt = std::min(mt, nt);

    slate::Matrix<scalar_t> U = A.emptyLike();
    U.insertLocalTiles();
    slate::TriangularFactors<scalar_t> T;

    // ----------
    generate_sigma( params, dist, false, cond, sigma_max, A, Sigma, iseed );

    // for generate correlation factor, need sum sigma_i^2 = n
    // scaling doesn't change cond
    if (condD != 1) {
        real_t sum_sq = blas::dot( Sigma.size(), Sigma.data(), 1, Sigma.data(), 1 );
        real_t scale = sqrt( Sigma.size() / sum_sq );
        blas::scal( Sigma.size(), scale, Sigma.data(), 1 );

        // copy Sigma to diag(A)
        int64_t S_index = 0;
        #pragma omp parallel for
        for (int64_t i = 0; i < min_mt_nt; ++i) {
            if (A.tileIsLocal(i, i)) {
                A.tileGetForWriting( i, i, LayoutConvert::ColMajor );
                auto Aii = A(i, i);
                for (int ii = 0; ii < A.tileNb(i); ++ii) {
                    Aii.at(ii, ii) = Sigma[S_index + ii];
                }
            }
            S_index += A.tileNb(i);
        }
    }

    // random U, m-by-min_mn
    auto Tmp = U.emptyLike();
    #pragma omp parallel for collapse(2)
    for (int64_t j = 0; j < nt; ++j) {
        for (int64_t i = 0; i < mt; ++i) {
            if (U.tileIsLocal(i, j)) {
                // lapack assume input tile is contigous in memory
                // if the tiles are not contigous in memory, then
                // insert temperory tiles to be passed to lapack
                // then copy output tile to U.tile
                Tmp.tileInsert(i, j);
                auto Tmpij = Tmp(i, j);
                scalar_t* data = Tmpij.data();
                int64_t ldt = Tmpij.stride();
                //lapack::larnv( idist_randn, params.iseed,
                //    U.tileMb(i)*U.tileNb(j), Tmpij.data() );

                // Added local seed array for each process to prevent race condition contention of iseed
                int64_t tile_iseed[4];
                tile_iseed[0] = (iseed[0] + i/4096) % 4096;
                tile_iseed[1] = (iseed[1] + j/2048) % 4096;
                tile_iseed[2] = (iseed[2] + i)      % 4096;
                tile_iseed[3] = (iseed[3] + j*2)    % 4096;
                for (int64_t k = 0; k < Tmpij.nb(); ++k) {
                    lapack::larnv(idist_randn, tile_iseed, Tmpij.mb(), &data[k*ldt]);
                }
                slate::tile::gecopy( Tmp(i, j), U(i, j) );
                Tmp.tileErase(i, j);
            }
        }
    }
    // Hack to update iseed between matrices.
    iseed[2] = (iseed[2]*31) % 4096;

    // we need to make each random column into a Householder vector;
    // no need to update subsequent columns (as in geqrf).
    // However, currently we do geqrf here,
    // since we don't have a way to make Householder vectors (no distributed larfg).
    slate::geqrf(U, T);

    // A = U*A
    slate::unmqr( slate::Side::Left, slate::Op::NoTrans, U, T, A);

    // random V, n-by-min_mn (stored column-wise in U)
    auto V = U.slice(0, n-1, 0, n-1);
    auto Tmp_V = Tmp.slice(0, n-1, 0, n-1);
    #pragma omp parallel for collapse(2)
    for (int64_t j = 0; j < min_mt_nt; ++j) {
        for (int64_t i = 0; i < nt; ++i) {
            if (V.tileIsLocal(i, j)) {
                Tmp_V.tileInsert(i, j);
                auto Tmpij = Tmp_V(i, j);
                scalar_t* data = Tmpij.data();
                int64_t ldt = Tmpij.stride();

                // Added local seed array for each process to prevent race condition contention of iseed
                int64_t tile_iseed[4];
                tile_iseed[0] = (iseed[0] + i/4096) % 4096;
                tile_iseed[1] = (iseed[1] + j/2048) % 4096;
                tile_iseed[2] = (iseed[2] + i)      % 4096;
                tile_iseed[3] = (iseed[3] + j*2)    % 4096;
                for (int64_t k = 0; k < Tmpij.nb(); ++k) {
                    lapack::larnv(idist_randn, tile_iseed, Tmpij.mb(), &data[k*ldt]);
                }
                slate::tile::gecopy( Tmp_V(i, j), V(i, j) );
                Tmp_V.tileErase(i, j);
            }
        }
    }
    // Hack to update iseed between matrices.
    iseed[2] = (iseed[2]*31) % 4096;

    slate::geqrf(V, T);

    // A = A*V^H
    slate::unmqr( slate::Side::Right, slate::Op::ConjTrans, V, T, A);

    if (condD != 1) {
        // A = A*W, W orthogonal, such that A has unit column norms,
        // i.e., A^H A is a correlation matrix with unit diagonal
        // TODO: uncomment generate_correlation_factor
        //generate_correlation_factor( A );

        // A = A*D col scaling
        std::vector<real_t> D( n );
        real_t range = log( condD );
        lapack::larnv( idist_rand, iseed, D.size(), D.data() );
        for (auto& D_i: D) {
            D_i = exp( D_i * range );
        }
        // TODO: add argument to return D to caller?
        if (params.verbose) {
            printf( "D = [" );
            for (auto& D_i: D) {
                printf( " %11.8g", D_i );
            }
            printf( " ];\n" );
        }

        int64_t J_index = 0;
        for (int64_t j = 0; j < nt; ++j) {
            for (int64_t i = 0; i < mt; ++i) {
                if (A.tileIsLocal(i, j)) {
                    A.tileGetForWriting( i, j, LayoutConvert::ColMajor );
                    auto Aij = A(i, j);
                    for (int jj = 0; jj < A.tileNb(j); ++jj) {
                        for (int ii = 0; ii < A.tileMb(i); ++ii) {
                            Aij.at(ii, jj) *= D[J_index + jj];
                        }
                    }
                }
            }
            J_index += A.tileNb(j);
        }
    }
    A.tileUpdateAllOrigin();
}

// -----------------------------------------------------------------------------
/// Generates matrix using Hermitian eigenvalue decomposition, $A = U Sigma U^H$.
///
/// Internal function, called from generate_matrix().
///
/// @ingroup generate_matrix
template <typename scalar_t>
void generate_heev(
    MatrixParams& params,
    TestMatrixDist dist, bool rand_sign,
    blas::real_type<scalar_t> cond,
    blas::real_type<scalar_t> condD,
    blas::real_type<scalar_t> sigma_max,
    slate::Matrix<scalar_t>& A,
    std::vector< blas::real_type<scalar_t> >& Sigma,
    int64_t iseed[4] )
{
    using real_t = blas::real_type<scalar_t>;

    // check inputs
    assert( A.m() == A.n() );

    // locals
    int64_t n = A.n();
    slate::Matrix<scalar_t> U = A.emptyLike();
    U.insertLocalTiles();
    slate::TriangularFactors<scalar_t> T;

    // ----------
    generate_sigma( params, dist, rand_sign, cond, sigma_max, A, Sigma, iseed );

    // random U, m-by-min_mn
    int64_t nt = U.nt();
    int64_t mt = U.mt();
    auto Tmp = U.emptyLike();

    #pragma omp parallel for collapse(2)
    for (int64_t j = 0; j < nt; ++j) {
        for (int64_t i = 0; i < mt; ++i) {
            if (U.tileIsLocal(i, j)) {
                Tmp.tileInsert(i, j);
                auto Tmpij = Tmp(i, j);
                scalar_t* data = Tmpij.data();
                int64_t ldt = Tmpij.stride();

                // Added local seed array for each process to prevent race condition contention of iseed
                int64_t tile_iseed[4];
                tile_iseed[0] = (iseed[0] + i/4096) % 4096;
                tile_iseed[1] = (iseed[1] + j/2048) % 4096;
                tile_iseed[2] = (iseed[2] + i)      % 4096;
                tile_iseed[3] = (iseed[3] + j*2)    % 4096;
                for (int64_t k = 0; k < Tmpij.nb(); ++k) {
                    lapack::larnv(idist_rand, tile_iseed, Tmpij.mb(), &data[k*ldt]);
                }
                slate::tile::gecopy( Tmp(i, j), U(i, j) );
                Tmp.tileErase(i, j);
            }
        }
    }
    // Hack to update iseed between matrices.
    iseed[2] = (iseed[2]*31) % 4096;

    // we need to make each random column into a Householder vector;
    // no need to update subsequent columns (as in geqrf).
    // However, currently we do geqrf here,
    // since we don't have a way to make Householder vectors (no distributed larfg).
    slate::geqrf(U, T);

    // A = U*A
    slate::unmqr( slate::Side::Left, slate::Op::NoTrans, U, T, A);

    // A = A*U^H
    slate::unmqr( slate::Side::Right, slate::Op::ConjTrans, U, T, A);

    // make diagonal real
    // usually LAPACK ignores imaginary part anyway, but Matlab doesn't
    #pragma omp parallel for
    for (int64_t i = 0; i < nt; ++i) {
        if (A.tileIsLocal(i, i)) {
            A.tileGetForWriting( i, i, LayoutConvert::ColMajor );
            auto Aii = A(i, i);
            for (int ii = 0; ii < A.tileMb(i); ++ii) {
                Aii.at(ii, ii) = std::real( Aii.at(ii, ii) );
            }
        }
    }

    if (condD != 1) {
        // A = D*A*D row & column scaling
        std::vector<real_t> D( n );
        real_t range = log( condD );
        lapack::larnv( idist_rand, iseed, n, D.data() );
        for (int64_t i = 0; i < n; ++i) {
            D[i] = exp( D[i] * range );
        }

        int64_t J_index = 0;
        #pragma omp parallel for
        for (int64_t j = 0; j < nt; ++j) {
            int64_t I_index = 0;
            for (int64_t i = 0; i < mt; ++i) {
                if (A.tileIsLocal(i, j)) {
                    A.tileGetForWriting( i, j, LayoutConvert::ColMajor );
                    auto Aij = A(i, j);
                    for (int jj = 0; jj < A.tileMb(j); ++jj) {
                        for (int ii = 0; ii < A.tileMb(i); ++ii) {
                            Aij.at(ii, jj) *= D[I_index + ii] * D[J_index + jj];
                        }
                    }
                }
                I_index += A.tileNb(i);
            }
            J_index += A.tileMb(j);
        }
    }
    A.tileUpdateAllOrigin();
}

// -----------------------------------------------------------------------------
/// Generates matrix using general eigenvalue decomposition, $A = V T V^H$,
/// with orthogonal eigenvectors.
/// Not yet implemented.
///
/// Internal function, called from generate_matrix().
///
/// @ingroup generate_matrix
template <typename scalar_t>
void generate_geev(
    MatrixParams& params,
    TestMatrixDist dist,
    blas::real_type<scalar_t> cond,
    blas::real_type<scalar_t> sigma_max,
    slate::Matrix<scalar_t>& A,
    std::vector< blas::real_type<scalar_t> >& Sigma,
    int64_t iseed[4] )
{
    throw std::exception();  // not implemented
}

// -----------------------------------------------------------------------------
/// Generates matrix using general eigenvalue decomposition, $A = X T X^{-1}$,
/// with random eigenvectors.
/// Not yet implemented.
///
/// Internal function, called from generate_matrix().
///
/// @ingroup generate_matrix
template <typename scalar_t>
void generate_geevx(
    MatrixParams& params,
    TestMatrixDist dist,
    blas::real_type<scalar_t> cond,
    blas::real_type<scalar_t> sigma_max,
    slate::Matrix<scalar_t>& A,
    std::vector< blas::real_type<scalar_t> >& Sigma,
    int64_t iseed[4] )
{
    throw std::exception();  // not implemented
}

// -----------------------------------------------------------------------------
void generate_matrix_usage()
{
    printf(
    "The --matrix, --cond, and --condD parameters specify a test matrix.\n"
    "See Test routines: generate_matrix in the HTML documentation for a\n"
    "complete description.\n"
    "\n"
    "%s--matrix%s is one of following:\n"
    "\n"
    "%sMatrix%s    |  %sDescription%s\n"
    "----------|-------------\n"
    "zero      |  all zero\n"
    "one       |  all one\n"
    "identity  |  ones on diagonal, rest zero\n"
    "ij        |  Aij = i + j / 10^ceil( log10( max( m, n ) ) )\n"
    "jordan    |  ones on diagonal and first subdiagonal, rest zero\n"
    "circul    |  circulant matrix where the first column is [1, 2, ..., n]^T\n"
    "fiedler   |  matrix entry i,j equal to |i - j|\n"
    "gfpp      |  growth factor for gesv of 1.5^n\n"
    "orthog    |  matrix entry i,j equal to sqrt(2/(n+1))sin(i*j*pi/(n+1))\n"
    "riemann   |  matrix entry i,j equal to i+1 if j+2 divides i+2 else -1\n"
    "ris       |  matrix entry i,j equal to 0.5/(n-i-j+1.5)\n"
    "          |  \n"
    "rand@     |  matrix entries random uniform on (0, 1)\n"
    "rands@    |  matrix entries random uniform on (-1, 1)\n"
    "randn@    |  matrix entries random normal with mean 0, std 1\n"
    "randb@    |  matrix entries random uniform from {0, 1}\n"
    "          |  \n"
    "diag^@    |  A = Sigma\n"
    "svd^@     |  A = U Sigma V^H\n"
    "poev^@    |  A = V Sigma V^H  (eigenvalues positive, i.e., matrix SPD)\n"
    "spd^@     |  alias for poev\n"
    "heev^@    |  A = V Lambda V^H (eigenvalues mixed signs)\n"
    "syev^@    |  alias for heev\n"
    "geev^@    |  A = V T V^H, Schur-form T                       [not yet implemented]\n"
    "geevx^@   |  A = X T X^{-1}, Schur-form T, X ill-conditioned [not yet implemented]\n"
    "\n"
    "^ and @ denote optional suffixes described below.\n"
    "\n"
    "%s^ Distribution%s  |  %sDescription%s\n"
    "----------------|-------------\n"
    "_logrand        |  log(sigma_i) random uniform on [ log(1/cond), log(1) ]; default\n"
    "_arith          |  sigma_i = 1 - frac{i - 1}{n - 1} (1 - 1/cond); arithmetic: sigma_{i+1} - sigma_i is constant\n"
    "_geo            |  sigma_i = (cond)^{ -(i-1)/(n-1) };             geometric:  sigma_{i+1} / sigma_i is constant\n"
    "_cluster0       |  Sigma = [ 1, 1/cond, ..., 1/cond ];  1  unit value,  n-1 small values\n"
    "_cluster1       |  Sigma = [ 1, ..., 1, 1/cond ];      n-1 unit values,  1  small value\n"
    "_rarith         |  _arith,    reversed order\n"
    "_rgeo           |  _geo,      reversed order\n"
    "_rcluster0      |  _cluster0,  reversed order\n"
    "_rcluster1      |  _cluster1, reversed order\n"
    "_specified      |  user specified Sigma on input\n"
    "                |  \n"
    "_rand           |  sigma_i random uniform on (0, 1)\n"
    "_rands          |  sigma_i random uniform on (-1, 1)\n"
    "_randn          |  sigma_i random normal with mean 0, std 1\n"
    "\n"
    "%s@ Scaling%s       |  %sDescription%s\n"
    "----------------|-------------\n"
    "_ufl            |  scale near underflow         = 1e-308 for double\n"
    "_ofl            |  scale near overflow          = 2e+308 for double\n"
    "_small          |  scale near sqrt( underflow ) = 1e-154 for double\n"
    "_large          |  scale near sqrt( overflow  ) = 6e+153 for double\n"
    "\n"
    "%s@ Modifier%s      |  %sDescription%s\n"
    "----------------|-------------\n"
    "_dominant       |  make matrix diagonally dominant\n"
    "\n",
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal
    );
}

// -----------------------------------------------------------------------------
/// Decode matrix type, distribution, scaling and modifier.
///
template <typename scalar_t>
void decode_matrix(
    MatrixParams& params,
    BaseMatrix<scalar_t>& A,
    TestMatrixType& type,
    TestMatrixDist& dist,
    blas::real_type<scalar_t>& cond,
    blas::real_type<scalar_t>& condD,
    blas::real_type<scalar_t>& sigma_max,
    bool& dominant)
{
    using real_t = blas::real_type<scalar_t>;

    const real_t ufl = std::numeric_limits< real_t >::min();      // == lamch("safe min")  ==  1e-38 or  2e-308
    const real_t ofl = 1 / ufl;                                   //                            8e37 or   4e307
    const real_t eps = std::numeric_limits< real_t >::epsilon();  // == lamch("precision") == 1.2e-7 or 2.2e-16

    // locals
    std::string kind = params.kind();

    //---------------
    cond = params.cond();
    bool cond_default = std::isnan( cond );
    if (cond_default) {
        cond = 1 / sqrt( eps );
    }

    condD = params.condD();
    bool condD_default = std::isnan( condD );
    if (condD_default) {
        condD = 1;
    }

    //---------------
    sigma_max = 1;
    std::vector< std::string > tokens = split( kind, "-_" );

    // ----- decode matrix type
    auto token = tokens.begin();
    if (token == tokens.end()) {
        throw std::runtime_error( "Error: empty matrix kind\n" );
    }
    std::string base = *token;
    ++token;
    type = TestMatrixType::identity;
    if      (base == "zero"    ) { type = TestMatrixType::zero;     }
    else if (base == "one"     ) { type = TestMatrixType::one;      }
    else if (base == "identity") { type = TestMatrixType::identity; }
    else if (base == "ij"      ) { type = TestMatrixType::ij;       }
    else if (base == "jordan"  ) { type = TestMatrixType::jordan;   }
    else if (base == "circul"  ) { type = TestMatrixType::circul;   }
    else if (base == "fiedler" ) { type = TestMatrixType::fiedler;  }
    else if (base == "gfpp"    ) { type = TestMatrixType::gfpp;     }
    else if (base == "orthog"  ) { type = TestMatrixType::orthog;   }
    else if (base == "riemann" ) { type = TestMatrixType::riemann;  }
    else if (base == "ris"     ) { type = TestMatrixType::ris;      }
    else if (base == "randb"   ) { type = TestMatrixType::randb;    }
    else if (base == "randn"   ) { type = TestMatrixType::randn;    }
    else if (base == "rands"   ) { type = TestMatrixType::rands;    }
    else if (base == "rand"    ) { type = TestMatrixType::rand;     }
    else if (base == "diag"    ) { type = TestMatrixType::diag;     }
    else if (base == "svd"     ) { type = TestMatrixType::svd;      }
    else if (base == "poev" ||
             base == "spd"     ) { type = TestMatrixType::poev;     }
    else if (base == "heev" ||
             base == "syev"    ) { type = TestMatrixType::heev;     }
    else if (base == "geevx"   ) { type = TestMatrixType::geevx;    }
    else if (base == "geev"    ) { type = TestMatrixType::geev;     }
    else {
        fprintf( stderr, "%sUnrecognized matrix '%s'%s\n",
                 ansi_red, kind.c_str(), ansi_normal );
        throw std::exception();
    }

    // ----- decode distribution
    std::string suffix;
    dist = TestMatrixDist::none;
    if (token != tokens.end()) {
        suffix = *token;
        if      (suffix == "randn"    ) { dist = TestMatrixDist::randn;     }
        else if (suffix == "rands"    ) { dist = TestMatrixDist::rands;     }
        else if (suffix == "rand"     ) { dist = TestMatrixDist::rand;      }
        else if (suffix == "logrand"  ) { dist = TestMatrixDist::logrand;   }
        else if (suffix == "arith"    ) { dist = TestMatrixDist::arith;     }
        else if (suffix == "geo"      ) { dist = TestMatrixDist::geo;       }
        else if (suffix == "cluster1" ) { dist = TestMatrixDist::cluster1;  }
        else if (suffix == "cluster0" ) { dist = TestMatrixDist::cluster0;  }
        else if (suffix == "rarith"   ) { dist = TestMatrixDist::rarith;    }
        else if (suffix == "rgeo"     ) { dist = TestMatrixDist::rgeo;      }
        else if (suffix == "rcluster1") { dist = TestMatrixDist::rcluster1; }
        else if (suffix == "rcluster0") { dist = TestMatrixDist::rcluster0; }
        else if (suffix == "specified") { dist = TestMatrixDist::specified; }

        // if found, move to next token
        if (dist != TestMatrixDist::none) {
            ++token;

            // error if matrix type doesn't support it
            if (! (type == TestMatrixType::diag ||
                   type == TestMatrixType::svd  ||
                   type == TestMatrixType::poev ||
                   type == TestMatrixType::heev ||
                   type == TestMatrixType::geev ||
                   type == TestMatrixType::geevx))
            {
                fprintf( stderr, "%sError in '%s': matrix '%s' doesn't support"
                         " distribution suffix.%s\n",
                         ansi_red, kind.c_str(), base.c_str(), ansi_normal );
                throw std::exception();
            }
        }
    }
    if (dist == TestMatrixDist::none)
        dist = TestMatrixDist::logrand;  // default

    // ----- decode scaling
    sigma_max = 1;
    if (token != tokens.end()) {
        suffix = *token;
        if      (suffix == "small") { sigma_max = sqrt( ufl ); }
        else if (suffix == "large") { sigma_max = sqrt( ofl ); }
        else if (suffix == "ufl"  ) { sigma_max = ufl; }
        else if (suffix == "ofl"  ) { sigma_max = ofl; }

        // if found, move to next token
        if (sigma_max != 1) {
            ++token;

            // error if matrix type doesn't support it
            if (! (type == TestMatrixType::rand  ||
                   type == TestMatrixType::rands ||
                   type == TestMatrixType::randn ||
                   type == TestMatrixType::randb ||
                   type == TestMatrixType::svd   ||
                   type == TestMatrixType::poev  ||
                   type == TestMatrixType::heev  ||
                   type == TestMatrixType::geev  ||
                   type == TestMatrixType::geevx))
            {
                fprintf( stderr, "%sError in '%s': matrix '%s' doesn't support"
                         " scaling suffix.%s\n",
                         ansi_red, kind.c_str(), base.c_str(), ansi_normal );
                throw std::exception();
            }
        }
    }

    // ----- decode modifier
    dominant = false;
    if (token != tokens.end()) {
        suffix = *token;
        if (suffix == "dominant") {
            dominant = true;

            // move to next token
            ++token;

            // error if matrix type doesn't support it
            if (! (type == TestMatrixType::rand  ||
                   type == TestMatrixType::rands ||
                   type == TestMatrixType::randn ||
                   type == TestMatrixType::randb ||
                   type == TestMatrixType::svd   ||
                   type == TestMatrixType::poev  ||
                   type == TestMatrixType::heev  ||
                   type == TestMatrixType::geev  ||
                   type == TestMatrixType::geevx))
            {
                fprintf( stderr, "%sError in '%s': matrix '%s' doesn't support"
                         " modifier suffix.%s\n",
                         ansi_red, kind.c_str(), base.c_str(), ansi_normal );
                throw std::exception();
            }
        }
    }

    if (token != tokens.end()) {
        fprintf( stderr, "%sError in '%s': unknown suffix '%s'.%s\n",
                 ansi_red, kind.c_str(), token->c_str(), ansi_normal );
        throw std::exception();
    }

    // ----- check compatability of options
    if (A.m() != A.n() &&
        (type == TestMatrixType::jordan ||
         type == TestMatrixType::poev   ||
         type == TestMatrixType::heev   ||
         type == TestMatrixType::geev   ||
         type == TestMatrixType::geevx))
    {
        fprintf( stderr, "%sError: matrix '%s' requires m == n.%s\n",
                 ansi_red, kind.c_str(), ansi_normal );
        throw std::exception();
    }

    if (type == TestMatrixType::zero      ||
        type == TestMatrixType::one       ||
        type == TestMatrixType::identity  ||
        type == TestMatrixType::jordan    ||
        type == TestMatrixType::circul    ||
        type == TestMatrixType::fiedler   ||
        type == TestMatrixType::gfpp      ||
        type == TestMatrixType::orthog    ||
        type == TestMatrixType::riemann   ||
        type == TestMatrixType::ris       ||
        type == TestMatrixType::randb     ||
        type == TestMatrixType::randn     ||
        type == TestMatrixType::rands     ||
        type == TestMatrixType::rand)
    {
        // warn first time if user set cond and matrix doesn't use it
        static std::string last;
        if (! cond_default && last != kind) {
            last = kind;
            fprintf( stderr, "%sWarning: matrix '%s' ignores cond %.2e.%s\n",
                     ansi_red, kind.c_str(), params.cond(), ansi_normal );
        }
        params.cond_used() = testsweeper::no_data_flag;
    }
    else if (dist == TestMatrixDist::randn ||
             dist == TestMatrixDist::rands ||
             dist == TestMatrixDist::rand)
    {
        // warn first time if user set cond and distribution doesn't use it
        static std::string last;
        if (! cond_default && last != kind) {
            last = kind;
            fprintf( stderr, "%sWarning: matrix '%s': rand, randn, and rands "
                     "singular/eigenvalue distributions ignore cond %.2e.%s\n",
                     ansi_red, kind.c_str(), params.cond(), ansi_normal );
        }
        params.cond_used() = testsweeper::no_data_flag;
    }
    else {
        params.cond_used() = cond;
    }

    if (! (type == TestMatrixType::svd ||
           type == TestMatrixType::heev ||
           type == TestMatrixType::poev))
    {
        // warn first time if user set condD and matrix doesn't use it
        static std::string last;
        if (! condD_default && last != kind) {
            last = kind;
            fprintf( stderr, "%sWarning: matrix '%s' ignores condD %.2e.%s\n",
                     ansi_red, kind.c_str(), params.condD(), ansi_normal );
        }
    }

    if (type == TestMatrixType::poev &&
        (dist == TestMatrixDist::rands ||
         dist == TestMatrixDist::randn))
    {
        fprintf( stderr, "%sWarning: matrix '%s' using rands or randn "
                 "will not generate SPD matrix; use rand instead.%s\n",
                 ansi_red, kind.c_str(), ansi_normal );
    }
}

// -----------------------------------------------------------------------------
/// Generates an arbitrary seed that is unlikely to be repeated
void configure_seed(MPI_Comm comm, int64_t user_seed, int64_t iseed[4])
{

    // if the given seed is -1, generate a new seed
    if (user_seed == -1) {
        // use the highest resolution clock as the seed
        using namespace std::chrono;
        user_seed = duration_cast<high_resolution_clock::duration>(
                        high_resolution_clock::now().time_since_epoch()).count();
        // higher order bits cannot affect the final seed
        user_seed %= int64_t(1) << 48;
        // scramble low order bits with a prime number less than 2^15
        user_seed *= 10477;
    }


    // ensure seeds are uniform across MPI ranks
    slate_mpi_call(MPI_Bcast(&user_seed, 1, MPI_INT64_T, 0, comm));

    // Assign bits across output seed
    iseed[0] = (user_seed >> 35) % 4096;
    iseed[1] = (user_seed >> 23) % 4096;
    iseed[2] = (user_seed >> 11) % 4096;
    iseed[3] = 2*(user_seed % 2048) + 1;
}

// -----------------------------------------------------------------------------
/// Generates an m-by-n test matrix.
/// Similar to LAPACK's libtmg functionality, but a level 3 BLAS implementation.
///
/// @param[in] params
///     Test matrix parameters. Uses matrix, cond, condD parameters;
///     see further details.
///
/// @param[out] A
///     Complex array, dimension (lda, n).
///     On output, the m-by-n test matrix A in an lda-by-n array.
///
/// @param[in,out] Sigma
///     Real array, dimension (min(m,n))
///     - On input with matrix distribution "_specified",
///       contains user-specified singular or eigenvalues.
///     - On output, contains singular or eigenvalues, if known,
///       else set to NaN. Sigma is not necesarily sorted.
///
/// ### Further Details
///
/// The **matrix** parameter specifies the matrix kind according to the
/// tables below. As indicated, kinds take an optional distribution suffix (^)
/// and an optional scaling and modifier suffix (@).
/// The default distribution is logrand.
/// Examples: rand, rand_small, svd_arith, heev_geo_small.
///
/// The **cond** parameter specifies the condition number $cond(S)$, where $S$ is either
/// the singular values $\Sigma$ or the eigenvalues $\Lambda$, as described by the
/// distributions below. It does not apply to some matrices and distributions.
/// For geev and geevx, cond(A) is generally much worse than cond(S).
/// If _dominant is applied, cond(A) generally improves.
/// By default, cond(S) = sqrt( 1/eps ) = 6.7e7 for double, 2.9e3 for single.
///
/// The **condD** parameter specifies the condition number cond(D), where D is
/// a diagonal scaling matrix [1]. By default, condD = 1. If condD != 1, then:
/// - For matrix = svd, set $A = A_0 K D$, where $A_0 = U \Sigma V^H$,
///   $D$ has log-random entries in $[ \log(1/condD), \log(1) ]$, and
///   $K$ is diagonal such that columns of $B = A_0 K$ have unit norm,
///   hence $B^T B$ has unit diagonal.
///
/// - For matrix = heev, set $A = D A_0 D$, where $A_0 = U \Lambda U^H$,
///   $D$ has log-random entries in $[ \log(1/condD), \log(1) ]$.
///   TODO: set $A = D K A_0 K D$ where
///   $K$ is diagonal such that $B = K A_0 K$ has unit diagonal.
///
/// Note using condD changes the singular or eigenvalues of $A$;
/// on output, Sigma contains the singular or eigenvalues of $A_0$, not of $A$.
///
/// Notation used below:
/// $\Sigma$ is a diagonal matrix with entries $\sigma_i$ for $i = 1, \dots, n$;
/// $\Lambda$ is a diagonal matrix with entries $\lambda_i = \pm \sigma_i$,
/// with random sign;
/// $U$ and $V$ are random orthogonal matrices from the Haar distribution [2],
/// $X$ is a random matrix.
///
/// See LAPACK Working Note (LAWN) 41:\n
/// Table  5 (Test matrices for the nonsymmetric eigenvalue problem)\n
/// Table 10 (Test matrices for the symmetric eigenvalue problem)\n
/// Table 11 (Test matrices for the singular value decomposition)
///
/// Matrix   | Description
/// ---------|------------
/// zero     | all zero
/// one      | all one
/// identity | ones on diagonal, rest zero
/// jordan   | ones on diagonal and first subdiagonal, rest zero
/// circul   | A circulant matrix where the first column is [1, 2, ..., n]^T
/// fiedler  | A matrix with entry i,j equal to |i - j|
/// gfpp     | A matrix with a growth factor of 1.5^n for gesv
/// orthog   | A matrix with entry i,j equal to sqrt(2/(n+1))sin(i*j*pi/(n+1))
/// riemann  | A matrix with entry i,j equal to i+1 if j+2 divides i+2 elso -1
/// ris      | A matrix with entry i,j equal to 0.5/(n-i-j+1.5)
/// --       | --
/// rand@    | matrix entries random uniform on (0, 1)
/// rands@   | matrix entries random uniform on (-1, 1)
/// randn@   | matrix entries random normal with mean 0, std 1
/// randb@   | matrix entries random uniform in {0, 1}
/// --       | --
/// diag^@   | $A = \Sigma$
/// svd^@    | $A = U \Sigma V^H$
/// poev^@   | $A = V \Sigma V^H$  (eigenvalues positive, i.e., matrix SPD)
/// spd^@    | alias for poev
/// heev^@   | $A = V \Lambda V^H$ (eigenvalues mixed signs)
/// syev^@   | alias for heev
/// geev^@   | $A = V T V^H$, Schur-form $T$                         [not yet implemented]
/// geevx^@  | $A = X T X^{-1}$, Schur-form $T$, $X$ ill-conditioned [not yet implemented]
///
/// Note for geev that $cond(\Lambda)$ is specified, where $\Lambda = diag(T)$;
/// while $cond(T)$ and $cond(A)$ are usually much worse.
///
/// ^ and @ denote optional suffixes described below.
///
/// ^ Distribution  |   Description
/// ----------------|--------------
/// _logrand        |  $\log(\sigma_i)$ random uniform on $[ \log(1/cond), \log(1) ]$; default
/// _arith          |  $\sigma_i = 1 - \frac{i - 1}{n - 1} (1 - 1/cond)$; arithmetic: $\sigma_{i+1} - \sigma_i$ is constant
/// _geo            |  $\sigma_i = (cond)^{ -(i-1)/(n-1) }$;              geometric:  $\sigma_{i+1} / \sigma_i$ is constant
/// _cluster0       |  $\Sigma = [ 1, 1/cond, ..., 1/cond ]$;  1     unit value,  $n-1$ small values
/// _cluster1       |  $\Sigma = [ 1, ..., 1, 1/cond ]$;       $n-1$ unit values, 1     small value
/// _rarith         |  _arith,    reversed order
/// _rgeo           |  _geo,      reversed order
/// _rcluster0      |  _cluster0,  reversed order
/// _rcluster1      |  _cluster1, reversed order
/// _specified      |  user specified Sigma on input
/// --              |  --
/// _rand           |  $\sigma_i$ random uniform on (0, 1)
/// _rands          |  $\sigma_i$ random uniform on (-1, 1)
/// _randn          |  $\sigma_i$ random normal with mean 0, std 1
///
/// Note _rand, _rands, _randn do not use cond; the condition number is random.
///
/// Note for _rands and _randn, $\Sigma$ contains negative values.
/// This means poev_rands and poev_randn will not generate an SPD matrix.
///
/// @ Scaling       |  Description
/// ----------------|-------------
/// _ufl            |  scale near underflow         = 1e-308 for double
/// _ofl            |  scale near overflow          = 2e+308 for double
/// _small          |  scale near sqrt( underflow ) = 1e-154 for double
/// _large          |  scale near sqrt( overflow  ) = 6e+153 for double
///
/// Note scaling changes the singular or eigenvalues, but not the condition number.
///
/// @ Modifier      |  Description
/// ----------------|-------------
/// _dominant       |  diagonally dominant: set $A_{i,i} = \pm \max_i( \sum_j |A_{i,j}|, \sum_j |A_{j,i}| )$.
///
/// Note _dominant changes the singular or eigenvalues, and the condition number.
///
/// ### References
///
/// [1] Demmel and Veselic, Jacobi's method is more accurate than QR, 1992.
///
/// [2] Stewart, The efficient generation of random orthogonal matrices
///     with an application to condition estimators, 1980.
///
/// @ingroup generate_matrix
template <typename scalar_t>
void generate_matrix(
    MatrixParams& params,
    slate::Matrix<scalar_t>& A,
    std::vector< blas::real_type<scalar_t> >& Sigma )
{
    using real_t = blas::real_type<scalar_t>;

    // Constants
    const real_t nan = std::numeric_limits<real_t>::quiet_NaN();
    const real_t d_zero = 0;
    const real_t d_one  = 1;
    const scalar_t zero = 0;
    const scalar_t one  = 1;

    // ----------
    // set Sigma to unknown (nan)
    lapack::laset( lapack::MatrixType::General, Sigma.size(), 1,
                   nan, nan, Sigma.data(), Sigma.size() );

    TestMatrixType type;
    TestMatrixDist dist;
    real_t cond;
    real_t condD;
    real_t sigma_max;
    bool dominant;
    decode_matrix<scalar_t>(params, A, type, dist, cond, condD, sigma_max, dominant);

    int64_t iseed[4];
    configure_seed(A.mpiComm(), params.seed(), iseed);

    int64_t n = A.n();
    int64_t m = A.m();
    int64_t nt = A.nt();
    int64_t mt = A.mt();
    // ----- generate matrix
    switch (type) {
        case TestMatrixType::zero:
            set(zero, zero, A);
            lapack::laset( lapack::MatrixType::General, Sigma.size(), 1,
                d_zero, d_zero, Sigma.data(), Sigma.size() );
            break;

        case TestMatrixType::one:
            set(one, one, A);
            lapack::laset( lapack::MatrixType::General, Sigma.size(), 1,
                d_zero, d_one, Sigma.data(), Sigma.size() );
            if (Sigma.size() >= 1) {
                Sigma[0] = sqrt(m*n);
            }
            break;

        case TestMatrixType::identity:
            set(zero, one, A);
            lapack::laset( lapack::MatrixType::General, Sigma.size(), 1,
                d_one, d_one, Sigma.data(), Sigma.size() );
            break;

        case TestMatrixType::ij: {
            // Scale so j*s < 1.
            real_t s = 1 / pow( 10, ceil( log10( n ) ) );
            #pragma omp parallel
            #pragma omp master
            {
                int64_t jj = 0;
                for (int64_t j = 0; j < nt; ++j) {
                    int64_t ii = 0;
                    for (int64_t i = 0; i < mt; ++i) {
                        #pragma omp task slate_omp_default_none \
                            firstprivate( i, j, ii, jj, s ) shared( A )
                        {
                            if (A.tileIsLocal( i, j )) {
                                A.tileGetForWriting( i, j, LayoutConvert::ColMajor );
                                auto Aij = A( i, j );
                                scalar_t* data = Aij.data();
                                int64_t lda = Aij.stride();
                                for (int64_t jjj = 0; jjj < Aij.nb(); ++jjj)
                                    for (int64_t iii = 0; iii < Aij.mb(); ++iii)
                                        data[ iii + jjj*lda ] = ii + iii + (jj + jjj)*s;
                            }
                        }
                        ii += A.tileMb( i );
                    }
                    jj += A.tileNb( j );
                }
            }
            break;
        }

        case TestMatrixType::jordan: {
            set(zero, one, A ); // ones on diagonal
            // ones on sub-diagonal
            for (int64_t i = 0; i < nt; ++i) {
                // Set 1 element from sub-diagonal tile to 1.
                if (i > 0) {
                    if (A.tileIsLocal(i, i-1)) {
                        A.tileGetForWriting( i, i-1, LayoutConvert::ColMajor );
                        auto T = A(i, i-1);
                        T.at(0, T.nb()-1) = 1.;
                    }
                }

                // Set 1 element from sub-diagonal tile to 1.
                if (A.tileIsLocal(i, i)) {
                    A.tileGetForWriting( i, i, LayoutConvert::ColMajor );
                    auto T = A(i, i);
                    auto len = T.nb();
                    for (int j = 0; j < len-1; ++j) {
                        T.at(j+1, j) = 1.;
                    }
                }
            }
            break;
        }

        // circulant matrix for the vector 1:n
        case TestMatrixType::circul: {
            const int64_t max_mn = std::max(n, m);
            #pragma omp parallel
            #pragma omp master
            {
                int64_t i_global = 1;
                for (int64_t i = 0; i < mt; ++i) {
                    const int64_t mb = A.tileMb(i);
                    int64_t j_global = 1;
                    for (int64_t j = 0; j < nt; ++j) {
                        const int64_t nb = A.tileNb(j);
                        if (A.tileIsLocal(i, j)) {
                            #pragma omp task firstprivate(i, j, mb, nb, \
                                                          i_global, j_global)
                            {
                                auto A_ij = A(i, j);
                                for (int64_t ii = 0; ii < mb; ++ii) {
                                    for (int64_t jj = 0; jj < nb; ++jj) {
                                        auto diff = (j_global+jj) - (i_global+ii);
                                        A_ij.at(ii, jj) = diff
                                                          + (diff < 0 ? max_mn : 0)
                                                          + 1;
                                    }
                                }
                            }
                        }
                        j_global += nb;
                    }
                    i_global += mb;
                }
                #pragma omp taskwait
            }
            break;
        }

        case TestMatrixType::fiedler: {
            #pragma omp parallel
            #pragma omp master
            {
                int64_t i_global = 0;
                for (int64_t i = 0; i < mt; ++i) {
                    const int64_t mb = A.tileMb(i);
                    int64_t j_global = 0;
                    for (int64_t j = 0; j < nt; ++j) {
                        const int64_t nb = A.tileNb(j);
                        if (A.tileIsLocal(i, j)) {
                            #pragma omp task firstprivate(i, j, mb, nb, i_global, j_global)
                            {
                                auto A_ij = A(i, j);
                                for (int64_t ii = 0; ii < mb; ++ii) {
                                    for (int64_t jj = 0; jj < nb; ++jj) {
                                        A_ij.at(ii, jj) = std::abs(i_global + ii
                                                                   - (j_global + jj));
                                    }
                                }
                            }
                        }
                        j_global += nb;
                    }
                    i_global += mb;
                }
                #pragma omp taskwait
            }
            break;
        }

        case TestMatrixType::gfpp: {
            set(zero, one, A);
            #pragma omp parallel for collapse(2)
            for (int64_t i = 0; i < mt; ++i) {
                for (int64_t j = 0; j < nt; ++j) {
                    if (A.tileIsLocal(i, j)) {
                        auto A_ij = A(i, j);
                        const int64_t mb = A.tileMb(i);
                        const int64_t nb = A.tileNb(j);
                        if (i == j) {
                            A_ij.set(-0.5, one);
                            for (int64_t ii = 0; ii < mb-1; ++ii) {
                                for (int64_t jj = ii+1; jj < nb; ++jj) {
                                    A_ij.at(ii, jj) = -zero;
                                }
                            }
                        }
                        else if (i < j) {
                            A_ij.set(zero);
                        }
                        else if (i > j) {
                            A_ij.set(-0.5);
                        }
                        if (j == nt-1) {
                            const int64_t jj = nb - 1;
                            for (int64_t ii = 0; ii < mb; ++ii) {
                                A_ij.at(ii, jj) = 1;
                            }
                        }
                    }
                }
            }
            break;
        }

        case TestMatrixType::orthog: {
            const int64_t max_mn = std::max(n, m);
            const scalar_t outer_const = sqrt(scalar_t(2)/scalar_t(max_mn+1));
            // pi = acos(-1)
            const scalar_t inner_const = scalar_t(acos(-1))/scalar_t(max_mn+1);
            #pragma omp parallel
            #pragma omp master
            {
                int64_t i_global = 1;
                for (int64_t i = 0; i < mt; ++i) {
                    const int64_t mb = A.tileMb(i);
                    int64_t j_global = 1;
                    for (int64_t j = 0; j < nt; ++j) {
                        const int64_t nb = A.tileNb(j);
                        if (A.tileIsLocal(i, j)) {
                            #pragma omp task firstprivate(i, j, mb, nb, \
                                                          i_global, j_global)
                            {
                                auto A_ij = A(i, j);
                                for (int64_t ii = 0; ii < mb; ++ii) {
                                    for (int64_t jj = 0; jj < nb; ++jj) {
                                        scalar_t a = scalar_t(i_global + ii)
                                                     * scalar_t(j_global + jj)
                                                     * inner_const;
                                        A_ij.at(ii, jj) = outer_const*sin(a);
                                    }
                                }
                            }
                        }
                        j_global += nb;
                    }
                    i_global += mb;
                }
                #pragma omp taskwait
            }
            break;
        }

        case TestMatrixType::riemann: {
            #pragma omp parallel
            #pragma omp master
            {
                int64_t i_global = 1;
                for (int64_t i = 0; i < mt; ++i) {
                    const int64_t mb = A.tileMb(i);
                    int64_t j_global = 1;
                    for (int64_t j = 0; j < nt; ++j) {
                        const int64_t nb = A.tileNb(j);
                        if (A.tileIsLocal(i, j)) {
                            #pragma omp task firstprivate(i, j, mb, nb, \
                                                          i_global, j_global)
                            {
                                auto A_ij = A(i, j);
                                for (int64_t ii = 0; ii < mb; ++ii) {
                                    for (int64_t jj = 0; jj < nb; ++jj) {
                                        int64_t B_i = i_global + ii + 2;
                                        int64_t B_j = j_global + jj + 2;
                                        if (B_i % B_j == 0) {
                                            A_ij.at(ii, jj) = B_i - 1;
                                        }
                                        else {
                                            A_ij.at(ii, jj) = -1;
                                        }
                                    }
                                }
                            }
                        }
                        j_global += nb;
                    }
                    i_global += mb;
                }
                #pragma omp taskwait
            }
            break;
        }

        case TestMatrixType::ris: {
            const int64_t max_mn = std::max(n, m);
            #pragma omp parallel
            #pragma omp master
            {
                int64_t i_global = 1;
                for (int64_t i = 0; i < mt; ++i) {
                    const int64_t mb = A.tileMb(i);
                    int64_t j_global = 1;
                    for (int64_t j = 0; j < nt; ++j) {
                        const int64_t nb = A.tileNb(j);
                        if (A.tileIsLocal(i, j)) {
                            #pragma omp task firstprivate(i, j, mb, nb, \
                                                          i_global, j_global)
                            {
                                auto A_ij = A(i, j);
                                for (int64_t ii = 0; ii < mb; ++ii) {
                                    for (int64_t jj = 0; jj < nb; ++jj) {
                                        A_ij.at(ii, jj) = 0.5 / (max_mn-(i_global+ii+1)-(j_global+jj+1)+1.5);
                                    }
                                }
                            }
                        }
                        j_global += nb;
                    }
                    i_global += mb;
                }
                #pragma omp taskwait
            }
            break;
        }

        case TestMatrixType::rand:
        case TestMatrixType::rands:
        case TestMatrixType::randn:
        case TestMatrixType::randb: {
            int64_t idist = (int64_t) type;
            if (type == TestMatrixType::randb) {
                idist = (int64_t) TestMatrixType::rand;
            }
            auto Tmp = A.emptyLike();
            #pragma omp parallel for collapse(2)
            for (int64_t j = 0; j < nt; ++j) {
                for (int64_t i = 0; i < mt; ++i) {
                    if (A.tileIsLocal(i, j)) {
                        A.tileGetForWriting( i, j, LayoutConvert::ColMajor );
                        auto Aij = A(i, j);
                        scalar_t* data = Aij.data();
                        int64_t lda = Aij.stride();
                        // Added local seed array for each process to prevent race condition contention of iseed
                        int64_t tile_iseed[4];
                        tile_iseed[0] = (iseed[0] + i/4096) % 4096;
                        tile_iseed[1] = (iseed[1] + j/2048) % 4096;
                        tile_iseed[2] = (iseed[2] + i)      % 4096;
                        tile_iseed[3] = (iseed[3] + j*2)    % 4096;
                        for (int64_t k = 0; k < Aij.nb(); ++k) {
                            lapack::larnv(idist, tile_iseed, Aij.mb(), &data[k*lda]);
                        }
                        if (type == TestMatrixType::randb) {
                            for (int64_t jj = 0; jj < Aij.nb(); ++jj) {
                                for (int64_t ii = 0; ii < Aij.mb(); ++ii) {
                                    Aij.at(ii, jj) = (std::fabs(Aij.at(ii, jj)) >= 0.5);
                                }
                            }
                        }

                        // Make it diagonally dominant
                        if (dominant) {
                            if (i == j) {
                                int bound = std::min( Aij.mb(), Aij.nb() );
                                for (int ii = 0; ii < bound; ++ii) {
                                    Aij.at(ii, ii) += n;
                                }
                            }
                        }
                        // Scale the matrix
                        if (sigma_max != 1) {
                            scalar_t s = sigma_max;
                            tile::scale( s, Aij );
                        }
                    }
                }
            }
            break;
        }

        case TestMatrixType::diag:
            generate_sigma( params, dist, false, cond, sigma_max, A, Sigma, iseed );
            break;

        case TestMatrixType::svd:
            generate_svd( params, dist, cond, condD, sigma_max, A, Sigma, iseed );
            break;

        case TestMatrixType::poev:
            generate_heev( params, dist, false, cond, condD, sigma_max, A, Sigma, iseed );
            break;

        case TestMatrixType::heev:
            generate_heev( params, dist, true, cond, condD, sigma_max, A, Sigma, iseed );
            break;

        case TestMatrixType::geev:
            generate_geev( params, dist, cond, sigma_max, A, Sigma, iseed );
            break;

        case TestMatrixType::geevx:
            generate_geevx( params, dist, cond, sigma_max, A, Sigma, iseed );
            break;
    }

    if (! (type == TestMatrixType::rand  ||
           type == TestMatrixType::rands ||
           type == TestMatrixType::randn ||
           type == TestMatrixType::randb) && dominant) {
        // make diagonally dominant; strict unless diagonal has zeros
        slate_error("Not implemented yet");
        throw std::exception();  // not implemented
    }
    A.tileUpdateAllOrigin();
}

// -----------------------------------------------------------------------------
/// Generates an m-by-n trapezoid-storage test matrix.
/// Handles Trapezoid, Triangular, Symmetric, and Hermitian matrices.
/// @see generate_matrix
/// @ingroup generate_matrix
///
template <typename scalar_t>
void generate_matrix(
    MatrixParams& params,
    slate::BaseTrapezoidMatrix<scalar_t>& A,
    std::vector< blas::real_type<scalar_t> >& Sigma )
{
    using real_t = blas::real_type<scalar_t>;

    // Constants
    const real_t nan = std::numeric_limits<real_t>::quiet_NaN();
    const real_t d_zero = 0;
    const real_t d_one  = 1;
    const scalar_t zero = 0;
    const scalar_t one  = 1;

    // ----------
    // set Sigma to unknown (nan)
    lapack::laset( lapack::MatrixType::General, Sigma.size(), 1,
                   nan, nan, Sigma.data(), Sigma.size() );

    TestMatrixType type;
    TestMatrixDist dist;
    real_t cond;
    real_t condD;
    real_t sigma_max;
    bool dominant;
    decode_matrix<scalar_t>(params, A, type, dist, cond, condD, sigma_max, dominant);

    int64_t iseed[4];
    configure_seed(A.mpiComm(), params.seed(), iseed);

    int64_t n = A.n();
    int64_t nt = A.nt();
    int64_t mt = A.mt();
    // ----- generate matrix
    switch (type) {
        case TestMatrixType::zero:
            set(zero, zero, A);
            lapack::laset( lapack::MatrixType::General, Sigma.size(), 1,
                d_zero, d_zero, Sigma.data(), Sigma.size() );
            break;

        case TestMatrixType::one:
            set(one, one, A);
            lapack::laset( lapack::MatrixType::General, Sigma.size(), 1,
                d_zero, d_one, Sigma.data(), Sigma.size() );
            if (Sigma.size() >= 1) {
                Sigma[0] = n;
            }
            break;

        case TestMatrixType::identity:
            set(zero, one, A);
            lapack::laset( lapack::MatrixType::General, Sigma.size(), 1,
                d_one, d_one, Sigma.data(), Sigma.size() );
            break;

        case TestMatrixType::jordan: {
            set(zero, one, A ); // ones on diagonal
            if (A.uplo() == Uplo::Lower) {
                // ones on sub-diagonal
                for (int64_t i = 0; i < nt; ++i) {
                    // Set 1 element from sub-diagonal tile to 1.
                    if (i > 0) {
                        if (A.tileIsLocal(i, i-1)) {
                            A.tileGetForWriting( i, i-1, LayoutConvert::ColMajor );
                            auto T = A(i, i-1);
                            T.at(0, T.nb()-1) = 1.;
                        }
                    }

                    // Set 1 element from sub-diagonal tile to 1.
                    if (A.tileIsLocal(i, i)) {
                        A.tileGetForWriting( i, i, LayoutConvert::ColMajor );
                        auto T = A(i, i);
                        auto len = T.nb();
                        for (int j = 0; j < len-1; ++j) {
                            T.at(j+1, j) = 1.;
                        }
                    }
                }
            }
            else { // upper
                // ones on sub-diagonal
                for (int64_t i = 0; i < nt; ++i) {
                    // Set 1 element from sub-diagonal tile to 1.
                    if (i > 0) {
                        if (A.tileIsLocal(i-1, i)) {
                            A.tileGetForWriting( i-1, i, LayoutConvert::ColMajor );
                            auto T = A(i-1, i);
                            T.at(T.nb()-1, 0) = 1.;
                        }
                    }

                    // Set 1 element from sub-diagonal tile to 1.
                    if (A.tileIsLocal(i, i)) {
                        A.tileGetForWriting( i, i, LayoutConvert::ColMajor );
                        auto T = A(i, i);
                        auto len = T.nb();
                        for (int j = 0; j < len-1; ++j) {
                            T.at(j, j+1) = 1.;
                        }
                    }
                }
            }
            break;
        }

        case TestMatrixType::rand:
        case TestMatrixType::rands:
        case TestMatrixType::randn:
        case TestMatrixType::randb: {
            int64_t idist = (int64_t) type;
            if (type == TestMatrixType::randb) {
                idist = (int64_t) TestMatrixType::rand;
            }
            if (A.uplo() == Uplo::Lower) {
                // TODO: Enable the following pragma to collapse loops for OpenMP 5.0.
                // OpenMP can parallelize the outer loop,
                // but since the inner loop depends on the outer loop,
                // it runs into issues. It appears this is solved in OpenMP 5.0,
                // but that requires gcc 11, which is under development.
                #pragma omp parallel for
                for (int64_t j=0; j < nt; ++j) {
                    for (int64_t i = j; i < mt; ++i) {
                        if (A.tileIsLocal(i, j)) {
                            A.tileGetForWriting( i, j, LayoutConvert::ColMajor );
                            auto Aij = A(i, j);
                            scalar_t* data = Aij.data();
                            int64_t lda = Aij.stride();
                            // Added local seed array for each process to prevent race condition contention of iseed
                            int64_t tile_iseed[4];
                            tile_iseed[0] = (iseed[0] + i/4096) % 4096;
                            tile_iseed[1] = (iseed[1] + j/2048) % 4096;
                            tile_iseed[2] = (iseed[2] + i)      % 4096;
                            tile_iseed[3] = (iseed[3] + j*2)    % 4096;
                            for (int64_t k = 0; k < Aij.nb(); ++k) {
                                lapack::larnv(idist, tile_iseed, Aij.mb(), &data[k*lda]);
                            }
                            if (type == TestMatrixType::randb) {
                                for (int64_t jj = 0; jj < Aij.nb(); ++jj) {
                                    for (int64_t ii = 0; ii < Aij.mb(); ++ii) {
                                        Aij.at(ii, jj) = (std::fabs(Aij.at(ii, jj)) >= 0.5);
                                    }
                                }
                            }

                            // Make it diagonally dominant
                            if (dominant) {
                                if (i == j) {
                                    int bound = std::min( Aij.mb(), Aij.nb() );
                                    for (int ii = 0; ii < bound; ++ii) {
                                        Aij.at(ii, ii) += n;
                                    }
                                }
                            }
                            // Scale the matrix
                            if (sigma_max != 1) {
                                scalar_t s = sigma_max;
                                tile::scale( s, Aij );
                            }
                        }
                    }
                }
            }
            else { // upper
                // TODO: Enable the following pragma to collapse loops for OpenMP 5.0.
                // OpenMP can parallelize the outer loop,
                // but since the inner loop depends on the outer loop,
                // it runs into issues. It appears this is solved in OpenMP 5.0,
                // but that requires gcc 11, which is under development.
                #pragma omp parallel for
                for (int64_t j = 0; j < nt; ++j) {
                    for (int64_t i = 0; i <= j && i < mt; ++i) {  // upper trapezoid
                        if (A.tileIsLocal(i, j)) {
                            A.tileGetForWriting( i, j, LayoutConvert::ColMajor );
                            auto Aij = A(i, j);
                            scalar_t* data = Aij.data();
                            int64_t lda = Aij.stride();
                            // Added local seed array for each process to prevent race condition contention of iseed
                            int64_t tile_iseed[4];
                            tile_iseed[0] = (iseed[0] + i/4096) % 4096;
                            tile_iseed[1] = (iseed[1] + j/2048) % 4096;
                            tile_iseed[2] = (iseed[2] + i)      % 4096;
                            tile_iseed[3] = (iseed[3] + j*2)    % 4096;
                            for (int64_t k = 0; k < Aij.nb(); ++k) {
                                lapack::larnv(idist, tile_iseed, Aij.mb(), &data[k*lda]);
                            }
                            if (type == TestMatrixType::randb) {
                                for (int64_t jj = 0; jj < Aij.nb(); ++jj) {
                                    for (int64_t ii = 0; ii < Aij.mb(); ++ii) {
                                        Aij.at(ii, jj) = (std::fabs(Aij.at(ii, jj)) >= 0.5);
                                    }
                                }
                            }

                            // Make it diagonally dominant
                            if (dominant) {
                                if (i == j) {
                                    int bound = std::min( Aij.mb(), Aij.nb() );
                                    for (int ii = 0; ii < bound; ++ii) {
                                        Aij.at(ii, ii) += n;
                                    }
                                }
                            }
                            // Scale the matrix
                            if (sigma_max != 1) {
                                scalar_t s = sigma_max;
                                tile::scale( s, Aij );
                            }
                        }
                    }
                }
            }
            break;
        }

        case TestMatrixType::diag:
            generate_sigma( params, dist, false, cond, sigma_max, A, Sigma, iseed );
            break;

        case TestMatrixType::poev:
        case TestMatrixType::heev:
        default:
            slate_error("Not implemented yet");
            throw std::exception();  // not implemented
    }

    if (! (type == TestMatrixType::rand  ||
           type == TestMatrixType::rands ||
           type == TestMatrixType::randn ||
           type == TestMatrixType::randb) && dominant) {
        // make diagonally dominant; strict unless diagonal has zeros
        slate_error("Not implemented yet");
        throw std::exception();  // not implemented
    }

    A.tileUpdateAllOrigin();
}

// -----------------------------------------------------------------------------
/// Generates an m-by-n Hermitian-storage test matrix.
/// Handles Hermitian matrices.
/// Diagonal elements of a Hermitian matrix must be real;
/// their imaginary part must be 0.
/// @see generate_matrix
/// @ingroup generate_matrix
///
template <typename scalar_t>
void generate_matrix(
    MatrixParams& params,
    slate::HermitianMatrix<scalar_t>& A,
    std::vector< blas::real_type<scalar_t> >& Sigma )
{
    slate::BaseTrapezoidMatrix<scalar_t>& TZ = A;
    generate_matrix( params, TZ, Sigma );

    // Set diagonal to real.
    #pragma omp parallel for
    for (int64_t i = 0; i < A.mt(); ++i) {
        if (A.tileIsLocal( i, i )) {
            A.tileGetForWriting( i, i, LayoutConvert::ColMajor );
            auto T = A( i, i );
            int64_t mb = T.mb();
            for (int64_t ii = 0; ii < mb; ++ii) {
                T.at( ii, ii ) = std::real( T( ii, ii ) );
            }
        }
    }
    A.tileUpdateAllOrigin();
}
// -----------------------------------------------------------------------------
/// Overload without Sigma.
/// @see generate_matrix()
/// @ingroup generate_matrix
///
template <typename scalar_t>
void generate_matrix(
    MatrixParams& params,
    slate::Matrix<scalar_t>& A )
{
    using real_t = blas::real_type<scalar_t>;
    std::vector<real_t> dummy;
    generate_matrix( params, A, dummy );
}

/// Overload without Sigma.
/// @see generate_matrix()
/// @ingroup generate_matrix
///
template <typename scalar_t>
void generate_matrix(
    MatrixParams& params,
    slate::BaseTrapezoidMatrix<scalar_t>& A )
{
    using real_t = blas::real_type<scalar_t>;
    std::vector<real_t> dummy;
    generate_matrix( params, A, dummy );
}

/// Overload without Sigma.
/// @see generate_matrix()
/// @ingroup generate_matrix
///
template <typename scalar_t>
void generate_matrix(
    MatrixParams& params,
    slate::HermitianMatrix<scalar_t>& A )
{
    using real_t = blas::real_type<scalar_t>;
    std::vector<real_t> dummy;
    generate_matrix( params, A, dummy );
}
// -----------------------------------------------------------------------------
// explicit instantiations
template
void generate_matrix(
    MatrixParams& params,
    slate::Matrix<float>& A );

template
void generate_matrix(
    MatrixParams& params,
    slate::Matrix<double>& A );

template
void generate_matrix(
    MatrixParams& params,
    slate::Matrix< std::complex<float> >& A );

template
void generate_matrix(
    MatrixParams& params,
    slate::Matrix< std::complex<double> >& A );

template
void generate_matrix(
    MatrixParams& params,
    slate::BaseTrapezoidMatrix<float>& A);

template
void generate_matrix(
    MatrixParams& params,
    slate::BaseTrapezoidMatrix<double>& A);

template
void generate_matrix(
    MatrixParams& params,
    slate::BaseTrapezoidMatrix< std::complex<float> >& A);

template
void generate_matrix(
    MatrixParams& params,
    slate::BaseTrapezoidMatrix< std::complex<double> >& A);

template
void generate_matrix(
    MatrixParams& params,
    slate::HermitianMatrix<float>& A);

template
void generate_matrix(
    MatrixParams& params,
    slate::HermitianMatrix<double>& A);

template
void generate_matrix(
    MatrixParams& params,
    slate::HermitianMatrix< std::complex<float> >& A);

template
void generate_matrix(
    MatrixParams& params,
    slate::HermitianMatrix< std::complex<double> >& A);

template
void decode_matrix<float>(
    MatrixParams& params,
    BaseMatrix<float>& A,
    TestMatrixType& type,
    TestMatrixDist& dist,
    blas::real_type<float>& cond,
    blas::real_type<float>& condD,
    blas::real_type<float>& sigma_max,
    bool& dominant);

template
void decode_matrix<double>(
    MatrixParams& params,
    BaseMatrix<double>& A,
    TestMatrixType& type,
    TestMatrixDist& dist,
    blas::real_type<double>& cond,
    blas::real_type<double>& condD,
    blas::real_type<double>& sigma_max,
    bool& dominant);

template
void decode_matrix<std::complex<float>>(
    MatrixParams& params,
    BaseMatrix<std::complex<float>>& A,
    TestMatrixType& type,
    TestMatrixDist& dist,
    blas::real_type<std::complex<float>>& cond,
    blas::real_type<std::complex<float>>& condD,
    blas::real_type<std::complex<float>>& sigma_max,
    bool& dominant);

template
void decode_matrix<std::complex<double>>(
    MatrixParams& params,
    BaseMatrix<std::complex<double>>& A,
    TestMatrixType& type,
    TestMatrixDist& dist,
    blas::real_type<std::complex<double>>& cond,
    blas::real_type<std::complex<double>>& condD,
    blas::real_type<std::complex<double>>& sigma_max,
    bool& dominant);
} // namespace slate
