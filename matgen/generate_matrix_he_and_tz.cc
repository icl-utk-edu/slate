// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "../test/test.hh"

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

#include "../test/matrix_params.hh"
#include "generate_matrix.hh"
#include "../test/random.hh"
#include "generate_matrix_utils.hh"
#include "generate_sigma.hh"
#include "set_lambdas.hh"

namespace slate {

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
    std::vector< blas::real_type<scalar_t> >& Sigma,
    slate::Options const& opts)
{
    using entry_type = std::function< scalar_t (int64_t, int64_t) >;
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

    int64_t seed = configure_seed(A.mpiComm(), params.seed());

    int64_t n = A.n();
    int64_t nt = A.nt();
    int64_t mt = A.mt();

    // ----- generate matrix
    switch (type) {
        case TestMatrixType::zeros:
            set(zero, zero, A);
            lapack::laset( lapack::MatrixType::General, Sigma.size(), 1,
                d_zero, d_zero, Sigma.data(), Sigma.size() );
            break;

        case TestMatrixType::ones:
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
        case TestMatrixType::randb:
        case TestMatrixType::randr: {
            auto rand_dist = slate::random::Dist(int(type));
            #pragma omp parallel
            #pragma omp master
            {
                int64_t j_global = 0;
                for (int64_t j = 0; j < nt; ++j) {
                    int64_t i_global = 0;
                    // lower trapezoid is [ j .. mt )
                    // upper trapezoid is [ 0 .. min( j+1, mt ) )
                    int64_t i_start = A.uplo() == Uplo::Lower ? j  : 0;
                    int64_t i_end   = A.uplo() == Uplo::Lower ? mt : std::min( j+1, mt );
                    for (int64_t i = 0; i < i_start; ++i) {
                        i_global += A.tileMb( i );
                    }
                    for (int64_t i = i_start; i < i_end; ++i) {
                        if (A.tileIsLocal( i, j )) {
                            #pragma omp task slate_omp_default_none shared( A ) \
                                firstprivate( i, j, j_global, i_global, dominant, \
                                              n, sigma_max, seed, rand_dist )
                            {
                                A.tileGetForWriting( i, j, LayoutConvert::ColMajor );
                                auto Aij = A( i, j );
                                slate::random::generate(
                                    rand_dist, seed,
                                    Aij.mb(), Aij.nb(), i_global, j_global,
                                    Aij.data(), Aij.stride() );

                                // Make it diagonally dominant
                                if (dominant && i == j) {
                                    int bound = std::min( Aij.mb(), Aij.nb() );
                                    for (int ii = 0; ii < bound; ++ii) {
                                        Aij.at( ii, ii ) += n;
                                    }
                                }
                                // Scale the matrix
                                if (sigma_max != 1) {
                                    scalar_t s = sigma_max;
                                    tile::scale( s, Aij );
                                }
                            }
                        }
                        i_global += A.tileMb( i );
                    }
                    j_global += A.tileNb( j );
                }
            }
            break;
        }

        case TestMatrixType::diag:
            generate_sigma( params, dist, false, cond, sigma_max, A, Sigma, seed );
            break;

        case TestMatrixType::poev:
        case TestMatrixType::heev:
        default:
            snprintf( msg, sizeof( msg ), "'%s' not yet implemented",
                      params.kind().c_str() );
            throw std::runtime_error( msg );
    }

    if (! (type == TestMatrixType::rand  ||
           type == TestMatrixType::rands ||
           type == TestMatrixType::randn ||
           type == TestMatrixType::randb) && dominant) {
        // make diagonally dominant; strict unless diagonal has zeros
        snprintf( msg, sizeof( msg ), "in '%s', dominant not yet implemented",
                  params.kind().c_str() );
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
                    // lower trapezoid is i in [ j, mt )
                    // upper trapezoid is i in [ 0, min( j+1, mt ) )
                    int64_t i_start = A.uplo() == Uplo::Lower ? j  : 0;
                    int64_t i_end   = A.uplo() == Uplo::Lower ? mt : std::min( j+1, mt );
                    for (int64_t i = i_start; i < i_end; ++i) {
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

        // Set row A( zero_col, : ) = 0.
        // Merging this with the above omp parallel would require
        // todo: this should apply only to Hermitian or symmetric matrices,
        // not to triangular or trapezoid matrices. Perhaps
        //     generate_matrix( ..., HermitianMatrix, ... )
        // could set a flag when calling this routine.
        #pragma omp parallel
        #pragma omp master
        {
            int64_t i_global = 0;
            for (int64_t i = 0; i < mt; ++i) {
                int64_t ii = zero_col - i_global;
                if (0 <= ii && ii < A.tileMb( i )) {
                    // upper trapezoid is j in [ i, nt )
                    // lower trapezoid is j in [ 0, min( i+1, nt ) )
                    int64_t j_start = A.uplo() == Uplo::Upper ? i  : 0;
                    int64_t j_end   = A.uplo() == Uplo::Upper ? nt : std::min( i+1, nt );
                    for (int64_t j = j_start; j < j_end; ++j) {
                        if (A.tileIsLocal( i, j )) {
                            #pragma omp task slate_omp_default_none shared( A ) \
                                firstprivate( i, j, ii, zero )
                            {
                                A.tileGetForWriting( i, j, LayoutConvert::ColMajor );
                                auto Aij = A( i, j );
                                lapack::laset(
                                    lapack::MatrixType::General,
                                    1, Aij.nb(), zero, zero,
                                    &Aij.at( ii, 0 ), Aij.stride() );
                            }
                        }
                    }
                }
                i_global += A.tileMb( i );
            }
        }
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
    std::vector< blas::real_type<scalar_t> >& Sigma,
    slate::Options const& opts)
{
    slate::BaseTrapezoidMatrix<scalar_t>& TZ = A;
    generate_matrix( params, TZ, Sigma, opts );

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
    slate::HermitianMatrix<scalar_t>& A,
    slate::Options const& opts)
{
    using real_t = blas::real_type<scalar_t>;
    std::vector<real_t> dummy;
    generate_matrix( params, A, dummy, opts );
}

template <typename scalar_t>
void generate_matrix(
    MatrixParams& params,
    slate::BaseTrapezoidMatrix<scalar_t>& A,
    slate::Options const& opts)
{
    using real_t = blas::real_type<scalar_t>;
    std::vector<real_t> dummy;
    generate_matrix( params, A, dummy, opts );
}

//------------------------------------------------------------------------------
// Explicit instantiations - hermitian matrix.
template
void generate_matrix(
    MatrixParams& params,
    slate::HermitianMatrix<float>& A,
    slate::Options const& opts);

template
void generate_matrix(
    MatrixParams& params,
    slate::HermitianMatrix<double>& A,
    slate::Options const& opts);

template
void generate_matrix(
    MatrixParams& params,
    slate::HermitianMatrix< std::complex<float> >& A,
    slate::Options const& opts);

template
void generate_matrix(
    MatrixParams& params,
    slate::HermitianMatrix< std::complex<double> >& A,
    slate::Options const& opts);

//------------------------------------------------------------------------------
// Explicit insantiations - trapezoid matrix.
template
void generate_matrix(
    MatrixParams& params,
    slate::BaseTrapezoidMatrix<float>& A,
    slate::Options const& opts);

template
void generate_matrix(
    MatrixParams& params,
    slate::BaseTrapezoidMatrix<double>& A,
    slate::Options const& opts);

template
void generate_matrix(
    MatrixParams& params,
    slate::BaseTrapezoidMatrix< std::complex<float> >& A,
    slate::Options const& opts);

template
void generate_matrix(
    MatrixParams& params,
    slate::BaseTrapezoidMatrix< std::complex<double> >& A,
    slate::Options const& opts);

} // namespace slate
