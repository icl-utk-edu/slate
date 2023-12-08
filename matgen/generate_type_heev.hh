// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_GENERATE_TYPE_HEEV_HH
#define SLATE_GENERATE_TYPE_HEEV_HH

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
#include "slate/generate_matrix.hh"
#include "../test/random.hh"

namespace slate {
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
    int64_t seed,
    slate::Options const& opts )
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
    generate_sigma( params, dist, rand_sign, cond, sigma_max, A, Sigma, seed );
    seed += 1;

    // random U, m-by-min_mn
    int64_t nt = U.nt();
    int64_t mt = U.mt();

    #pragma omp parallel
    #pragma omp master
    {
        int64_t j_global = 0;
        for (int64_t j = 0; j < nt; ++j) {
            int64_t i_global = 0;
            for (int64_t i = 0; i < mt; ++i) {
                if (A.tileIsLocal(i, j)) {
                    #pragma omp task slate_omp_default_none shared( U ) \
                        firstprivate( i, j, i_global, j_global, seed )
                    {
                        U.tileGetForWriting( i, j, LayoutConvert::ColMajor );
                        auto Uij = U(i, j);
                        slate::random::generate(slate::random::Dist::Normal, seed,
                                                Uij.mb(), Uij.nb(), i_global, j_global,
                                                Uij.data(), Uij.stride());
                    }
                }
                i_global += A.tileMb(i);
            }
            j_global += A.tileNb(j);
        }
    }
    seed += 1;

    // we need to make each random column into a Householder vector;
    // no need to update subsequent columns (as in geqrf).
    // However, currently we do geqrf here,
    // since we don't have a way to make Householder vectors (no distributed larfg).
    slate::geqrf(U, T, opts);

    // A = U*A
    slate::unmqr( slate::Side::Left, slate::Op::NoTrans, U, T, A, opts );

    // A = A*U^H
    slate::unmqr( slate::Side::Right, slate::Op::ConjTrans, U, T, A, opts );

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
        slate::random::generate(slate::random::Dist::Uniform, seed,
                                n, 1, 0, 0,
                                D.data(), n);
        for (int64_t i = 0; i < n; ++i) {
            D[i] = exp( D[i] * range );
        }

        int64_t J_index = 0;
        #pragma omp parallel
        #pragma omp master
        for (int64_t j = 0; j < nt; ++j) {
            int64_t I_index = 0;
            for (int64_t i = 0; i < mt; ++i) {
                if (A.tileIsLocal(i, j)) {
                    #pragma omp task slate_omp_default_none shared( A, D ) \
                        firstprivate( i, j, I_index, J_index )
                    {
                        A.tileGetForWriting( i, j, LayoutConvert::ColMajor );
                        auto Aij = A(i, j);
                        for (int jj = 0; jj < A.tileMb(j); ++jj) {
                            for (int ii = 0; ii < A.tileMb(i); ++ii) {
                                Aij.at(ii, jj) *= D[I_index + ii] * D[J_index + jj];
                            }
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

} // namespace slate

#endif // SLATE_GENERATE_TYPE_HEEV_HH
