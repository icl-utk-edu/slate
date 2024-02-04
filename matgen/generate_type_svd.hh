// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_GENERATE_TYPE_SVD_HH
#define SLATE_GENERATE_TYPE_SVD_HH

#include "slate/slate.hh"
#include "slate/generate_matrix.hh"
#include "generate_sigma.hh"
#include "random.hh"

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
/// Generates matrix using SVD, $A = U Sigma V^H$.
///
/// Internal function, called from generate_matrix().
///
/// @ingroup generate_matrix
template <typename scalar_t>
void generate_svd(
    MatgenParams& params,
    TestMatrixDist dist,
    blas::real_type<scalar_t> cond,
    blas::real_type<scalar_t> condD,
    blas::real_type<scalar_t> sigma_max,
    slate::Matrix<scalar_t>& A,
    std::vector< blas::real_type<scalar_t> >& Sigma,
    int64_t seed,
    slate::Options const& opts )
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
    generate_sigma( params, dist, false, cond, sigma_max, A, Sigma, seed );
    seed += 1;

    // for generate correlation factor, need sum sigma_i^2 = n
    // scaling doesn't change cond
    if (condD != 1) {
        real_t sum_sq = blas::dot( Sigma.size(), Sigma.data(), 1, Sigma.data(), 1 );
        real_t scale = sqrt( Sigma.size() / sum_sq );
        blas::scal( Sigma.size(), scale, Sigma.data(), 1 );

        // copy Sigma to diag(A)
        int64_t S_index = 0;
        #pragma omp parallel
        #pragma omp master
        for (int64_t i = 0; i < min_mt_nt; ++i) {
            if (A.tileIsLocal(i, i)) {
                #pragma omp task slate_omp_default_none shared( A, Sigma ) \
                    firstprivate( i, S_index )
                {
                    A.tileGetForWriting( i, i, LayoutConvert::ColMajor );
                    auto Aii = A(i, i);
                    for (int ii = 0; ii < A.tileNb(i); ++ii) {
                        Aii.at(ii, ii) = Sigma[S_index + ii];
                    }
                }
            }
            S_index += A.tileNb(i);
        }
    }

    // random U, m-by-min_mn
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
    slate::unmqr( slate::Side::Left, slate::Op::NoTrans, U, T, A, opts);

    // random V, n-by-min_mn (stored column-wise in U)
    auto V = U.slice(0, n-1, 0, n-1);
    int64_t V_mt = V.mt();
    int64_t V_nt = V.nt();
    #pragma omp parallel
    #pragma omp master
    {
        int64_t j_global = 0;
        for (int64_t j = 0; j < V_nt; ++j) {
            int64_t i_global = 0;
            for (int64_t i = 0; i < V_mt; ++i) {
                if (A.tileIsLocal(i, j)) {
                    #pragma omp task slate_omp_default_none shared( V ) \
                        firstprivate( i, j, i_global, j_global, seed )
                    {
                        V.tileGetForWriting( i, j, LayoutConvert::ColMajor );
                        auto Vij = V(i, j);
                        slate::random::generate(slate::random::Dist::Normal, seed,
                                                Vij.mb(), Vij.nb(), i_global, j_global,
                                                Vij.data(), Vij.stride());
                    }
                }
                i_global += A.tileMb(i);
            }
            j_global += A.tileNb(j);
        }
    }
    seed += 1;

    slate::geqrf(V, T, opts);

    // A = A*V^H
    slate::unmqr( slate::Side::Right, slate::Op::ConjTrans, V, T, A, opts);

    if (condD != 1) {
        // A = A*W, W orthogonal, such that A has unit column norms,
        // i.e., A^H A is a correlation matrix with unit diagonal
        // TODO: uncomment generate_correlation_factor
        //generate_correlation_factor( A );

        // A = A*D col scaling
        std::vector<real_t> D( n );
        real_t range = log( condD );
        slate::random::generate(slate::random::Dist::Uniform, seed,
                                n, 1, 0, 0,
                                D.data(), n);
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

} // namespace slate

#endif // SLATE_GENERATE_TYPE_SVD_HH
