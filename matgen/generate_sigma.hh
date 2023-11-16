// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_GENERATE_SIGMA_HH
#define SLATE_GENERATE_SIGMA_HH

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
#include "generate_matrix.hh"
#include "random.hh"

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
    int64_t seed )
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
            slate::random::generate(slate::random::Dist::Uniform, seed,
                                    Sigma.size(), 1, 0, 0,
                                    Sigma.data(), Sigma.size());
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
            auto rand_dist = slate::random::Dist(int(dist));
            slate::random::generate(rand_dist, seed,
                                    Sigma.size(), 1, 0, 0,
                                    Sigma.data(), 1);
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
            // use j_offset==1 to get different random values than above
            slate::random::generate(slate::random::Dist::Uniform, seed,
                                    1, 1, i, 1,
                                    &rand, 1);
            if (rand > 0.5) {
                Sigma[i] = -Sigma[i];
            }
        }
    }

    // copy Sigma => A
    int64_t min_mt_nt = std::min(A.mt(), A.nt());
    set(zero, zero, A);
    int64_t S_index = 0;
    #pragma omp parallel
    #pragma omp master
    for (int64_t i = 0; i < min_mt_nt; ++i) {
        if (A.tileIsLocal(i, i)) {
            #pragma omp task slate_omp_default_none shared( A, Sigma ) \
                firstprivate( i, S_index )
            {
                A.tileGetForWriting( i, i, LayoutConvert::ColMajor );
                auto T = A(i, i);
                for (int ii = 0; ii < A.tileNb(i); ++ii) {
                    T.at(ii, ii) = Sigma[S_index + ii];
                }
            }
        }
        S_index += A.tileNb(i);
    }

    A.tileUpdateAllOrigin();
}

} // namespace slate

#endif // SLATE_GENERATE_SIGMA_HH
