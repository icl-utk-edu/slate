// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_GENERATE_TYPE_RAND_HH
#define SLATE_GENERATE_TYPE_RAND_HH

#include "slate/slate.hh"
#include "slate/generate_matrix.hh"
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

// -----------------------------------------------------------------------------
/// Generates matrix using either:
/// random uniform entries on (0, 1)
/// random uniform entries on (-1 ,1)
/// random normal entries with mean 0, std 1
/// random uniform entries from {0, 1}
/// random uniform from {-1, 1} 
///
/// with orthogonal eigenvectors.
///
/// Internal function, called from generate_matrix().
///
/// @ingroup generate_matrix
template <typename scalar_t>
void generate_rand( 
    Matrix<scalar_t>& A, 
    TestMatrixType type, bool dominant, blas::real_type<scalar_t> sigma_max,
    int64_t seed, Options const& opt ) 
{

    int64_t n = A.n();
    int64_t mt = A.mt();
    int64_t nt = A.nt();    

    auto rand_dist = slate::random::Dist(int(type));

    #pragma omp parallel
    #pragma omp master
    {
        int64_t j_global = 0;
        for (int64_t j = 0; j < nt; ++j) {
            int64_t i_global = 0;
            for (int64_t i = 0; i < mt; ++i) {
                if (A.tileIsLocal(i, j)) {
                    #pragma omp task slate_omp_default_none shared( A ) \
                        firstprivate( i, j, j_global, i_global, dominant, \
                                      n, sigma_max, seed, rand_dist )
                    {
                        A.tileGetForWriting( i, j, LayoutConvert::ColMajor );
                        auto Aij = A(i, j);
                        slate::random::generate( rand_dist, seed,
                                                 Aij.mb(), Aij.nb(), i_global, j_global,
                                                 Aij.data(), Aij.stride() );

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
                i_global += A.tileMb(i);
            }
            j_global += A.tileNb(j);
        }
    }

} 

} // namespace slate

#endif // SLATE_GENERATE_TYPE_RAND_HH
