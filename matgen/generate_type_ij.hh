// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_GENERATE_TYPE_IJ_HH
#define SLATE_GENERATE_TYPE_IJ_HH


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

template <typename scalar_t>
void generate_ij( slate::Matrix<scalar_t>& A,
                  blas::real_type<scalar_t> s, 
                  slate::Options const& opts ) 
{

    int64_t nt = A.nt();
    int64_t mt = A.mt();

    #pragma omp parallel
    #pragma omp master

    int64_t jj = 0;
    for (int64_t j = 0; j < nt; ++j) {
        int64_t ii = 0;
        for (int64_t i = 0; i < mt; ++i) {
            #pragma omp task slate_omp_default_none shared( A ) \
                firstprivate( i, j, ii, jj, s )
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

} // namespace slate

#endif // SLATE_GENERATE_TYPE_IJ_HH
