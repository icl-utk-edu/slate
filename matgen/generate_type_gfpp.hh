// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_GENERATE_TYPE_GFPP_HH
#define SLATE_GENERATE_TYPE_GFPP_HH


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


void generate_gfpp( slate::Matrix<scalar_t>& A,
                    blas::real_type<scalar_t> zero,
	            blas::real_type<scalar_t> one,
                    slate::Options const& opts ) 
{

    int64_t mt = A.mt();
    int64_t nt = A.nt();

    #pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < mt; ++i) {
        for (int64_t j = 0; j < nt; ++j) {
            if (A.tileIsLocal(i, j)) {
                A.tileGetForWriting( i, j, LayoutConvert::ColMajor );
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

}

} // namespace slate

#endif // SLATE_GENERATE_TYPE_GFPP_HH
