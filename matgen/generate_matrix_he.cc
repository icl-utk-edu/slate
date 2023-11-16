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
#include "generate_matrix.hh"
#include "random.hh"

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

