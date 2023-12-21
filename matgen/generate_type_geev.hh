// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_GENERATE_TYPE_GEEV_HH
#define SLATE_GENERATE_TYPE_GEEV_HH

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
/// Generates matrix using general eigenvalue decomposition, $A = V T V^H$,
/// with orthogonal eigenvectors.
/// Not yet implemented.
///
/// Internal function, called from generate_matrix().
///
/// @ingroup generate_matrix
template <typename scalar_t>
void generate_geev(
    MatgenParams& params,
    TestMatrixDist dist,
    blas::real_type<scalar_t> cond,
    blas::real_type<scalar_t> sigma_max,
    slate::Matrix<scalar_t>& A,
    std::vector< blas::real_type<scalar_t> >& Sigma,
    int64_t seed,
    slate::Options const& opts )
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
    MatgenParams& params,
    TestMatrixDist dist,
    blas::real_type<scalar_t> cond,
    blas::real_type<scalar_t> sigma_max,
    slate::Matrix<scalar_t>& A,
    std::vector< blas::real_type<scalar_t> >& Sigma,
    int64_t seed,
    slate::Options const& opts )
{
    throw std::exception();  // not implemented
}

} // namespace slate

#endif // SLATE_GENERATE_TYPE_GEEV_HH
