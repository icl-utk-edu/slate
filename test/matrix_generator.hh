// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_MATRIX_GENERATOR_HH
#define SLATE_MATRIX_GENERATOR_HH

#include <exception>
#include <complex>
#include <ctype.h>

#include "testsweeper.hh"
#include "blas.hh"
#include "lapack.hh"
#include "slate/slate.hh"

#include "matrix_params.hh"

namespace slate {

// -----------------------------------------------------------------------------
template <typename scalar_t>
void generate_matrix(
    MatrixParams& params,
    slate::Matrix< scalar_t >& A,
    std::vector< blas::real_type<scalar_t> >& Sigma,
    slate::Options const& opts = slate::Options());

template <typename scalar_t>
void generate_matrix(
    MatrixParams& params,
    slate::BaseTrapezoidMatrix< scalar_t >& A,
    std::vector< blas::real_type<scalar_t> >& Sigma,
    slate::Options const& opts = slate::Options());

template <typename scalar_t>
void generate_matrix(
    MatrixParams& params,
    slate::HermitianMatrix< scalar_t >& A,
    std::vector< blas::real_type<scalar_t> >& Sigma,
    slate::Options const& opts = slate::Options());

// Overload without sigma.
template <typename scalar_t>
void generate_matrix(
    MatrixParams& params,
    slate::Matrix< scalar_t >& A,
    slate::Options const& opts = slate::Options());

template <typename scalar_t>
void generate_matrix(
    MatrixParams& params,
    slate::BaseTrapezoidMatrix< scalar_t >& A,
    slate::Options const& opts = slate::Options());

template <typename scalar_t>
void generate_matrix(
    MatrixParams& params,
    slate::HermitianMatrix< scalar_t >& A,
    slate::Options const& opts = slate::Options());

void generate_matrix_usage();

} // namespace slate

#endif // SLATE_MATRIX_GENERATOR_HH
