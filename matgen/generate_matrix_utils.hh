// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_GENERATE_MATRIX_UTILS_HH
#define SLATE_GENERATE_MATRIX_UTILS_HH

#include "slate/slate.hh"
#include "random.hh"
#include "slate/generate_matrix.hh"

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
// Enum declarations.
enum class TestMatrixType {
    rand  = int(slate::random::Dist::Uniform),
    rands = int(slate::random::Dist::UniformSigned),
    randn = int(slate::random::Dist::Normal),
    randb = int(slate::random::Dist::Binary),
    randr = int(slate::random::Dist::BinarySigned),
    zeros,
    ones,
    identity,
    ij,
    jordan,
    chebspec,
    circul,
    fiedler,
    gfpp,
    kms,
    orthog,
    riemann,
    ris,
    zielkeNS,
    diag,
    svd,
    poev,
    heev,
    geev,
    geevx,
};

enum class TestMatrixDist {
    rand  = int(slate::random::Dist::Uniform),
    rands = int(slate::random::Dist::UniformSigned),
    randn = int(slate::random::Dist::Normal),
    arith,
    geo,
    cluster0,
    cluster1,
    rarith,
    rgeo,
    rcluster0,
    rcluster1,
    logrand,
    specified,
    none,
};

//------------------------------------------------------------------------------
// Function forward declarations. 
template <typename scalar_t>
void decode_matrix(
    MatgenParams& params,
    BaseMatrix<scalar_t>& A,
    TestMatrixType& type,
    TestMatrixDist& dist,
    blas::real_type<scalar_t>& cond,
    blas::real_type<scalar_t>& condD,
    blas::real_type<scalar_t>& sigma_max,
    bool& dominant,
    int64_t& zero_col );

void generate_matrix_usage();

std::vector< std::string > split( const std::string& str, 
                                  const std::string& delims );

int64_t configure_seed(MPI_Comm comm, int64_t user_seed);

} // namespace slate

#endif // SLATE_GENERATE_MATRIX_UTILS_HH
