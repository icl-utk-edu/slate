// Copyright (c) 2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

//------------------------------------------------------------------------------
/// @file
///
#ifndef SLATE_ADDMOD_HH
#define SLATE_ADDMOD_HH

#include "slate/Matrix.hh"

#include <vector>

namespace slate {

//------------------------------------------------------------------------------
// auxiliary type for modifications and Woodbury matrices

template <typename scalar_t>
class AddModFactors {
    using real_t = blas::real_type<scalar_t>;
public:
    int64_t block_size;
    int64_t num_modifications;
    BlockFactor factorType;

    Matrix<scalar_t> A;
    Matrix<scalar_t> U_factors;
    Matrix<scalar_t> VT_factors;
    std::vector<std::vector<real_t>> singular_values;
    std::vector<std::vector<scalar_t>> modifications;
    std::vector<std::vector<int64_t>>  modification_indices;
    Matrix<scalar_t> capacitance_matrix;
    Pivots capacitance_pivots;

    Matrix<scalar_t> S_VT_Rinv;
    Matrix<scalar_t> Linv_U;
};

//------------------------------------------------------------------------------
// Routines

template <typename scalar_t>
void gesv_addmod(Matrix<scalar_t>& A, AddModFactors<scalar_t>& W, Matrix<scalar_t>& B,
                 Options const& opts = Options());

template <typename scalar_t>
void gesv_addmod_ir( Matrix<scalar_t>& A, AddModFactors<scalar_t>& W,
                     Matrix<scalar_t>& B,
                     Matrix<scalar_t>& X,
                     int& iter,
                     Options const& opts);

template <typename scalar_t>
void getrf_addmod(Matrix<scalar_t>& A, AddModFactors<scalar_t>& W,
                  Options const& opts = Options());

template <typename scalar_t>
void getrs_addmod(AddModFactors<scalar_t>& W,
                  Matrix<scalar_t>& B,
                  Options const& opts);

template <typename scalar_t>
void trsm_addmod(
    Side side, Uplo uplo,
    scalar_t alpha, AddModFactors<scalar_t>& W,
                           Matrix<scalar_t>& B,
    Options const& opts = Options());

template <typename scalar_t>
void trsmA_addmod(
    Side side, Uplo uplo,
    scalar_t alpha, AddModFactors<scalar_t>& W,
                           Matrix<scalar_t>& B,
    Options const& opts = Options());

template <typename scalar_t>
void trsmB_addmod(
    Side side, Uplo uplo,
    scalar_t alpha, AddModFactors<scalar_t>& W,
                           Matrix<scalar_t>& B,
    Options const& opts = Options());

} // namespace slate

#endif // SLATE_ADDMOD_HH
