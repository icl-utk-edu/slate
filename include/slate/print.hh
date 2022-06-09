// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_PRINT_HH
#define SLATE_PRINT_HH

#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/SymmetricMatrix.hh"
#include "slate/TriangularMatrix.hh"
#include "slate/BandMatrix.hh"
#include "slate/TriangularBandMatrix.hh"
#include "slate/HermitianBandMatrix.hh"
#include "slate/types.hh"

namespace slate {

//------------------------------------------------------------------------------
template <typename real_t>
int snprintf_value(
    char* buf, size_t buf_len,
    int width, int precision,
    real_t value);

// Overload for complex.
template <typename real_t>
int snprintf_value(
    char* buf, size_t buf_len,
    int width, int precision,
    std::complex<real_t> value);

//------------------------------------------------------------------------------
// Tile
template <typename scalar_t>
void print(
    const char* label,
    slate::Tile<scalar_t>& A,
    blas::Queue& queue,
    slate::Options const& opts = Options());

//------------------------------------------------------------------------------
// Matrix
template <typename scalar_t>
void print(
    const char* label,
    slate::Matrix<scalar_t>& A,
    slate::Options const& opts = Options());

//------------------------------------------------------------------------------
template <typename scalar_t>
void print(
    const char* label,
    slate::BandMatrix<scalar_t>& A,
    slate::Options const& opts = Options());

//------------------------------------------------------------------------------
template <typename scalar_t>
void print(
    const char* label,
    slate::BaseTriangularBandMatrix<scalar_t>& A,
    slate::Options const& opts = Options());

//------------------------------------------------------------------------------
template <typename scalar_t>
void print(
    const char* label,
    slate::HermitianMatrix<scalar_t>& A,
    slate::Options const& opts = Options());

//------------------------------------------------------------------------------
template <typename scalar_t>
void print(
    const char* label,
    slate::SymmetricMatrix<scalar_t>& A,
    slate::Options const& opts = Options());

//------------------------------------------------------------------------------
template <typename scalar_t>
void print(
    const char* label,
    slate::TrapezoidMatrix<scalar_t>& A,
    slate::Options const& opts = Options());

//------------------------------------------------------------------------------
template <typename scalar_t>
void print(
    const char* label,
    slate::TriangularMatrix<scalar_t>& A,
    slate::Options const& opts = Options());

//------------------------------------------------------------------------------
// Vector (array)
template <typename scalar_t>
void print(
    const char* label,
    int64_t n, scalar_t const* x, int64_t incx,
    slate::Options const& opts = Options());

//------------------------------------------------------------------------------
// std::vector
template <typename scalar_type>
void print(
    const char* label,
    std::vector<scalar_type> const& x,
    slate::Options const& opts = Options());

} // namespace slate

#endif // SLATE_PRINT_HH
