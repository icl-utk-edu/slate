// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
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
/// Print real value to a buffer buf of length buf_len.
/// For w = width and p = precision:
/// - integers are printed with %v.0f where v = w-p
/// - small values < 0.01 or large values > threshold are printed with %w.pg
/// - modest values are printed with %w.pf.
/// To ensure data fits, set threshold = 10^(w - p - 2) and w >= p + 6.
///
/// @ingroup print
///
template <typename real_t,
          enable_if_t< std::is_floating_point< real_t >::value >* = nullptr>
int snprintf_value(
    char* buf, size_t buf_len,
    int width, int precision,
    real_t value );

//------------------------------------------------------------------------------
/// Overload to print integer values.
///
/// @ingroup print
///
template <typename integer_t,
          enable_if_t< std::is_integral< integer_t >::value >* = nullptr>
int snprintf_value(
    char* buf, size_t buf_len,
    int width, int precision,
    integer_t value );

//------------------------------------------------------------------------------
/// Overload to print complex values as " <real> + <imag>i".
///
/// @ingroup print
///
template <typename real_t>
int snprintf_value(
    char* buf, size_t buf_len,
    int width, int precision,
    std::complex<real_t> value);

//------------------------------------------------------------------------------
// Tile

//------------------------------------------------------------------------------
/// Print a SLATE tile, on either CPU or GPU.
/// Does not change MSI status. No MPI is involved.
///
/// @param[in] label
///     Label to print before the tile.
///
/// @param[in] A
///     The tile to print.
///
/// @param[in] queue
///     BLAS++ queue for device operations.
///
/// @param[in] opts
///     Options for printing:
///     - Option::PrintPrecision: number of digits (default 4).
///     - Option::PrintWidth: total width of fields (default precision + 6).
///     - Option::PrintVerbose: 0 to disable (default 4).
///
/// @ingroup print
///
template <typename scalar_t>
void print(
    const char* label,
    slate::Tile<scalar_t>& A,
    blas::Queue& queue,
    slate::Options const& opts = Options());

//------------------------------------------------------------------------------
// Matrix

//------------------------------------------------------------------------------
/// Print a SLATE distributed matrix.
/// Rank 0 does the printing, and must have enough memory to fit one entire
/// block row of the matrix.
/// For block-sparse matrices, missing tiles are printed as "nan".
///
/// @param[in] label
///     Label to print before the matrix.
///
/// @param[in] A
///     The matrix to print.
///
/// @param[in] opts
///     Options for printing:
///     - Option::PrintPrecision: number of digits (default 4).
///     - Option::PrintWidth: total width of fields (default 10 or precision + 6).
///     - Option::PrintVerbose:
///         - 0: disable
///         - 1: print header only
///         - 2: abbreviate rows and columns (default for large matrices)
///         - 3: abbreviate rows
///         - 4: print all (default)
///         - 5: abbreviate columns
///     - Option::PrintEdgeItems: number of edge items to print in abbreviated mode.
///
/// @ingroup print
///
template <typename scalar_t>
void print(
    const char* label,
    slate::Matrix<scalar_t>& A,
    slate::Options const& opts = Options());

//------------------------------------------------------------------------------
/// Print a SLATE distributed band matrix.
/// Rank 0 does the printing, and must have enough memory to fit one entire
/// block row of the matrix.
/// Tiles outside the bandwidth are printed as "0", with no trailing decimals.
/// For block-sparse matrices, missing tiles are printed as "nan".
///
/// @param[in] label
///     Label to print before the matrix.
///
/// @param[in] A
///     The matrix to print.
///
/// @param[in] opts
///     See print( const char*, Matrix<scalar_t>&, Options const& ) for options.
///
/// @ingroup print
///
template <typename scalar_t>
void print(
    const char* label,
    slate::BandMatrix<scalar_t>& A,
    slate::Options const& opts = Options());

//------------------------------------------------------------------------------
/// Print a SLATE distributed BaseTriangular (triangular, symmetric, and
/// Hermitian) band matrix.
/// Rank 0 does the printing, and must have enough memory to fit one entire
/// block row of the matrix.
/// Tiles outside the bandwidth are printed as "0", with no trailing decimals.
/// For block-sparse matrices, missing tiles are printed as "nan".
///
/// Entries in the A.uplo triangle are printed; entries in the opposite
/// triangle are printed as "nan".
///
/// @param[in] label
///     Label to print before the matrix.
///
/// @param[in] A
///     The matrix to print.
///
/// @param[in] opts
///     See print( const char*, Matrix<scalar_t>&, Options const& ) for options.
///
/// @ingroup print
///
template <typename scalar_t>
void print(
    const char* label,
    slate::BaseTriangularBandMatrix<scalar_t>& A,
    slate::Options const& opts = Options());

//------------------------------------------------------------------------------
/// Print a SLATE distributed Hermitian matrix.
/// Also prints Matlab tril or triu command to fix entries in opposite triangle.
///
/// @param[in] label
///     Label to print before the matrix.
///
/// @param[in] A
///     The matrix to print.
///
/// @param[in] opts
///     See print( const char*, Matrix<scalar_t>&, Options const& ) for options.
///
/// @ingroup print
///
template <typename scalar_t>
void print(
    const char* label,
    slate::HermitianMatrix<scalar_t>& A,
    slate::Options const& opts = Options());

//------------------------------------------------------------------------------
/// Print a SLATE distributed symmetric matrix.
/// Also prints Matlab tril or triu command to fix entries in opposite triangle.
///
/// @param[in] label
///     Label to print before the matrix.
///
/// @param[in] A
///     The matrix to print.
///
/// @param[in] opts
///     See print( const char*, Matrix<scalar_t>&, Options const& ) for options.
///
/// @ingroup print
///
template <typename scalar_t>
void print(
    const char* label,
    slate::SymmetricMatrix<scalar_t>& A,
    slate::Options const& opts = Options());

//------------------------------------------------------------------------------
/// Print a SLATE distributed trapezoid matrix.
/// Also prints Matlab tril or triu command to fix entries in opposite triangle.
///
/// @param[in] label
///     Label to print before the matrix.
///
/// @param[in] A
///     The matrix to print.
///
/// @param[in] opts
///     See print( const char*, Matrix<scalar_t>&, Options const& ) for options.
///
/// @ingroup print
///
template <typename scalar_t>
void print(
    const char* label,
    slate::TrapezoidMatrix<scalar_t>& A,
    slate::Options const& opts = Options());

//------------------------------------------------------------------------------
/// Print a SLATE distributed triangular matrix.
/// Also prints Matlab tril or triu command to fix entries in opposite triangle.
///
/// @param[in] label
///     Label to print before the matrix.
///
/// @param[in] A
///     The matrix to print.
///
/// @param[in] opts
///     See print( const char*, Matrix<scalar_t>&, Options const& ) for options.
///
/// @ingroup print
///
template <typename scalar_t>
void print(
    const char* label,
    slate::TriangularMatrix<scalar_t>& A,
    slate::Options const& opts = Options());

//------------------------------------------------------------------------------
// Vector (array)

//------------------------------------------------------------------------------
/// Print a vector.
/// Every MPI rank does its own printing, so protect with `if (mpi_rank == 0)`
/// as desired.
///
/// @param[in] label
///     Label to print before the vector.
///
/// @param[in] n
///     Size of the vector.
///
/// @param[in] x
///     Vector data.
///
/// @param[in] incx
///     Stride of the vector.
///
/// @param[in] opts
///     See print( const char*, Matrix<scalar_t>&, Options const& ) for options.
///
/// @ingroup print
///
template <typename scalar_t>
void print(
    const char* label,
    int64_t n, scalar_t const* x, int64_t incx,
    slate::Options const& opts = Options());

//------------------------------------------------------------------------------
// std::vector

//------------------------------------------------------------------------------
/// Print a std::vector.
/// Every MPI rank does its own printing, so protect with `if (mpi_rank == 0)`
/// as desired.
///
/// @param[in] label
///     Label to print before the vector.
///
/// @param[in] x
///     Vector data.
///
/// @param[in] opts
///     See print( const char*, Matrix<scalar_t>&, Options const& ) for options.
///
/// @ingroup print
///
template <typename scalar_type>
void print(
    const char* label,
    std::vector<scalar_type> const& x,
    slate::Options const& opts = Options());

} // namespace slate

#endif // SLATE_PRINT_HH
