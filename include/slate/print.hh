// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
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
/// Print a LAPACK matrix. Should be called from only one rank.
///
template <typename scalar_t>
void print(
    const char* label,
    int64_t m, int64_t n, scalar_t* A, int64_t lda,
    slate::Options const& opts = Options());

//------------------------------------------------------------------------------
/// Print a ScaLAPACK distributed matrix.
/// Prints each rank's data as a contiguous block, numbered by the block row &
/// column indices. Rank 0 does the printing.
///
template <typename scalar_t>
void print(
    const char* label,
    int64_t mlocal, int64_t nlocal, scalar_t* A, int64_t lda,
    int p, int q, MPI_Comm comm,
    slate::Options const& opts = Options());

//------------------------------------------------------------------------------
/// Print a SLATE distributed matrix.
/// Rank 0 does the printing, and must have enough memory to fit one entire
/// block row of the matrix.
/// For block-sparse matrices, missing tiles are print as "nan".
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
/// For block-sparse matrices, missing tiles are print as "nan".
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
/// For block-sparse matrices, missing tiles are print as "nan".
///
/// Entries in the A.uplo triangle are printed; entries in the opposite
/// triangle are printed as "nan".
///
/// Having said that, if the printed matrix is a lower triangular matrix,
/// then the routine will print the tiles of upper part of the matrix as "nan",
/// and the lower part tiles that are inside the bandwidth will be printed
/// as they are, whereas the non existing tiles, tiles outside the bandwidth,
/// will be printed as "0", with no trailing decimals.
/// This is to follow MATLAB convention and to make it easier for debugging.
///
template <typename scalar_t>
void print(
    const char* label,
    slate::BaseTriangularBandMatrix<scalar_t>& A,
    slate::Options const& opts = Options());

//------------------------------------------------------------------------------
/// Print a SLATE distributed Hermitian matrix.
/// Also prints Matlab tril or triu command to fix entries in opposite triangle.
/// todo: fix complex diag in Matlab? (Sca)LAPACK ignores imag part.
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
template <typename scalar_t>
void print(
    const char* label,
    slate::SymmetricMatrix<scalar_t>& A,
    slate::Options const& opts = Options());

//------------------------------------------------------------------------------
/// Print a SLATE distributed trapezoid matrix.
/// Also prints Matlab tril or triu command to fix entries in opposite triangle.
/// todo: fix unit diag in Matlab.
///
template <typename scalar_t>
void print(
    const char* label,
    slate::TrapezoidMatrix<scalar_t>& A,
    slate::Options const& opts = Options());

//------------------------------------------------------------------------------
/// Print a SLATE distributed triangular matrix.
/// Also prints Matlab tril or triu command to fix entries in opposite triangle.
/// todo: fix unit diag in Matlab.
///
template <typename scalar_t>
void print(
    const char* label,
    slate::TriangularMatrix<scalar_t>& A,
    slate::Options const& opts = Options());

//------------------------------------------------------------------------------
/// Print a vector.
/// Every MPI rank does its own printing, so protect with `if (mpi_rank == 0)`
/// as desired.
///
template <typename scalar_t>
void print(
    const char* label,
    int64_t n, scalar_t const* x, int64_t incx,
    slate::Options const& opts = Options());

//------------------------------------------------------------------------------
/// Print a vector.
/// Every MPI rank does its own printing, so protect with `if (mpi_rank == 0)`
/// as desired.
///
template <typename scalar_t>
void print(
    const char* label,
    int64_t n, scalar_t const* x, int64_t incx,
    int width=10, int precision=4 );

//------------------------------------------------------------------------------
/// Print a vector.
/// Every MPI rank does its own printing, so protect with `if (mpi_rank == 0)`
/// as desired.
///
template <typename scalar_type>
void print(
    const char* label,
    std::vector<scalar_type> const& x,
    slate::Options const& opts = Options());
//------------------------------------------------------------------------------
/// Print a vector.
/// Every MPI rank does its own printing, so protect with `if (mpi_rank == 0)`
/// as desired.
///
template <typename scalar_type>
void print(
    const char* label,
    std::vector<scalar_type> const& x,
    int width=10, int precision=4 );

} // namespace slate

#endif // SLATE_PRINT_HH
