// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_PRINT_MATRIX_HH
#define SLATE_PRINT_MATRIX_HH

#include "test.hh"

//------------------------------------------------------------------------------
/// Print a LAPACK matrix. Should be called from only one rank.
///
template <typename scalar_t>
void print_matrix(
    const char* label,
    int64_t m, int64_t n, scalar_t* A, int64_t lda, Params params)
{
    if (params.verbose() == 0)
        return;

    const slate::Options opts = {
        { slate::Option::PrintWidth, params.print_width() },
        { slate::Option::PrintPrecision, params.print_precision() },
        { slate::Option::PrintVerbose, params.verbose() },
        { slate::Option::PrintEdgeItems, params.print_edgeitems() },
    };

    slate::print( label, m, n, A, lda, opts );
}

//------------------------------------------------------------------------------
/// Print a ScaLAPACK distributed matrix.
/// Prints each rank's data as a contiguous block, numbered by the block row &
/// column indices. Rank 0 does the printing.
///
template <typename scalar_t>
void print_matrix(
    const char* label,
    int64_t mlocal, int64_t nlocal, scalar_t* A, int64_t lda,
    int p, int q, MPI_Comm comm,
    Params& params)
{
    if (params.verbose() == 0)
        return;

    const slate::Options opts = {
        { slate::Option::PrintWidth, params.print_width() },
        { slate::Option::PrintPrecision, params.print_precision() },
        { slate::Option::PrintVerbose, params.verbose() },
        { slate::Option::PrintEdgeItems, params.print_edgeitems() },
    };
    slate::print( label, mlocal, nlocal, A, lda, p, q, comm, opts );
}

//------------------------------------------------------------------------------
/// Print a SLATE distributed matrix.
/// Rank 0 does the printing, and must have enough memory to fit one entire
/// block row of the matrix.
/// For block-sparse matrices, missing tiles are print as "nan".
///
template <typename scalar_t>
void print_matrix(
    const char* label,
    slate::Matrix<scalar_t>& A,
    Params& params)
{
    if (params.verbose() == 0)
        return;

    const slate::Options opts = {
        { slate::Option::PrintWidth, params.print_width() },
        { slate::Option::PrintPrecision, params.print_precision() },
        { slate::Option::PrintVerbose, params.verbose() },
        { slate::Option::PrintEdgeItems, params.print_edgeitems() },
    };

    slate::print( label, A, opts );
}

//------------------------------------------------------------------------------
/// Print a SLATE distributed band matrix.
/// Rank 0 does the printing, and must have enough memory to fit one entire
/// block row of the matrix.
/// Tiles outside the bandwidth are printed as "0", with no trailing decimals.
/// For block-sparse matrices, missing tiles are print as "nan".
///
template <typename scalar_t>
void print_matrix(
    const char* label,
    slate::BandMatrix<scalar_t>& A,
    Params& params)
{
    if (params.verbose() == 0)
        return;

    const slate::Options opts = {
        { slate::Option::PrintWidth, params.print_width() },
        { slate::Option::PrintPrecision, params.print_precision() },
        { slate::Option::PrintVerbose, params.verbose() },
        { slate::Option::PrintEdgeItems, params.print_edgeitems() },
    };

    slate::print( label, A, opts );
}

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
void print_matrix(
    const char* label,
    slate::BaseTriangularBandMatrix<scalar_t>& A,
    Params& params)
{
    if (params.verbose() == 0)
        return;

    const slate::Options opts = {
        { slate::Option::PrintWidth, params.print_width() },
        { slate::Option::PrintPrecision, params.print_precision() },
        { slate::Option::PrintVerbose, params.verbose() },
        { slate::Option::PrintEdgeItems, params.print_edgeitems() },
    };

    slate::print( label, A, opts );
}

//------------------------------------------------------------------------------
/// Print a SLATE distributed Hermitian matrix.
/// Also prints Matlab tril or triu command to fix entries in opposite triangle.
/// todo: fix complex diag in Matlab? (Sca)LAPACK ignores imag part.
///
template <typename scalar_t>
void print_matrix(
    const char* label,
    slate::HermitianMatrix<scalar_t>& A,
    Params& params)
{
    if (params.verbose() == 0)
        return;

    const slate::Options opts = {
        { slate::Option::PrintWidth, params.print_width() },
        { slate::Option::PrintPrecision, params.print_precision() },
        { slate::Option::PrintVerbose, params.verbose() },
        { slate::Option::PrintEdgeItems, params.print_edgeitems() },
    };

    slate::print( label, A, opts );
}

//------------------------------------------------------------------------------
/// Print a SLATE distributed symmetric matrix.
/// Also prints Matlab tril or triu command to fix entries in opposite triangle.
///
template <typename scalar_t>
void print_matrix(
    const char* label,
    slate::SymmetricMatrix<scalar_t>& A,
    Params& params)
{
    if (params.verbose() == 0)
        return;

    // Set defaults
    const slate::Options opts = {
        { slate::Option::PrintWidth, params.print_width() },
        { slate::Option::PrintPrecision, params.print_precision() },
        { slate::Option::PrintVerbose, params.verbose() },
        { slate::Option::PrintEdgeItems, params.print_edgeitems() },
    };

    slate::print( label, A, opts );
}

//------------------------------------------------------------------------------
/// Print a SLATE distributed trapezoid matrix.
/// Also prints Matlab tril or triu command to fix entries in opposite triangle.
/// todo: fix unit diag in Matlab.
///
template <typename scalar_t>
void print_matrix(
    const char* label,
    slate::TrapezoidMatrix<scalar_t>& A,
    Params& params)
{
    if (params.verbose() == 0)
        return;

    // Set defaults
    const slate::Options opts = {
        { slate::Option::PrintWidth, params.print_width() },
        { slate::Option::PrintPrecision, params.print_precision() },
        { slate::Option::PrintVerbose, params.verbose() },
        { slate::Option::PrintEdgeItems, params.print_edgeitems() },
    };

    slate::print( label, A, opts );
}

//------------------------------------------------------------------------------
/// Print a SLATE distributed triangular matrix.
/// Also prints Matlab tril or triu command to fix entries in opposite triangle.
/// todo: fix unit diag in Matlab.
///
template <typename scalar_t>
void print_matrix(
    const char* label,
    slate::TriangularMatrix<scalar_t>& A,
    Params& params)
{
    if (params.verbose() == 0)
        return;

    // Set defaults
    const slate::Options opts = {
        { slate::Option::PrintWidth, params.print_width() },
        { slate::Option::PrintPrecision, params.print_precision() },
        { slate::Option::PrintVerbose, params.verbose() },
        { slate::Option::PrintEdgeItems, params.print_edgeitems() },
    };

    slate::print( label, A, opts );
}

#endif // SLATE_PRINT_MATRIX_HH
