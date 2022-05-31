// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
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
void print(
    const char* label,
    int64_t m, int64_t n, scalar_t* A, int64_t lda,
    slate::Options const& opts = slate::Options())
{
    int64_t width = slate::get_option<int64_t>( opts, slate::Option::PrintWidth, 10 );
    int64_t precision = slate::get_option<int64_t>( opts, slate::Option::PrintPrecision, 4 );
    int64_t verbose = slate::get_option<int64_t>( opts, slate::Option::PrintVerbose, 4 );

    if (verbose == 0)
        return;

    width = std::max(width, precision + 6);

    char buf[ 1024 ];
    std::string msg;

    printf("%% LAPACK matrix\n");
    printf("%s = [\n", label);
    for (int64_t i = 0; i < m; ++i) {
        msg = "";
        for (int64_t j = 0; j < n; ++j) {
            slate::snprintf_value( buf, sizeof(buf), width, precision,
                                   A[i + j*lda] );
            msg += buf;
        }
        printf( "%s\n", msg.c_str() );
    }
    printf("];\n");
}

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
    slate::Options const& opts = slate::Options())
{
    int64_t width = slate::get_option<int64_t>( opts, slate::Option::PrintWidth, 10 );
    int64_t precision = slate::get_option<int64_t>( opts, slate::Option::PrintPrecision, 4 );
    int64_t verbose = slate::get_option<int64_t>( opts, slate::Option::PrintVerbose, 4 );
    int64_t edgeitems = slate::get_option<int64_t>( opts, slate::Option::PrintEdgeItems, 16 );

    if (verbose == 0)
        return;

    if ((verbose == 2) && (mlocal <= 2*edgeitems) && (nlocal <= 2*edgeitems)) {
        verbose = 4;
    }

    width = std::max(width, precision + 6);
    const int64_t abbrev_rows = edgeitems;
    const int64_t abbrev_cols = edgeitems;

    int mpi_rank;
    MPI_Comm_rank(comm, &mpi_rank);

    char buf[ 1024 ];
    std::string msg;

    // loop over process rows & cols
    for (int prow = 0; prow < q; ++prow) {
        for (int pcol = 0; pcol < p; ++pcol) {
            int rank = prow + pcol*p;
            // Print only from rank 0 here
            // to print roughly the same amount of data as for
            // SLATE matrices (up to 32*32 = 1024 entries).
            // But in general, wrapping a ScaLAPACK matrix in a SLATE matrix
            // and printing it is better.
            if (verbose == 2 && rank != 0)
                continue;
            if (rank == mpi_rank) {
                snprintf(buf, sizeof(buf),
                         "%% %s%d_%d: ScaLAPACK matrix\n",
                         label, prow, pcol);
                msg += buf;
                if (verbose != 1) {
                    snprintf(buf, sizeof(buf),
                             "%s%d_%d = [\n", label, prow, pcol);
                    msg += buf;
                }
                if (verbose == 2) {
                    // first abbrev_rows
                    int64_t max_rows = (mlocal < abbrev_rows ? mlocal : abbrev_rows);
                    int64_t max_cols = (nlocal < abbrev_cols ? nlocal : abbrev_cols);
                    int64_t start_col = blas::max( nlocal - abbrev_cols, 0 );
                    for (int64_t i = 0; i < max_rows; ++i) {
                        // first abbrev_cols
                        for (int64_t j = 0; j < max_cols; ++j) {
                            slate::snprintf_value( buf, sizeof(buf), width, precision,
                                                   A[i + j*lda] );
                            msg += buf;
                        }
                        if (nlocal > 2*abbrev_cols)
                            msg += " ..."; // column abbreviation indicator
                        // last abbrev_cols columns
                        for (int64_t j = start_col; j < nlocal; ++j) {
                            slate::snprintf_value( buf, sizeof(buf), width, precision,
                                                   A[i + j*lda] );
                            msg += buf;
                        }
                        msg += "\n";
                    }
                    if (mlocal > 2*abbrev_rows)
                        msg += " ...\n"; // row abbreviation indicator
                    // last abbrev_rows
                    int64_t start_row = (mlocal - abbrev_rows < abbrev_rows
                                         ? abbrev_rows
                                         : mlocal-abbrev_rows);
                    for (int64_t i = start_row; i < mlocal; ++i) {
                        // first abbrev_cols
                        for (int64_t j = 0; j < max_cols; ++j) {
                            slate::snprintf_value( buf, sizeof(buf), width, precision,
                                                   A[i + j*lda] );
                            msg += buf;
                        }
                        if (nlocal > 2*abbrev_cols)
                            msg += " ..."; // column abbreviation indicator
                        // last abbrev_cols columns
                        for (int64_t j = start_col; j < nlocal; ++j) {
                            slate::snprintf_value( buf, sizeof(buf), width, precision,
                                                   A[i + j*lda] );
                            msg += buf;
                        }
                        msg += "\n";
                    }
                    msg += "];\n";
                }
                else if (verbose == 3 || verbose == 4) {
                    int64_t row_step = (verbose == 3 && mlocal > 1 ? mlocal - 1 : 1);
                    int64_t col_step = (verbose == 3 && nlocal > 1 ? nlocal - 1 : 1);
                    for (int64_t i = 0; i < mlocal; i += row_step) {
                        // for verbose=3 only row i = 0 and i = mlocal-1
                        for (int64_t j = 0; j < nlocal; j += col_step) {
                            // for verbose=3 only column j = 0 and j = nlocal-1
                            slate::snprintf_value( buf, sizeof(buf), width, precision,
                                                   A[i + j*lda] );
                            msg += buf;
                        }
                        msg += "\n";
                    }
                    msg += "];\n";
                }

                if (mpi_rank != 0) {
                    // Send msg to root, which handles actual I/O.
                    int len = int(msg.size());
                    MPI_Send(&len, 1, MPI_INT, 0, 0, comm);
                    MPI_Send(msg.c_str(), len, MPI_CHAR, 0, 0, comm);
                }
                else {
                    // Already on root, just print it.
                    printf("%s", msg.c_str());
                }
            }
            else if (mpi_rank == 0) {
                // Root receives msg and handles actual I/O.
                MPI_Status status;
                int len;
                MPI_Recv(&len, 1, MPI_INT, rank, 0, comm, &status);
                msg.resize(len);
                MPI_Recv(&msg[0], len, MPI_CHAR, rank, 0, comm, &status);
                printf("%s", msg.c_str());
            }
        }
    }
    if (mpi_rank == 0) {
        fflush(stdout);
    }
    MPI_Barrier(comm);
}

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

    print( label, m, n, A, lda, opts );
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
    print( label, mlocal, nlocal, A, lda, p, q, comm, opts );
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
