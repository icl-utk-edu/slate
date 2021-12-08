// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/print.hh"
#include "blas.hh"

#include <string>
#include <cstdio>

namespace slate {

//------------------------------------------------------------------------------
/// @return 10^y for 0 <= y <= 20.
static inline double pow10( int y )
{
    static double values[] = {
        1, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12,
        1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20 };
    assert( 0 <= y && y <= 20 );
    return values[ y ];
}

//------------------------------------------------------------------------------
/// Print real value to a buffer buf of length buf_len.
/// For w = width and p = precision:
/// - integers are printed with %v.0f where v = w-p
/// - small values < 0.01 or large values > threshold are printed with %w.pg
/// - modest values are printed with %w.pf.
/// To ensure data fits, set threshold = 10^(w - p - 2) and w >= p + 6.
template <typename real_t>
static int snprintf_value(
    char* buf, size_t buf_len,
    int width, int precision,
    real_t value)
{
    real_t abs_val = std::abs( value );
    real_t threshold = pow10( width - precision - 2 );

    int len;
    if (value == int64_t( value )) {
        // exactly integer, print without digits after decimal point
        len = snprintf( buf, buf_len,
                        " %#*.0f%*s", width - precision, value,
                        precision, "" );
    }
    else if (abs_val < 0.01 || abs_val >= threshold) {
        // small or large value: print with %g
        len = snprintf( buf, buf_len,
                        " %#*.*g", width, precision, value );
    }
    else {
        // between 1 and threshold = 10^(w-p-2): %f will fit in width.
        len = snprintf( buf, buf_len,
                        " %#*.*f", width, precision, value );
    }
    return len;
}

//------------------------------------------------------------------------------
/// Print complex values as " <real> + <imag>i".
template <typename real_t>
static void snprintf_value(
    char* buf, size_t buf_len,
    int width, int precision,
    std::complex<real_t> value)
{
    // " real"
    real_t re = std::real( value );
    int len = snprintf_value( buf, buf_len, width, precision, re );
    buf     += len;
    buf_len -= len;

    real_t im = std::imag( value );
    if (im == 0) {
        // blank padding
        snprintf( buf, buf_len, "   %*s ", width, "" );
    }
    else {
        // " + imagi"
        len = snprintf( buf, buf_len, " +" );
        buf     += len;
        buf_len -= len;

        len = snprintf_value( buf, buf_len, width, precision, im );
        buf     += len;
        buf_len -= len;

        snprintf( buf, buf_len, "i" );
    }
}

//------------------------------------------------------------------------------
/// Sends tiles A(i, j) and receives it on rank 0.
/// If rank != 0 and tile A(i, j) is local, sends it to rank 0.
/// If rank == 0, inserts and receives tile A(i, j),
/// unless tile didn't exist on sender.
///
template <typename scalar_t>
void send_recv_tile(
    slate::BaseMatrix<scalar_t>& A, int64_t i, int64_t j,
    int mpi_rank, MPI_Comm comm)
{
    int flag_exist   = 0;
    int flag_missing = 1;
    int flag;
    int err;
    MPI_Status status;

    int tile_rank = A.tileRank(i, j);
    if (tile_rank != 0) {
        if (A.tileIsLocal(i, j)) {
            try {
                auto T = A(i, j);
                err = MPI_Send( &flag_exist, 1, MPI_INT, 0, 0, comm );
                slate_assert(err == 0);
                T.send(0, comm);
            }
            catch (std::out_of_range const& ex) {
                err = MPI_Send( &flag_missing, 1, MPI_INT, 0, 0, comm );
                slate_assert(err == 0);
            }
        }
        else if (mpi_rank == 0) {
            err = MPI_Recv(&flag, 1, MPI_INT, tile_rank, 0, comm, &status);
            slate_assert(err == 0);
            if (flag == flag_exist) {
                A.tileInsert(i, j);
                A(i, j).recv(tile_rank, comm, A.layout());
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Returns string for row ti of tile A(i, j).
/// If tile doesn't exist, returns string with NAN values.
/// For upper or lower tiles, uses opposite for values in the opposite
/// (lower or upper, respectively) triangle.
/// Works for all matrix types.
///
template <typename scalar_t>
std::string tile_row_string(
    slate::BaseMatrix<scalar_t>& A, int64_t i, int64_t j, int64_t ti,
    slate::Options const& opts,
    const char* opposite="",
    bool is_last_abbrev_cols = false)
{
    int64_t width = slate::get_option<int64_t>( opts, slate::Option::PrintWidth, 10 );
    int64_t precision = slate::get_option<int64_t>( opts, slate::Option::PrintPrecision, 4 );
    int64_t verbose = slate::get_option<int64_t>( opts, slate::Option::PrintVerbose, 4 );
    int64_t edgeitems = slate::get_option<int64_t>( opts, slate::Option::PrintEdgeItems, 16 );
    assert( verbose >= 2 );

    if (verbose == 5)
        verbose = 2;

    const int64_t abbrev_cols = edgeitems;

    using real_t = blas::real_type<scalar_t>;

    real_t nan_ = nan("");

    int64_t tile_columns = 0;
    char buf[ 80 ];
    std::string msg;
    try {
        A.tileGetForReading( i, j, slate::LayoutConvert::None );
        tile_columns = A.tileNb(j);
        auto T = A(i, j);
        slate::Uplo uplo = T.uplo();
        int64_t nb    = T.nb();
        int64_t begin = 0;
        int64_t end   = nb;
        int64_t step  = 1;
        if (! is_last_abbrev_cols && verbose == 2) {
            // first abbrev_cols
            end = std::min( nb, abbrev_cols );
        }
        else if (is_last_abbrev_cols && verbose == 2) {
            // last abbrev_cols
            if ((A.nt() == 1) && (nb < abbrev_cols*2)) // only 1 column tile
                begin = abbrev_cols;
            else
                begin = blas::max( nb - abbrev_cols, 0 );
        }
        else {
            // for verbose=3 only j = 0 and j = tileNb-1
            step = (verbose == 3 && nb > 1 ? nb - 1 : 1);
        }
        for (int64_t tj = begin; tj < end; tj += step) {
            if ((uplo == slate::Uplo::General)
                || (uplo == slate::Uplo::Lower && ti >= tj)
                || (uplo == slate::Uplo::Upper && ti <= tj))
            {
                snprintf_value( buf, sizeof(buf), width, precision,
                                T(ti, tj) );
                msg += buf;
            }
            else {
                msg += opposite;
            }
        }
    }
    catch (std::out_of_range const& ex) {
        // tile missing: print NAN
        snprintf_value( buf, sizeof(buf), width, precision, nan_ );

        if (! is_last_abbrev_cols && verbose == 2) {
            // first abbrev_cols
            int64_t max_cols = std::min( tile_columns, abbrev_cols );
            for (int64_t tj = 0; tj < max_cols; ++tj) {
                msg += buf;
            }
        }
        else if (is_last_abbrev_cols && verbose == 2) {
            // last abbrev_cols
            int64_t start_col;
            if ((A.nt() == 1) && (tile_columns < abbrev_cols*2)) // only 1 column tile
                start_col = abbrev_cols;
            else
                start_col = blas::max( tile_columns - abbrev_cols, 0 );

            for (int64_t tj = start_col; tj < tile_columns; ++tj) {
                msg += buf;
            }
        }
        else {
            int64_t col_step = (verbose == 3 && tile_columns > 1 ? tile_columns - 1 : 1);
            for (int64_t tj = 0; tj < tile_columns; tj += col_step) {
                // for verbose=3 only j = 0 and j = tileNb-1
                msg += buf;
            }
        }
    }
    return msg;
}

//------------------------------------------------------------------------------
/// Returns string for row ti of tile A(i, j).
/// If tile doesn't exist, returns string with NAN values.
/// For upper or lower tiles, uses opposite for values in the opposite
/// (lower or upper, respectively) triangle.
/// Works for all matrix types.
///
template <typename scalar_t>
std::string tile_row_string(
    slate::BaseMatrix<scalar_t>& A, int64_t i, int64_t j, int64_t ti,
    int width, int precision,
    const char* opposite="")
{
    const slate::Options opts = {
        { slate::Option::PrintWidth, width},
        { slate::Option::PrintPrecision, precision},
        { slate::Option::PrintVerbose, 4 }
    };
    return tile_row_string( A, i, j, ti, opts, opposite );
}

//------------------------------------------------------------------------------
/// Print a SLATE distributed trapezoid matrix.
/// Rank 0 does the printing, and must have enough memory to fit one entire
/// block row of the matrix.
/// For block-sparse matrices, missing tiles are print as "nan".
///
/// This version handles trapezoid, triangular, symmetric, and Hermitian
/// matrices. Entries in the A.uplo triangle are printed; entries in the
/// opposite triangle are printed as "nan".
///
template <typename scalar_t>
void print_work(
    const char* label,
    slate::BaseMatrix<scalar_t>& A,
    int64_t klt,
    int64_t kut,
    slate::Options const& opts_)
{
    using real_t = blas::real_type<scalar_t>;
    slate::Options opts(opts_);

    int64_t width = slate::get_option<int64_t>( opts, slate::Option::PrintWidth, 10 );
    int64_t precision = slate::get_option<int64_t>( opts, slate::Option::PrintPrecision, 4 );
    int64_t verbose = slate::get_option<int64_t>( opts, slate::Option::PrintVerbose, 4 );
    int64_t edgeitems = slate::get_option<int64_t>( opts, slate::Option::PrintEdgeItems, 16 );
    if (verbose <= 1)
        return;

    int64_t nrows = A.m();
    int64_t ncolumns = A.n();
    if (verbose == 2) { // abbreviate rows and columns
        if ((nrows <= 2*edgeitems) && (ncolumns <= 2*edgeitems))
            verbose = 4; // print all rows and columns
        else if ((nrows <= 2*edgeitems) && (ncolumns > 2*edgeitems))
            verbose = 5; // print all rows, abbreviate columns
        else if ((nrows > 2*edgeitems) && (ncolumns <= 2*edgeitems))
            verbose = 6; // abbreviate rows, print all columns

        opts[slate::Option::PrintVerbose] = verbose;
    }

    width = std::max(width, precision + 6);
    real_t nan_ = nan("");

    const int64_t abbrev_rows = edgeitems;
    const int64_t abbrev_cols = edgeitems;

    int mpi_rank = A.mpiRank();
    MPI_Comm comm = A.mpiComm();
    MPI_Barrier(comm);

    std::string msg = std::string( label ) + " = [\n";

    // for entries in opposite triangle from A.uplo
    char opposite[ 80 ];
    if (slate::is_complex<scalar_t>::value) {
        snprintf(opposite, sizeof(opposite), " %*f   %*s ",
                 (int)width, nan_, (int)width, "");
    }
    else {
        snprintf(opposite, sizeof(opposite), " %*f",
                 (int)width, nan_);
    }

    // for tiles outside bandwidth
    char outside_bandwidth[ 80 ];
    if (slate::is_complex<scalar_t>::value) {
        snprintf(outside_bandwidth, sizeof(outside_bandwidth), " %*.0f   %*s ",
                 (int)width, 0., (int)width, "");
    }
    else {
        snprintf(outside_bandwidth, sizeof(outside_bandwidth), " %*.0f",
                 (int)width, 0.);
    }

    int64_t tile_row_step = 1;
    int64_t tile_col_step = 1;

    if (verbose == 2) { // abbreviate rows and columns
        tile_row_step = (A.mt() > 1 ? A.mt()-1 : 1);
        tile_col_step = (A.nt() > 1 ? A.nt()-1 : 1);
    }
    else if (verbose == 5) { // abbreviate columns only
        tile_col_step = (A.nt() > 1 ? A.nt()-1 : 1);
    }
    else if (verbose == 6) { // abbreviate rows only
        tile_row_step = (A.mt() > 1 ? A.mt()-1 : 1);
    }
    for (int64_t i = 0; i < A.mt(); i += tile_row_step) {
        // for verbose=2 only tile row i = 0 and i = mt-1
        // gather block row to rank 0
        for (int64_t j = 0; j < A.nt(); j += tile_col_step) {
            // for verbose=2 only tile column j = 0 and j = nt-1
            if ((A.uplo() == slate::Uplo::General && -klt <= j - i && j - i <= kut)
                || (A.uplo() == slate::Uplo::Lower && i <= j + klt && j <= i)
                || (A.uplo() == slate::Uplo::Upper && i >= j - kut && j >= i))
            {
                // inside bandwidth
                send_recv_tile(A, i, j, mpi_rank, comm);
            }
        }

        if (mpi_rank == 0) {
            // print block row
            if (verbose == 2 || verbose == 5 || verbose == 6) {
                // only first & last abbrev_rows & abbrev_cols
                // (of 1st & last block-row & block-col)
                // so just the 4 corner tiles:
                // A( 0, 0 )  ... A( nt-1, 0 )
                // ...
                // A( mt-1, 0 ) ... A( mt-1, nt-1 )
                bool last_row_tile = false;
                int max_pass = 1;
                if (A.mt() <= 1)
                    max_pass = 2; // pass twice on same tile
                for (int pass = 0; pass < max_pass; ++pass) {
                    int64_t start_row = 0;
                    int64_t max_rows = 0;
                    if (i == 0 && pass == 0) { // first row tile
                        // first abbrev_rows
                        max_rows = std::min( A.tileMb(i), abbrev_rows );
                        if (verbose == 5)
                            max_rows = A.tileMb(i);
                    }
                    else if (i == A.mt()-1) { // last row tile
                        last_row_tile = true;
                        if (verbose != 5 || A.mt() > 1) {
                            if ((verbose != 5) && (nrows > 2 * abbrev_rows || A.mt() > 1))
                                msg += " ...\n"; // row abbreviation indicator

                            // last abbrev_rows
                            start_row = blas::max( 0, A.tileMb(i) - abbrev_rows);
                            max_rows = A.tileMb(i);
                        }
                    }
                    for (int64_t ti = start_row; ti < max_rows; ++ti) {
                        bool is_last_abbrev_cols = false;
                        int64_t j = 0; // first column tile
                        int64_t start_col = 0;
                        int64_t max_cols = A.tileNb(j);
                        for (int loop = 0; loop < 2; ++loop) {
                            if (loop == 0) { // first column tile
                                if (verbose == 2 || verbose == 5)
                                    max_cols = std::min( A.tileNb(j), abbrev_cols );
                            }
                            else if (verbose != 6 || A.nt() > 1) { // last column tile
                                is_last_abbrev_cols = true;
                                if ((verbose != 6) && (ncolumns > 2 * abbrev_cols || A.nt() > 1))
                                    msg += " ..."; // column abbreviation indicator

                                j = A.nt()-1; // last column tile
                                max_cols = A.tileNb(j);
                                if (j>0)
                                    msg += "    "; // space between column tiles

                                if (verbose == 2 || verbose == 5) {
                                    if ((A.nt() == 1) && (A.tileNb(j) < abbrev_cols*2))
                                        start_col = abbrev_cols; // only 1 column tile
                                    else
                                        start_col = blas::max( A.tileNb(j) - abbrev_cols, 0 );
                                }
                            }
                            else
                                break;

                            if ((A.uplo() == slate::Uplo::General && -klt <= j - i && j - i <= kut)
                                || (A.uplo() == slate::Uplo::Lower && i <= j + klt && j <= i)
                                || (A.uplo() == slate::Uplo::Upper && i >= j - kut && j >= i))
                            {
                                // inside bandwidth
                                msg += tile_row_string(A, i, j, ti, opts, opposite, is_last_abbrev_cols);
                            }
                            else {
                                for (int64_t tj = start_col; tj < max_cols; ++tj) {
                                    if ((A.uplo() == slate::Uplo::Lower && j <= i)
                                        || (A.uplo() == slate::Uplo::Upper && j >= i))
                                    {
                                        msg += outside_bandwidth;
                                    }
                                    else {
                                        msg += opposite;
                                    }
                                }
                            }
                        }
                        msg += "\n";
                    }
                }
                if (last_row_tile)
                    msg += "];\n";
            }
            else if (verbose == 3 || verbose == 4) {
                // for verbose=3 only rows ti = 0 and ti = tileMb-1
                int64_t row_step = (verbose == 3 && A.tileMb(i) > 1 ? A.tileMb(i) - 1 : 1);
                for (int64_t ti = 0; ti < A.tileMb(i); ti += row_step) {
                    for (int64_t j = 0; j < A.nt(); ++j) {
                        if ((A.uplo() == slate::Uplo::General && -klt <= j - i && j - i <= kut)
                            || (A.uplo() == slate::Uplo::Lower && i <= j + klt && j <= i)
                            || (A.uplo() == slate::Uplo::Upper && i >= j - kut && j >= i))
                        {
                            // inside bandwidth
                            msg += tile_row_string(A, i, j, ti, opts, opposite);
                        }
                        else {
                            // for verbose=3 only j = 0 and j = tileNb-1
                            int64_t col_step = (verbose == 3 && A.tileNb(j) > 1 ? A.tileNb(j) - 1 : 1);
                            for (int64_t tj = 0; tj < A.tileNb(j); tj += col_step) {
                                if ((A.uplo() == slate::Uplo::Lower && j <= i)
                                    || (A.uplo() == slate::Uplo::Upper && j >= i))
                                {
                                    msg += outside_bandwidth;
                                }
                                else {
                                    msg += opposite;
                                }
                            }
                        }
                        if (j < A.nt() - 1)
                            msg += "    "; // space between column tiles
                        else
                            msg += "\n";
                    }
                }
            }

            if (verbose != 2 && verbose != 6) {
                if (i < A.mt() - 1)
                    msg += "\n"; // line between row tiles
                else if (verbose != 5)
                    msg += "];\n";
            }
            printf("%s", msg.c_str());
            msg.clear();

            // cleanup data
            for (int64_t j = 0; j < A.nt(); ++j) {
                if (! A.tileIsLocal(i, j)) {
                    A.tileErase(i, j);
                }
            }
        }
    }

    MPI_Barrier(comm);
}

//------------------------------------------------------------------------------
/// Print a LAPACK matrix. Should be called from only one rank.
///
template <typename scalar_t>
void print(
    const char* label,
    int64_t m, int64_t n, scalar_t* A, int64_t lda,
    slate::Options const& opts)
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
            snprintf_value( buf, sizeof(buf), width, precision,
                            A[i + j*lda] );
            msg += buf;
        }
        printf( "%s\n", msg.c_str() );
    }
    printf("];\n");
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void print(
    const char* label,
    int64_t m, int64_t n, float* A, int64_t lda,
    slate::Options const& opts);

template
void print(
    const char* label,
    int64_t m, int64_t n, double* A, int64_t lda,
    slate::Options const& opts);

template
void print(
    const char* label,
    int64_t m, int64_t n, std::complex<float>* A, int64_t lda,
    slate::Options const& opts);

template
void print(
    const char* label,
    int64_t m, int64_t n, std::complex<double>* A, int64_t lda,
    slate::Options const& opts);

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
    slate::Options const& opts)
{
    int64_t width = slate::get_option<int64_t>( opts, slate::Option::PrintWidth, 10 );
    int64_t precision = slate::get_option<int64_t>( opts, slate::Option::PrintPrecision, 4 );
    int64_t verbose = slate::get_option<int64_t>( opts, slate::Option::PrintVerbose, 4 );
    int64_t edgeitems = slate::get_option<int64_t>( opts, slate::Option::PrintEdgeItems, 16 );

    if (verbose == 0)
        return;

    if ((verbose == 2) && (mlocal <= 2*edgeitems) && (nlocal <= 2*edgeitems))
    {
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
                            snprintf_value( buf, sizeof(buf), width, precision,
                                            A[i + j*lda] );
                            msg += buf;
                        }
                        if (nlocal > 2*abbrev_cols)
                            msg += " ..."; // column abbreviation indicator
                        // last abbrev_cols columns
                        for (int64_t j = start_col; j < nlocal; ++j) {
                            snprintf_value( buf, sizeof(buf), width, precision,
                                            A[i + j*lda] );
                            msg += buf;
                        }
                        msg += "\n";
                    }
                    if (mlocal > 2*abbrev_rows)
                        msg += " ...\n";// row abbreviation indicator
                    // last abbrev_rows
                    int64_t start_row = (mlocal - abbrev_rows < abbrev_rows
                                      ? abbrev_rows : mlocal-abbrev_rows);
                    for (int64_t i = start_row; i < mlocal; ++i) {
                        // first abbrev_cols
                        for (int64_t j = 0; j < max_cols; ++j) {
                            snprintf_value( buf, sizeof(buf), width, precision,
                                            A[i + j*lda] );
                            msg += buf;
                        }
                        if (nlocal > 2*abbrev_cols)
                            msg += " ..."; // column abbreviation indicator
                        // last abbrev_cols columns
                        for (int64_t j = start_col; j < nlocal; ++j) {
                            snprintf_value( buf, sizeof(buf), width, precision,
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
                            snprintf_value( buf, sizeof(buf), width, precision,
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
// Explicit instantiations.
// ----------------------------------------
template
void print(
    const char* label,
    int64_t mlocal, int64_t nlocal, float* A, int64_t lda,
    int p, int q, MPI_Comm comm,
    slate::Options const& opts);

template
void print(
    const char* label,
    int64_t mlocal, int64_t nlocal, double* A, int64_t lda,
    int p, int q, MPI_Comm comm,
    slate::Options const& opts);

template
void print(
    const char* label,
    int64_t mlocal, int64_t nlocal, std::complex<float>* A, int64_t lda,
    int p, int q, MPI_Comm comm,
    slate::Options const& opts);

template
void print(
    const char* label,
    int64_t mlocal, int64_t nlocal, std::complex<double>* A, int64_t lda,
    int p, int q, MPI_Comm comm,
    slate::Options const& opts);

//------------------------------------------------------------------------------
/// Print a ScaLAPACK distributed matrix.
/// Prints each rank's data as a contiguous block, numbered by the block row &
/// column indices. Rank 0 does the printing.
///
template <typename scalar_t>
void print(
    const char* label,
    int64_t mlocal, int64_t nlocal, scalar_t* A, int64_t lda,
    int p, int q, MPI_Comm comm )
{
    const slate::Options opts = {
        { slate::Option::PrintWidth, 10 },
        { slate::Option::PrintPrecision, 4 },
        { slate::Option::PrintVerbose, 4 }
    };

    print( label, mlocal, nlocal, A, lda, p, q, comm, opts);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void print(
    const char* label,
    int64_t mlocal, int64_t nlocal, float* A, int64_t lda,
    int p, int q, MPI_Comm comm );

template
void print(
    const char* label,
    int64_t mlocal, int64_t nlocal, double* A, int64_t lda,
    int p, int q, MPI_Comm comm );

template
void print(
    const char* label,
    int64_t mlocal, int64_t nlocal, std::complex<float>* A, int64_t lda,
    int p, int q, MPI_Comm comm );

template
void print(
    const char* label,
    int64_t mlocal, int64_t nlocal, std::complex<double>* A, int64_t lda,
    int p, int q, MPI_Comm comm );

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
    slate::Options const& opts)
{
    int64_t verbose = slate::get_option<int64_t>( opts, slate::Option::PrintVerbose, 4 );
    if (verbose == 0)
        return;

    if (A.mpiRank() == 0) {
        std::string msg = std::string( "% " ) + label + ": slate::Matrix ";
        msg += std::to_string( A.m() ) + "-by-" + std::to_string( A.n() ) + ", "
            +  std::to_string( A.mt() ) + "-by-" + std::to_string( A.nt() )
            +  " tiles, tileSize " + std::to_string( A.tileMb(0) ) + "-by-"
            +  std::to_string( A.tileNb(0) ) + "\n";

        printf( "%s", msg.c_str() );
    }

    int64_t klt, kut;
    klt = kut = std::max( A.mt(), A.nt() );
    print_work( label, A, klt, kut, opts );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void print(
    const char* label,
    slate::Matrix<float>& A,
    slate::Options const& opts);

template
void print(
    const char* label,
    slate::Matrix<double>& A,
    slate::Options const& opts);

template
void print(
    const char* label,
    slate::Matrix<std::complex<float>>& A,
    slate::Options const& opts);

template
void print(
    const char* label,
    slate::Matrix<std::complex<double>>& A,
    slate::Options const& opts);

//------------------------------------------------------------------------------
/// Print a SLATE distributed matrix.
/// Rank 0 does the printing, and must have enough memory to fit one entire
/// block row of the matrix.
/// For block-sparse matrices, missing tiles are print as "nan".
///
template <typename scalar_t>
void print(
    const char* label,
    slate::Matrix<scalar_t>& A)
{
    // Set defaults
    const slate::Options opts = {
        { slate::Option::PrintWidth, 10 },
        { slate::Option::PrintPrecision, 4 },
        { slate::Option::PrintVerbose, 4 } // default 4 prints full matrix
    };

    print( label, A, opts );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void print(
    const char* label,
    slate::Matrix<float>& A);

template
void print(
    const char* label,
    slate::Matrix<double>& A);

template
void print(
    const char* label,
    slate::Matrix<std::complex<float>>& A);

template
void print(
    const char* label,
    slate::Matrix<std::complex<double>>& A);

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
    slate::Options const& opts)
{
    int64_t verbose = slate::get_option<int64_t>( opts, slate::Option::PrintVerbose, 4 );
    if (verbose == 0)
        return;

    if (A.mpiRank() == 0) {
        std::string msg = "\n% slate::BandMatrix ";
        msg += std::to_string( A.m()  ) + "-by-" + std::to_string( A.n()  )
            + ", "
            +  std::to_string( A.mt() ) + "-by-" + std::to_string( A.nt() )
            +  " tiles, tileSize " + std::to_string( A.tileMb(0) ) + "-by-"
            +  std::to_string( A.tileNb(0) ) + ","
            +  " kl " + std::to_string( A.lowerBandwidth() )
            +  " ku " + std::to_string( A.upperBandwidth() ) + "\n";

        printf( "%s", msg.c_str() );
    }

    // todo: initially, assume fixed size, square tiles for simplicity
    int64_t klt = slate::ceildiv(A.lowerBandwidth(), A.tileNb(0));
    int64_t kut = slate::ceildiv(A.upperBandwidth(), A.tileNb(0));
    print_work( label, A, klt, kut, opts );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void print(
    const char* label,
    slate::BandMatrix<float>& A,
    slate::Options const& opts);

template
void print(
    const char* label,
    slate::BandMatrix<double>& A,
    slate::Options const& opts);

template
void print(
    const char* label,
    slate::BandMatrix<std::complex<float>>& A,
    slate::Options const& opts);

template
void print(
    const char* label,
    slate::BandMatrix<std::complex<double>>& A,
    slate::Options const& opts);

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
    slate::Options const& opts)
{
    int64_t verbose = slate::get_option<int64_t>( opts, slate::Option::PrintVerbose, 4 );
    if (verbose == 0)
        return;

    if (A.mpiRank() == 0) {
        std::string msg = "\n% slate::BaseTriangularBandMatrix ";
        msg += std::to_string( A.m()  ) + "-by-" + std::to_string( A.n()  ) + ", "
            +  std::to_string( A.mt() ) + "-by-" + std::to_string( A.nt() )
            +  " tiles, tileSize " + std::to_string( A.tileMb(0) ) + "-by-"
            +  std::to_string( A.tileNb(0) ) + ","
            +  " kd " + std::to_string( A.bandwidth() )
            +  " uplo " + char( A.uplo() ) + "\n";

        printf( "%s", msg.c_str() );
    }

    int64_t kdt = slate::ceildiv(A.bandwidth(), A.tileNb(0));
    int64_t klt, kut;
    if (A.uplo() == slate::Uplo::Lower) {
        klt = kdt;
        kut = 0;
    }
    else {
        kut = kdt;
        klt = 0;
    }
    print_work( label, A, klt, kut, opts );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void print(
    const char* label,
    slate::BaseTriangularBandMatrix<float>& A,
    slate::Options const& opts);

template
void print(
    const char* label,
    slate::BaseTriangularBandMatrix<double>& A,
    slate::Options const& opts);

template
void print(
    const char* label,
    slate::BaseTriangularBandMatrix<std::complex<float>>& A,
    slate::Options const& opts);

template
void print(
    const char* label,
    slate::BaseTriangularBandMatrix<std::complex<double>>& A,
    slate::Options const& opts);

//------------------------------------------------------------------------------
/// Print a SLATE distributed Hermitian matrix.
/// Also prints Matlab tril or triu command to fix entries in opposite triangle.
/// todo: fix complex diag in Matlab? (Sca)LAPACK ignores imag part.
///
template <typename scalar_t>
void print(
    const char* label,
    slate::HermitianMatrix<scalar_t>& A,
    slate::Options const& opts)
{
    int64_t verbose = slate::get_option<int64_t>( opts, slate::Option::PrintVerbose, 4 );
    if (verbose == 0)
        return;

    if (A.mpiRank() == 0) {
        printf( "\n"
                "%% %s: slate::HermitianMatrix %lld-by-%lld, %lld-by-%lld tiles, "
                "tileSize %lld-by-%lld, uplo %c\n",
                label,
                llong( A.m() ), llong( A.n() ),
                llong( A.mt() ), llong( A.nt() ),
                llong( A.tileMb(0) ), llong( A.tileNb(0) ),
                char( A.uplo() ) );
    }
    char buf[ 80 ];
    snprintf( buf, sizeof(buf), "%s_", label );

    int64_t klt, kut;
    if (A.uplo() == slate::Uplo::Lower) {
        klt = std::max ( A.mt(), A.nt() );
        kut = 0;
    }
    else {
        kut = std::max( A.mt(), A.nt() );
        klt = 0;
    }

    print_work( buf, A, klt, kut, opts );
    if (A.mpiRank() == 0) {
        if (A.uplo() == slate::Uplo::Lower) {
            printf( "%s = tril( %s_ ) + tril( %s_, -1 )';\n\n",
                    label, label, label );
        }
        else {
            printf( "%s = triu( %s_ ) + triu( %s_,  1 )';\n\n",
                    label, label, label );
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void print(
    const char* label,
    slate::HermitianMatrix<float>& A,
    slate::Options const& opts);

template
void print(
    const char* label,
    slate::HermitianMatrix<double>& A,
    slate::Options const& opts);

template
void print(
    const char* label,
    slate::HermitianMatrix<std::complex<float>>& A,
    slate::Options const& opts);

template
void print(
    const char* label,
    slate::HermitianMatrix<std::complex<double>>& A,
    slate::Options const& opts);

//------------------------------------------------------------------------------
/// Print a SLATE distributed symmetric matrix.
/// Also prints Matlab tril or triu command to fix entries in opposite triangle.
///
template <typename scalar_t>
void print(
    const char* label,
    slate::SymmetricMatrix<scalar_t>& A,
    slate::Options const& opts)
{
    int64_t verbose = slate::get_option<int64_t>( opts, slate::Option::PrintVerbose, 4 );
    if (verbose == 0)
        return;

    if (A.mpiRank() == 0) {
        printf( "\n"
                "%% %s: slate::SymmetricMatrix %lld-by-%lld, %lld-by-%lld tiles, "
                "tileSize %lld-by-%lld, uplo %c\n",
                label,
                llong( A.m() ), llong( A.n() ),
                llong( A.mt() ), llong( A.nt() ),
                llong( A.tileMb(0) ), llong( A.tileNb(0) ),
                char( A.uplo() ) );
    }

    int64_t klt, kut;
    if (A.uplo() == slate::Uplo::Lower) {
        klt = std::max ( A.mt(), A.nt() );
        kut = 0;
    }
    else {
        kut = std::max( A.mt(), A.nt() );
        klt = 0;
    }
    print_work( label, A, klt, kut, opts );
    if (A.mpiRank() == 0) {
        if (A.uplo() == slate::Uplo::Lower) {
            printf( "%s = tril( %s_ ) + tril( %s_, -1 ).';\n\n",
                    label, label, label );
        }
        else {
            printf( "%s = triu( %s_ ) + triu( %s_,  1 ).';\n\n",
                    label, label, label );
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void print(
    const char* label,
    slate::SymmetricMatrix<float>& A,
    slate::Options const& opts);

template
void print(
    const char* label,
    slate::SymmetricMatrix<double>& A,
    slate::Options const& opts);

template
void print(
    const char* label,
    slate::SymmetricMatrix<std::complex<float>>& A,
    slate::Options const& opts);

template
void print(
    const char* label,
    slate::SymmetricMatrix<std::complex<double>>& A,
    slate::Options const& opts);

//------------------------------------------------------------------------------
/// Print a SLATE distributed trapezoid matrix.
/// Also prints Matlab tril or triu command to fix entries in opposite triangle.
/// todo: fix unit diag in Matlab.
///
template <typename scalar_t>
void print(
    const char* label,
    slate::TrapezoidMatrix<scalar_t>& A,
    slate::Options const& opts)
{
    int64_t verbose = slate::get_option<int64_t>( opts, slate::Option::PrintVerbose, 4 );
    if (verbose == 0)
        return;

    if (A.mpiRank() == 0) {
        printf( "\n"
                "%% %s: slate::TrapezoidMatrix %lld-by-%lld, %lld-by-%lld tiles, "
                "tileSize %lld-by-%lld, uplo %c diag %c\n",
                label,
                llong( A.m() ), llong( A.n() ),
                llong( A.mt() ), llong( A.nt() ),
                llong( A.tileMb(0) ), llong( A.tileNb(0) ),
                char( A.uplo() ), char( A.diag() ) );
    }
    char buf[ 80 ];
    snprintf( buf, sizeof(buf), "%s_", label );

    int64_t klt, kut;
    if (A.uplo() == slate::Uplo::Lower) {
        klt = std::max ( A.mt(), A.nt() );
        kut = 0;
    }
    else {
        kut = std::max( A.mt(), A.nt() );
        klt = 0;
    }
    print_work( buf, A, klt, kut, opts );
    if (A.mpiRank() == 0) {
        if (A.uplo() == slate::Uplo::Lower) {
            printf( "%s = tril( %s_ );\n\n", label, label );
        }
        else {
            printf( "%s = triu( %s_ );\n\n", label, label );
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void print(
    const char* label,
    slate::TrapezoidMatrix<float>& A,
    slate::Options const& opts);

template
void print(
    const char* label,
    slate::TrapezoidMatrix<double>& A,
    slate::Options const& opts);

template
void print(
    const char* label,
    slate::TrapezoidMatrix<std::complex<float>>& A,
    slate::Options const& opts);

template
void print(
    const char* label,
    slate::TrapezoidMatrix<std::complex<double>>& A,
    slate::Options const& opts);

//------------------------------------------------------------------------------
/// Print a SLATE distributed triangular matrix.
/// Also prints Matlab tril or triu command to fix entries in opposite triangle.
/// todo: fix unit diag in Matlab.
///
template <typename scalar_t>
void print(
    const char* label,
    slate::TriangularMatrix<scalar_t>& A,
    slate::Options const& opts)
{
    int64_t verbose = slate::get_option<int64_t>( opts, slate::Option::PrintVerbose, 4 );
    if (verbose == 0)
        return;

    if (A.mpiRank() == 0) {
        printf( "\n"
                "%% %s: slate::TriangularMatrix %lld-by-%lld, %lld-by-%lld tiles, "
                "tileSize %lld-by-%lld, uplo %c diag %c\n",
                label,
                llong( A.m() ), llong( A.n() ),
                llong( A.mt() ), llong( A.nt() ),
                llong( A.tileMb(0) ), llong( A.tileNb(0) ),
                char( A.uplo() ), char( A.diag() ) );
    }
    char buf[ 80 ];
    snprintf( buf, sizeof(buf), "%s_", label );

    int64_t klt, kut;
    if (A.uplo() == slate::Uplo::Lower) {
        klt = std::max ( A.mt(), A.nt() );
        kut = 0;
    }
    else {
        kut = std::max( A.mt(), A.nt() );
        klt = 0;
    }
    print_work( buf, A, klt, kut, opts );
    if (A.mpiRank() == 0) {
        if (A.uplo() == slate::Uplo::Lower) {
            printf( "%s = tril( %s_ );\n\n", label, label );
        }
        else {
            printf( "%s = triu( %s_ );\n\n", label, label );
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void print(
    const char* label,
    slate::TriangularMatrix<float>& A,
    slate::Options const& opts);

template
void print(
    const char* label,
    slate::TriangularMatrix<double>& A,
    slate::Options const& opts);

template
void print(
    const char* label,
    slate::TriangularMatrix<std::complex<float>>& A,
    slate::Options const& opts);

template
void print(
    const char* label,
    slate::TriangularMatrix<std::complex<double>>& A,
    slate::Options const& opts);

//------------------------------------------------------------------------------
/// Print a vector.
/// Every MPI rank does its own printing, so protect with `if (mpi_rank == 0)`
/// as desired.
///
template <typename scalar_t>
void print(
    const char* label,
    int64_t n, scalar_t const* x, int64_t incx,
    slate::Options const& opts)
{
    slate_assert( n >= 0 );
    slate_assert( incx != 0 );

    int64_t width = slate::get_option<int64_t>( opts, slate::Option::PrintWidth, 10 );
    int64_t precision = slate::get_option<int64_t>( opts, slate::Option::PrintPrecision, 4 );

    width = std::max(width, precision + 6);

    char buf[ 80 ];
    std::string msg;

    int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
    for (int64_t i = 0; i < n; ++i) {
        snprintf_value( buf, sizeof(buf), width, precision, x[ix] );
        msg += buf;
        ix += incx;
    }
    printf( "%s = [ %s ]';\n", label, msg.c_str() );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void print(
    const char* label,
    int64_t n, float const* x, int64_t incx,
    slate::Options const& opts);

template
void print(
    const char* label,
    int64_t n, double const* x, int64_t incx,
    slate::Options const& opts);

template
void print(
    const char* label,
    int64_t n, std::complex<float> const* x, int64_t incx,
    slate::Options const& opts);

template
void print(
    const char* label,
    int64_t n, std::complex<double> const* x, int64_t incx,
    slate::Options const& opts);

//------------------------------------------------------------------------------
/// Print a vector.
/// Every MPI rank does its own printing, so protect with `if (mpi_rank == 0)`
/// as desired.
///
template <typename scalar_t>
void print(
    const char* label,
    int64_t n, scalar_t const* x, int64_t incx,
    int width, int precision )
{
    const slate::Options opts = {
        { slate::Option::PrintWidth, width },
        { slate::Option::PrintPrecision, precision }
    };

    print( label, n, x, incx, opts );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void print(
    const char* label,
    int64_t n, float const* x, int64_t incx,
    int width=10, int precision=4 );

template
void print(
    const char* label,
    int64_t n, double const* x, int64_t incx,
    int width=10, int precision=4 );

template
void print(
    const char* label,
    int64_t n, std::complex<float> const* x, int64_t incx,
    int width=10, int precision=4 );

template
void print(
    const char* label,
    int64_t n, std::complex<double> const* x, int64_t incx,
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
    slate::Options const& opts)
{
    print( label, x.size(), x.data(), 1, opts );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void print(
    const char* label,
    std::vector<float> const& x,
    slate::Options const& opts);

template
void print(
    const char* label,
    std::vector<double> const& x,
    slate::Options const& opts);

template
void print(
    const char* label,
    std::vector<std::complex<float>> const& x,
    slate::Options const& opts);

template
void print(
    const char* label,
    std::vector<std::complex<double>> const& x,
    slate::Options const& opts);

//------------------------------------------------------------------------------
/// Print a vector.
/// Every MPI rank does its own printing, so protect with `if (mpi_rank == 0)`
/// as desired.
///
template <typename scalar_type>
void print(
    const char* label,
    std::vector<scalar_type> const& x,
    int width, int precision )
{
    const slate::Options opts = {
        { slate::Option::PrintWidth, width },
        { slate::Option::PrintPrecision, precision }
    };

    print( label, x.size(), x.data(), 1, opts );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void print(
    const char* label,
    std::vector<float> const& x,
    int width=10, int precision=4 );

template
void print(
    const char* label,
    std::vector<double> const& x,
    int width=10, int precision=4 );

template
void print(
    const char* label,
    std::vector<std::complex<float>> const& x,
    int width=10, int precision=4 );

template
void print(
    const char* label,
    std::vector<std::complex<double>> const& x,
    int width=10, int precision=4 );

} // namespace slate
