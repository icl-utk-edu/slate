// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_GRID_UTILS_HH
#define SLATE_GRID_UTILS_HH

#include "slate/slate.hh"
#include "scalapack_wrappers.hh"

#include <stdint.h>

//------------------------------------------------------------------------------
// Similar to ScaLAPACK numroc (number of rows or columns).
inline int64_t num_local_rows_cols(int64_t n, int64_t nb, int iproc, int nprocs)
{
    int64_t nblocks = n / nb;
    int64_t num = (nblocks / nprocs) * nb;
    int64_t extra_blocks = nblocks % nprocs;
    if (iproc < extra_blocks) {
        // extra full blocks
        num += nb;
    }
    else if (iproc == extra_blocks) {
        // last partial block
        num += n % nb;
    }
    return num;
}

//------------------------------------------------------------------------------
// Similar to BLACS gridinfo
// (local row ID and column ID in 2D block cyclic distribution).
inline void gridinfo(
    int mpi_rank, slate::GridOrder order, int p, int q,
    int*  my_row, int*  my_col)
{
    if (order == slate::GridOrder::Col) {
        *my_row = mpi_rank % p;
        *my_col = mpi_rank / p;
    }
    else if (order == slate::GridOrder::Row) {
        *my_row = mpi_rank / q;
        *my_col = mpi_rank % q;
    }
    else {
        slate_error( "Unknown GridOrder" );
    }
}

// Overload with grid order == Col.
inline void gridinfo(
    int mpi_rank, int p, int q,
    int*  my_row, int*  my_col)
{
    gridinfo( mpi_rank, slate::GridOrder::Col, p, q, my_row, my_col );
}

//------------------------------------------------------------------------------
#ifdef SLATE_HAVE_SCALAPACK
// Sets up a BLACS context with the given grid
inline void create_ScaLAPACK_context( slate::GridOrder grid_order,
                                      int p, int q,
                                      blas_int* ictxt )
{
    // BLACS/MPI variables
    int mpi_rank, myrow, mycol;
    blas_int p_, q_, myrow_, mycol_;
    blas_int mpi_rank_ = 0, nprocs = 1;

    // initialize BLACS and ScaLAPACK
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
    Cblacs_pinfo( &mpi_rank_, &nprocs );
    slate_assert( mpi_rank_ == mpi_rank );

    Cblacs_get( -1, 0, ictxt );

    slate_assert( p*q <= nprocs );
    Cblacs_gridinit( ictxt, grid_order2str( grid_order ), p, q );
    gridinfo( mpi_rank, grid_order, p, q, &myrow, &mycol );
    Cblacs_gridinfo( *ictxt, &p_, &q_, &myrow_, &mycol_ );
    slate_assert( p == p_ );
    slate_assert( q == q_ );
    slate_assert( myrow == myrow_ );
    slate_assert( mycol == mycol_ );
}
#endif // SLATE_HAVE_SCALAPACK

#endif // SLATE_GRID_UTILS_HH
