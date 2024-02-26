// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_SCALAPACK_COPY_HH
#define SLATE_SCALAPACK_COPY_HH

#include "slate/Matrix.hh"
#include "slate/BaseTrapezoidMatrix.hh"

#include "blas.hh"
#include "lapack.hh"

#include "scalapack_wrappers.hh"

//------------------------------------------------------------------------------
/// Indices for ScaLAPACK descriptor
/// 0:  dtype:   1 for dense
/// 1:  context: BLACS context handle
/// 2:  m:       global number of rows
/// 3:  n:       global number of cols
/// 4:  mb:      row blocking factor
/// 5:  nb:      col blocking factor
/// 6:  rowsrc:  process row over which the first row of array is distributed
/// 7:  colsrc:  process col over which the first col of array is distributed
/// 8:  ld:      local leading dimension

enum Descriptor {
    dtype   = 0,
    context = 1,
    m       = 2,
    n       = 3,
    mb      = 4,
    nb      = 5,
    rowsrc  = 6,
    colsrc  = 7,
    ld      = 8
};

//------------------------------------------------------------------------------
/// Copy tile (i, j) from SLATE matrix A to ScaLAPACK matrix B.
///
template <typename scalar_t>
void copyTile(
    slate::BaseMatrix<scalar_t>& A,
    scalar_t* B, lapack_int descB[9],
    int64_t i, int64_t j,
    blas_int p, blas_int q )
{
    int64_t mb  = descB[ Descriptor::mb ];
    int64_t nb  = descB[ Descriptor::nb ];
    int64_t ldb = descB[ Descriptor::ld ];

    int64_t ii_local = int64_t( i / p )*mb;
    int64_t jj_local = int64_t( j / q )*nb;
    if (A.tileIsLocal(i, j)) {
        int dev = A.tileDevice(i, j);
        if (A.tileExists(i, j) &&
            A.tileState(i, j) != slate::MOSI::Invalid)
        {
            // Copy from host tile, if it exists, to ScaLAPACK.
            auto Aij = A(i, j);
            lapack::lacpy(
                lapack::MatrixType::General,
                Aij.mb(), Aij.nb(),
                Aij.data(), Aij.stride(),
                &B[ ii_local + jj_local*ldb ], ldb );
        }
        else if (A.tileExists(i, j, dev) &&
                 A.tileState(i, j, dev) != slate::MOSI::Invalid)
        {
            // Copy from device tile, if it exists, to ScaLAPACK.
            auto Aij = A(i, j, dev);
            blas::Queue queue( dev );
            blas::device_copy_matrix(
                Aij.mb(), Aij.nb(),
                Aij.data(), Aij.stride(),
                &B[ ii_local + jj_local*ldb ], ldb, queue);
            queue.sync();
        }
        else {
            // todo: what to throw?
            throw std::runtime_error("missing tile");
        }
    }
}

//------------------------------------------------------------------------------
/// Copy tile (i, j) from ScaLAPACK matrix B to SLATE matrix A.
///
template <typename scalar_t>
void copyTile(
    scalar_t const* B, lapack_int descB[9],
    slate::BaseMatrix<scalar_t>& A,
    int64_t i, int64_t j,
    blas_int p, blas_int q )
{
    int64_t mb  = descB[ Descriptor::mb ];
    int64_t nb  = descB[ Descriptor::nb ];
    int64_t ldb = descB[ Descriptor::ld ];

    int64_t ii_local = int64_t( i / p )*mb;
    int64_t jj_local = int64_t( j / q )*nb;
    if (A.tileIsLocal(i, j)) {
        int dev = A.tileDevice(i, j);
        if (A.tileExists(i, j) &&
            A.tileState(i, j) != slate::MOSI::Invalid)
        {
            // Copy from ScaLAPACK to host tile, if it exists.
            A.tileGetForWriting(i, j, slate::LayoutConvert::ColMajor);
            auto Aij = A(i, j);
            lapack::lacpy(
                lapack::MatrixType::General,
                Aij.mb(), Aij.nb(),
                &B[ ii_local + jj_local*ldb ], ldb,
                Aij.data(), Aij.stride() );
        }
        else if (A.tileExists(i, j, dev) &&
                 A.tileState(i, j, dev) != slate::MOSI::Invalid)
        {
            // Copy from ScaLAPACK to device tile, if it exists.
            A.tileGetForWriting(i, j, dev, slate::LayoutConvert::ColMajor);
            auto Aij = A(i, j, dev);
            blas::Queue queue( dev );
            blas::device_copy_matrix(
                Aij.mb(), Aij.nb(),
                &B[ ii_local + jj_local*ldb ], ldb,
                Aij.data(), Aij.stride(), queue);
        }
        else {
            // todo: what to throw?
            throw std::runtime_error("missing tile");
        }
    }
}

//------------------------------------------------------------------------------
/// Copies the ScaLAPACK-style matrix B to SLATE general matrix A.
/// Assumes both matrices have the same 2D block cyclic distribution.
///
/// @param[in] B
///     Local ScaLAPACK matrix.
///
/// @param[in] descB
///     Descriptor for ScaLAPACK matrix B.
///
/// @param[in,out] A
///     Matrix to copy data to.
///     The tiles, on either host or device, must already exist.
///
template <typename scalar_t>
void copy(
    scalar_t* B, lapack_int descB[9],
    slate::Matrix<scalar_t>& A )
{
    // todo: verify A and B have same distribution.
    blas_int p, q, myrow, mycol;
    Cblacs_gridinfo( descB[ Descriptor::context ], &p, &q, &myrow, &mycol );

    // Code assumes A is not transposed.
    slate_assert(A.op() == slate::Op::NoTrans);

    #pragma omp parallel for
    for (int64_t j = 0; j < A.nt(); ++j) {
        for (int64_t i = 0; i < A.mt(); ++i) {
            copyTile( B, descB, A, i, j, p, q );
        }
    }
}

//------------------------------------------------------------------------------
/// Copies SLATE general matrix A to ScaLAPACK-style matrix B.
/// Assumes both matrices have the same 2D block cyclic distribution.
///
/// @param[in] B
///     Local ScaLAPACK matrix.
///
/// @param[in] descB
///     Descriptor for ScaLAPACK matrix B.
///
/// @param[in,out] A
///     Matrix to copy data to.
///     The tiles, on either host or device, must already exist.
///
template <typename scalar_t>
void copy(
    slate::Matrix<scalar_t>& A,
    scalar_t* B, lapack_int descB[9] )
{
    // todo: verify A and B have same distribution.
    blas_int p, q, myrow, mycol;
    Cblacs_gridinfo( descB[ Descriptor::context ], &p, &q, &myrow, &mycol );

    // Code assumes A is not transposed.
    slate_assert(A.op() == slate::Op::NoTrans);

    #pragma omp parallel for
    for (int64_t j = 0; j < A.nt(); ++j) {
        for (int64_t i = 0; i < A.mt(); ++i) {
            copyTile( A, B, descB, i, j, p, q );
        }
    }
}

//------------------------------------------------------------------------------
/// Copies the ScaLAPACK-style matrix B to SLATE trapezoid-storage matrix A.
/// Handles Trapezoid, Triangular, Symmetric, and Hermitian matrices.
/// Assumes both matrices have the same 2D block cyclic distribution.
///
/// @param[in] B
///     Local ScaLAPACK matrix.
///
/// @param[in] descB
///     Descriptor for ScaLAPACK matrix B.
///
/// @param[in,out] A
///     Matrix to copy data to.
///     The tiles, on either host or device, must already exist.
///
template <typename scalar_t>
void copy(
    scalar_t* B, lapack_int descB[9],
    slate::BaseTrapezoidMatrix<scalar_t>& A )
{
    // todo: verify A and B have same distribution.
    blas_int p, q, myrow, mycol;
    Cblacs_gridinfo( descB[ Descriptor::context ], &p, &q, &myrow, &mycol );

    // Code assumes A is not transposed.
    slate_assert(A.op() == slate::Op::NoTrans);

    bool lower = A.uplo() == slate::Uplo::Lower;
    #pragma omp parallel for
    for (int64_t j = 0; j < A.nt(); ++j) {
        int64_t ibegin = (lower ? j : 0);
        int64_t iend   = (lower ? A.mt() : blas::min(j+1, A.mt()));
        for (int64_t i = ibegin; i < iend; ++i) {
            copyTile( B, descB, A, i, j, p, q );
        }
    }
}

//------------------------------------------------------------------------------
/// Copies SLATE trapezoid-storage matrix A to ScaLAPACK-style matrix B.
/// Handles Trapezoid, Triangular, Symmetric, and Hermitian matrices.
/// Assumes both matrices have the same 2D block cyclic distribution.
///
/// @param[in] B
///     Local ScaLAPACK matrix.
///
/// @param[in] descB
///     Descriptor for ScaLAPACK matrix B.
///
/// @param[in,out] A
///     Matrix to copy data to.
///     The tiles, on either host or device, must already exist.
///
template <typename scalar_t>
void copy(
    slate::BaseTrapezoidMatrix<scalar_t>& A,
    scalar_t* B, lapack_int descB[9] )
{
    // todo: verify A and B have same distribution.
    blas_int p, q, myrow, mycol;
    Cblacs_gridinfo( descB[ Descriptor::context ], &p, &q, &myrow, &mycol );

    // Code assumes A is not transposed.
    slate_assert(A.op() == slate::Op::NoTrans);

    bool lower = A.uplo() == slate::Uplo::Lower;
    #pragma omp parallel for
    for (int64_t j = 0; j < A.nt(); ++j) {
        int64_t ibegin = (lower ? j : 0);
        int64_t iend   = (lower ? A.mt() : blas::min(j+1, A.mt()));
        for (int64_t i = ibegin; i < iend; ++i) {
            copyTile( A, B, descB, i, j, p, q );
        }
    }
}

#endif // SLATE_SCALAPACK_COPY_HH
