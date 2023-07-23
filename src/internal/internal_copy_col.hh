// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

//------------------------------------------------------------------------------
#ifndef SLATE_COPY_COL_HH
#define SLATE_COPY_COL_HH

#include "slate/Matrix.hh"

#include <numeric>

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Copy local rows of column from matrix A, tile j, column jj, to vector x.
///
/// todo: This currently assumes tiles exist on the host.
///
/// @param[in] A
///     Matrix to copy from.
///
/// @param[in] j
///     Block column to copy from.
///
/// @param[in] jj
///     Offset within block column j of column to copy.
///
/// @param[out] x
///     Vector of length >= mlocal, the local number of rows of A.
///     On output, copy of local rows of A.at( :, j ).at( :, jj ).
///
template <typename real_t>
void copy_col(
    Matrix<real_t>& A, int64_t j, int64_t jj,
    real_t* x )
{
    int64_t mt = A.mt();
    int64_t ii = 0;
    for (int64_t i = 0; i < mt; ++i) {
        if (A.tileIsLocal( i, j )) {
            auto Aij = A( i, j );
            int64_t mb = Aij.mb();
            blas::copy( mb, &Aij.at( 0, jj ), 1, &x[ ii ], 1 );
            ii += mb;
        }
    }
    //assert( ii == mlocal );
}

//------------------------------------------------------------------------------
/// Copy local rows of column from vector x to matrix A, tile j, column jj.
///
/// todo: This currently assumes tiles exist on the host.
///
/// @param[in] x
///     Vector to copy from, of length >= mlocal, the local number of rows of A.
///
/// @param[in,out] A
///     Matrix to copy to.
///     On output, local rows of A.at( :, j ).at( :, jj ) are a copy of x.
///
/// @param[in] j
///     Block column to copy to.
///
/// @param[in] jj
///     Offset within block column j of column to copy.
///
template <typename real_t>
void copy_col(
    real_t* x,
    Matrix<real_t>& A, int64_t j, int64_t jj )
{
    int64_t mt = A.mt();
    int64_t ii = 0;
    for (int64_t i = 0; i < mt; ++i) {
        if (A.tileIsLocal( i, j )) {
            auto Aij = A( i, j );
            int64_t mb = Aij.mb();
            blas::copy( mb, &x[ ii ], 1, &Aij.at( 0, jj ), 1 );
            ii += mb;
        }
    }
    //assert( ii == mlocal );
}

//------------------------------------------------------------------------------
/// Copy local rows of column from matrix A, tile j, column jj,
/// to matrix B, tile k, column kk.
/// A and B must have the same distribution, number of rows, and tile mb;
/// they may differ in the number of columns.
///
/// todo: This currently assumes tiles exist on the host.
///
/// @param[in] A
///     Matrix to copy from.
///
/// @param[in] j
///     Block column to copy from.
///
/// @param[in] jj
///     Offset within block column j of column to copy.
///
/// @param[in,out] B
///     Matrix to copy to.
///     On output, local rows of
///     B.at( :, k ).at( :, kk ) = A.at( :, j ).at( :, jj ).
///
/// @param[in] k
///     Block column to copy to.
///
/// @param[in] kk
///     Offset within block column k of column to copy.
///
template <typename real_t>
void copy_col(
    Matrix<real_t>& A, int64_t j, int64_t jj,
    Matrix<real_t>& B, int64_t k, int64_t kk )
{
    assert( A.mt() == B.mt() );

    int64_t mt = A.mt();
    for (int64_t i = 0; i < mt; ++i) {
        if (A.tileIsLocal( i, j )) {
            assert( B.tileIsLocal( i, j ) );
            auto Aij = A( i, j );
            auto Bik = B( i, k );
            int64_t mb = Aij.mb();
            assert( mb == Bik.mb() );
            blas::copy( mb, &Aij.at( 0, jj ), 1, &Bik.at( 0, kk ), 1 );
        }
    }
}

} // namespace internal
} // namespace slate

#endif // SLATE_COPY_COL_HH
