// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_TILE_AUX_HH
#define SLATE_TILE_AUX_HH

// #include "slate/Tile.hh"
#include "slate/internal/util.hh"
#include "slate/internal/device.hh"

namespace slate {

// forward declaration
template <typename scalar_t>
class Tile;

namespace tile {

//------------------------------------------------------------------------------
/// Copy and precision conversion.
/// @ingroup copy_tile
///
template <typename src_scalar_t, typename dst_scalar_t>
void gecopy(Tile<src_scalar_t> const& A, Tile<dst_scalar_t>& B)
{
//  trace::Block trace_block("aux::copy");

    assert(A.mb() == B.mb());
    assert(A.nb() == B.nb());
    // Quick return
    if (A.mb() == 0 || A.nb() == 0)
        return;

    const src_scalar_t* A00 = &A.at(0, 0);
    int64_t a_col_inc = A.colIncrement();
    int64_t a_row_inc = A.rowIncrement();
    dst_scalar_t* B00 = &B.at(0, 0);
    int64_t b_col_inc = B.colIncrement();
    int64_t b_row_inc = B.rowIncrement();

    for (int64_t j = 0; j < B.nb(); ++j) {
        const src_scalar_t* Aj = &A00[j*a_row_inc];
        dst_scalar_t* Bj = &B00[j*b_row_inc];
        for (int64_t i = 0; i < B.mb(); ++i)
            Bj[i*b_col_inc] = Aj[i*a_col_inc];
    }

}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup copy_tile
///
template <typename src_scalar_t, typename dst_scalar_t>
void gecopy(Tile<src_scalar_t> const&& A, Tile<dst_scalar_t>&& B)
{
    gecopy(A, B);
}

//------------------------------------------------------------------------------
/// Copy and precision conversion.
/// @ingroup copy_tile
///
template <typename src_scalar_t, typename dst_scalar_t>
void tzcopy(Tile<src_scalar_t> const& A, Tile<dst_scalar_t>& B)
{
//  trace::Block trace_block("aux::copy");

    // TODO: Can be loosened?
    assert(A.uplo() != Uplo::General);
    assert(B.uplo() == A.uplo());

    assert(A.op() == Op::NoTrans);
    assert(B.op() == A.op());

    assert(A.mb() == B.mb());
    assert(A.nb() == B.nb());

    const src_scalar_t* A00 = &A.at(0, 0);
    int64_t a_col_inc = A.colIncrement();
    int64_t a_row_inc = A.rowIncrement();
    dst_scalar_t* B00 = &B.at(0, 0);
    int64_t b_col_inc = B.colIncrement();
    int64_t b_row_inc = B.rowIncrement();

    for (int64_t j = 0; j < B.nb(); ++j) {
        const src_scalar_t* Aj = &A00[j*a_row_inc];
        dst_scalar_t* Bj = &B00[j*b_row_inc];
        if (j < B.mb()) {
            Bj[j*b_col_inc] = Aj[j*a_col_inc];
        }
        if (B.uplo() == Uplo::Lower) {
            for (int64_t i = j; i < B.mb(); ++i) {
                Bj[i*b_col_inc] = Aj[i*a_col_inc];
            }
        }
        else {
            for (int64_t i = 0; i <= j && i < B.mb(); ++i) {
                Bj[i*b_col_inc] = Aj[i*a_col_inc];
            }
        }
    }
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup copy_tile
///
template <typename src_scalar_t, typename dst_scalar_t>
void tzcopy(Tile<src_scalar_t> const&& A, Tile<dst_scalar_t>&& B)
{
    tzcopy(A, B);
}

//------------------------------------------------------------------------------
/// Set entries in the matrix $A$ to the value of $\alpha$.
/// Only set the strictly-lower or the strictly-upper part.
/// @ingroup set_tile
///
template <typename scalar_t>
void tzset(scalar_t alpha, Tile<scalar_t>& A)
{
//  trace::Block trace_block("aux::tzset");

    // TODO: Can be loosened?
    assert(A.uplo() != Uplo::General);
    assert(A.op() == Op::NoTrans);

    scalar_t* A00 = &A.at(0, 0);
    int64_t a_col_inc = A.colIncrement();
    int64_t a_row_inc = A.rowIncrement();

    for (int64_t j = 0; j < A.nb(); ++j) {
        scalar_t* Aj = &A00[j*a_row_inc];
        if (A.uplo() == Uplo::Lower) {
            for (int64_t i = j+1; i < A.mb(); ++i) {
                Aj[i*a_col_inc] = alpha;
            }
        }
        else {
            for (int64_t i = 0; i < j && i < A.mb(); ++i) {
                Aj[i*a_col_inc] = alpha;
            }
        }
    }
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup set_tile
///
template <typename scalar_t>
void tzset(scalar_t alpha, Tile<scalar_t>&& A)
{
    tzset(alpha, A);
}

//------------------------------------------------------------------------------
/// Transpose a square matrix in-place, $A = A^T$.
/// Host implementation.
///
/// @param[in] n
///     Number of rows and columns of matrix A.
///
/// @param[in,out] A
///     The n-by-n matrix A, stored in an lda-by-n array, of input data.
///     On output: holds the transposed data.
///
/// @param[in] lda
///     Leading dimension of matrix A. lda >= n.
///
template <typename scalar_t>
void transpose(int64_t n,
               scalar_t* A, int64_t lda)
{
    assert(lda >= n);
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < j; ++i) { // upper
            std::swap(A[i + j*lda], A[j + i*lda]);
        }
    }
}

//------------------------------------------------------------------------------
/// Transpose a rectangular matrix out-of-place, $AT = A^T$.
/// Host implementation.
///
/// @param[in] m
///     Number of rows of matrix A.
///
/// @param[in] n
///     Number of columns of matrix A.
///
/// @param[in] A
///     The m-by-n matrix A, stored in an lda-by-n array, of input data.
///
/// @param[in] lda
///     Leading dimension of matrix A. lda >= m.
///
/// @param[out] AT
///     The n-by-m matrix AT, stored in an lda-by-m array, of output data.
///     On output: holds the transposed data.
///
/// @param[in] ldat
///     Leading dimension of matrix AT. ldat >= n.
///
template <typename scalar_t>
void transpose(int64_t m, int64_t n,
               scalar_t* A, int64_t lda,
               scalar_t* AT, int64_t ldat)
{
    assert(lda >= m);
    assert(ldat >= n);
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) {
            AT[j + i*ldat] = A[i + j*lda];
        }
    }
}

//------------------------------------------------------------------------------
/// Conjugate transpose a square matrix in-place, $A = A^H$.
/// Host implementation.
///
/// @param[in] n
///     Number of rows and columns of matrix A.
///
/// @param[in,out] A
///     The n-by-n matrix A, stored in an lda-by-n array, of input data.
///     On output: holds the conjugate-transposed data.
///
/// @param[in] lda
///     Leading dimension of matrix A. lda >= n.
///
template <typename scalar_t>
void conjTranspose(int64_t n,
                   scalar_t* A, int64_t lda)
{
    using blas::conj;
    assert(lda >= n);
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < j; ++i) { // upper
            scalar_t tmp = A[i + j*lda];
            A[i + j*lda] = conj(A[j + i*lda]);
            A[j + i*lda] = conj(tmp);
        }
        A[j + j*lda] = conj(A[j + j*lda]); // diag
    }
}

//------------------------------------------------------------------------------
/// Conjugate transpose a rectangular matrix out-of-place, $AT = A^H$.
/// Host implementation.
///
/// @param[in] m
///     Number of rows of matrix A.
///
/// @param[in] n
///     Number of columns of matrix A.
///
/// @param[in] A
///     The m-by-n matrix A, stored in an lda-by-n array, of input data.
///
/// @param[in] lda
///     Leading dimension of matrix A. lda >= m.
///
/// @param[out] AT
///     The n-by-m matrix AT, stored in an lda-by-m array, of output data.
///     On output: holds the conjugate-transposed data.
///
/// @param[in] ldat
///     Leading dimension of matrix AT. ldat >= n.
///
template <typename scalar_t>
void conjTranspose(int64_t m, int64_t n,
                   scalar_t* A, int64_t lda,
                   scalar_t* AT, int64_t ldat)
{
    using blas::conj;
    assert(lda >= m);
    assert(ldat >= n);
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) {
            AT[j + i*ldat] = conj(A[i + j*lda]);
        }
    }
}

//------------------------------------------------------------------------------
/// Transpose a square matrix in-place, $A = A^T$.
/// Host implementation.
///
template <typename scalar_t>
void deepTranspose(Tile<scalar_t>&& A)
{
    assert(A.mb() == A.nb());
    transpose(A.nb(), A.data(), A.stride());
}

//------------------------------------------------------------------------------
/// Transpose a rectangular matrix out-of-place, $AT = A^T$.
/// Host implementation.
///
template <typename scalar_t>
void deepTranspose(Tile<scalar_t>&& A, Tile<scalar_t>&& AT)
{
    assert(A.mb() == AT.nb());
    assert(A.nb() == AT.mb());
    transpose(A.mb(), A.nb(), A.data(), A.stride(), AT.data(), AT.stride());
}

//------------------------------------------------------------------------------
/// Conjugate transpose a square matrix in-place, $A = A^H$.
/// Host implementation.
///
template <typename scalar_t>
void deepConjTranspose(Tile<scalar_t>&& A)
{
    assert(A.mb() == A.nb());
    conjTranspose(A.nb(), A.data(), A.stride());
}

//------------------------------------------------------------------------------
/// Conjugate transpose a rectangular matrix out-of-place, $AT = A^H$.
/// Host implementation.
///
template <typename scalar_t>
void deepConjTranspose(Tile<scalar_t>&& A, Tile<scalar_t>&& AT)
{
    assert(A.mb() == AT.nb());
    assert(A.nb() == AT.mb());
    conjTranspose(A.mb(), A.nb(), A.data(), A.stride(), AT.data(), AT.stride());
}

//------------------------------------------------------------------------------
/// Copy a row of a tile to a vector.
///
template <typename scalar_t>
void copyRow(int64_t n,
             Tile<scalar_t>& A, int64_t i_offs, int64_t j_offs,
             scalar_t* V)
{
    // todo: size assertions
    for (int64_t j = 0; j < n; ++j)
        V[j] = A(i_offs, j_offs+j);
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
///
template <typename scalar_t>
void copyRow(int64_t n,
             Tile<scalar_t>&& A, int64_t i_offs, int64_t j_offs,
             scalar_t* V)
{
    copyRow(n, A, i_offs, j_offs, V);
}

//------------------------------------------------------------------------------
/// Copy a vector to a row of a tile.
///
template <typename scalar_t>
void copyRow(int64_t n,
             scalar_t* V,
             Tile<scalar_t>& A, int64_t i_offs, int64_t j_offs)
{
    // todo: size assertions
    if (n <= 0) return;

    scalar_t* A00 = &A.at(0, 0);
    int64_t a_col_inc = A.colIncrement();
    int64_t a_row_inc = A.rowIncrement();

    scalar_t* Ai = &A00[i_offs*a_col_inc];
    for (int64_t j = 0; j < n; ++j)
        Ai[(j_offs+j)*a_row_inc] = V[j];
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
///
template <typename scalar_t>
void copyRow(int64_t n,
             scalar_t* V,
             Tile<scalar_t>&& A, int64_t i_offs, int64_t j_offs)
{
    copyRow(n, V, A, i_offs, j_offs);
}

} // namespace tile

} // namespace slate

#endif // SLATE_TILE_AUX_HH
