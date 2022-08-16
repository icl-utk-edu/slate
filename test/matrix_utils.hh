// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_MATRIX_UTILS_HH
#define SLATE_MATRIX_UTILS_HH

#include "slate/slate.hh"

//------------------------------------------------------------------------------
// Zero out B, then copy band matrix B from A.
// B is stored as a non-symmetric matrix, so we can apply Q from left
// and right separately.
template <typename scalar_t>
void he2gb(slate::HermitianMatrix< scalar_t > A, slate::Matrix< scalar_t > B)
{
    // It must be defined here to avoid having numerical error with complex
    // numbers when calling conj();
    using blas::conj;
    int64_t nt = A.nt();
    const scalar_t zero = 0;
    set(zero, B);
    for (int64_t i = 0; i < nt; ++i) {
        int tag_i = i+1;
        if (B.tileIsLocal(i, i)) {
            // diagonal tile
            A.tileGetForReading(i, i, slate::LayoutConvert::ColMajor);
            auto Aii = A(i, i);
            auto Bii = B(i, i);
            Aii.uplo(slate::Uplo::Lower);
            Bii.uplo(slate::Uplo::Lower);
            slate::tile::tzcopy( Aii, Bii );
            // Symmetrize the tile.
            for (int64_t jj = 0; jj < Bii.nb(); ++jj)
                for (int64_t ii = jj; ii < Bii.mb(); ++ii)
                    Bii.at(jj, ii) = conj(Bii(ii, jj));
        }
        if (i+1 < nt && B.tileIsLocal(i+1, i)) {
            // sub-diagonal tile
            A.tileGetForReading(i+1, i, slate::LayoutConvert::ColMajor);
            auto Ai1i = A(i+1, i);
            auto Bi1i = B(i+1, i);
            Ai1i.uplo(slate::Uplo::Upper);
            Bi1i.uplo(slate::Uplo::Upper);
            slate::tile::tzcopy( Ai1i, Bi1i );
            if (! B.tileIsLocal(i, i+1))
                B.tileSend(i+1, i, B.tileRank(i, i+1), tag_i);
        }
        if (i+1 < nt && B.tileIsLocal(i, i+1)) {
            if (! B.tileIsLocal(i+1, i)) {
                // Remote copy-transpose B(i+1, i) => B(i, i+1);
                // assumes square tiles!
                B.tileRecv(i+1, i, B.tileRank(i+1, i), slate::Layout::ColMajor, tag_i);
                slate::tile::deepConjTranspose( B(i+1, i), B(i, i+1) );
            }
            else {
                // Local copy-transpose B(i+1, i) => B(i, i+1).
                slate::tile::deepConjTranspose( B(i+1, i), B(i, i+1) );
            }
        }
    }
}

//------------------------------------------------------------------------------
// Convert a HermitianMatrix into a General Matrix, ConjTrans/Trans the opposite
// off-diagonal tiles
// todo: shouldn't assume the input HermitianMatrix has uplo=lower
template <typename scalar_t>
inline void he2ge(slate::HermitianMatrix<scalar_t> A, slate::Matrix<scalar_t> B)
{
    // todo:: shouldn't assume the input matrix has uplo=lower
    assert(A.uplo() == slate::Uplo::Lower);

    using blas::conj;
    const scalar_t zero = 0;
    set(zero, B);
    for (int64_t j = 0; j < A.nt(); ++j) {
        // todo: shouldn't assume uplo=lowwer
        for (int64_t i = j; i < A.nt(); ++i) {
            if (i == j) { // diagonal tiles
                if (B.tileIsLocal(i, j)) {
                    auto Aij = A(i, j);
                    auto Bij = B(i, j);
                    Aij.uplo(slate::Uplo::Lower);
                    Bij.uplo(slate::Uplo::Lower);
                    slate::tile::tzcopy( Aij, Bij );
                    for (int64_t jj = 0; jj < Bij.nb(); ++jj) {
                        for (int64_t ii = jj; ii < Bij.mb(); ++ii) {
                            Bij.at(jj, ii) = conj(Bij(ii, jj));
                        }
                    }
                }
            }
            else {
                if (B.tileIsLocal(i, j)) {
                    auto Aij = A(i, j);
                    auto Bij = B(i, j);
                    slate::tile::gecopy( Aij, Bij );
                    if (! B.tileIsLocal(j, i)) {
                        B.tileSend(i, j, B.tileRank(j, i));
                    }
                }
                if (B.tileIsLocal(j, i)) {
                    if (! B.tileIsLocal(i, j)) {
                        B.tileRecv(
                            j, i, B.tileRank(i, j), slate::Layout::ColMajor);
                        slate::tile::deepConjTranspose( B(j, i) );
                    }
                    else {
                        slate::tile::deepConjTranspose( B(i, j), B(j, i) );
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// Class to carry matrix_type for overloads with return type, in lieu of
// partial specialization. See
// https://www.fluentcpp.com/2017/08/15/function-templates-partial-specialization-cpp/
template <typename T>
struct MatrixType {};

//------------------------------------------------------------------------------
// Overloads for each matrix type. dummy carries the return type.

/// Cast to General m-by-n matrix. Nothing to do. uplo and diag unused.
template <typename scalar_t>
slate::Matrix<scalar_t> matrix_cast_impl(
    MatrixType< slate::Matrix<scalar_t> > dummy,
    slate::Matrix<scalar_t>& A,
    slate::Uplo uplo,
    slate::Diag diag )
{
    return A;
}

/// Cast to Trapezoid m-by-n matrix.
template <typename scalar_t>
slate::TrapezoidMatrix<scalar_t> matrix_cast_impl(
    MatrixType< slate::TrapezoidMatrix<scalar_t> > dummy,
    slate::Matrix<scalar_t>& A,
    slate::Uplo uplo,
    slate::Diag diag )
{
    return slate::TrapezoidMatrix<scalar_t>( uplo, diag, A );
}

/// Cast to Triangular n-by-n matrix.
template <typename scalar_t>
slate::TriangularMatrix<scalar_t> matrix_cast_impl(
    MatrixType< slate::TriangularMatrix<scalar_t> > dummy,
    slate::Matrix<scalar_t>& A,
    slate::Uplo uplo,
    slate::Diag diag )
{
    slate_assert( A.m() == A.n() );  // must be square
    return slate::TriangularMatrix<scalar_t>( uplo, diag, A );
}

/// Cast to Symmetric n-by-n matrix. diag unused.
template <typename scalar_t>
slate::SymmetricMatrix<scalar_t> matrix_cast_impl(
    MatrixType< slate::SymmetricMatrix<scalar_t> > dummy,
    slate::Matrix<scalar_t>& A,
    slate::Uplo uplo,
    slate::Diag diag )
{
    slate_assert( A.m() == A.n() );  // must be square
    return slate::SymmetricMatrix<scalar_t>( uplo, A );
}

/// Cast to Hermitian n-by-n matrix. diag unused.
template <typename scalar_t>
slate::HermitianMatrix<scalar_t> matrix_cast_impl(
    MatrixType< slate::HermitianMatrix<scalar_t> > dummy,
    slate::Matrix<scalar_t>& A,
    slate::Uplo uplo,
    slate::Diag diag )
{
    slate_assert( A.m() == A.n() );  // must be square
    return slate::HermitianMatrix<scalar_t>( uplo, A );
}

//------------------------------------------------------------------------------
/// Casts general m-by-n matrix to matrix_type, which can be
/// Matrix, Trapezoid, Triangular*, Symmetric*, or Hermitian*.
/// uplo and diag are ignored when not applicable.
/// * types require input A to be square.
///
template <typename matrix_type>
matrix_type matrix_cast(
    slate::Matrix< typename matrix_type::value_type >& A,
    slate::Uplo uplo,
    slate::Diag diag )
{
    return matrix_cast_impl( MatrixType< matrix_type >(), A, uplo, diag );
}

#endif // SLATE_MATRIX_UTILS_HH
