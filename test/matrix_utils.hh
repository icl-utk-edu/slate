// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_MATRIX_UTILS_HH
#define SLATE_MATRIX_UTILS_HH

#include "slate/slate.hh"
#include "test.hh"

#include "scalapack_wrappers.hh"
#include "grid_utils.hh"

//------------------------------------------------------------------------------
// Zero out B, then copy band matrix B from A.
// B is stored as a non-symmetric matrix, so we can apply Q from left
// and right separately.
template <typename scalar_t>
void copy_he2gb(
    slate::HermitianMatrix<scalar_t> A,
    slate::Matrix<scalar_t> B )
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
void copy_he2ge(
    slate::HermitianMatrix<scalar_t> A,
    slate::Matrix<scalar_t> B )
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


//------------------------------------------------------------------------------
// Functions for allocating test matrices

template <typename MatrixType>
class TestMatrix {
    using scalar_t = typename MatrixType::value_type;

public:
    TestMatrix() {}

    TestMatrix(int64_t m_, int64_t n_, int64_t nb_,
               int p_, int q_, slate::GridOrder grid_order_)
        : m(m_), n(n_), nb(nb_), p(p_), q(q_), grid_order(grid_order_)
    {
        int mpi_rank, myrow, mycol;
        MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
        gridinfo( mpi_rank, grid_order, p, q, &myrow, &mycol );
        this->mloc = num_local_rows_cols( m, nb, myrow, p );
        this->nloc = num_local_rows_cols( n, nb, mycol, q );
        this->lld  = blas::max( 1, mloc ); // local leading dimension of A
    }

    // SLATE matrices
    MatrixType A;
    MatrixType Aref;

    // Storage for ScaLAPACK matrices
    std::vector<scalar_t> A_data;
    std::vector<scalar_t> Aref_data;

    // ScaLAPACK configuration
    int64_t m, n, mloc, nloc, lld, nb;
    int p, q;
    slate::GridOrder grid_order;

    #ifdef SLATE_HAVE_SCALAPACK
    void ScaLAPACK_descriptor( blas_int ictxt, blas_int A_desc[9] )
    {
        int64_t info;
        scalapack_descinit(A_desc, m, n, nb, nb, 0, 0, ictxt, mloc, &info);
        slate_assert(info == 0);
    }

    void create_ScaLAPACK_context( blas_int* ictxt )
    {
        // Call free function version
        ::create_ScaLAPACK_context( grid_order, p, q, ictxt );
    }
    #endif
};

//------------------------------------------------------------------------------
/// Marks the paramters used by allocate_test_Matrix
inline void mark_params_for_test_Matrix(Params& params)
{
    params.grid.m();
    params.grid.n();
    params.dev_dist();
    params.nb();
    params.nonuniform_nb();
    params.origin();
    params.grid_order();
}

//------------------------------------------------------------------------------
/// Marks the paramters used by allocate_test_HermitianMatrix
inline void mark_params_for_test_HermitianMatrix(Params& params)
{
    params.uplo();
    mark_params_for_test_Matrix( params );
}

//------------------------------------------------------------------------------
/// Marks the paramters used by allocate_test_SymmetricMatrix
inline void mark_params_for_test_SymmetricMatrix(Params& params)
{
    mark_params_for_test_HermitianMatrix( params );
}

//------------------------------------------------------------------------------
/// Marks the paramters used by allocate_test_HermitianMatrix
inline void mark_params_for_test_TriangularMatrix(Params& params)
{
    params.uplo();
    params.diag();
    mark_params_for_test_Matrix( params );
}

template <typename scalar_t>
TestMatrix<slate::Matrix<scalar_t>> allocate_test_Matrix(
        bool ref_matrix,
        bool nonuniform_ref,
        int64_t m,
        int64_t n,
        Params& params);

template <typename scalar_t>
TestMatrix<slate::HermitianMatrix<scalar_t>> allocate_test_HermitianMatrix(
        bool ref_matrix,
        bool nonuniform_ref,
        int64_t n,
        Params& params);

template <typename scalar_t>
TestMatrix<slate::SymmetricMatrix<scalar_t>> allocate_test_SymmetricMatrix(
        bool ref_matrix,
        bool nonuniform_ref,
        int64_t n,
        Params& params);

template <typename scalar_t>
TestMatrix<slate::TriangularMatrix<scalar_t>> allocate_test_TriangularMatrix(
        bool ref_matrix,
        bool nonuniform_ref,
        int64_t n,
        Params& params);

#endif // SLATE_MATRIX_UTILS_HH
