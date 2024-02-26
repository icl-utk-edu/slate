// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_TRIANGULAR_MATRIX_HH
#define SLATE_TRIANGULAR_MATRIX_HH

#include "slate/BaseTrapezoidMatrix.hh"
#include "slate/TrapezoidMatrix.hh"
#include "slate/Matrix.hh"
#include "slate/Tile.hh"
#include "slate/types.hh"

#include "lapack.hh"

#include <algorithm>
#include <utility>

#include "slate/internal/mpi.hh"

namespace slate {

template <typename scalar_t>
class TriangularBandMatrix;

//==============================================================================
/// Triangular, n-by-n, distributed, tiled matrices.
template <typename scalar_t>
class TriangularMatrix: public TrapezoidMatrix<scalar_t> {
public:
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    // constructors
    TriangularMatrix();

    TriangularMatrix(Uplo uplo, Diag diag, int64_t n,
                     std::function<int64_t (int64_t j)>& inTileNb,
                     std::function<int (ij_tuple ij)>& inTileRank,
                     std::function<int (ij_tuple ij)>& inTileDevice,
                     MPI_Comm mpi_comm);

    //----------
    TriangularMatrix(
        Uplo uplo, Diag diag, int64_t n, int64_t nb,
        GridOrder order, int p, int q, MPI_Comm mpi_comm );

    // With order = Col.
    TriangularMatrix(
        Uplo uplo, Diag diag, int64_t n, int64_t nb,
        int p, int q, MPI_Comm mpi_comm )
        : TriangularMatrix( uplo, diag, n, nb, GridOrder::Col, p, q, mpi_comm )
    {}

    //----------
    static
    TriangularMatrix fromLAPACK(Uplo uplo, Diag diag, int64_t n,
                                scalar_t* A, int64_t lda, int64_t nb,
                                int p, int q, MPI_Comm mpi_comm);

    //----------
    static
    TriangularMatrix fromScaLAPACK(
        Uplo uplo, Diag diag, int64_t n,
        scalar_t* A, int64_t lda, int64_t nb,
        GridOrder order, int p, int q, MPI_Comm mpi_comm);

    /// With order = Col.
    static
    TriangularMatrix fromScaLAPACK(
        Uplo uplo, Diag diag, int64_t n,
        scalar_t* A, int64_t lda, int64_t nb,
        int p, int q, MPI_Comm mpi_comm)
    {
        return fromScaLAPACK( uplo, diag, n, A, lda, nb,
                              GridOrder::Col, p, q, mpi_comm );
    }

    //----------
    static
    TriangularMatrix fromDevices(Uplo uplo, Diag diag, int64_t n,
                                 scalar_t** Aarray, int num_devices, int64_t lda,
                                 int64_t nb, int p, int q, MPI_Comm mpi_comm);

    // conversion
    TriangularMatrix(TrapezoidMatrix<scalar_t>& orig);

    TriangularMatrix(Diag diag, BaseTrapezoidMatrix<scalar_t>& orig);

    TriangularMatrix(Uplo uplo, Diag diag, BaseMatrix<scalar_t>& orig);

    // conversion sub-matrix
    TriangularMatrix(Diag diag, BaseTrapezoidMatrix<scalar_t>& orig,
                     int64_t i1, int64_t i2,
                     int64_t j1, int64_t j2);

    TriangularMatrix(Uplo uplo, Diag diag, Matrix<scalar_t>& orig,
                     int64_t i1, int64_t i2,
                     int64_t j1, int64_t j2);

    TriangularMatrix(TriangularBandMatrix<scalar_t>& orig,
                     int64_t i1, int64_t i2);

    // on-diagonal sub-matrix
    TriangularMatrix sub(int64_t i1, int64_t i2);
    TriangularMatrix slice(int64_t index1, int64_t index2);

    // off-diagonal sub-matrix
    Matrix<scalar_t> sub(int64_t i1, int64_t i2, int64_t j1, int64_t j2);
    Matrix<scalar_t> slice(int64_t row1, int64_t row2,
                           int64_t col1, int64_t col2);

    template <typename out_scalar_t=scalar_t>
    TriangularMatrix<out_scalar_t> emptyLike(int64_t nb=0,
                                             Op deepOp=Op::NoTrans);

protected:
    // used by fromLAPACK and fromScaLAPACK
    TriangularMatrix( Uplo uplo, Diag diag, int64_t n,
                      scalar_t* A, int64_t lda, int64_t nb,
                      GridOrder order, int p, int q, MPI_Comm mpi_comm,
                      bool is_scalapack );

    // used by fromDevices
    TriangularMatrix(Uplo uplo, Diag diag, int64_t n,
                     scalar_t** Aarray, int num_devices, int64_t lda,
                     int64_t nb, int p, int q, MPI_Comm mpi_comm);

    // used by sub
    TriangularMatrix(TriangularMatrix& orig,
                     int64_t i1, int64_t i2);

    // used by slice
    TriangularMatrix(TriangularMatrix& orig,
                     typename BaseMatrix<scalar_t>::Slice slice);

public:
    template <typename T>
    friend void swap(TriangularMatrix<T>& A, TriangularMatrix<T>& B);
};

//------------------------------------------------------------------------------
/// Default constructor creates an empty matrix.
template <typename scalar_t>
TriangularMatrix<scalar_t>::TriangularMatrix()
    : TrapezoidMatrix<scalar_t>()
{}

//------------------------------------------------------------------------------
/// Constructor creates an n-by-n matrix, with no tiles allocated,
/// where tileNb, tileRank, tileDevice are given as functions.
/// Tiles can be added with tileInsert().
///
/// @see slate::func for common functions.
///
template <typename scalar_t>
TriangularMatrix<scalar_t>::TriangularMatrix(
    Uplo uplo, Diag diag, int64_t n,
    std::function<int64_t (int64_t j)>& inTileNb,
    std::function<int (ij_tuple ij)>& inTileRank,
    std::function<int (ij_tuple ij)>& inTileDevice,
    MPI_Comm mpi_comm)
    : TrapezoidMatrix<scalar_t>(uplo, diag, n, n, inTileNb, inTileRank,
                                inTileDevice, mpi_comm)
{}

//------------------------------------------------------------------------------
/// Constructor creates an n-by-n matrix, with no tiles allocated,
/// with fixed nb-by-nb tile size and 2D block cyclic distribution.
/// Tiles can be added with tileInsert().
///
template <typename scalar_t>
TriangularMatrix<scalar_t>::TriangularMatrix(
    Uplo uplo, Diag diag, int64_t n, int64_t nb,
    GridOrder order, int p, int q, MPI_Comm mpi_comm)
    : TrapezoidMatrix<scalar_t>( uplo, diag, n, n, nb, order, p, q, mpi_comm )
{}

//------------------------------------------------------------------------------
/// [static]
/// Named constructor returns a new Matrix from LAPACK layout.
/// Construct matrix by wrapping existing memory of an n-by-n lower
/// or upper triangular LAPACK-style matrix.
///
/// @param[in] uplo
///     - Upper: upper triangle of A is stored.
///     - Lower: lower triangle of A is stored.
///
/// @param[in] diag
///     - NonUnit: A does not have unit diagonal.
///     - Unit:    A has unit diagonal; diagonal elements are not referenced
///                and are assumed to be one.
///
/// @param[in] n
///     Number of rows and columns of the matrix. n >= 0.
///
/// @param[in,out] A
///     The n-by-n triangular matrix A, stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of the array A. lda >= m.
///
/// @param[in] nb
///     Block size in 2D block-cyclic distribution. nb > 0.
///
/// @param[in] p
///     Number of block rows in 2D block-cyclic distribution. p > 0.
///
/// @param[in] q
///     Number of block columns of 2D block-cyclic distribution. q > 0.
///
/// @param[in] mpi_comm
///     MPI communicator to distribute matrix across.
///     p*q == MPI_Comm_size(mpi_comm).
///
template <typename scalar_t>
TriangularMatrix<scalar_t> TriangularMatrix<scalar_t>::fromLAPACK(
    Uplo uplo, Diag diag, int64_t n,
    scalar_t* A, int64_t lda, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
{
    return TriangularMatrix<scalar_t>( uplo, diag, n, A, lda, nb,
                                       GridOrder::Col, p, q, mpi_comm, false );
}

//------------------------------------------------------------------------------
/// [static]
/// Named constructor returns a new Matrix from ScaLAPACK layout.
/// Construct matrix by wrapping existing memory of an n-by-n lower
/// or upper triangular ScaLAPACK-style matrix.
/// @see BaseTrapezoidMatrix
///
/// @param[in] uplo
///     - Upper: upper triangle of A is stored.
///     - Lower: lower triangle of A is stored.
///
/// @param[in] diag
///     - NonUnit: A does not have unit diagonal.
///     - Unit:    A has unit diagonal; diagonal elements are not referenced
///                and are assumed to be one.
///
/// @param[in] n
///     Number of rows and columns of the matrix. n >= 0.
///
/// @param[in,out] A
///     The local portion of the 2D block cyclic distribution of
///     the n-by-n matrix A, with local leading dimension lda.
///
/// @param[in] lda
///     Local leading dimension of the array A. lda >= local number of rows.
///
/// @param[in] nb
///     Block size in 2D block-cyclic distribution. nb > 0.
///
/// @param[in] order
///     Order to map MPI processes to tile grid,
///     GridOrder::ColMajor (default) or GridOrder::RowMajor.
///
/// @param[in] p
///     Number of block rows in 2D block-cyclic distribution. p > 0.
///
/// @param[in] q
///     Number of block columns of 2D block-cyclic distribution. q > 0.
///
/// @param[in] mpi_comm
///     MPI communicator to distribute matrix across.
///     p*q == MPI_Comm_size(mpi_comm).
///
template <typename scalar_t>
TriangularMatrix<scalar_t> TriangularMatrix<scalar_t>::fromScaLAPACK(
    Uplo uplo, Diag diag, int64_t n,
    scalar_t* A, int64_t lda, int64_t nb,
    GridOrder order, int p, int q, MPI_Comm mpi_comm)
{
    return TriangularMatrix<scalar_t>( uplo, diag, n, A, lda, nb,
                                       order, p, q, mpi_comm, true);
}

//------------------------------------------------------------------------------
/// [static]
/// TODO
/// Named constructor returns a new Matrix from ScaLAPACK layout.
/// Construct matrix by wrapping existing memory of an n-by-n lower
/// or upper triangular ScaLAPACK-style matrix.
/// @see BaseTrapezoidMatrix
///
/// @param[in] uplo
///     - Upper: upper triangle of A is stored.
///     - Lower: lower triangle of A is stored.
///
/// @param[in] diag
///     - NonUnit: A does not have unit diagonal.
///     - Unit:    A has unit diagonal; diagonal elements are not referenced
///                and are assumed to be one.
///
/// @param[in] n
///     Number of rows and columns of the matrix. n >= 0.
///
/// @param[in,out] Aarray
///     TODO
///     The local portion of the 2D block cyclic distribution of
///     the n-by-n matrix A, with local leading dimension lda.
///
/// @param[in] num_devices
///     TODO
///
/// @param[in] lda
///     Local leading dimension of the array A. lda >= local number of rows.
///
/// @param[in] nb
///     Block size in 2D block-cyclic distribution. nb > 0.
///
/// @param[in] p
///     Number of block rows in 2D block-cyclic distribution. p > 0.
///
/// @param[in] q
///     Number of block columns of 2D block-cyclic distribution. q > 0.
///
/// @param[in] mpi_comm
///     MPI communicator to distribute matrix across.
///     p*q == MPI_Comm_size(mpi_comm).
///
template <typename scalar_t>
TriangularMatrix<scalar_t> TriangularMatrix<scalar_t>::fromDevices(
    Uplo uplo, Diag diag, int64_t n,
    scalar_t** Aarray, int num_devices, int64_t lda, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
{
    return TriangularMatrix<scalar_t>(uplo, diag, n, Aarray, num_devices, lda, nb,
                                      p, q, mpi_comm);
}

//------------------------------------------------------------------------------
/// @see fromLAPACK
/// @see fromScaLAPACK
///
/// @param[in] is_scalapack
///     If true,  A is a ScaLAPACK matrix.
///     If false, A is an LAPACK matrix.
///
template <typename scalar_t>
TriangularMatrix<scalar_t>::TriangularMatrix(
    Uplo uplo, Diag diag, int64_t n,
    scalar_t* A, int64_t lda, int64_t nb,
    GridOrder order, int p, int q, MPI_Comm mpi_comm, bool is_scalapack)
    : TrapezoidMatrix<scalar_t>( uplo, diag, n, n, A, lda, nb,
                                 order, p, q, mpi_comm, is_scalapack )
{}

//------------------------------------------------------------------------------
/// @see fromDevices
///
template <typename scalar_t>
TriangularMatrix<scalar_t>::TriangularMatrix(
    Uplo uplo, Diag diag, int64_t n,
    scalar_t** Aarray, int num_devices, int64_t lda, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
    : TrapezoidMatrix<scalar_t>(uplo, diag, n, n, Aarray, num_devices, lda, nb,
                                p, q, mpi_comm)
{}

//------------------------------------------------------------------------------
/// Conversion from trapezoid or triangular matrix
/// creates a shallow copy view of the original matrix.
/// Orig must be square -- slice beforehand if needed.
///
/// @param[in,out] orig
///     Original matrix.
///
template <typename scalar_t>
TriangularMatrix<scalar_t>::TriangularMatrix(
    TrapezoidMatrix<scalar_t>& orig)
    : TrapezoidMatrix<scalar_t>(orig)
{
    slate_assert(orig.mt() == orig.nt());
    slate_assert(orig.m() == orig.n());
}

//------------------------------------------------------------------------------
/// Conversion from trapezoid, triangular, symmetric, or Hermitian matrix
/// creates a shallow copy view of the original matrix.
/// Orig must be square -- slice beforehand if needed.
///
/// @param[in] diag
///     - NonUnit: A does not have unit diagonal.
///     - Unit:    A has unit diagonal; diagonal elements are not referenced
///                and are assumed to be one.
///
/// @param[in,out] orig
///     Original matrix.
///
template <typename scalar_t>
TriangularMatrix<scalar_t>::TriangularMatrix(
    Diag diag, BaseTrapezoidMatrix<scalar_t>& orig)
    : TrapezoidMatrix<scalar_t>(diag, orig)
{
    slate_assert(orig.mt() == orig.nt());
    slate_assert(orig.m() == orig.n());
}

//------------------------------------------------------------------------------
/// Conversion from trapezoid, triangular, symmetric, or Hermitian matrix
/// creates a shallow copy view of the original matrix, A[ i1:i2, j1:j2 ].
///
/// @param[in] diag
///     - NonUnit: A does not have unit diagonal.
///     - Unit:    A has unit diagonal; diagonal elements are not referenced
///                and are assumed to be one.
///
/// @param[in,out] orig
///     Original matrix.
///
/// @param[in] i1
///     Starting block row index. 0 <= i1 < mt.
///
/// @param[in] i2
///     Ending block row index (inclusive). i2 < mt.
///
/// @param[in] j1
///     Starting block column index. 0 <= j1 < nt.
///
/// @param[in] j2
///     Ending block column index (inclusive). j2 < nt.
///     j2 - j1 = i2 - i1, i.e., it is square.
///
template <typename scalar_t>
TriangularMatrix<scalar_t>::TriangularMatrix(
    Diag diag, BaseTrapezoidMatrix<scalar_t>& orig,
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
    : TrapezoidMatrix<scalar_t>(diag, orig, i1, i2, j1, j2)
{
    slate_assert(i2 - i1 == j2 - j1);
}

//------------------------------------------------------------------------------
/// Conversion from general matrix
/// creates a shallow copy view of the original matrix.
/// Orig must be square -- slice beforehand if needed.
///
/// @param[in] uplo
///     - Upper: upper triangle of A is stored.
///     - Lower: lower triangle of A is stored.
///
/// @param[in] diag
///     - NonUnit: A does not have unit diagonal.
///     - Unit:    A has unit diagonal; diagonal elements are not referenced
///                and are assumed to be one.
///
/// @param[in,out] orig
///     Original matrix.
///
template <typename scalar_t>
TriangularMatrix<scalar_t>::TriangularMatrix(
    Uplo uplo, Diag diag, BaseMatrix<scalar_t>& orig)
    : TrapezoidMatrix<scalar_t>(uplo, diag, orig)
{
    slate_assert(orig.mt() == orig.nt());
    slate_assert(orig.m() == orig.n());
}

//------------------------------------------------------------------------------
/// Conversion from general matrix, sub-matrix constructor
/// creates shallow copy view of original matrix, A[ i1:i2, j1:j2 ].
///
/// @param[in] diag
///     - NonUnit: A does not have unit diagonal.
///     - Unit:    A has unit diagonal; diagonal elements are not referenced
///                and are assumed to be one.
///
/// @param[in,out] orig
///     Original matrix.
///
/// @param[in] i1
///     Starting block row index. 0 <= i1 < mt.
///
/// @param[in] i2
///     Ending block row index (inclusive). i2 < mt.
///
/// @param[in] j1
///     Starting block column index. 0 <= j1 < nt.
///
/// @param[in] j2
///     Ending block column index (inclusive). j2 < nt.
///     j2 - j1 = i2 - i1, i.e., it is square.
///
template <typename scalar_t>
TriangularMatrix<scalar_t>::TriangularMatrix(
    Uplo uplo, Diag diag, Matrix<scalar_t>& orig,
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
    : TrapezoidMatrix<scalar_t>(uplo, diag, orig, i1, i2, j1, j2)
{
    if ((i2 - i1) != (j2 - j1))
        throw std::runtime_error("i2 - i1 != j2 - j1, Matrix");
}

//------------------------------------------------------------------------------
/// Sub-matrix constructor creates shallow copy view of parent matrix,
/// A[ i1:i2, i1:i2 ]. The new view is still a triangular matrix, with the
/// same diagonal as the parent matrix.
///
/// @param[in,out] orig
///     Original matrix.
///
/// @param[in] i1
///     Starting block row and column index. 0 <= i1 < mt.
///
/// @param[in] i2
///     Ending block row and column index (inclusive). i2 < mt.
///
template <typename scalar_t>
TriangularMatrix<scalar_t>::TriangularMatrix(
    TriangularMatrix& orig,
    int64_t i1, int64_t i2)
    : TrapezoidMatrix<scalar_t>(orig, i1, i2, i1, i2)
{}

//------------------------------------------------------------------------------
/// Sub-matrix constructor creates shallow copy view of parent matrix,
/// A[ i1:i2, i1:i2 ]. The new view is still a triangular matrix, with the
/// same diagonal as the parent matrix.
///
/// @param[in,out] orig
///     Original matrix.
///
/// @param[in] i1
///     Starting block row and column index. 0 <= i1 < mt.
///
/// @param[in] i2
///     Ending block row and column index (inclusive). i2 < mt.
///
template <typename scalar_t>
TriangularMatrix<scalar_t>::TriangularMatrix(
    TriangularBandMatrix<scalar_t>& orig,
    int64_t i1, int64_t i2)
    : TrapezoidMatrix<scalar_t>(orig.uploPhysical(), orig.diag(), orig, i1, i2, i1, i2)
{}

//------------------------------------------------------------------------------
/// Returns sub-matrix that is a shallow copy view of the
/// parent matrix, A[ i1:i2, i1:i2 ].
/// This version returns a TriangularMatrix with the same diagonal as the
/// parent matrix.
/// @see Matrix TrapezoidMatrix::sub(int64_t i1, int64_t i2,
///                                  int64_t j1, int64_t j2)
///
/// @param[in] i1
///     Starting block row and column index. 0 <= i1 < mt.
///
/// @param[in] i2
///     Ending block row and column index (inclusive). i2 < mt.
///
template <typename scalar_t>
TriangularMatrix<scalar_t> TriangularMatrix<scalar_t>::sub(
    int64_t i1, int64_t i2)
{
    return TriangularMatrix<scalar_t>(*this, i1, i2);
}

//------------------------------------------------------------------------------
/// Returns off-diagonal sub-matrix that is a shallow copy view of the
/// parent matrix, A[ i1:i2, j1:j2 ].
/// This version returns a general Matrix, which:
/// - if uplo = Lower, is strictly below the diagonal, or
/// - if uplo = Upper, is strictly above the diagonal.
/// @see TrapezoidMatrix sub(int64_t i1, int64_t i2)
///
/// @param[in] i1
///     Starting block row index. 0 <= i1 < mt.
///
/// @param[in] i2
///     Ending block row index (inclusive). i2 < mt.
///
/// @param[in] j1
///     Starting block column index. 0 <= j1 < nt.
///
/// @param[in] j2
///     Ending block column index (inclusive). j2 < nt.
///
template <typename scalar_t>
Matrix<scalar_t> TriangularMatrix<scalar_t>::sub(
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
{
    return BaseTrapezoidMatrix<scalar_t>::sub(i1, i2, j1, j2);
}

//------------------------------------------------------------------------------
/// Sliced matrix constructor creates shallow copy view of parent matrix,
/// A[ row1:row2, col1:col2 ].
/// This takes row & col indices instead of block row & block col indices.
/// Assumes that row1 == col1 and row2 == col2 (@see slice()).
///
/// @param[in] orig
///     Original matrix of which to make sub-matrix.
///
/// @param[in] slice
///     Contains start and end row and column indices.
///
template <typename scalar_t>
TriangularMatrix<scalar_t>::TriangularMatrix(
    TriangularMatrix<scalar_t>& orig,
    typename BaseMatrix<scalar_t>::Slice slice)
    : TrapezoidMatrix<scalar_t>(orig, slice)
{}

//------------------------------------------------------------------------------
/// Returns sliced matrix that is a shallow copy view of the
/// parent matrix, A[ index1:index2, index1:index2 ].
/// This takes row & col indices instead of block row & block col indices.
///
/// @param[in] index1
///     Starting row and col index. 0 <= index1 < n.
///
/// @param[in] index2
///     Ending row and col index (inclusive). index1 <= index2 < n.
///
template <typename scalar_t>
TriangularMatrix<scalar_t> TriangularMatrix<scalar_t>::slice(
    int64_t index1, int64_t index2)
{
    return TriangularMatrix<scalar_t>(*this,
        typename BaseMatrix<scalar_t>::Slice(index1, index2, index1, index2));
}

//------------------------------------------------------------------------------
/// Returns sliced matrix that is a shallow copy view of the
/// parent matrix, A[ row1:row2, col1:col2 ].
/// This takes row & col indices instead of block row & block col indices.
/// The sub-matrix cannot overlap the diagonal.
/// - if uplo = Lower, 0 <= col1 <= col2 <= row1 <= row2 < n;
/// - if uplo = Upper, 0 <= row1 <= row2 <= col1 <= col2 < n.
///
/// @param[in] row1
///     Starting row index.
///
/// @param[in] row2
///     Ending row index (inclusive).
///
/// @param[in] col1
///     Starting column index.
///
/// @param[in] col2
///     Ending column index (inclusive).
///
template <typename scalar_t>
Matrix<scalar_t> TriangularMatrix<scalar_t>::slice(
    int64_t row1, int64_t row2,
    int64_t col1, int64_t col2)
{
    return Matrix<scalar_t>(*this,
        typename BaseMatrix<scalar_t>::Slice(row1, row2, col1, col2));
}

//------------------------------------------------------------------------------
/// Swaps contents of matrices A and B.
//
// (This isn't really needed over TrapezoidMatrix swap, but is here as a
// reminder in case any members are added that aren't in TrapezoidMatrix.)
template <typename scalar_t>
void swap(TriangularMatrix<scalar_t>& A, TriangularMatrix<scalar_t>& B)
{
    using std::swap;
    swap(static_cast< TrapezoidMatrix<scalar_t>& >(A),
         static_cast< TrapezoidMatrix<scalar_t>& >(B));
}

//------------------------------------------------------------------------------
/// Named constructor returns a new, empty Matrix with the same structure
/// (size and distribution) as this matrix. Tiles are not allocated.
///
template <typename scalar_t>
template <typename out_scalar_t>
TriangularMatrix<out_scalar_t> TriangularMatrix<scalar_t>::emptyLike(
    int64_t nb, Op deepOp)
{
    auto B = this->template baseEmptyLike<out_scalar_t>(nb, nb, deepOp);
    return TriangularMatrix<out_scalar_t>(this->uplo(), this->diag(), B);
}

} // namespace slate

#endif // SLATE_TRIANGULAR_MATRIX_HH
