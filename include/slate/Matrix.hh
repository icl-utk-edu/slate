// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_MATRIX_HH
#define SLATE_MATRIX_HH

#include "slate/BaseMatrix.hh"
#include "slate/Tile.hh"
#include "slate/types.hh"

#include "lapack.hh"

#include <algorithm>
#include <utility>

#include "slate/internal/mpi.hh"
#include "slate/internal/openmp.hh"

namespace slate {

//==============================================================================
/// General non-symmetric, m-by-n, distributed, tiled matrices.
template <typename scalar_t>
class Matrix: public BaseMatrix<scalar_t> {
public:
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    // constructors
    Matrix();

    Matrix(int64_t m, int64_t n,
           std::function<int64_t (int64_t i)>& inTileMb,
           std::function<int64_t (int64_t j)>& inTileNb,
           std::function<int (ij_tuple ij)>& inTileRank,
           std::function<int (ij_tuple ij)>& inTileDevice,
           MPI_Comm mpi_comm);

    //----------
    Matrix( int64_t m, int64_t n, int64_t mb, int64_t nb,
            GridOrder order, int p, int q, MPI_Comm mpi_comm );

    /// With order = Col.
    Matrix( int64_t m, int64_t n, int64_t mb, int64_t nb,
            int p, int q, MPI_Comm mpi_comm )
        : Matrix( m, n, mb, nb, GridOrder::Col, p, q, mpi_comm )
    {}

    /// With mb = nb, order = Col.
    Matrix( int64_t m, int64_t n, int64_t nb,
            int p, int q, MPI_Comm mpi_comm )
        : Matrix( m, n, nb, nb, GridOrder::Col, p, q, mpi_comm )
    {}

    //----------
    static
    Matrix fromLAPACK(int64_t m, int64_t n,
                      scalar_t* A, int64_t lda, int64_t mb, int64_t nb,
                      int p, int q, MPI_Comm mpi_comm);

    /// With mb = nb.
    static
    Matrix fromLAPACK(int64_t m, int64_t n,
                      scalar_t* A, int64_t lda, int64_t nb,
                      int p, int q, MPI_Comm mpi_comm)
    {
        return fromLAPACK(m, n, A, lda, nb, nb, p, q, mpi_comm);
    }

    //----------
    static
    Matrix fromScaLAPACK(
        int64_t m, int64_t n,
        scalar_t* A, int64_t lda, int64_t mb, int64_t nb,
        GridOrder order, int p, int q, MPI_Comm mpi_comm );

    /// With order = Col.
    static
    Matrix fromScaLAPACK(
        int64_t m, int64_t n,
        scalar_t* A, int64_t lda, int64_t mb, int64_t nb,
        int p, int q, MPI_Comm mpi_comm )
    {
        return fromScaLAPACK( m, n, A, lda, mb, nb,
                              GridOrder::Col, p, q, mpi_comm );
    }

    /// With mb = nb, order = Col.
    static
    Matrix fromScaLAPACK(
        int64_t m, int64_t n,
        scalar_t* A, int64_t lda, int64_t nb,
        int p, int q, MPI_Comm mpi_comm )
    {
        return fromScaLAPACK( m, n, A, lda, nb, nb,
                              GridOrder::Col, p, q, mpi_comm );
    }

    //----------
    static
    Matrix fromDevices(int64_t m, int64_t n,
                       scalar_t** Aarray, int num_devices, int64_t lda,
                       int64_t mb, int64_t nb, int p, int q, MPI_Comm mpi_comm);

    /// With mb = nb.
    static
    Matrix fromDevices(int64_t m, int64_t n,
                       scalar_t** Aarray, int num_devices, int64_t lda,
                       int64_t nb, int p, int q, MPI_Comm mpi_comm)
    {
        return fromDevices(m, n, Aarray, num_devices, lda, nb, nb, p, q, mpi_comm);
    }

    //----------
    template <typename out_scalar_t=scalar_t>
    Matrix<out_scalar_t> emptyLike(int64_t mb=0, int64_t nb=0,
                                   Op deepOp=Op::NoTrans);

    template <typename out_scalar_t=scalar_t>
    static
    Matrix<out_scalar_t> emptyLike(BaseMatrix<scalar_t>& orig, int64_t mb=0,
                                   int64_t nb=0, Op deepOp=Op::NoTrans);

    // conversion sub-matrix
    Matrix(BaseMatrix<scalar_t>& orig,
           int64_t i1, int64_t i2,
           int64_t j1, int64_t j2);

    // sub-matrix
    Matrix sub(int64_t i1, int64_t i2,
               int64_t j1, int64_t j2);

    // sliced matrix
    Matrix slice(int64_t row1, int64_t row2,
                 int64_t col1, int64_t col2);

    // used by slice and conversion slice
    Matrix(BaseMatrix<scalar_t>& orig,
           typename BaseMatrix<scalar_t>::Slice slice);

protected:
    // used by fromLAPACK and fromScaLAPACK
    Matrix(int64_t m, int64_t n,
           scalar_t* A, int64_t lda, int64_t mb, int64_t nb,
           GridOrder order, int p, int q, MPI_Comm mpi_comm, bool is_scalapack);

    // used by fromDevices
    Matrix(int64_t m, int64_t n, scalar_t** Aarray,
           int num_devices, int64_t lda, int64_t mb, int64_t nb,
           int p, int q, MPI_Comm mpi_comm);

public:
    template <typename T>
    friend void swap(Matrix<T>& A, Matrix<T>& B);

    int64_t getMaxHostTiles();
    int64_t getMaxDeviceTiles(int device);
    void allocateBatchArrays(int64_t batch_size=0, int64_t num_arrays=1);
    void reserveHostWorkspace();
    void reserveDeviceWorkspace();
    void gather(scalar_t* A, int64_t lda);
    void insertLocalTiles(Target origin=Target::Host);
    void redistribute(Matrix<scalar_t>& A);
};

//------------------------------------------------------------------------------
/// Default constructor creates an empty matrix.
template <typename scalar_t>
Matrix<scalar_t>::Matrix():
    BaseMatrix<scalar_t>()
{}

//------------------------------------------------------------------------------
/// Constructor creates an m-by-n matrix, with no tiles allocated,
/// where tileMb, tileNb, tileRank, tileDevice are given as functions.
/// Tiles can be added with tileInsert().
///
/// @param[in] m
///     Number of rows of the matrix. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix. n >= 0.
///
/// @param[in] inTileMb
///     Function that takes block-row index, returns block-row size.
///
/// @param[in] inTileNb
///     Function that takes block-col index, returns block-col size.
///
/// @param[in] inTileRank
///     Function that takes tuple of { block-row, block-col } indices,
///     returns MPI rank for that tile.
///
/// @param[in] inTileDevice
///     Function that takes tuple of { block-row, block-col } indices,
///     returns local GPU device ID for that tile.
///
/// @param[in] mpi_comm
///     MPI communicator to distribute matrix across.
///
template <typename scalar_t>
Matrix<scalar_t>::Matrix(
    int64_t m, int64_t n,
    std::function<int64_t (int64_t i)>& inTileMb,
    std::function<int64_t (int64_t j)>& inTileNb,
    std::function<int (ij_tuple ij)>& inTileRank,
    std::function<int (ij_tuple ij)>& inTileDevice,
    MPI_Comm mpi_comm)
    : BaseMatrix<scalar_t>(m, n, inTileMb, inTileNb, inTileRank, inTileDevice,
                           mpi_comm)
{}

//------------------------------------------------------------------------------
/// Constructor creates an m-by-n matrix, with no tiles allocated,
/// with fixed mb-by-nb tile size and 2D block cyclic distribution.
/// Tiles can be added with tileInsert().
///
/// @param[in] m
///     Number of rows of the matrix. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix. n >= 0.
///
/// @param[in] mb
///     Row block size in 2D block-cyclic distribution. mb > 0.
///
/// @param[in] nb
///     Column block size in 2D block-cyclic distribution. nb > 0.
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
Matrix<scalar_t>::Matrix(
    int64_t m, int64_t n, int64_t mb, int64_t nb,
    GridOrder order, int p, int q, MPI_Comm mpi_comm)
    : BaseMatrix<scalar_t>( m, n, mb, nb, order, p, q, mpi_comm )
{}

//------------------------------------------------------------------------------
/// [static]
/// Named constructor returns a new Matrix from LAPACK layout.
/// Construct matrix by wrapping existing memory of an m-by-n LAPACK matrix.
/// The caller must ensure that the memory remains valid for the lifetime
/// of the Matrix object and any shallow copies of it.
/// Input format is an LAPACK-style column-major matrix with leading
/// dimension (column stride) lda >= m, that is replicated across all nodes.
/// Matrix gets tiled with mb-by-nb tiles.
///
/// @param[in] m
///     Number of rows of the matrix. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix. n >= 0.
///
/// @param[in,out] A
///     The m-by-n matrix A, stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of the array A. lda >= m.
///
/// @param[in] mb
///     Row block size in 2D block-cyclic distribution. mb > 0.
///
/// @param[in] nb
///     Column block size in 2D block-cyclic distribution. nb > 0.
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
Matrix<scalar_t> Matrix<scalar_t>::fromLAPACK(
    int64_t m, int64_t n,
    scalar_t* A, int64_t lda, int64_t mb, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
{
    return Matrix<scalar_t>( m, n, A, lda, mb, nb,
                             GridOrder::Col, p, q, mpi_comm, false );
}

//------------------------------------------------------------------------------
/// [static]
/// Named constructor returns a new Matrix from ScaLAPACK layout.
/// Construct matrix by wrapping existing memory of an m-by-n ScaLAPACK matrix.
/// The caller must ensure that the memory remains valid for the lifetime
/// of the Matrix object and any shallow copies of it.
/// Input format is a ScaLAPACK-style 2D block-cyclic column-major matrix
/// with local leading dimension (column stride) lda,
/// p block rows and q block columns.
/// Matrix gets tiled with mb-by-nb tiles.
///
/// @param[in] m
///     Number of rows of the matrix. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix. n >= 0.
///
/// @param[in,out] A
///     The local portion of the 2D block cyclic distribution of
///     the m-by-n matrix A, with local leading dimension lda.
///
/// @param[in] lda
///     Local leading dimension of the array A. lda >= local number of rows.
///
/// @param[in] mb
///     Row block size in 2D block-cyclic distribution. mb > 0.
///
/// @param[in] nb
///     Column block size in 2D block-cyclic distribution. nb > 0.
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
Matrix<scalar_t> Matrix<scalar_t>::fromScaLAPACK(
    int64_t m, int64_t n,
    scalar_t* A, int64_t lda, int64_t mb, int64_t nb,
    GridOrder order, int p, int q, MPI_Comm mpi_comm)
{
    return Matrix<scalar_t>( m, n, A, lda, mb, nb, order, p, q, mpi_comm, true );
}

//------------------------------------------------------------------------------
/// [static]
/// Named constructor returns a new Matrix from data in GPU device memory,
/// distributed 2D block-cyclic across MPI ranks,
/// and 1D block-cyclic across GPU devices within an MPI rank.
///
/// Aarray contains pointers to data on each GPU device in this MPI rank.
///
/// @param[in] m
///     Number of rows of the matrix. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix. n >= 0.
///
/// @param[in,out] Aarray
///     Array of
///     The local portion of the 2D block cyclic distribution of
///     the m-by-n matrix A, with local leading dimension lda.
///
/// @param[in] num_devices
///     Dimension of Aarray.
///
/// @param[in] lda
///     Local leading dimension of the array A. lda >= local number of rows.
///
/// @param[in] mb
///     Row block size in 2D block-cyclic distribution. mb > 0.
///
/// @param[in] nb
///     Column block size in 2D block-cyclic distribution. nb > 0.
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
Matrix<scalar_t> Matrix<scalar_t>::fromDevices(
    int64_t m, int64_t n,
    scalar_t** Aarray, int num_devices, int64_t lda,
    int64_t mb, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    return Matrix<scalar_t>(m, n, Aarray, num_devices, lda, mb, nb,
                            p, q, mpi_comm);
}

//------------------------------------------------------------------------------
/// Named constructor returns a new, empty Matrix with the same structure
/// (distribution and number of tiles) as this matrix. Tiles are not allocated.
///
/// @param[in] mb
///     Row block size of new matrix.
///     If mb = 0, uses the same mb and m as this matrix;
///     otherwise, m = mb * mt.
///
/// @param[in] nb
///     Column block size of new matrix.
///     If nb = 0, uses the same nb and n as this matrix;
///     otherwise, n = nb * nt.
///
/// @param[in] deepOp
///     Additional deep-transposition operation to apply. If deepOp=Trans, the
///     new matrix has the transposed structure (distribution and number of
///     tiles) of this matrix, but its shallow-transpose op() flag is set to
///     NoTrans. For a 1x4 matrix A, compare:
///     - transpose(A).emptyLike() creates a new 1x4 matrix, then transposes it
///       to return a 4x1 matrix with its op set to Trans.
///     - A.emptyLike(mb, nb, Op::Trans) creates and returns a new 4x1 matrix
///       with its op set to NoTrans.
///
template <typename scalar_t>
template <typename out_scalar_t>
Matrix<out_scalar_t> Matrix<scalar_t>::emptyLike(
    int64_t mb, int64_t nb, Op deepOp)
{
    auto B = this->template baseEmptyLike<out_scalar_t>(mb, nb, deepOp);
    return Matrix<out_scalar_t>(B, 0, B.mt()-1, 0, B.nt()-1);
}

//------------------------------------------------------------------------------
/// Named constructor returns a new, empty Matrix with the same structure
/// (size and distribution) as the matrix orig. Tiles are not allocated.
///
/// @param[in] orig
///     Original matrix of which to make an empty matrix with the same structure
///     (size and distribution) as this original matrix.
///
/// @param[in] mb
///     Row block size of new matrix.
///     If mb = 0, uses the same mb and m as this matrix;
///     otherwise, m = mb * mt.
///
/// @param[in] nb
///     Column block size of new matrix.
///     If nb = 0, uses the same nb and n as this matrix;
///     otherwise, n = nb * nt.
///
/// @param[in] deepOp
///     Additional deep-transposition operation to apply. If deepOp=Trans, the
///     new matrix has the transposed structure (distribution and number of
///     tiles) of this matrix, but its shallow-transpose op() flag is set to
///     NoTrans. For a 1x4 matrix A, compare:
///     - transpose(A).emptyLike() creates a new 1x4 matrix, then transposes it
///       to return a 4x1 matrix with its op set to Trans.
///     - A.emptyLike(mb, nb, Op::Trans) creates and returns a new 4x1 matrix
///       with its op set to NoTrans.
///
template <typename scalar_t>
template <typename out_scalar_t>
Matrix<out_scalar_t> Matrix<scalar_t>::emptyLike(
    BaseMatrix<scalar_t>& orig, int64_t mb, int64_t nb, Op deepOp)
{
    auto B = orig.template baseEmptyLike<out_scalar_t>(mb, nb, deepOp);
    return Matrix<out_scalar_t>(B, 0, B.mt()-1, 0, B.nt()-1);
}

//------------------------------------------------------------------------------
/// [internal]
/// @see fromLAPACK
/// @see fromScaLAPACK
///
/// @param[in] is_scalapack
///     If true,  A is a ScaLAPACK matrix.
///     If false, A is an LAPACK matrix.
///
template <typename scalar_t>
Matrix<scalar_t>::Matrix(
    int64_t m, int64_t n,
    scalar_t* A, int64_t lda, int64_t mb, int64_t nb,
    GridOrder order, int p, int q, MPI_Comm mpi_comm, bool is_scalapack)
    : BaseMatrix<scalar_t>( m, n, mb, nb, order, p, q, mpi_comm )
{
    // ii, jj are row, col indices
    // ii_local and jj_local are the local array indices in A
    // block-cyclic layout (indxg2l)
    // i, j are tile (block row, block col) indices
    int64_t jj = 0;
    for (int64_t j = 0; j < this->nt(); ++j) {
        int64_t jb = this->tileNb(j);
        int64_t jj_local = jj;
        if (is_scalapack) {
            jj_local = indexGlobal2Local(jj, nb, q);
        }

        int64_t ii = 0;
        for (int64_t i = 0; i < this->mt(); ++i) {
            int64_t ib = this->tileMb(i);
            if (this->tileIsLocal(i, j)) {
                int64_t ii_local = ii;
                if (is_scalapack) {
                    ii_local = indexGlobal2Local(ii, mb, p);
                }

                this->tileInsert( i, j, HostNum,
                                  &A[ ii_local + jj_local*lda ], lda );
            }
            ii += ib;
        }
        jj += jb;
    }
}

//------------------------------------------------------------------------------
/// [internal]
/// @see fromDevices
///
template <typename scalar_t>
Matrix<scalar_t>::Matrix(
    int64_t m, int64_t n,
    scalar_t** Aarray, int num_devices, int64_t lda,
    int64_t mb, int64_t nb, int p, int q, MPI_Comm mpi_comm)
    : BaseMatrix<scalar_t>(m, n, mb, nb, p, q, mpi_comm)
{
    slate_error_if(this->num_devices() != num_devices);

    // ii, jj are row, col indices
    // ii_local and jj_local are the local array indices in A
    // 2D block-cyclic layout.
    // jj_dev is the local array index for the current device in A
    // 1D block-cyclic layout within a node.
    // i, j are tile (block row, block col) indices
    int64_t jj = 0;
    for (int64_t j = 0; j < this->nt(); ++j) {
        int64_t jb = this->tileNb(j);
        int64_t jj_local = indexGlobal2Local(jj, nb, q);
        int64_t ii = 0;
        for (int64_t i = 0; i < this->mt(); ++i) {
            int64_t ib = this->tileMb(i);
            if (this->tileIsLocal(i, j)) {
                int64_t ii_local = indexGlobal2Local(ii, mb, p);
                int dev = this->tileDevice(i, j);
                int64_t jj_dev = indexGlobal2Local(jj_local, nb, num_devices);
                this->tileInsert(i, j, dev,
                                 &Aarray[ dev ][ ii_local + jj_dev*lda ], lda);
            }
            ii += ib;
        }
        jj += jb;
    }
}

//------------------------------------------------------------------------------
/// Sub-matrix constructor creates shallow copy view of parent matrix,
/// A[ i1:i2, j1:j2 ].
/// This is called from Matrix::sub(i1, i2, j1, j2) and
/// off-diagonal sub(i1, i2, j1, j2) of
/// TriangularMatrix, SymmetricMatrix, HermitianMatrix, etc.
///
/// @param[in] orig
///     Original matrix of which to make sub-matrix.
///
/// @param[in] i1
///     Starting block-row index. 0 <= i1 < mt.
///
/// @param[in] i2
///     Ending block-row index (inclusive). i2 < mt.
///
/// @param[in] j1
///     Starting block-column index. 0 <= j1 < nt.
///
/// @param[in] j2
///     Ending block-column index (inclusive). j2 < nt.
///
template <typename scalar_t>
Matrix<scalar_t>::Matrix(
    BaseMatrix< scalar_t >& orig,
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
    : BaseMatrix<scalar_t>(orig, i1, i2, j1, j2)
{
    this->uplo_ = Uplo::General;
}

//------------------------------------------------------------------------------
/// Returns sub-matrix that is a shallow copy view of the
/// parent matrix, A[ i1:i2, j1:j2 ].
///
/// @param[in] i1
///     Starting block-row index. 0 <= i1 < mt.
///
/// @param[in] i2
///     Ending block-row index (inclusive). i2 < mt.
///
/// @param[in] j1
///     Starting block-column index. 0 <= j1 < nt.
///
/// @param[in] j2
///     Ending block-column index (inclusive). j2 < nt.
///
template <typename scalar_t>
Matrix<scalar_t> Matrix<scalar_t>::sub(
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
{
    return Matrix<scalar_t>(*this, i1, i2, j1, j2);
}

//------------------------------------------------------------------------------
/// Sliced matrix constructor creates shallow copy view of parent matrix,
/// A[ row1:row2, col1:col2 ].
/// This takes row & col indices instead of block row & block col indices.
///
/// @param[in] orig
///     Original matrix of which to make sub-matrix.
///
/// @param[in] slice
///     Contains start and end row and column indices.
///
template <typename scalar_t>
Matrix<scalar_t>::Matrix(
    BaseMatrix<scalar_t>& orig, typename BaseMatrix<scalar_t>::Slice slice)
    : BaseMatrix<scalar_t>(orig, slice)
{
    this->uplo_ = Uplo::General;
}

//------------------------------------------------------------------------------
/// Returns sliced matrix that is a shallow copy view of the
/// parent matrix, A[ row1:row2, col1:col2 ].
/// This takes row & col indices instead of block row & block col indices.
///
/// @param[in] row1
///     Starting row index. 0 <= row1 < m.
///
/// @param[in] row2
///     Ending row index (inclusive). row2 < m.
///
/// @param[in] col1
///     Starting column index. 0 <= col1 < n.
///
/// @param[in] col2
///     Ending column index (inclusive). col2 < n.
///
template <typename scalar_t>
Matrix<scalar_t> Matrix<scalar_t>::slice(
    int64_t row1, int64_t row2,
    int64_t col1, int64_t col2)
{
    return Matrix<scalar_t>(*this,
        typename BaseMatrix<scalar_t>::Slice(row1, row2, col1, col2));
}

//------------------------------------------------------------------------------
/// Swap contents of matrices A and B.
//
// (This isn't really needed over BaseMatrix swap, but is here as a reminder
// in case any members are added to Matrix that aren't in BaseMatrix.)
template <typename scalar_t>
void swap(Matrix<scalar_t>& A, Matrix<scalar_t>& B)
{
    using std::swap;
    swap(static_cast< BaseMatrix<scalar_t>& >(A),
         static_cast< BaseMatrix<scalar_t>& >(B));
}

//------------------------------------------------------------------------------
/// Returns number of local tiles of the matrix on this rank.
//
// todo: numLocalTiles? use for life as well?
template <typename scalar_t>
int64_t Matrix<scalar_t>::getMaxHostTiles()
{
    int64_t num_tiles = 0;
    for (int64_t j = 0; j < this->nt(); ++j)
        for (int64_t i = 0; i < this->mt(); ++i)
            if (this->tileIsLocal(i, j))
                ++num_tiles;

    return num_tiles;
}

//------------------------------------------------------------------------------
/// Returns number of local tiles of the matrix on this rank and given device.
//
// todo: numLocalDeviceTiles?
template <typename scalar_t>
int64_t Matrix<scalar_t>::getMaxDeviceTiles(int device)
{
    int64_t num_tiles = 0;
    for (int64_t j = 0; j < this->nt(); ++j)
        for (int64_t i = 0; i < this->mt(); ++i)
            if (this->tileIsLocal(i, j) && this->tileDevice(i, j) == device)
                ++num_tiles;

    return num_tiles;
}

//------------------------------------------------------------------------------
/// Allocates batch arrays and BLAS++ queues for all devices.
/// This overrides BaseMatrix::allocateBatchArrays
/// to use the number of local tiles in a general matrix.
///
/// @param[in] batch_size
///     Allocate batch arrays as needed so that
///     size of each batch array >= batch_size >= 0.
///     If batch_size = 0 (default), uses batch_size = getMaxDeviceTiles.
///
/// @param[in] num_arrays
///     Allocate batch arrays as needed so that
///     number of batch arrays per device >= num_arrays >= 1.
///
template <typename scalar_t>
void Matrix<scalar_t>::allocateBatchArrays(
    int64_t batch_size, int64_t num_arrays)
{
    if (batch_size == 0) {
        for (int device = 0; device < this->num_devices_; ++device)
            batch_size = std::max(batch_size, getMaxDeviceTiles(device));
    }
    this->storage_->allocateBatchArrays(batch_size, num_arrays);
}

//------------------------------------------------------------------------------
/// Reserve space for temporary workspace tiles on host.
template <typename scalar_t>
void Matrix<scalar_t>::reserveHostWorkspace()
{
    this->storage_->reserveHostWorkspace(getMaxHostTiles());
}

//------------------------------------------------------------------------------
/// Reserve space for temporary workspace tiles on all GPU devices.
template <typename scalar_t>
void Matrix<scalar_t>::reserveDeviceWorkspace()
{
    int64_t num_tiles = 0;
    for (int device = 0; device < this->num_devices_; ++device)
        num_tiles = std::max(num_tiles, getMaxDeviceTiles(device));
    this->storage_->reserveDeviceWorkspace(num_tiles);
}

//------------------------------------------------------------------------------
/// Gathers the entire matrix to the LAPACK-style matrix A on MPI rank 0.
/// Primarily for debugging purposes.
///
template <typename scalar_t>
void Matrix<scalar_t>::gather(scalar_t* A, int64_t lda)
{
    // this code assumes the matrix is not transposed
    Op op_save = this->op();
    this->op_ = Op::NoTrans;

    // ii, jj are row, col indices
    // i, j are tile (block row, block col) indices
    int64_t jj = 0;
    for (int64_t j = 0; j < this->nt(); ++j) {
        int64_t jb = this->tileNb(j);

        int64_t ii = 0;
        for (int64_t i = 0; i < this->mt(); ++i) {
            int64_t ib = this->tileMb(i);

            if (this->mpi_rank_ == 0) {
                if (! this->tileIsLocal(i, j)) {
                    // erase any existing non-local tile and insert new one
                    this->tileErase( i, j, HostNum );
                    this->tileInsert( i, j, HostNum,
                                      &A[(size_t)lda*jj + ii], lda );
                    auto Aij = this->at(i, j);
                    Aij.recv(this->tileRank(i, j), this->mpi_comm_, this->layout());
                    this->tileLayout(i, j, this->layout_);
                }
                else {
                    this->tileGetForReading(i, j, LayoutConvert(this->layout()));
                    // copy local tiles if needed.
                    auto Aij = this->at(i, j);
                    if (Aij.data() != &A[(size_t)lda*jj + ii]) {
                        lapack::lacpy(lapack::MatrixType::General, ib, jb,
                                      Aij.data(), Aij.stride(),
                                      &A[(size_t)lda*jj + ii], lda);
                    }
                }
            }
            else if (this->tileIsLocal(i, j)) {
                this->tileGetForReading(i, j, LayoutConvert(this->layout()));
                auto Aij = this->at(i, j);
                Aij.send(0, this->mpi_comm_);
            }
            ii += ib;
        }
        jj += jb;
    }

    this->op_ = op_save;
}

//------------------------------------------------------------------------------
/// Inserts all local tiles into an empty matrix.
///
/// @param[in] target
///     - if target = Devices, inserts tiles on appropriate GPU devices, or
///     - if target = Host,    inserts tiles on CPU host.
///
template <typename scalar_t>
void Matrix<scalar_t>::insertLocalTiles(Target origin)
{
    bool on_devices = (origin == Target::Devices);
    if (on_devices)
        reserveDeviceWorkspace();

    for (int64_t j = 0; j < this->nt(); ++j) {
        for (int64_t i = 0; i < this->mt(); ++i) {
            if (this->tileIsLocal(i, j)) {
                int dev = (on_devices ? this->tileDevice(i, j)
                                      : HostNum);
                this->tileInsert(i, j, dev);
            }
        }
    }
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void Matrix<scalar_t>::redistribute(Matrix<scalar_t>& A)
{
    int64_t mt = this->mt();
    int64_t nt = this->nt();

    for (int64_t j = 0; j < nt; ++j) {
        for (int64_t i = 0; i < mt; ++i) {
            if (this->tileIsLocal(i, j)) {
                if (! A.tileIsLocal(i, j)) {
                    auto Bij = this->at(i, j);
                    Bij.recv(A.tileRank(i, j), A.mpiComm(),  A.layout());
                }
                else {
                    A.tileGetForReading(i, j, LayoutConvert::None);
                    // copy local tiles if needed.
                    auto Aij = A(i, j);
                    auto Bij = this->at(i, j);
                    if (Aij.data() != Bij.data() ) {
                        tile::gecopy( Aij, Bij );
                    }
                }
            }
            else if (A.tileIsLocal(i, j)) {
                A.tileGetForReading(i, j, LayoutConvert::None);
                auto Aij = A(i, j);
                Aij.send(this->tileRank(i, j), this->mpiComm());
            }
        }
    }
}


} // namespace slate

#endif // SLATE_MATRIX_HH
