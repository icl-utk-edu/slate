// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_BAND_MATRIX_HH
#define SLATE_BAND_MATRIX_HH

#include "slate/BaseBandMatrix.hh"
#include "slate/Tile.hh"
#include "slate/types.hh"

#include "lapack.hh"

#include <algorithm>
#include <utility>

#include "slate/internal/mpi.hh"
#include "slate/internal/openmp.hh"

namespace slate {

//==============================================================================
/// General banded, non-symmetric, m-by-n, distributed, tiled matrices.
template <typename scalar_t>
class BandMatrix: public BaseBandMatrix<scalar_t> {
public:
    using ij_tuple = std::tuple<int64_t, int64_t>;

    // constructors
    BandMatrix();

    BandMatrix(int64_t m, int64_t n, int64_t kl, int64_t ku,
               std::function<int64_t (int64_t j)>& inTileNb,
               std::function<int (ij_tuple ij)>& inTileRank,
               std::function<int (ij_tuple ij)>& inTileDevice,
               MPI_Comm mpi_comm);

    BandMatrix(int64_t m, int64_t n, int64_t kl, int64_t ku,
               int64_t nb, int p, int q, MPI_Comm mpi_comm);

    BandMatrix(int64_t kl, int64_t ku, Matrix<scalar_t>& orig);

    BandMatrix<scalar_t> slice(
        int64_t row1, int64_t row2,
        int64_t col1, int64_t col2);

    template <typename out_scalar_t=scalar_t>
    static
    BandMatrix<out_scalar_t> emptyLike(BaseMatrix<scalar_t>& orig, int64_t kl,
                                       int64_t ku, int64_t mb=0, int64_t nb=0,
                                       Op deepOp=Op::NoTrans);

protected:
    // used by slice
    BandMatrix(BaseBandMatrix<scalar_t>& orig,
           typename BaseMatrix<scalar_t>::Slice slice);

public:
    template <typename T>
    friend void swap(BandMatrix<T>& A, BandMatrix<T>& B);

    int64_t lowerBandwidth() const;
    void    lowerBandwidth(int64_t kl);

    int64_t upperBandwidth() const;
    void    upperBandwidth(int64_t ku);

    void insertLocalTiles(Target origin=Target::Host);
};

//------------------------------------------------------------------------------
/// Default constructor creates an empty band matrix with bandwidth = 0.
template <typename scalar_t>
BandMatrix<scalar_t>::BandMatrix():
    BaseBandMatrix<scalar_t>()
{}

//------------------------------------------------------------------------------
/// Constructor creates an m-by-n band matrix, with no tiles allocated,
/// where tileMb, tileNb, tileRank, tileDevice are given as functions.
/// Tiles can be added with tileInsert().
//
/// @param[in] m
///     Number of rows of the matrix. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix. n >= 0.
///
/// @param[in] kl
///     Number of subdiagonals within band. kl >= 0.
///
/// @param[in] ku
///     Number of superdiagonals within band. ku >= 0.
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
BandMatrix<scalar_t>::BandMatrix(int64_t m, int64_t n, int64_t kl, int64_t ku,
    std::function<int64_t (int64_t j)>& inTileNb,
    std::function<int (ij_tuple ij)>& inTileRank,
    std::function<int (ij_tuple ij)>& inTileDevice,
    MPI_Comm mpi_comm)
    : BaseBandMatrix<scalar_t>(m, n, kl, ku, inTileNb, inTileRank, inTileDevice,
    mpi_comm)
{}

//------------------------------------------------------------------------------
/// Constructor creates an m-by-n band matrix, with no tiles allocated,
/// with fixed nb-by-nb tile size and 2D block cyclic distribution.
/// Tiles can be added with tileInsert().
///
/// @param[in] m
///     Number of rows of the matrix. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix. n >= 0.
///
/// @param[in] kl
///     Number of subdiagonals within band. kl >= 0.
///
/// @param[in] ku
///     Number of superdiagonals within band. ku >= 0.
///
/// @param[in] nb
///     Block size in 2D block-cyclic distribution.
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
BandMatrix<scalar_t>::BandMatrix(
    int64_t m, int64_t n, int64_t kl, int64_t ku, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
    : BaseBandMatrix<scalar_t>(m, n, kl, ku, nb, p, q, mpi_comm)
{}

//------------------------------------------------------------------------------
/// Conversion from general Matrix.
/// Creates a shallow copy view of the band region [kl, ku] of the original matrix.
///
/// @param[in] kl
///     Lower bandwidth.
///
/// @param[in] ku
///     Upper bandwidth.
///
/// @param[in] orig
///     Original matrix.
///
template <typename scalar_t>
BandMatrix<scalar_t>::BandMatrix(
    int64_t kl, int64_t ku, Matrix<scalar_t>& orig)
    : BaseBandMatrix<scalar_t>(kl, ku, orig)
{}

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
BandMatrix<scalar_t>::BandMatrix(
    BaseBandMatrix<scalar_t>& orig, typename BaseMatrix<scalar_t>::Slice slice)
    : BaseBandMatrix<scalar_t>(orig, slice)
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
BandMatrix<scalar_t> BandMatrix<scalar_t>::slice(
    int64_t row1, int64_t row2,
    int64_t col1, int64_t col2)
{
    return BandMatrix<scalar_t>(*this,
        typename BaseMatrix<scalar_t>::Slice(row1, row2, col1, col2));
}

//------------------------------------------------------------------------------
/// Named constructor returns a new, empty Matrix with the same structure
/// (size and distribution) as the matrix orig. Tiles are not allocated.
///
/// @param[in] orig
///     Original matrix of which to make an empty matrix with the same structure
///     (size and distribution) as this original matrix.
///
/// @param[in] kl
///     Number of subdiagonals within band. kl >= 0.
///
/// @param[in] ku
///     Number of superdiagonals within band. ku >= 0.
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
BandMatrix<out_scalar_t> BandMatrix<scalar_t>::emptyLike(
    BaseMatrix<scalar_t>& orig, int64_t kl, int64_t ku, int64_t mb, int64_t nb,
    Op deepOp)
{
    auto B = orig.template baseEmptyLike<out_scalar_t>(mb, nb, deepOp);
    auto M = Matrix<out_scalar_t>(B, 0, B.mt()-1, 0, B.nt()-1);
    return BandMatrix<out_scalar_t>(kl, ku, M);
}

//------------------------------------------------------------------------------
/// Swap contents of band matrices A and B.
template <typename scalar_t>
void swap(BandMatrix<scalar_t>& A, BandMatrix<scalar_t>& B)
{
    using std::swap;
    swap(static_cast< BaseBandMatrix<scalar_t>& >(A),
         static_cast< BaseBandMatrix<scalar_t>& >(B));
}

//------------------------------------------------------------------------------
/// @return number of subdiagonals within band.
template <typename scalar_t>
int64_t BandMatrix<scalar_t>::lowerBandwidth() const
{
    return (this->op() == Op::NoTrans ? this->kl_ : this->ku_);
}

//------------------------------------------------------------------------------
/// Sets number of subdiagonals within band.
template <typename scalar_t>
void BandMatrix<scalar_t>::lowerBandwidth(int64_t kl)
{
    if (this->op() == Op::NoTrans)
        this->kl_ = kl;
    else
        this->ku_ = kl;
}

//------------------------------------------------------------------------------
/// @return number of superdiagonals within band.
template <typename scalar_t>
int64_t BandMatrix<scalar_t>::upperBandwidth() const
{
    return (this->op() == Op::NoTrans ? this->ku_ : this->kl_);
}

//------------------------------------------------------------------------------
/// Sets number of superdiagonals within band.
template <typename scalar_t>
void BandMatrix<scalar_t>::upperBandwidth(int64_t ku)
{
    if (this->op() == Op::NoTrans)
        this->ku_ = ku;
    else
        this->kl_ = ku;
}

//------------------------------------------------------------------------------
/// Inserts all local tiles into an empty band matrix.
///
/// @param[in] target
///     - if target = Devices, inserts tiles on appropriate GPU devices, or
///     - if target = Host, inserts on tiles on CPU host.
///
template <typename scalar_t>
void BandMatrix<scalar_t>::insertLocalTiles(Target origin)
{
    bool on_devices = (origin == Target::Devices);
    int64_t mt = this->mt();
    int64_t nt = this->nt();
    for (int64_t j = 0; j < nt; ++j) {
        for (int64_t i = 0; i < mt; ++i) {
            if (this->tileIsLocal(i, j) && this->tileIsInBand(i, j)) {
                int dev = (on_devices ? this->tileDevice(i, j)
                                      : HostNum);
                this->tileInsert(i, j, dev);
            }
        }
    }
}

} // namespace slate

#endif // SLATE_BAND_MATRIX_HH
