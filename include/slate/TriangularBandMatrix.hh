// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_TRIANGULAR_BAND_MATRIX_HH
#define SLATE_TRIANGULAR_BAND_MATRIX_HH

#include "slate/BaseTriangularBandMatrix.hh"
#include "slate/BandMatrix.hh"
#include "slate/TriangularMatrix.hh"
#include "slate/Tile.hh"
#include "slate/types.hh"

#include "lapack.hh"

#include <algorithm>
#include <utility>

#include "slate/internal/mpi.hh"
#include "slate/internal/openmp.hh"

namespace slate {

//==============================================================================
/// Triangular banded, n-by-n, distributed, tiled matrices.
template <typename scalar_t>
class TriangularBandMatrix: public BaseTriangularBandMatrix<scalar_t> {
public:
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    // constructors
    TriangularBandMatrix();

    TriangularBandMatrix(Uplo uplo, Diag diag, int64_t n, int64_t kd,
                         std::function<int64_t (int64_t j)>& inTileNb,
                         std::function<int (ij_tuple ij)>& inTileRank,
                         std::function<int (ij_tuple ij)>& inTileDevice,
                         MPI_Comm mpi_comm);

    TriangularBandMatrix(
        Uplo uplo, Diag diag,
        int64_t n, int64_t kd,
        int64_t nb, int p, int q, MPI_Comm mpi_comm);

    // conversion
    TriangularBandMatrix(Uplo uplo, Diag diag, BandMatrix<scalar_t>& orig);
    TriangularBandMatrix(Diag diag, BaseTriangularBandMatrix<scalar_t>& orig);

    TriangularMatrix<scalar_t> sub(int64_t i1, int64_t i2);

    // sub-matrix
    Matrix<scalar_t> sub(int64_t i1, int64_t i2,
                         int64_t j1, int64_t j2);
    Matrix<scalar_t> slice(int64_t row1, int64_t row2,
                           int64_t col1, int64_t col2);

public:
    template <typename T>
    friend void swap(TriangularBandMatrix<T>& A, TriangularBandMatrix<T>& B);

    void    gatherAll(std::set<int>& rank_set, int tag = 0, int64_t life_factor = 1);
    void    ge2tbGather(Matrix<scalar_t>& A);

    Diag diag() { return diag_; }
    void diag(Diag in_diag) { diag_ = in_diag; }

protected:
    Diag diag_;
};

//------------------------------------------------------------------------------
/// Default constructor creates an empty band matrix with bandwidth = 0.
template <typename scalar_t>
TriangularBandMatrix<scalar_t>::TriangularBandMatrix()
    : BaseTriangularBandMatrix<scalar_t>(),
      diag_(Diag::NonUnit)
{}

//------------------------------------------------------------------------------
/// Constructor creates an n-by-n matrix, with no tiles allocated,
/// where tileNb, tileRank, tileDevice are given as functions.
/// Tiles can be added with tileInsert().
///
template <typename scalar_t>
TriangularBandMatrix<scalar_t>::TriangularBandMatrix(
    Uplo uplo, Diag diag, int64_t n, int64_t kd,
    std::function<int64_t (int64_t j)>& inTileNb,
    std::function<int (ij_tuple ij)>& inTileRank,
    std::function<int (ij_tuple ij)>& inTileDevice,
    MPI_Comm mpi_comm)
    : BaseTriangularBandMatrix<scalar_t>(uplo, n, kd, inTileNb, inTileRank,
                                inTileDevice, mpi_comm),
      diag_(diag)
{}

//------------------------------------------------------------------------------
/// Constructor creates an n-by-n band matrix, with no tiles allocated,
/// with fixed nb-by-nb tile size and 2D block cyclic distribution.
/// Tiles can be added with tileInsert().
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
/// @param[in] kd
///     Number of sub (if lower) or super (if upper) diagonals within band.
///     kd >= 0.
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
TriangularBandMatrix<scalar_t>::TriangularBandMatrix(
    Uplo uplo, Diag diag,
    int64_t n, int64_t kd, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
    : BaseTriangularBandMatrix<scalar_t>(uplo, n, kd, nb, p, q, mpi_comm),
      diag_(diag)
{}

//------------------------------------------------------------------------------
/// [explicit]
/// todo:
/// Conversion from general band matrix
/// creates a shallow copy view of the original matrix.
/// Uses only square portion, Aorig[ 0:min(mt,nt)-1, 0:min(mt,nt)-1 ].
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
TriangularBandMatrix<scalar_t>::TriangularBandMatrix(
    Uplo uplo, Diag diag, BandMatrix<scalar_t>& orig)
    : BaseTriangularBandMatrix<scalar_t>(uplo, orig),
      diag_(diag)
{}

//------------------------------------------------------------------------------
/// [explicit]
/// todo:
/// Conversion from base triangular band matrix
/// creates a shallow copy view of the original matrix.
/// Uses only square portion, Aorig[ 0:min(mt,nt)-1, 0:min(mt,nt)-1 ].
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
TriangularBandMatrix<scalar_t>::TriangularBandMatrix(
    Diag diag, BaseTriangularBandMatrix<scalar_t>& orig)
    : BaseTriangularBandMatrix<scalar_t>(orig),
      diag_(diag)
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
TriangularMatrix<scalar_t> TriangularBandMatrix<scalar_t>::sub(
    int64_t i1, int64_t i2)
{
    return TriangularMatrix<scalar_t>(*this, i1, i2);
}

//------------------------------------------------------------------------------
/// Returns sub-matrix that is a shallow copy view of the
/// parent matrix, A[ i1:i2, j1:j2 ].
/// This version returns a Matrix.
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
Matrix<scalar_t> TriangularBandMatrix<scalar_t>::sub(
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
{
    // todo: assert that sub-matrix falls within upper/lower band
    return Matrix<scalar_t>(*this, i1, i2, j1, j2);
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
Matrix<scalar_t> TriangularBandMatrix<scalar_t>::slice(
    int64_t row1, int64_t row2,
    int64_t col1, int64_t col2)
{
    // todo: assert that sub-matrix falls within upper/lower band
    return Matrix<scalar_t>(*this,
        typename BaseMatrix<scalar_t>::Slice(row1, row2, col1, col2));
}

//------------------------------------------------------------------------------
/// Swap contents of band matrices A and B.
template <typename scalar_t>
void swap(TriangularBandMatrix<scalar_t>& A, TriangularBandMatrix<scalar_t>& B)
{
    using std::swap;
    swap(static_cast< BaseTriangularBandMatrix<scalar_t>& >(A),
         static_cast< BaseTriangularBandMatrix<scalar_t>& >(B));
    swap(A.diag_, B.diag_);
}

//------------------------------------------------------------------------------
/// Gather all tiles on each rank
// WARNING: this is a demanding process storage and communication wise,
// avoid if possible.
//
template <typename scalar_t>
void TriangularBandMatrix<scalar_t>::gatherAll(std::set<int>& rank_set, int tag, int64_t life_factor)
{
    trace::Block trace_block("slate::gatherAll");

    auto upper = this->uplo() == Uplo::Upper;
    auto layout = this->layout(); // todo: is this correct?

    // If this rank is not in the set.
    if (rank_set.find(this->mpiRank()) == rank_set.end())
        return;

    int64_t mt = this->mt();
    int64_t nt = this->nt();
    int64_t kdt = ceildiv( this->bandwidth(), this->tileNb(0) );
    for (int64_t j = 0; j < nt; ++j) {
        int64_t istart = upper ? blas::max( 0, j-kdt ) : j;
        int64_t iend   = upper ? j : blas::min( j+kdt, mt-1 );
        for (int64_t i = istart; i <= iend; ++i) {

            // If receiving the tile.
            if (! this->tileIsLocal(i, j)) {
                // Create tile to receive data, with life span.
                // If tile already exists, add to its life span.
                LockGuard guard(this->storage_->getTilesMapLock()); // todo: accessor
                auto iter = this->storage_->find( this->globalIndex( i, j, HostNum ) );

                int64_t life = life_factor;
                if (iter == this->storage_->end())
                    this->tileInsertWorkspace( i, j, HostNum );
                else
                    life += this->tileLife(i, j); // todo: use temp tile to receive
                this->tileLife(i, j, life);
            }

            // Send across MPI ranks.
            // Previous used MPI bcast: tileBcastToSet(i, j, rank_set);
            // Currently uses 2D hypercube p2p send.
            this->tileBcastToSet(i, j, rank_set, 2, tag, layout, Target::HostTask);
        }
    }
}

//------------------------------------------------------------------------------
/// Gather the distributed triangular band portion of a general Matrix A
/// to TriangularBandMatrix B on MPI rank 0.
/// Primarily for SVD code
///
// todo: parameter for rank to collect on, default 0
template <typename scalar_t>
void TriangularBandMatrix<scalar_t>::ge2tbGather(Matrix<scalar_t>& A)
{
    Op op_save = this->op();
    this->op_ = Op::NoTrans;
    auto upper = this->uplo() == Uplo::Upper;

    int64_t mt = A.mt();
    int64_t nt = A.nt();
    int64_t kdt = ceildiv( this->bandwidth(), this->tileNb(0) );
    // i, j are tile (block row, block col) indices
    for (int64_t j = 0; j < nt; ++j) {

        int64_t istart = upper ? blas::max( 0, j-kdt ) : j;
        int64_t iend   = upper ? j : blas::min( j+kdt, mt-1 );
        for (int64_t i = 0; i < mt; ++i) {
            if (i >= istart && i <= iend) {
                if (this->mpi_rank_ == 0) {
                    if (! A.tileIsLocal(i, j)) {
                        this->tileInsert( i, j, HostNum );
                        auto Bij = this->at(i, j);
                        Bij.recv(A.tileRank(i, j), this->mpi_comm_, this->layout());
                    }
                    else {
                        A.tileGetForReading(i, j, LayoutConvert(this->layout()));
                        // copy local tiles if needed.
                        auto Aij = A(i, j);
                        auto Bij = this->at(i, j);
                        if (Aij.data() != Bij.data() ) {
                            tile::gecopy( A(i, j), Bij );
                        }
                    }
                }
                else if (A.tileIsLocal(i, j)) {
                    A.tileGetForReading(i, j, LayoutConvert(this->layout()));
                    auto Aij = A(i, j);
                    Aij.send(0, this->mpi_comm_);
                }
            }
        }
    }

    this->op_ = op_save;
}


} // namespace slate

#endif // SLATE_TRIANGULAR_BAND_MATRIX_HH
