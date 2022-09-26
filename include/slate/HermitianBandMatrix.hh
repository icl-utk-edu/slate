// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_HERMITIAN_BAND_MATRIX_HH
#define SLATE_HERMITIAN_BAND_MATRIX_HH

#include "slate/BaseTriangularBandMatrix.hh"
#include "slate/BandMatrix.hh"
#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/Tile.hh"
#include "slate/types.hh"

#include "lapack.hh"

#include <algorithm>
#include <utility>

#include "slate/internal/mpi.hh"
#include "slate/internal/openmp.hh"

namespace slate {

//==============================================================================
/// Hermitian banded, n-by-n, distributed, tiled matrices.
template <typename scalar_t>
class HermitianBandMatrix: public BaseTriangularBandMatrix<scalar_t> {
public:
    // constructors
    HermitianBandMatrix();

    HermitianBandMatrix(
        Uplo uplo,
        int64_t n, int64_t kd,
        int64_t nb, int p, int q, MPI_Comm mpi_comm);

    // conversion
    HermitianBandMatrix(Uplo uplo, BandMatrix<scalar_t>& orig);
    HermitianBandMatrix(int64_t kd, HermitianMatrix<scalar_t>& orig);

    // on-diagonal sub-matrix
    HermitianMatrix<scalar_t> sub(int64_t i1, int64_t i2);
    HermitianMatrix<scalar_t> slice(int64_t index1, int64_t index2);


    // sub-matrix
    Matrix<scalar_t> sub(int64_t i1, int64_t i2,
                         int64_t j1, int64_t j2);
    Matrix<scalar_t> slice(int64_t row1, int64_t row2,
                           int64_t col1, int64_t col2);

public:
    template <typename T>
    friend void swap(HermitianBandMatrix<T>& A, HermitianBandMatrix<T>& B);

    void    gatherAll(std::set<int>& rank_set, int tag = 0, int64_t life_factor = 1);
    void    he2hbGather(HermitianMatrix<scalar_t>& A);
};

//------------------------------------------------------------------------------
/// Default constructor creates an empty Hermitian band matrix with bandwidth = 0.
template <typename scalar_t>
HermitianBandMatrix<scalar_t>::HermitianBandMatrix()
    : BaseTriangularBandMatrix<scalar_t>()
{}

//------------------------------------------------------------------------------
/// Constructor creates an n-by-n Hermitian band matrix, with no tiles allocated.
/// Tiles can be added with tileInsert().
///
/// @param[in] uplo
///     - Upper: upper triangle of A is stored.
///     - Lower: lower triangle of A is stored.
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
HermitianBandMatrix<scalar_t>::HermitianBandMatrix(
    Uplo uplo,
    int64_t n, int64_t kd, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
    : BaseTriangularBandMatrix<scalar_t>(uplo, n, kd, nb, p, q, mpi_comm)
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
/// @param[in,out] orig
///     Original matrix.
///
template <typename scalar_t>
HermitianBandMatrix<scalar_t>::HermitianBandMatrix(
    Uplo uplo, BandMatrix<scalar_t>& orig)
    : BaseTriangularBandMatrix<scalar_t>(uplo, orig)
{}

//------------------------------------------------------------------------------
/// [explicit]
/// todo:
/// Conversion from Hermitian matrix
/// creates a shallow copy view of the original matrix.
///
/// @param[in] kd
///     Number of sub (if lower) or super (if upper) diagonals within band.
///     kd >= 0.
///
/// @param[in,out] orig
///     Original matrix.
///
template <typename scalar_t>
HermitianBandMatrix<scalar_t>::HermitianBandMatrix(
    int64_t kd, HermitianMatrix<scalar_t>& orig)
    : BaseTriangularBandMatrix<scalar_t>(kd, orig)
{}

//------------------------------------------------------------------------------
/// Returns sub-matrix that is a shallow copy view of the
/// parent matrix, A[ i1:i2, i1:i2 ].
/// This version returns a HermitianMatrix with the same diagonal as the
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
HermitianMatrix<scalar_t> HermitianBandMatrix<scalar_t>::sub(
    int64_t i1, int64_t i2)
{
    // return HermitianMatrix<scalar_t>(this->uplo(), *this, i1, i2);
    return HermitianMatrix<scalar_t>(this->uploPhysical(), *this, i1, i2);
}

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
// todo: should check indices within band
template <typename scalar_t>
HermitianMatrix<scalar_t> HermitianBandMatrix<scalar_t>::slice(
    int64_t index1, int64_t index2)
{
    return HermitianMatrix<scalar_t>(this->uplo_, *this,
        typename BaseMatrix<scalar_t>::Slice(index1, index2, index1, index2));
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
Matrix<scalar_t> HermitianBandMatrix<scalar_t>::sub(
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
{

    if (this->uplo() == Uplo::Lower) {
        // require top-right corner (i1, j2) to be at or below diagonal
        if (i1 < j2)
            slate_error("submatrix outside lower triangle; requires i1 >= j2");
    }
    else {
        // require bottom-left corner (i2, j1) to be at or above diagonal
        if (i2 > j1)
            slate_error("submatrix outside upper triangle; requires i2 <= j1");
    }
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
Matrix<scalar_t> HermitianBandMatrix<scalar_t>::slice(
    int64_t row1, int64_t row2,
    int64_t col1, int64_t col2)
{
    return Matrix<scalar_t>(*this,
        typename BaseMatrix<scalar_t>::Slice(row1, row2, col1, col2));
}

//------------------------------------------------------------------------------
/// Swap contents of Hermitian band matrices A and B.
template <typename scalar_t>
void swap(HermitianBandMatrix<scalar_t>& A, HermitianBandMatrix<scalar_t>& B)
{
    using std::swap;
    swap(static_cast< BaseTriangularBandMatrix<scalar_t>& >(A),
         static_cast< BaseTriangularBandMatrix<scalar_t>& >(B));
}

//------------------------------------------------------------------------------
/// Gather all tiles on each rank
// WARNING: this is a demanding process storage and communication wise,
// avoid if possible.
//
template <typename scalar_t>
void HermitianBandMatrix<scalar_t>::gatherAll(std::set<int>& rank_set, int tag, int64_t life_factor)
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
/// Gather the distributed triangular band portion of a HermitianMatrix A
//  to HermitianBandMatrix B on MPI rank 0.
/// Primarily for EVD code
///
template <typename scalar_t>
void HermitianBandMatrix<scalar_t>::he2hbGather(HermitianMatrix<scalar_t>& A)
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
                        // TODO add: this->tileGetForWriting(i, j, LayoutConvert(this->layout()));
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

#endif // SLATE_HERMITIAN_BAND_MATRIX_HH
