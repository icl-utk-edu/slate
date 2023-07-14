// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_BASE_TRIANGULAR_BAND_MATRIX_HH
#define SLATE_BASE_TRIANGULAR_BAND_MATRIX_HH

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
/// Triangular banded, n-by-n, distributed, tiled matrices.
template <typename scalar_t>
class BaseTriangularBandMatrix: public BaseBandMatrix<scalar_t> {
public:
    using ij_tuple = std::tuple<int64_t, int64_t>;

protected:
    // constructors
    BaseTriangularBandMatrix();

    BaseTriangularBandMatrix(Uplo uplo, int64_t n, int64_t kd,
                            std::function<int64_t (int64_t j)>& inTileNb,
                            std::function<int (ij_tuple ij)>& inTileRank,
                            std::function<int (ij_tuple ij)>& inTileDevice,
                            MPI_Comm mpi_comm);

    BaseTriangularBandMatrix(
        Uplo uplo,
        int64_t n, int64_t kd,
        int64_t nb, int p, int q, MPI_Comm mpi_comm);

    // conversion
    BaseTriangularBandMatrix(Uplo uplo, BaseBandMatrix<scalar_t>& orig);

    BaseTriangularBandMatrix(int64_t kd, BaseMatrix<scalar_t>& orig);

public:
    template <typename T>
    friend void swap(BaseTriangularBandMatrix<T>& A, BaseTriangularBandMatrix<T>& B);

    int64_t bandwidth() const;
    void    bandwidth(int64_t kd);

    void    gather(scalar_t* A, int64_t lda);

    void    insertLocalTiles(Target origin=Target::Host);
};

//------------------------------------------------------------------------------
/// Default constructor creates an empty band matrix with bandwidth = 0.
template <typename scalar_t>
BaseTriangularBandMatrix<scalar_t>::BaseTriangularBandMatrix()
    : BaseBandMatrix<scalar_t>()
{
    this->uplo_ = Uplo::Lower;
}

//------------------------------------------------------------------------------
/// Constructor creates an n-by-n band matrix, with no tiles allocated,
/// where tileNb, tileRank, tileDevice are given as functions.
/// Tiles can be added with tileInsert().
///
template <typename scalar_t>
BaseTriangularBandMatrix<scalar_t>::BaseTriangularBandMatrix(
    Uplo uplo, int64_t n, int64_t kd,
    std::function<int64_t (int64_t j)>& inTileNb,
    std::function<int (ij_tuple ij)>& inTileRank,
    std::function<int (ij_tuple ij)>& inTileDevice,
    MPI_Comm mpi_comm)
    : BaseBandMatrix<scalar_t>(n, n, kd, kd, inTileNb, inTileRank,
                               inTileDevice, mpi_comm)
{
    slate_error_if(uplo == Uplo::General);
    this->uplo_ = uplo;
    this->kl_ = (uplo == Uplo::Lower) ? this->kl_ : 0;
    this->ku_ = (uplo == Uplo::Lower) ? 0 : this->ku_;
}

//------------------------------------------------------------------------------
/// Constructor creates an n-by-n band matrix, with no tiles allocated,
/// with fixed nb-by-nb tile size and 2D block cyclic distribution.
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
BaseTriangularBandMatrix<scalar_t>::BaseTriangularBandMatrix(
    Uplo uplo,
    int64_t n, int64_t kd, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
    : BaseBandMatrix<scalar_t>(n, n, kd, kd, nb, p, q, mpi_comm)
{
    slate_error_if(uplo == Uplo::General);
    this->uplo_ = uplo;
    this->kl_ = (uplo == Uplo::Lower) ? this->kl_ : 0;
    this->ku_ = (uplo == Uplo::Lower) ? 0 : this->ku_;
}

//------------------------------------------------------------------------------
/// [explicit]
///
/// Conversion from general band matrix
/// creates a shallow copy view of the original matrix.
///
/// @param[in] uplo
///     - Upper: upper triangle of A is stored.
///     - Lower: lower triangle of A is stored.
///
/// @param[in,out] orig
///     Original matrix.
///
// todo: should the trans be reflected in kl_ and ku_ values?
template <typename scalar_t>
BaseTriangularBandMatrix<scalar_t>::BaseTriangularBandMatrix(
    Uplo uplo, BaseBandMatrix<scalar_t>& orig)
    : BaseBandMatrix<scalar_t>(orig)
{
    slate_error_if(uplo == Uplo::General);
    this->uplo_ = uplo;
    this->kl_ = (uplo == Uplo::Lower) ? this->kl_ : 0;
    this->ku_ = (uplo == Uplo::Lower) ? 0 : this->ku_;
}

//------------------------------------------------------------------------------
/// [explicit]
///
/// Conversion from general matrix
/// creates a shallow copy view of the original matrix.
/// Uses only square portion, Aorig[ 0:min(mt,nt)-1, 0:min(mt,nt)-1 ].
///
/// @param[in] kd
///     Number of sub (if lower) or super (if upper) diagonals within band.
///     kd >= 0.
///
/// @param[in,out] orig
///     Original matrix.
///
// todo: should the trans be reflected in kl_ and ku_ values?
template <typename scalar_t>
BaseTriangularBandMatrix<scalar_t>::BaseTriangularBandMatrix(
    int64_t kd, BaseMatrix<scalar_t>& orig)
    : BaseBandMatrix<scalar_t>(kd, kd, orig)
{
    this->kl_ = (this->uploPhysical() == Uplo::Lower) ? this->kl_ : 0;
    this->ku_ = (this->uploPhysical() == Uplo::Lower) ? 0 : this->ku_;
}

//------------------------------------------------------------------------------
/// Swap contents of band matrices A and B.
template <typename scalar_t>
void swap(BaseTriangularBandMatrix<scalar_t>& A, BaseTriangularBandMatrix<scalar_t>& B)
{
    using std::swap;
    swap(static_cast< BaseBandMatrix<scalar_t>& >(A),
         static_cast< BaseBandMatrix<scalar_t>& >(B));
}

//------------------------------------------------------------------------------
/// @return number of subdiagonals within band.
template <typename scalar_t>
int64_t BaseTriangularBandMatrix<scalar_t>::bandwidth() const
{
    // todo: is this correct? should the trans be accounted for here?
    return (this->uploPhysical() == Uplo::Lower)
            ? this->kl_
            : this->ku_;
}

//------------------------------------------------------------------------------
/// Sets number of subdiagonals within band.
template <typename scalar_t>
void BaseTriangularBandMatrix<scalar_t>::bandwidth(int64_t kd)
{
    // todo: is this correct? should the trans be accounted for here?
    if (this->uploPhysical() == Uplo::Lower)
        this->kl_ = kd;
    else
        this->ku_ = kd;
}

//------------------------------------------------------------------------------
/// Gathers the entire matrix to the LAPACK-style matrix A on MPI rank 0.
/// Primarily for debugging purposes.
///
template <typename scalar_t>
void BaseTriangularBandMatrix<scalar_t>::gather(scalar_t* A, int64_t lda)
{
    // this code assumes the matrix is not transposed
    Op op_save = this->op();
    this->op_ = Op::NoTrans;
    auto upper = this->uplo() == Uplo::Upper;

    int64_t mt = this->mt();
    int64_t nt = this->nt();
    int64_t kdt = ceildiv( this->bandwidth(), this->tileNb(0) );
    // ii, jj are row, col indices
    // i, j are tile (block row, block col) indices
    int64_t jj = 0;
    for (int64_t j = 0; j < nt; ++j) {
        int64_t jb = this->tileNb(j);

        int64_t ii = 0;
        int64_t istart = upper ? blas::max( 0, j-kdt ) : j;
        int64_t iend   = upper ? j : blas::min( j+kdt, mt-1 );
        for (int64_t i = 0; i < this->mt(); ++i) {
            int64_t ib = this->tileMb(i);
            if (i >= istart && i <= iend) {
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
///     - if target = Host, inserts on tiles on CPU host.
///
template <typename scalar_t>
void BaseTriangularBandMatrix<scalar_t>::insertLocalTiles(Target origin)
{
    this->origin_ = origin;
    bool on_devices = (origin == Target::Devices);
    auto upper = this->uplo() == Uplo::Upper;
    int64_t mt = this->mt();
    int64_t nt = this->nt();
    int64_t kdt = ceildiv( this->bandwidth(), this->tileNb(0) );
    for (int64_t j = 0; j < nt; ++j) {
        int64_t istart = upper ? blas::max( 0, j-kdt ) : j;
        int64_t iend   = upper ? j : blas::min( j+kdt, mt-1 );
        for (int64_t i = istart; i <= iend; ++i) {
            if (this->tileIsLocal(i, j)) {
                int dev = (on_devices ? this->tileDevice(i, j)
                                      : HostNum);
                this->tileInsert(i, j, dev);
            }
        }
    }
}

} // namespace slate

#endif // SLATE_BASE_TRIANGULAR_BAND_MATRIX_HH
