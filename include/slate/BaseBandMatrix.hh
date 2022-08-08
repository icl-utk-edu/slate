// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_BASE_BAND_MATRIX_HH
#define SLATE_BASE_BAND_MATRIX_HH

#include "slate/BaseMatrix.hh"
#include "slate/Matrix.hh"
#include "slate/Tile.hh"
#include "slate/types.hh"

#include "lapack.hh"

#include <algorithm>
#include <utility>

#include "slate/internal/mpi.hh"
#include "slate/internal/openmp.hh"

namespace slate {

//==============================================================================
/// Base class for all SLATE distributed, tiled banded storage matrices.
template <typename scalar_t>
class BaseBandMatrix: public BaseMatrix<scalar_t> {
public:
    using ij_tuple = std::tuple<int64_t, int64_t>;

protected:
    // constructors
    BaseBandMatrix();

    BaseBandMatrix(int64_t m, int64_t n, int64_t kl, int64_t ku,
                   std::function<int64_t (int64_t j)>& inTileNb,
                   std::function<int (ij_tuple ij)>& inTileRank,
                   std::function<int (ij_tuple ij)>& inTileDevice,
                   MPI_Comm mpi_comm);

    BaseBandMatrix(int64_t m, int64_t n, int64_t kl, int64_t ku,
                   int64_t nb, int p, int q, MPI_Comm mpi_comm);

    // conversion sub-matrix
    BaseBandMatrix(int64_t kl, int64_t ku, BaseMatrix< scalar_t >& orig);

    // used by slice
    BaseBandMatrix(BaseBandMatrix<scalar_t>& orig,
                   typename BaseMatrix<scalar_t>::Slice slice);

public:
    template <typename T>
    friend void swap(BaseBandMatrix<T>& A, BaseBandMatrix<T>& B);

    int64_t getMaxDeviceTiles(int device);
    void    allocateBatchArrays(int64_t batch_size=0, int64_t num_arrays=1);
    void    reserveDeviceWorkspace();

    // sub-matrix
    Matrix<scalar_t> sub(int64_t i1, int64_t i2,
                         int64_t j1, int64_t j2);

    void tileUpdateAllOrigin();

protected:
    int64_t kl_, ku_;
};

//------------------------------------------------------------------------------
/// Default constructor creates an empty band matrix with bandwidth = 0.
template <typename scalar_t>
BaseBandMatrix<scalar_t>::BaseBandMatrix():
    BaseMatrix<scalar_t>(),
    kl_(0),
    ku_(0)
{}

//------------------------------------------------------------------------------
/// Constructor creates an m-by-n matrix, with no tiles allocated,
/// where tileMb, tileNb, tileRank, tileDevice are given as functions.
/// Tiles can be added with tileInsert().
///
template <typename scalar_t>
BaseBandMatrix<scalar_t>::BaseBandMatrix(
    int64_t m, int64_t n, int64_t kl, int64_t ku,
    std::function<int64_t (int64_t j)>& inTileNb,
    std::function<int (ij_tuple ij)>& inTileRank,
    std::function<int (ij_tuple ij)>& inTileDevice,
    MPI_Comm mpi_comm)
    : BaseMatrix<scalar_t>(m, n, inTileNb, inTileNb, inTileRank,
                           inTileDevice, mpi_comm),
      kl_(kl),
      ku_(ku)
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
BaseBandMatrix<scalar_t>::BaseBandMatrix(
    int64_t m, int64_t n, int64_t kl, int64_t ku, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
    : BaseMatrix<scalar_t>(m, n, nb, p, q, mpi_comm),
      kl_(kl),
      ku_(ku)
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
BaseBandMatrix<scalar_t>::BaseBandMatrix(
    BaseBandMatrix<scalar_t>& orig, typename BaseMatrix<scalar_t>::Slice slice)
    : BaseMatrix<scalar_t>(orig, slice),
      kl_(orig.kl_),
      ku_(orig.ku_)
{}

//------------------------------------------------------------------------------
/// Conversion from general matrix
/// creates shallow copy view of the band region [kl, ku] of the original matrix.
///
/// @param[in] kl
///     Number of subdiagonals within band. kl >= 0.
///
/// @param[in] ku
///     Number of superdiagonals within band. ku >= 0.
///
/// @param[in] orig
///     Original matrix of which to make sub-matrix.
///
template <typename scalar_t>
BaseBandMatrix<scalar_t>::BaseBandMatrix(
    int64_t kl, int64_t ku, BaseMatrix< scalar_t >& orig)
    : BaseMatrix<scalar_t>(orig),
      kl_(kl),
      ku_(ku)
{}

//------------------------------------------------------------------------------
/// Swap contents of band matrices A and B.
template <typename scalar_t>
void swap(BaseBandMatrix<scalar_t>& A, BaseBandMatrix<scalar_t>& B)
{
    using std::swap;
    swap(static_cast< BaseMatrix<scalar_t>& >(A),
         static_cast< BaseMatrix<scalar_t>& >(B));
    swap(A.kl_, B.kl_);
    swap(A.ku_, B.ku_);
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
// todo: any restrictions to be imposed?
template <typename scalar_t>
Matrix<scalar_t> BaseBandMatrix<scalar_t>::sub(
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
{
    return Matrix<scalar_t>(*this, i1, i2, j1, j2);
}

//------------------------------------------------------------------------------
/// Returns number of local tiles of the matrix on this rank and given device.
///
// todo: numLocalDeviceTiles
// todo: assumes uniform tile sizes.
template <typename scalar_t>
int64_t BaseBandMatrix<scalar_t>::getMaxDeviceTiles(int device)
{
    int64_t num_tiles = 0;
    int64_t mt = this->mt();
    int64_t nt = this->nt();
    int64_t klt = ceildiv( this->kl_, this->tileNb(0) );
    int64_t kut = ceildiv( this->ku_, this->tileNb(0) );
    for (int64_t j = 0; j < nt; ++j) {
        int64_t istart = blas::max( 0, j-kut );
        int64_t iend   = blas::min( j+klt+1, mt );
        for (int64_t i = istart; i < iend; ++i) {
            if (this->tileIsLocal(i, j) && this->tileDevice(i, j) == device)
                ++num_tiles;
        }
    }
    return num_tiles;
}

//------------------------------------------------------------------------------
/// Allocates batch arrays and BLAS++ queues for all devices.
/// This overrides BaseMatrix::allocateBatchArrays
/// to use the number of local tiles inside the band.
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
void BaseBandMatrix<scalar_t>::allocateBatchArrays(
    int64_t batch_size, int64_t num_arrays)
{
    if (batch_size == 0) {
        for (int device = 0; device < this->num_devices_; ++device)
            batch_size = std::max(batch_size, getMaxDeviceTiles(device));
    }
    this->storage_->allocateBatchArrays(batch_size, num_arrays);
}

//------------------------------------------------------------------------------
/// Reserve space for temporary workspace tiles on all GPU devices.
template <typename scalar_t>
void BaseBandMatrix<scalar_t>::reserveDeviceWorkspace()
{
    int64_t num_tiles = 0;
    for (int device = 0; device < this->num_devices_; ++device)
        num_tiles = std::max(num_tiles, getMaxDeviceTiles(device));
    this->storage_->reserveDeviceWorkspace(num_tiles);
}

//------------------------------------------------------------------------------
/// Move all tiles back to their origin.
//
// todo: Assuming fixed size, square tiles for simplicity, should generalize
template <typename scalar_t>
void BaseBandMatrix<scalar_t>::tileUpdateAllOrigin()
{
    int64_t mt = this->mt();
    int64_t nt = this->nt();
    // int64_t klt = ceildiv( this->kl_, this->tileNb(0) );
    // int64_t kut = ceildiv( this->ku_, this->tileNb(0) );
    // todo: Agree upon weather lowerBandwidth and upperBandwidth should
    // be in BaseBandMatrix class or BandMatrix class.
    int64_t klt = ceildiv(
            this->op() == Op::NoTrans ? this->kl_ : this->ku_, this->tileNb(0));
    int64_t kut = ceildiv(
            this->op() == Op::NoTrans ? this->ku_ : this->kl_, this->tileNb(0));

    std::vector< std::set<ij_tuple> > tiles_set_host(this->num_devices());
    std::vector< std::set<ij_tuple> > tiles_set_dev(this->num_devices());

    for (int64_t j = 0; j < nt; ++j) {
        int64_t istart = blas::max( 0, j-kut );
        int64_t iend   = blas::min( j+klt+1, mt );
        for (int64_t i = istart; i < iend; ++i) {
            if (this->tileIsLocal(i, j)) {
                // this->tileUpdateOrigin(i, j);
                auto& tile_node = this->storage_->at(this->globalIndex(i, j));

                // find on host
                if (tile_node.existsOn( HostNum )
                    && tile_node[ HostNum ].tile()->origin()) {
                    if (tile_node[ HostNum ].stateOn( MOSI::Invalid )) {
                        // tileGetForReading(i, j, LayoutConvert::None);
                        for (int d = 0; d < this->num_devices(); ++d) {
                            if (tile_node.existsOn(d)
                                && tile_node[d].getState() != MOSI::Invalid)
                            {
                                tiles_set_host[d].insert({i, j});
                                break;
                            }
                        }
                    }
                }
                else {
                    auto device = this->tileDevice(i, j);
                    if (tile_node.existsOn(device) &&
                        tile_node[device].tile()->origin()) {
                        if (tile_node[device].stateOn(MOSI::Invalid)) {
                            // tileGetForReading(i, j, device, LayoutConvert::None);
                            tiles_set_dev[device].insert({i, j});
                        }
                    }
                    else
                        slate_error( std::string("Origin tile not found! tile(")
                                    +std::to_string(i)+","+std::to_string(j)+")");
                }
            }
        }
    }

    #pragma omp taskgroup
    {
        for (int d = 0; d < this->num_devices(); ++d) {
            if (! tiles_set_host[d].empty()) {
                #pragma omp task slate_omp_default_none \
                    firstprivate( d ) shared( tiles_set_host )
                {
                    this->tileGetForReading(tiles_set_host[d], LayoutConvert::None, d);
                }
            }
            if (! tiles_set_dev[d].empty()) {
                #pragma omp task slate_omp_default_none \
                    firstprivate( d ) shared( tiles_set_dev )
                {
                    this->tileGetForReading(tiles_set_dev[d], d, LayoutConvert::None);
                }
            }
        }
    }
}

} // namespace slate

#endif // SLATE_BASE_BAND_MATRIX_HH
