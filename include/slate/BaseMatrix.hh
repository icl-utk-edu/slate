// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_BASE_MATRIX_HH
#define SLATE_BASE_MATRIX_HH

#include "slate/internal/comm.hh"
#include "slate/internal/Memory.hh"
#include "slate/internal/device.hh"
#include "slate/internal/MatrixStorage.hh"
#include "slate/Tile.hh"
#include "slate/Tile_blas.hh"
#include "slate/types.hh"
#include "slate/config.hh"

#include "lapack.hh"
#include "lapack/device.hh"

#include <algorithm>
#include <memory>
#include <set>
#include <list>
#include <tuple>
#include <utility>
#include <vector>

#include "slate/internal/mpi.hh"
#include "slate/internal/openmp.hh"

namespace slate {

//==============================================================================
/// Base class for all SLATE distributed, tiled matrices.
/// In general, the documentation refers to the current matrix object as op(A),
/// to emphasize that it may be transposed with respect to its parent matrix.
///
template <typename scalar_t>
class BaseMatrix {
public:
    using BcastList =
        std::vector<std::tuple<int64_t, int64_t,
                             std::list<BaseMatrix<scalar_t> > > >;

    // For multi-threaded bcast, each <i,j> needs a message tag
    using BcastListTag =
        std::vector<std::tuple<int64_t, int64_t,
                             std::list<BaseMatrix<scalar_t> >,
                             int64_t> >;

    using ReduceList =
        std::vector<std::tuple<int64_t, int64_t,
                             BaseMatrix<scalar_t>,
                             std::list<BaseMatrix<scalar_t> > > >;

    using ij_tuple = typename MatrixStorage<scalar_t>::ij_tuple;

    friend class Debug;

    // Make every class BaseMatrix<T2> a friend of BaseMatrix<scalar_t>.
    template <typename T2>
    friend class BaseMatrix;

    static constexpr bool is_complex = slate::is_complex<scalar_t>::value;
    static constexpr bool is_real    = ! is_complex;

    typedef scalar_t value_type;

protected:
    BaseMatrix();

    // Defaults okay for:
    // 1. destructor
    // 2. copy constructor
    // 3. move constructor
    // 4. copy assignment
    // 5. move assignment

    BaseMatrix(int64_t m, int64_t n,
               std::function<int64_t (int64_t i)>& inTileMb,
               std::function<int64_t (int64_t j)>& inTileNb,
               std::function<int (ij_tuple ij)>& inTileRank,
               std::function<int (ij_tuple ij)>& inTileDevice,
               MPI_Comm mpi_comm);

    //----------
    BaseMatrix( int64_t m, int64_t n, int64_t mb, int64_t nb,
                GridOrder order, int nprow, int npcol, MPI_Comm mpi_comm );

    /// With order = Col.
    BaseMatrix( int64_t m, int64_t n, int64_t mb, int64_t nb,
                int nprow, int npcol, MPI_Comm mpi_comm )
        : BaseMatrix( m, n, mb, nb, GridOrder::Col, nprow, npcol, mpi_comm )
    {}

    /// With mb = nb, order = Col.
    BaseMatrix( int64_t m, int64_t n, int64_t nb,
                int nprow, int npcol, MPI_Comm mpi_comm )
        : BaseMatrix( m, n, nb, nb, GridOrder::Col, nprow, npcol, mpi_comm )
    {}

    //----------
    BaseMatrix(BaseMatrix& orig,
               int64_t i1, int64_t i2,
               int64_t j1, int64_t j2);

    // internal utility class to differentiate slicing (row/col indices)
    // from sub-matrix (tile indices)
    class Slice {
    public:
        Slice( int64_t in_row1, int64_t in_row2,
               int64_t in_col1, int64_t in_col2 )
        : row1( in_row1 ),
          row2( in_row2 ),
          col1( in_col1 ),
          col2( in_col2 )
        {}
        int64_t row1, row2, col1, col2;
    };

    BaseMatrix(BaseMatrix& orig, Slice slice);

public:
    template <typename out_scalar_t>
    BaseMatrix<out_scalar_t> baseEmptyLike(int64_t mb, int64_t nb, Op deepOp);

private:
    void initSubmatrix(
        int64_t i1, int64_t i2,
        int64_t j1, int64_t j2);

    void initSlice(int64_t row0_offset, int64_t col0_offset,
                   int64_t last_mb, int64_t last_nb);

public:
    /// Returns shallow copy of op(A) that is transposed.
    /// @see conj_transpose
    template<typename MatrixType>
    friend MatrixType transpose(MatrixType& A);

    /// Returns shallow copy of op(A) that is conjugate-transpose.
    /// @see transpose
    template<typename MatrixType>
    friend MatrixType conj_transpose( MatrixType& A );

    /// @deprecated
    template<typename MatrixType>
    friend MatrixType conjTranspose( MatrixType& A );

    template <typename T>
    friend void swap(BaseMatrix<T>& A, BaseMatrix<T>& B);

    Tile<scalar_t> operator()( int64_t i, int64_t j, int device=HostNum );

    /// Alias of operator().
    Tile<scalar_t> at( int64_t i, int64_t j, int device=HostNum )
    {
        return (*this)(i, j, device);
    }

    /// Returns number of devices (per MPI process) to distribute matrix to.
    int num_devices() const { return num_devices_; }

    void gridinfo( GridOrder* order, int* nprow, int* npcol,
                   int* myrow, int* mycol ) const;

    /// Returns tileMb function. Useful to construct matrices with the
    /// same block size. For submatrices, this is of the parent matrix.
    std::function<int64_t (int64_t i)> tileMbFunc() const
    {
        return storage_->tileMb;
    }

    /// Returns tileNb function. Useful to construct matrices with the
    /// same block size. For submatrices, this is of the parent matrix.
    std::function<int64_t (int64_t j)> tileNbFunc() const
    {
        return storage_->tileNb;
    }

    /// Returns tileRank function. Useful to construct matrices with the
    /// same block size. For submatrices, this is of the parent matrix.
    std::function<int (ij_tuple ij)> tileRankFunc() const
    {
        return storage_->tileRank;
    }

    /// Returns tileDevice function. Useful to construct matrices with the
    /// same block size. For submatrices, this is of the parent matrix.
    std::function<int (ij_tuple ij)> tileDeviceFunc() const
    {
        return storage_->tileDevice;
    }

    int64_t m() const;
    int64_t n() const;

    /// Returns number of block rows in op(A).
    int64_t mt() const { return (op_ == Op::NoTrans ? mt_ : nt_); }

    /// Returns number of block cols in op(A).
    int64_t nt() const { return (op_ == Op::NoTrans ? nt_ : mt_); }

    /// Returns transposition operation op(A) as NoTrans, Trans, or ConjTrans.
    Op op() const { return op_; }

    /// returns true if tile exists on specified device
    bool tileExists( int64_t i, int64_t j, int device=HostNum )
    {
        return storage_->tileExists( globalIndex( i, j, device ) );
    }

    /// Returns MPI rank of tile {i, j} of op(A).
    int tileRank(int64_t i, int64_t j) const
    {
        return storage_->tileRank(globalIndex(i, j));
    }

    /// Returns device of tile {i, j} of op(A).
    int tileDevice(int64_t i, int64_t j) const
    {
        return storage_->tileDevice(globalIndex(i, j));
    }

    /// Returns whether tile {i, j} of op(A) is local.
    bool tileIsLocal(int64_t i, int64_t j) const
    {
        return storage_->tileIsLocal(globalIndex(i, j));
    }

    /// Returns whether op(A) is logically Lower, Upper, or General.
    Uplo uplo() const { return uploLogical(); }
    Uplo uploLogical() const;
    Uplo uploPhysical() const;

    Tile<scalar_t> originTile(int64_t i, int64_t j);
    /// Tile origin
    Target origin() const { return origin_; }

    int64_t tileMb(int64_t i) const;
    int64_t tileNb(int64_t j) const;
private:
    int64_t tileMbInternal(int64_t i) const;
    int64_t tileNbInternal(int64_t j) const;

public:
    Tile<scalar_t> tileInsert( int64_t i, int64_t j, int device=HostNum );
    Tile<scalar_t> tileInsert( int64_t i, int64_t j, int device,
                                scalar_t* A, int64_t ld );

    /// Insert tile with default device=HostNum. @see tileInsert.
    Tile<scalar_t> tileInsert(int64_t i, int64_t j,
                               scalar_t* A, int64_t ld)
    {
        return tileInsert( i, j, HostNum, A, ld );
    }

    Tile<scalar_t> tileInsertWorkspace(int64_t i, int64_t j, int device, Layout layout);

    Tile<scalar_t> tileInsertWorkspace(int64_t i, int64_t j, int device)
    {
        return tileInsertWorkspace(i, j, device, layout_);
    }

    Tile<scalar_t> tileInsertWorkspace(int64_t i, int64_t j, Layout layout)
    {
        return tileInsertWorkspace( i, j, HostNum, layout );
    }

    Tile<scalar_t> tileInsertWorkspace(int64_t i, int64_t j)
    {
        return tileInsertWorkspace( i, j, HostNum, layout_ );
    }

    scalar_t* allocWorkspaceBuffer(int device, int64_t size);
    void freeWorkspaceBuffer(int device, scalar_t* buffer);

    //--------------------------------------------------------------------------
    // MOSI
private:
    void tileGet(int64_t i, int64_t j, int dst_device,
                 LayoutConvert layout, bool modify, bool hold,
                 bool async);

    void tileGet(std::set<ij_tuple>& tile_set, int dst_device,
                 LayoutConvert layout, bool modify, bool hold,
                 bool async);

    void tileCopyDataLayout(Tile<scalar_t>* src_tile,
                            Tile<scalar_t>* dst_tile,
                            Layout target_layout,
                            bool async);

public:

    //--------------------------------------------------------------------------
    /// Returns tile(i, j)'s state on device (defaults to host).
    /// Asserts that the tile exists.
    ///
    /// @param[in] i
    ///     Tile's block row index. 0 <= i < mt.
    ///
    /// @param[in] j
    ///     Tile's block column index. 0 <= j < nt.
    ///
    /// @param[in] device
    ///     Tile's device ID.
    ///
    MOSI tileState( int64_t i, int64_t j, int device=HostNum )
    {
        return storage_->tileState( globalIndex( i, j, device ) );
    }

    //------------------------------------------------------------------------------
    /// Returns whether tile(i, j) is OnHold on device (defaults to host).
    /// Asserts that the tile exists.
    ///
    /// @param[in] i
    ///     Tile's block row index. 0 <= i < mt.
    ///
    /// @param[in] j
    ///     Tile's block column index. 0 <= j < nt.
    ///
    /// @param[in] device
    ///     Tile's device ID.
    ///
    bool tileOnHold( int64_t i, int64_t j, int device=HostNum )
    {
        return storage_->tileOnHold( globalIndex( i, j, device ) );
    }

    //------------------------------------------------------------------------------
    /// Unsets the hold of tile(i, j) on device (defaults to host) if it was OnHold.
    ///
    /// @param[in] i
    ///     Tile's block row index. 0 <= i < mt.
    ///
    /// @param[in] j
    ///     Tile's block column index. 0 <= j < nt.
    ///
    /// @param[in] device
    ///     Tile's device ID.
    ///
    void tileUnsetHold( int64_t i, int64_t j, int device=HostNum )
    {
        storage_->tileUnsetHold( globalIndex( i, j, device ) );
    }

    void tileUnsetHoldAll( int device=HostNum );

    void tileUnsetHoldAllOnDevices();

    void tileModified( int64_t i, int64_t j, int device=HostNum,
                       bool permissive=false );

    void tileAcquire(int64_t i, int64_t j, int device, Layout layout);

    void tileAcquire(int64_t i, int64_t j, Layout layout)
    {
        tileAcquire( i, j, HostNum, layout );
    }

    void tileGetForReading(int64_t i, int64_t j, int device, LayoutConvert layout);

    void tileGetForReading(std::set<ij_tuple>& tile_set, int device, LayoutConvert layout);

    /// Gets tile(i, j) for reading on host.
    /// @see tileGetForReading
    void tileGetForReading(int64_t i, int64_t j, LayoutConvert layout)
    {
        tileGetForReading( i, j, HostNum, layout );
    }

    /// Gets a set of tiles for reading on host.
    /// @see tileGetForReading
    void tileGetForReading(std::set<ij_tuple>& tile_set, LayoutConvert layout)
    {
        tileGetForReading( tile_set, HostNum, layout );
    }
    void tileGetForReading(std::set<ij_tuple>& tile_set, LayoutConvert layout,
                            int from_device);

    void tileGetAllForReading(int device, LayoutConvert layout);

    void tileGetAllForReadingOnDevices(LayoutConvert layout);

    void tileGetForWriting(int64_t i, int64_t j, int device, LayoutConvert layout);

    void tileGetForWriting(std::set<ij_tuple>& tile_set, int device, LayoutConvert layout);

    /// Gets tile(i, j) for writing on host.
    /// @see tileGetForWriting
    void tileGetForWriting(int64_t i, int64_t j, LayoutConvert layout)
    {
        tileGetForWriting( i, j, HostNum, layout );
    }

    /// Gets a set of tiles for writing on host.
    /// @see tileGetForWriting
    void tileGetForWriting(std::set<ij_tuple>& tile_set, LayoutConvert layout)
    {
        tileGetForWriting( tile_set, HostNum, layout );
    }

    void tileGetAllForWriting(int device, LayoutConvert layout);

    void tileGetAllForWritingOnDevices(LayoutConvert layout);

    void tileGetAndHold(int64_t i, int64_t j, int device, LayoutConvert layout);

    /// Gets tile(i, j) on host for reading and marks it as MOSI::OnHold.
    /// @see tileGetAndHold
    void tileGetAndHold(int64_t i, int64_t j, LayoutConvert layout)
    {
        tileGetAndHold( i, j, HostNum, layout );
    }

    void tileGetAndHold(std::set<ij_tuple>& tile_set, int device, LayoutConvert layout);

    /// Gets a set of tiles for reading on host and marks them as MOSI::OnHold.
    /// @see tileGetAndHold
    void tileGetAndHold(std::set<ij_tuple>& tile_set, LayoutConvert layout)
    {
        tileGetAndHold( tile_set, HostNum, layout );
    }

    void tileGetAndHoldAll(int device, LayoutConvert layout);

    void tileGetAndHoldAllOnDevices(LayoutConvert layout);

    Tile<scalar_t> tileUpdateOrigin(int64_t i, int64_t j);

    void tileUpdateAllOrigin();

    /// Returns life counter of tile {i, j} of op(A).
    int64_t tileLife(int64_t i, int64_t j) const
    {
        return storage_->tileLife(globalIndex(i, j));
    }

    /// Set life counter of tile {i, j} of op(A).
    void tileLife(int64_t i, int64_t j, int64_t life)
    {
        storage_->tileLife(globalIndex(i, j), life);
    }

    /// Decrements life counter of workspace tile {i, j} of op(A).
    /// Then, if life reaches 0, deletes tile on all devices.
    /// For local, non-workspace tiles, does nothing.
    void tileTick(int64_t i, int64_t j)
    {
        storage_->tileTick(globalIndex(i, j));
    }

    /// Returns how many times the tile {i, j} is received
    /// through MPI.
    /// This function is used to track tiles that may be
    /// communicated twice due to symmetricity
    /// during hemm and symm operations.
    int64_t tileReceiveCount(int64_t i, int64_t j) const
    {
        return storage_->tileReceiveCount( globalIndex( i, j ) );
    }

    /// Increments the number of times the tile {i, j} is received
    /// through MPI.
    void tileIncrementReceiveCount(int64_t i, int64_t j)
    {
        storage_->tileIncrementReceiveCount( globalIndex( i, j ) );
    }

    /// Decrements the number of times the tile {i, j} is received
    /// through MPI.
    void tileDecrementReceiveCount(int64_t i, int64_t j)
    {
        storage_->tileDecrementReceiveCount( globalIndex( i, j ) );
    }

    void tileErase( int64_t i, int64_t j, int device=HostNum );

    void tileRelease( int64_t i, int64_t j, int device=HostNum );

    //--------------------------------------------------------------------------
    template <Target target = Target::Host>
    void tileSend(int64_t i, int64_t j, int dst_rank, int tag = 0);

    template <Target target = Target::Host>
    void tileIsend(int64_t i, int64_t j, int dst_rank,
                   int tag, MPI_Request* request);

    template <Target target = Target::Host>
    void tileRecv(int64_t i, int64_t j, int dst_rank,
                  Layout layout, int tag = 0);

    void tileIrecv(int64_t i, int64_t j, int dst_rank,
                  Layout layout, int tag, MPI_Request* request);

    template <Target target = Target::Host>
    void tileBcast(int64_t i, int64_t j, BaseMatrix const& B,
                   Layout layout, int tag = 0, int64_t life_factor = 1);

    template <Target target = Target::Host>
    void listBcast(
        BcastList& bcast_list, Layout layout,
        int tag = 0, int64_t life_factor = 1,
        bool is_shared = false);

    // This variant takes a BcastListTag where each <i,j> tile has
    // its own message tag
    template <Target target = Target::Host>
    void listBcastMT(
        BcastListTag& bcast_list, Layout layout,
        int64_t life_factor = 1,
        bool is_shared = false);

    template <Target target = Target::Host>
    void listReduce(ReduceList& reduce_list, Layout layout, int tag = 0);

    //--------------------------------------------------------------------------
    // LAYOUT
public:
    /// Returns matrix layout flag
    Layout layout() const { return layout_; }

    /// Returns Layout of tile(i, j, device)
    Layout tileLayout( int64_t i, int64_t j, int device=HostNum )
    {
        return storage_->at( globalIndex(i, j, device) )->layout();
    }

    /// Sets Layout of tile(i, j, device)
    [[deprecated( "Use tileGetFor* instead. Will be removed 2024-10." )]]
    void tileLayout(int64_t i, int64_t j, int device, Layout layout)
    {
        storage_->at( globalIndex(i, j, device) )->setLayout(layout);
    }

    /// Sets Layout of tile(i, j, host)
    [[deprecated( "Use tileGetFor* instead. Will be removed 2024-10." )]]
    void tileLayout(int64_t i, int64_t j, Layout layout)
    {
        tileLayout( i, j, HostNum, layout );
    }

    bool tileLayoutIsConvertible( int64_t i, int64_t j, int device=HostNum );

    void tileLayoutConvert(int64_t i, int64_t j, int device, Layout layout,
                           bool reset = false, bool async = false);
    /// Convert layout of tile(i, j) to layout on host, optionally reset
    void tileLayoutConvert(int64_t i, int64_t j, Layout layout,
                           bool reset = false, bool async = false)
    {
        tileLayoutConvert( i, j, HostNum, layout, reset, async );
    }
    void tileLayoutConvert(std::set<ij_tuple>& tile_set, int device,
                           Layout layout, bool reset = false);
    /// Convert layout of a set of tiles to layout on host, optionally reset
    void tileLayoutConvert(std::set<ij_tuple>& tile_set, Layout layout, bool reset = false)
    {
        tileLayoutConvert( tile_set, HostNum, layout, reset );
    }
    void tileLayoutConvert(int device, Layout layout, bool reset = false);
    void tileLayoutConvertOnDevices(Layout layout, bool reset = false);

    void tileLayoutReset(int64_t i, int64_t j, int device, Layout layout);
    void tileLayoutReset(int64_t i, int64_t j, Layout layout)
    {
        tileLayoutReset( i, j, HostNum, layout );
    }
    void tileLayoutReset(std::set<ij_tuple>& tile_set, int device, Layout layout);
    void tileLayoutReset(std::set<ij_tuple>& tile_set, Layout layout)
    {
        tileLayoutReset( tile_set, HostNum, layout );
    }
    void tileLayoutReset();

    //--------------------------------------------------------------------------
protected:
    void tileBcastToSet(int64_t i, int64_t j, std::set<int> const& bcast_set);
    void tileBcastToSet(int64_t i, int64_t j, std::set<int> const& bcast_set,
                        int radix, int tag, Layout layout,
                        Target target);
    void tileIbcastToSet(int64_t i, int64_t j, std::set<int> const& bcast_set,
                        int radix, int tag, Layout layout,
                        std::vector<MPI_Request>& send_requests,
                        Target target);

public:
    // todo: should this be private?
    void tileReduceFromSet(int64_t i, int64_t j, int root_rank,
                           std::set<int>& reduce_set, int radix, int tag,
                           Layout layout);


    void getRanks(std::set<int>* bcast_set) const;
    void getLocalDevices(std::set<int>* dev_set) const;
    int64_t numLocalTiles() const;
    MPI_Comm  mpiComm()  const { return mpi_comm_; }
    int       mpiRank()  const { return mpi_rank_; }
    MPI_Group mpiGroup() const { return mpi_group_; }

    [[deprecated("use slate::HostNum constant")]]
    int       hostNum()  const { return HostNum; }

    /// Removes all tiles from matrix.
    /// WARNING: currently this clears the entire parent matrix,
    /// not just a sub-matrix.
    void clear()
    {
        storage_->clear();
    }

    /// Clears all workspace tiles that are not OnHold.
    /// For local tiles, it ensures that a valid copy remains.
    ///
    /// Note that local tiles are currently not released if it would leave all
    /// remaining tiles invalid, but this behavior may change in the future
    /// and should not be relied on.
    void releaseWorkspace()
    {
        storage_->releaseWorkspace();
    }

    void releaseLocalWorkspaceTile( int64_t i, int64_t j );
    void releaseLocalWorkspace();
    void releaseLocalWorkspace( std::set<ij_tuple>& tile_set );

    void releaseRemoteWorkspaceTile( int64_t i, int64_t j );
    void releaseRemoteWorkspace();
    void releaseRemoteWorkspace( std::set<ij_tuple>& tile_set );

    /// Removes all temporary host and device workspace tiles from matrix.
    /// WARNING: currently, this clears the entire parent matrix,
    /// not just a sub-matrix.
    void clearWorkspace()
    {
        storage_->clearWorkspace();
    }

    /// Allocates batch arrays and BLAS++ queues for all devices.
    /// Matrix classes override this with versions that can also allocate based
    /// on the number of local tiles.
    ///
    /// @param[in] batch_size
    ///     Allocate batch arrays as needed so that
    ///     size of each batch array >= batch_size >= 0.
    ///
    /// @param[in] num_arrays
    ///     Allocate batch arrays as needed so that
    ///     number of batch arrays per device >= num_arrays >= 1.
    ///
    void allocateBatchArrays(int64_t batch_size, int64_t num_arrays)
    {
        storage_->allocateBatchArrays(batch_size, num_arrays);
    }

    /// Removes batch arrays from matrix for all devices.
    /// WARNING: currently this clears the entire parent matrix,
    /// not just a sub-matrix.
    void clearBatchArrays()
    {
        storage_->clearBatchArrays();
    }

    /// @return currently allocated batch array size
    int64_t batchArraySize()
    {
        return storage_->batchArraySize();
    }

    //--------------------------------------------------------------------------
    /// @return batch arrays for the A, B, or C matrices,
    /// on host, to send to device
    scalar_t** array_host(int device, int64_t batch_arrays_index=0)
    {
        assert(batch_arrays_index >= 0);
        std::vector< scalar_t** >& array = storage_->array_host_.at(
                                                            batch_arrays_index);
        return array.at(device);
    }

    //--------------------------------------------------------------------------
    /// @return batch arrays for the A, B, or C matrices, on device
    scalar_t** array_device(int device, int64_t batch_arrays_index=0)
    {
        assert(batch_arrays_index >= 0);
        std::vector< scalar_t** >& array = storage_->array_dev_.at(
                                                            batch_arrays_index);
        return array.at(device);
    }

    //--------------------------------------------------------------------------
    /// @return BLAS++ communication queues
    ///
    /// @param[in] device
    ///     Tile's device ID.
    ///
    lapack::Queue* comm_queue(int device)
    {
        return storage_->comm_queues_.at(device);
    }

    //--------------------------------------------------------------------------
    /// @return BLAS++ compute queues
    ///
    /// @param[in] device
    ///     Tile's device ID
    ///
    /// @param[in] queue_index
    ///     The index of a specific set of queues
    ///
    lapack::Queue* compute_queue(int device, int queue_index=0)
    {
        assert((queue_index >= 0) &&
               (queue_index < int(storage_->compute_queues_.size())));
        return storage_->compute_queues_.at(queue_index).at(device);
    }

    //--------------------------------------------------------------------------
    /// @return number of allocated BLAS++ compute queues
    ///
    int numComputeQueues()
    {
        return int(storage_->compute_queues_.size());
    }

protected:
    std::tuple<int64_t, int64_t>
        globalIndex(int64_t i, int64_t j) const;

    std::tuple<int64_t, int64_t, int>
        globalIndex(int64_t i, int64_t j, int device) const;

    /// row offset of first block row.
    int64_t row0_offset() const { return row0_offset_; }

    /// col offset of first block col.
    int64_t col0_offset() const { return col0_offset_; }

    /// rows in last block row.
    int64_t last_mb() const { return last_mb_; }

    /// cols in last block col.
    int64_t last_nb() const { return last_nb_; }

    /// block row offset with respect to original matrix
    int64_t ioffset() const { return ioffset_; }

    /// block col offset with respect to original matrix
    int64_t joffset() const { return joffset_; }

private:
    //--------------------------------------------------------------------------
    int64_t row0_offset_;  ///< row offset in first block row
    int64_t col0_offset_;  ///< col offset in first block col
    int64_t last_mb_;      ///< size of last block row
    int64_t last_nb_;      ///< size of last block col
    int64_t ioffset_;   ///< block row offset with respect to original matrix
    int64_t joffset_;   ///< block col offset with respect to original matrix
    int64_t mt_;        ///< number of local block rows in this view
    int64_t nt_;        ///< number of local block cols in this view
    int64_t nprow_;     ///< number of process rows if 2D block cyclic
    int64_t npcol_;     ///< number of process cols if 2D block cyclic
    GridOrder order_;   ///< order to map MPI processes to tile grid

protected:
    Uplo uplo_;         ///< upper or lower storage
    Op op_;             ///< transpose operation with respect to original matrix
    Layout layout_;     ///< intended layout of the matrix. defaults to ColMajor.
    Target origin_;     ///< Tile origins. defaults to Host


    /// shared storage of tiles and buffers
    std::shared_ptr< MatrixStorage<scalar_t> > storage_;

    // ----- consider where to put these, here or in MatrixStorage
    static int num_devices_;

    MPI_Comm  mpi_comm_;
    MPI_Group mpi_group_;
    int mpi_rank_;
};

//------------------------------------------------------------------------------
/// [internal]
/// Default constructor creates an empty matrix.
/// Does not allocate any memory.
///
template <typename scalar_t>
BaseMatrix<scalar_t>::BaseMatrix()
    : row0_offset_(0),
      col0_offset_(0),
      last_mb_(0),
      last_nb_(0),
      ioffset_(0),
      joffset_(0),
      mt_(0),
      nt_(0),
      nprow_(-1),
      npcol_(-1),
      order_( GridOrder::Col ),
      uplo_(Uplo::General),
      op_(Op::NoTrans),
      layout_(Layout::ColMajor),
      origin_(Target::Host),
      storage_(nullptr),
      mpi_comm_( MPI_COMM_SELF )
{}

//------------------------------------------------------------------------------
/// [internal]
/// Construct matrix with mt block rows and nt block columns, such that
///     sum_{i = 0}^{mt-1} tileMb(i) >= m,
///     sum_{j = 0}^{nt-1} tileNb(j) >= n,
/// where tileMb, tileNb, tileRank, tileDevice are given as functions.
/// No tiles are allocated. Creates empty matrix storage.
///
/// @param[in] m
///     Number of rows of the matrix. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix. n >= 0
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
///     nprow * npcol <= MPI_Comm_size( mpi_comm ).
///
template <typename scalar_t>
BaseMatrix<scalar_t>::BaseMatrix(
    int64_t m, int64_t n,
    std::function<int64_t (int64_t i)>& inTileMb,
    std::function<int64_t (int64_t j)>& inTileNb,
    std::function<int (ij_tuple ij)>& inTileRank,
    std::function<int (ij_tuple ij)>& inTileDevice,
    MPI_Comm mpi_comm)
    : row0_offset_(0),
      col0_offset_(0),
      ioffset_(0),
      joffset_(0),
      nprow_(-1),
      npcol_(-1),
      order_( GridOrder::Unknown ),
      uplo_(Uplo::General),
      op_(Op::NoTrans),
      layout_(Layout::ColMajor),
      origin_(Target::Host),
      storage_(std::make_shared< MatrixStorage< scalar_t > >(
          inTileMb, inTileNb, inTileRank, inTileDevice, mpi_comm)),
      mpi_comm_(mpi_comm)
{
    // Count number of block rows.
    mt_ = 0;
    int64_t ii = 0;  // row index (not block row)
    while (ii < m) {
        last_mb_ = std::min(inTileMb(mt_), m - ii);
        assert(last_mb_ != 0);
        ii += last_mb_;
        ++mt_;
    }

    // Count number of block cols.
    nt_ = 0;
    int64_t jj = 0;  // col index (not block col)
    while (jj < n) {
        last_nb_ = std::min(inTileNb(nt_), n - jj);
        assert(last_nb_ != 0);
        jj += last_nb_;
        ++nt_;
    }

    slate_mpi_call(
        MPI_Comm_rank(mpi_comm_, &mpi_rank_));
    slate_mpi_call(
        MPI_Comm_group(mpi_comm_, &mpi_group_));

    // todo: these are static, but we (re-)initialize with each matrix.
    // todo: similar code in BaseMatrix(...) and MatrixStorage(...)
    num_devices_ = storage_->num_devices_;
}

//------------------------------------------------------------------------------
/// [internal]
/// Construct matrix with
///     mt = ceil( m / mb ) block rows and
///     nt = ceil( n / nb ) block columns,
/// with fixed mb-by-nb tile size and 2D block cyclic distribution.
/// No tiles are allocated. Creates empty matrix storage.
///
/// @param[in] m
///     Number of rows of the matrix. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix. n >= 0
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
/// @param[in] nprow
///     Number of process rows in 2D block-cyclic distribution. nprow > 0.
///
/// @param[in] npcol
///     Number of process cols of 2D block-cyclic distribution. npcol > 0.
///
/// @param[in] mpi_comm
///     MPI communicator to distribute matrix across.
///     nprow * npcol <= MPI_Comm_size( mpi_comm ).
///
template <typename scalar_t>
BaseMatrix<scalar_t>::BaseMatrix(
    int64_t m, int64_t n, int64_t mb, int64_t nb,
    GridOrder order, int nprow, int npcol, MPI_Comm mpi_comm)
    : row0_offset_(0),
      col0_offset_(0),
      last_mb_(m % mb == 0 ? mb : m % mb),
      last_nb_(n % nb == 0 ? nb : n % nb),
      ioffset_(0),
      joffset_(0),
      mt_(ceildiv(m, mb)),
      nt_(ceildiv(n, nb)),
      nprow_(nprow),
      npcol_(npcol),
      order_(order),
      uplo_(Uplo::General),
      op_(Op::NoTrans),
      layout_(Layout::ColMajor),
      origin_(Target::Host),
      storage_(std::make_shared< MatrixStorage< scalar_t > >(
          m, n, mb, nb, order, nprow, npcol, mpi_comm)),
      mpi_comm_(mpi_comm)
{
    slate_mpi_call(
        MPI_Comm_rank(mpi_comm_, &mpi_rank_));
    slate_mpi_call(
        MPI_Comm_group(mpi_comm_, &mpi_group_));

    // todo: these are static, but we (re-)initialize with each matrix.
    // todo: similar code in BaseMatrix(...) and MatrixStorage(...)
    num_devices_ = storage_->num_devices_;
}

//------------------------------------------------------------------------------
/// [internal]
/// Sub-matrix constructor
/// creates shallow copy view of parent matrix, B[ i1:i2, j1:j2 ].
/// B[ 0:mt-1, 0:nt-1 ] is the entire B matrix.
/// If i2 < i1 or j2 < j1, produces empty matrix.
/// See Matrix::sub().
///
/// @param[in,out] B
///     Parent matrix B.
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
BaseMatrix<scalar_t>::BaseMatrix(
    BaseMatrix<scalar_t>& B,
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
    : BaseMatrix(B)
{
    initSubmatrix(i1, i2, j1, j2);
}

//------------------------------------------------------------------------------
/// [private]
/// Sub-matrix constructor implementation used by
/// BaseMatrix( orig, i1, i2, j1, j2 ) and
/// BaseMatrix( orig, Slice( row1, row2, col1, col2 ) ).
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::initSubmatrix(
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
{
    assert((0 <= i1 && i1 < mt()) || (i1 > i2));  // todo: remove || ... clause?
    assert((0 <= i2 && i2 < mt()) || (i1 > i2));
    assert((0 <= j1 && j1 < nt()) || (j1 > j2));  // todo: remove || ... clause?
    assert((0 <= j2 && j2 < nt()) || (j1 > j2));

    const int64_t izero = 0;

    // adjust i2, j2 for empty matrix
    i2 = std::max(i2, i1 - 1);
    j2 = std::max(j2, j1 - 1);

    if (op_ == Op::NoTrans) {
        last_mb_ = tileMb(std::max(i2, izero));
        last_nb_ = tileNb(std::max(j2, izero));
        ioffset_ += i1;
        joffset_ += j1;
        mt_ = i2 - i1 + 1;
        nt_ = j2 - j1 + 1;
        // only if first tile remains in sub we have to keep row/col offset
        if (i1 > 0) {
            row0_offset_ = 0;
        }
        if (j1 > 0) {
            col0_offset_ = 0;
        }
    }
    else {
        last_nb_ = tileMb(std::max(i2, izero));
        last_mb_ = tileNb(std::max(j2, izero));
        ioffset_ += j1;
        joffset_ += i1;
        mt_ = j2 - j1 + 1;
        nt_ = i2 - i1 + 1;
        // only if first tile remains in sub we have to keep row/col offset
        if (j1 > 0) {
            row0_offset_ = 0;
        }
        if (i1 > 0) {
            col0_offset_ = 0;
        }
    }
}

//------------------------------------------------------------------------------
/// [internal]
/// Sliced matrix constructor
/// creates shallow copy view of parent matrix, B[ row1:row2, col1:col2 ].
/// This takes row & col indices instead of block row & block col indices.
/// B[ 0:m-1, 0:n-1 ] is the entire B matrix.
/// If row2 < row1 or col2 < col1, produces empty matrix.
/// See Matrix::slice().
///
/// @param[in,out] B
///     Parent matrix B.
///
/// @param[in] slice
///     Start and end row and column indices:
///
///     - slice.row1: Starting row index. 0 <= row1 < m.
///
///     - slice.row2: Ending row index (inclusive). row2 < m.
///
///     - slice.col1: Starting column index. 0 <= col1 < n.
///
///     - slice.col2: Ending column index (inclusive). col2 < n.
///
template <typename scalar_t>
BaseMatrix<scalar_t>::BaseMatrix(
    BaseMatrix<scalar_t>& B, Slice slice)
    : BaseMatrix(B)
{
    int64_t row1 = slice.row1;
    int64_t row2 = slice.row2;
    int64_t col1 = slice.col1;
    int64_t col2 = slice.col2;
    assert(0 <= row1 && row1 < m());
    assert(0 <= col1 && col1 < n());
    assert(row2 < m());
    assert(col2 < n());

    // Map row indices (row1, row2) => block-row indices (i1, i2).
    int64_t i1 = 0;
    int64_t i2;
    int64_t row = tileMb( i1 );  // past end of tile i1
    while (row <= row1) {
        ++i1;
        row += tileMb( i1 );
    }
    int64_t new_row0_offset = row1 - (row - tileMb( i1 ));  // beginning of tile i1
    i2 = i1;
    while (row <= row2) {
        ++i2;
        row += tileMb( i2 );
    }
    int64_t new_last_mb = row2 - (row - tileMb( i2 )) + 1;
    if (i1 == i2)
        new_last_mb -= new_row0_offset;

    // Map col indices (col1, col2) => block-col indices (j1, j2).
    int64_t j1 = 0;
    int64_t j2;
    int64_t col = tileNb( j1 );  // past end of tile j1
    while (col <= col1) {
        ++j1;
        col += tileNb( j1 );
    }
    int64_t new_col0_offset = col1 - (col - tileNb( j1 ));  // beginning of tile j1
    j2 = j1;
    while (col <= col2) {
        ++j2;
        col += tileNb( j2 );
    }
    int64_t new_last_nb = col2 - (col - tileNb( j2 )) + 1;
    if (j1 == j2)
        new_last_nb -= new_col0_offset;

    // Create sub-matrix of tiles A[ i1:i2, j1:j2 ]
    initSubmatrix(i1, i2, j1, j2);

    // Adjust offsets & sizes of first & last block rows & columns.
    if (op_ == Op::NoTrans) {
        if (i1 == 0)
            new_row0_offset += row0_offset_;
        if (j1 == 0)
            new_col0_offset += col0_offset_;
        initSlice(new_row0_offset, new_col0_offset, new_last_mb, new_last_nb);
    }
    else {
        if (i1 == 0)
            new_row0_offset += col0_offset_;
        if (j1 == 0)
            new_col0_offset += row0_offset_;
        initSlice(new_col0_offset, new_row0_offset, new_last_nb, new_last_mb);
    }
}

//------------------------------------------------------------------------------
/// [private]
/// Used by BaseMatrix( orig, Slice( row1, row2, col1, col2 ) ) constructor to set
/// offsets and sizes for the first and last block row and block col.
/// This ignores transposition, which is handled in the constructor.
///
/// @param[in] row0_offset
///     Row offset in first block-row of A.
///
/// @param[in] col0_offset
///     Col offset in first block-col of A.
///
/// @param[in] last_mb
///     Size of last block-row of A.
///
/// @param[in] last_nb
///     Size of last block-col of A.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::initSlice(
    int64_t row0_offset, int64_t col0_offset,
    int64_t last_mb, int64_t last_nb)
{
    row0_offset_ = row0_offset;
    col0_offset_ = col0_offset;
    last_mb_ = last_mb;
    last_nb_ = last_nb;
}

//------------------------------------------------------------------------------
/// [internal]
///
template <typename scalar_t>
template <typename out_scalar_t>
BaseMatrix<out_scalar_t> BaseMatrix<scalar_t>::baseEmptyLike(
    int64_t mb, int64_t nb, Op deepOp)
{
    // tileMb, tileNb are of A instead of op(A).
    auto newMb = this->storage_->tileMb;
    auto newNb = this->storage_->tileNb;

    // m, n, mt, nt are of op(A).
    int64_t m  = this->m();
    int64_t n  = this->n();
    int64_t mt = this->mt();
    int64_t nt = this->nt();

    // Undo transpose to get m, n, mt, nt, mb, nb of A instead of op(A).
    if (this->op() != Op::NoTrans) {
        std::swap(m, n);
        std::swap(mt, nt);
        std::swap(mb, nb);
    }

    // Override mb, nb if requested.
    if (mb != 0) {
        newMb = [mb](int64_t i) { return mb; };
        m = mb * mt;
    }
    if (nb != 0) {
        newNb = [nb](int64_t j) { return nb; };
        n = nb * nt;
    }

    // Adjust size to include parent matrix outside this sub-matrix.
    int64_t ioffset = this->ioffset();
    int64_t joffset = this->joffset();
    int64_t parent_m = m;
    for (int i = 0; i < ioffset; ++i) {
        parent_m += newMb(i);
    }
    int64_t parent_n = n;
    for (int j = 0; j < joffset; ++j) {
        parent_n += newNb(j);
    }

    // Create new parent matrix B.
    BaseMatrix<out_scalar_t> B;
    if (deepOp == Op::NoTrans) {
        B = BaseMatrix<out_scalar_t>(
            parent_m, parent_n, newMb, newNb,
            this->storage_->tileRank, this->storage_->tileDevice, this->mpiComm());
    }
    else {
        // todo: just swap and redefine newRank? then use above B constructor.
        auto oldRank = this->storage_->tileRank;
        auto newRank = slate::func::transpose_grid( oldRank );
        // todo: what about tileDevice?
        B = BaseMatrix<out_scalar_t>(
            parent_n, parent_m, newNb, newMb,  // transposed
            newRank, this->storage_->tileDevice, this->mpiComm());
        std::swap(ioffset, joffset);
        std::swap(mt, nt);
    }

    B.nprow_ = nprow_;
    B.npcol_ = npcol_;
    B.order_ = order_;

    // Apply operation and return sub-matrix.
    if (this->op() == Op::Trans) {
        B = transpose( B );
        std::swap(ioffset, joffset);
        std::swap(mt, nt);
    }
    else if (this->op() == Op::ConjTrans) {
        B = conj_transpose( B );
        std::swap(ioffset, joffset);
        std::swap(mt, nt);
    }
    B.initSubmatrix(ioffset, ioffset + mt - 1,
                    joffset, joffset + nt - 1);
    return B;
}

//------------------------------------------------------------------------------
/// Swap contents of matrices A and B.
template <typename scalar_t>
void swap(BaseMatrix<scalar_t>& A, BaseMatrix<scalar_t>& B)
{
    using std::swap;
    swap(A.ioffset_, B.ioffset_);
    swap(A.joffset_, B.joffset_);
    swap(A.mt_,      B.mt_);
    swap(A.nt_,      B.nt_);
    swap(A.uplo_,    B.uplo_);
    swap(A.op_,      B.op_);
    swap(A.storage_, B.storage_);
}

//------------------------------------------------------------------------------
/// Get nprow, npcol, myrow, mycol for 2D block cyclic (2DBC) distribution.
/// If SLATE doesn't know the distribution, sets all values to -1.
/// todo: Assumes col-major 2D block cyclic distribution, not row-major.
///
/// @param[out] order
///     Order to map MPI processes to tile grid.
///
/// @param[out] nprow
///     Number of process rows.
///
/// @param[out] npcol
///     Number of process cols.
///
/// @param[out] myrow
///     Process row for this process (MPI rank).
///
/// @param[out] mycol
///     Process col for this process (MPI rank).
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::gridinfo(
    GridOrder* order, int* nprow, int* npcol, int* myrow, int* mycol ) const
{
    if (nprow_ > 0) {
        *order = order_;
        *nprow = nprow_;
        *npcol = npcol_;
        if (order_ == GridOrder::Col) {
            *myrow = mpi_rank_ % nprow_;
            *mycol = mpi_rank_ / nprow_;
        }
        else {
            *myrow = mpi_rank_ / npcol_;
            *mycol = mpi_rank_ % npcol_;
        }
    }
    else {
        int mpi_size;
        slate_mpi_call(
            MPI_Comm_size(mpiComm(), &mpi_size));
        if (mpi_size == 1) {
            *order = GridOrder::Col;
            *nprow = *npcol = 1;
            *myrow = *mycol = 0;
        }
        else {
            *order = GridOrder::Unknown;
            *nprow = *npcol = *myrow = *mycol = -1;
        }
    }
}

//------------------------------------------------------------------------------
/// Get shallow copy of tile {i, j} of op(A) on given device,
/// with the tile's op flag set to match the matrix's.
///
/// @param[in] i
///     Tile's block row index. 0 <= i1 < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's device ID; default is HostNum.
///
/// @return Tile {i, j, device}.
///
// todo: should extended tiles (especially sliced-extended) be handled differently?
template <typename scalar_t>
Tile<scalar_t> BaseMatrix<scalar_t>::operator()(
    int64_t i, int64_t j, int device)
{
    auto tile = *(storage_->at( globalIndex(i, j, device) ));

    // Set op first, before setting offset, mb, nb!
    tile.op(op_);

    // Set row & col offset within first block-row & block-col; before mb, nb!
    if (op_ == Op::NoTrans) {
        tile.offset(i == 0 ? row0_offset_ : 0,
                    j == 0 ? col0_offset_ : 0);
    }
    else {
        tile.offset(i == 0 ? col0_offset_ : 0,
                    j == 0 ? row0_offset_ : 0);
    }

    // Set tile size.
    tile.mb(tileMb(i));
    tile.nb(tileNb(j));

    // Set uplo on diagonal tile (off-diagonal tiles are always general).
    if (i == j)
        tile.uplo(uplo_);

    return tile;
}

//------------------------------------------------------------------------------
/// Returns number of rows in op(A).
template <typename scalar_t>
int64_t BaseMatrix<scalar_t>::m() const
{
    int64_t sum = 0;
    for (int64_t i = 0; i < mt(); ++i)
        sum += tileMb(i);
    return sum;
}

//------------------------------------------------------------------------------
/// Returns number of columns in op(A).
template <typename scalar_t>
int64_t BaseMatrix<scalar_t>::n() const
{
    int64_t sum = 0;
    for (int64_t j = 0; j < nt(); ++j)
        sum += tileNb(j);
    return sum;
}

//------------------------------------------------------------------------------
/// Returns whether op(A) is logically Lower, Upper, or General storage,
/// taking the transposition operation into account.
///
template <typename scalar_t>
Uplo BaseMatrix<scalar_t>::uploLogical() const
{
    if (uplo_ == Uplo::General)
        return uplo_;
    else if ((uplo_ == Uplo::Lower) == (op_ == Op::NoTrans))
        return Uplo::Lower;
    else
        return Uplo::Upper;
}

//------------------------------------------------------------------------------
/// Returns whether A is physically Lower, Upper, or General storage,
/// ignoring the transposition operation.
///
template <typename scalar_t>
Uplo BaseMatrix<scalar_t>::uploPhysical() const
{
    return uplo_;
}

//------------------------------------------------------------------------------
/// Returns number of rows (mb) in block row i of op(A).
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
template <typename scalar_t>
int64_t BaseMatrix<scalar_t>::tileMb(int64_t i) const
{
    if (op_ == Op::NoTrans)
        return tileMbInternal(i);
    else
        return tileNbInternal(i);
}

//------------------------------------------------------------------------------
/// Returns number of cols (nb) in block col j of op(A).
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
template <typename scalar_t>
int64_t BaseMatrix<scalar_t>::tileNb(int64_t j) const
{
    if (op_ == Op::NoTrans)
        return tileNbInternal(j);
    else
        return tileMbInternal(j);
}

//------------------------------------------------------------------------------
/// [internal]
/// Returns number of rows (mb) in block row i of A,
/// ignoring transposition, which is handled in tileMb().
/// This handles the first and last block rows for slicing.
///
/// @param[in] i
///     Tile's block row index.
///
template <typename scalar_t>
int64_t BaseMatrix<scalar_t>::tileMbInternal(int64_t i) const
{
    if (i == mt_ - 1)
        return last_mb_;
    else if (i == 0)
        return storage_->tileMb(ioffset_ + i) - row0_offset_;
    else
        return storage_->tileMb(ioffset_ + i);
}

//------------------------------------------------------------------------------
/// [internal]
/// Returns number of cols (nb) in block col j of A,
/// ignoring transposition, which is handled in tileNb().
/// This handles the first and last block columns for slicing.
///
/// @param[in] j
///     Tile's block column index.
///
template <typename scalar_t>
int64_t BaseMatrix<scalar_t>::tileNbInternal(int64_t j) const
{
    if (j == nt_ - 1)
        return last_nb_;
    else if (j == 0)
        return storage_->tileNb(joffset_ + j) - col0_offset_;
    else
        return storage_->tileNb(joffset_ + j);
}

//------------------------------------------------------------------------------
/// Insert tile {i, j} of op(A) and allocate its data.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's device ID; default is HostNum.
///
/// @return Pointer to new tile.
///
template <typename scalar_t>
Tile<scalar_t> BaseMatrix<scalar_t>::tileInsert(
    int64_t i, int64_t j, int device)
{
    auto index = globalIndex(i, j, device);
    return *(storage_->tileInsert(index, TileKind::SlateOwned, layout_));
}

//------------------------------------------------------------------------------
/// Insert a workspace tile {i, j} of op(A) and allocate its data.
/// The tile will be freed
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's device ID; default is HostNum.
///
/// @return Pointer to new tile.
///
template <typename scalar_t>
Tile<scalar_t> BaseMatrix<scalar_t>::tileInsertWorkspace(
    int64_t i, int64_t j, int device, Layout layout)
{
    auto index = globalIndex(i, j, device);
    return *(storage_->tileInsert(index, TileKind::Workspace, layout));
}

//------------------------------------------------------------------------------
/// Allocates a workspace buffer using the matrix's memory pool.
/// The memory must be freed with BaseMatrix::freeWorkspaceBuffer
///
/// @param[in] device
///     Device ID (GPU or Host) where the memory block is needed.
///
/// @param[in] size
///     The required allocation size
///
/// @return Pointer to the buffer
///
template<typename scalar_t>
scalar_t* BaseMatrix<scalar_t>::allocWorkspaceBuffer(int device, int64_t size)
{
    assert(size <= storage_->tileMb(0)*storage_->tileNb(0));
    return storage_->allocWorkspaceBuffer(device);
}

//------------------------------------------------------------------------------
/// Frees a workspace buffer allocated with BaseMatrix::allocWorkspaceBuffer.
///
/// @param[in] device
///     Device ID (GPU or Host) where the memory block is needed.
///
/// @param[in] buffer
///     Pointer to the buffer
///
template<typename scalar_t>
void BaseMatrix<scalar_t>::freeWorkspaceBuffer(int device, scalar_t* buffer)
{
    storage_->releaseWorkspaceBuffer(buffer, device);
}

//------------------------------------------------------------------------------
/// Insert tile {i, j} of op(A) with existing data.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's device ID; default is HostNum (provided by overloaded function).
///
/// @param[in,out] data
///     Tile's data. The matrix uses this pointer directly, it does not copy
///     the data, so the data must remain valid while the matrix exists.
///
/// @param[in] ld
///     Leading dimension of data; column stride. ld >= tileMb(i).
///
/// @return Pointer to new tile.
///
template <typename scalar_t>
Tile<scalar_t> BaseMatrix<scalar_t>::tileInsert(
    int64_t i, int64_t j, int device, scalar_t* data, int64_t ld)
{
    auto index = globalIndex(i, j, device);
    // tile layout must match the matrix layout
    return *(storage_->tileInsert(index, data, ld, layout_)); // TileKind::UserOwned
}

//------------------------------------------------------------------------------
/// Erase tile {i, j} of op(A) on device (host, one device or all devices).
/// If tile's memory was allocated by SLATE,
/// via tileInsert(i, j, dev) or tileInsertWorkspace(i, j, dev),
/// then the memory is released to the allocator pool.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's device ID.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileErase(int64_t i, int64_t j, int device)
{
    if (device == AllDevices) {
        storage_->erase(globalIndex(i, j));
    }
    else {
        // todo: erase only workspace tiles? if so, rename with "Workspace"?
        storage_->erase(globalIndex(i, j, device));
    }
}

//------------------------------------------------------------------------------
/// Erase the tile {i, j}'s instance on device if it is a workspace tile with
/// no hold set on it.
/// If tile's memory was allocated by SLATE,
/// via tileInsert(i, j, dev) or tileInsertWorkspace(i, j, dev),
/// then the memory is released to the allocator pool.
/// For local tiles, it ensures that a valid copy remains.
///
/// Note that local tiles are currently not released if it would leave all
/// remaining tiles invalid, but this behavior may change in the future
/// and should not be relied on.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's device ID.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileRelease(int64_t i, int64_t j, int device)
{
    storage_->release(globalIndex(i, j, device));
}

//------------------------------------------------------------------------------
/// Unsets all local tiles' hold on device.
///
/// @param[in] device
///     Tile's device ID.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileUnsetHoldAll(int device)
{
    for (int64_t j = 0; j < nt(); ++j)
        for (int64_t i = 0; i < mt(); ++i)
            if (tileIsLocal(i, j))
                tileUnsetHold(i, j, device);
}

//------------------------------------------------------------------------------
/// Unsets all local tiles' hold on all devices.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileUnsetHoldAllOnDevices()
{
    #pragma omp parallel for slate_omp_default_none
    for (int64_t j = 0; j < nt(); ++j)
        for (int64_t i = 0; i < mt(); ++i)
            if (tileIsLocal(i, j))
                tileUnsetHold(i, j, tileDevice(i, j));
}


//------------------------------------------------------------------------------
/// Marks tile(i, j) as Modified on device.
/// Other instances will be invalidated.
/// Unless permissive, asserts if other instances are in Modified state.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's device ID, defaults to host.
///
/// @param[in] permissive
///     Defaults to false.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileModified(int64_t i, int64_t j, int device, bool permissive)
{
    auto& tile_node = storage_->at(globalIndex(i, j));

    LockGuard guard(tile_node.getLock());

    auto tile = tile_node[device];

    // if no need to update
    if (tile->stateOn(MOSI::Modified))
        return;

    tile->state(MOSI::Modified);

    for (int d = HostNum; d < num_devices(); ++d) {
        if (d != device && tile_node.existsOn(d)) {
            if (! permissive)
                slate_assert(tile_node[d]->stateOn(MOSI::Modified) == false);
            tile_node[d]->state(MOSI::Invalid);
        }
    }
}

//------------------------------------------------------------------------------
/// Send tile {i, j} of op(A) to the given MPI rank.
/// Destination rank must call tileRecv().
///
/// @tparam target
///     Destination to target; either Host (default) or Device.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] dst_rank
///     Destination MPI rank. If dst_rank == mpiRank, this is a no-op.
///
/// @param[in] tag
///     MPI tag, default 0.
///
template <typename scalar_t>
template <Target target>
void BaseMatrix<scalar_t>::tileSend(
    int64_t i, int64_t j, int dst_rank, int tag)
{
    if (dst_rank != mpiRank()) {
        // todo: need to acquire read access lock to TileNode(i, j)
        tileGetForReading(i, j, LayoutConvert::None);
        at(i, j).send(dst_rank, mpiComm(), tag);
    }
}

//------------------------------------------------------------------------------
/// Immediately send tile {i, j} of op(A) to the given MPI rank.
/// Destination rank must call tileRecv().
///
/// @tparam target
///     Destination to target; either Host (default) or Device.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] dst_rank
///     Destination MPI rank. If dst_rank == mpiRank, this is a no-op.
///
/// @param[in] tag
///     MPI tag, default 0.
///
/// @param[out] request
///     Pointer to an MPI_Request struct
///
template <typename scalar_t>
template <Target target>
void BaseMatrix<scalar_t>::tileIsend(
    int64_t i, int64_t j, int dst_rank, int tag, MPI_Request* request)
{
    if (dst_rank != mpiRank()) {
        //todo: need to qcquire read access lock to TileNode(i, j)
        tileGetForReading(i, j, LayoutConvert::None);
        at(i, j).isend(dst_rank, mpiComm(), tag, request);
    }
    else {
        *request = MPI_REQUEST_NULL;
    }
}

//------------------------------------------------------------------------------
/// Receive tile {i, j} of op(A) to the given MPI rank.
/// Tile is allocated as workspace with life = 1 if it doesn't yet exist,
/// or 1 is added to life if it does exist.
/// Source rank must call tileSend().
/// Data received must be in 'layout' (ColMajor/RowMajor) major.
///
/// @tparam target
///     Destination to target; either Host (default) or Device.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] src_rank
///     Source MPI rank. If src_rank == mpiRank, this is a no-op.
///
/// @param[in] layout
///     Indicates the Layout (ColMajor/RowMajor) of the received data.
///     WARNING: must match the layout of the tile in the sender MPI rank.
///
/// @param[in] tag
///     MPI tag, default 0.
///
template <typename scalar_t>
template <Target target>
void BaseMatrix<scalar_t>::tileRecv(
    int64_t i, int64_t j, int src_rank, Layout layout, int tag)
{
    if (src_rank != mpiRank()) {
        storage_->tilePrepareToReceive( globalIndex( i, j ), 1, layout );
        tileAcquire(i, j, layout);

        // Receive data.
        at(i, j).recv(src_rank, mpiComm(), layout, tag);

        tileModified( i, j, HostNum, true );

        // Copy to devices.
        if (target == Target::Devices) {
            #pragma omp task slate_omp_default_none \
                firstprivate( i, j )
            {
                tileGetForReading(i, j, tileDevice(i, j), LayoutConvert::None);
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Receive tile {i, j} of op(A) to the given MPI rank using immediate mode..
/// Tile is allocated as workspace with life = 1 if it doesn't yet exist,
/// or 1 is added to life if it does exist.
/// Source rank must call tileSend().
/// Data received must be in 'layout' (ColMajor/RowMajor) major.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] src_rank
///     Source MPI rank. If src_rank == mpiRank, this is a no-op.
///
/// @param[in] layout
///     Indicates the Layout (ColMajor/RowMajor) of the received data.
///     WARNING: must match the layout of the tile in the sender MPI rank.
///
/// @param[in] tag
///     MPI tag
///
///
/// @param[out] request
///     MPI request object
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileIrecv(
    int64_t i, int64_t j, int src_rank, Layout layout, int tag, MPI_Request* request)
{
    if (src_rank != mpiRank()) {
        storage_->tilePrepareToReceive( globalIndex( i, j ), 1, layout );
        tileAcquire(i, j, layout);

        // Receive data.
        at(i, j).irecv(src_rank, mpiComm(), layout, tag, request);

        tileModified(i, j, HostNum, true);
    }
    else {
        *request = MPI_REQUEST_NULL;
    }
}

//------------------------------------------------------------------------------
/// Send tile {i, j} of op(A) to all MPI ranks in matrix B.
/// If target is Devices, also copies tile to all devices on each MPI rank.
/// This should be called by at least all ranks with local tiles in B;
/// ones that do not have any local tiles are excluded from the broadcast.
/// Data received must be in 'layout' (ColMajor/RowMajor) major.
///
/// @tparam target
///     Destination to target; either Host (default) or Device.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] B
///     Sub-matrix B defines the MPI ranks to send to.
///     Usually it is the portion of the matrix to be updated by tile {i, j}.
///
/// @param[in] layout
///     Indicates the Layout (ColMajor/RowMajor) of the broadcasted data.
///     WARNING: must match the layout of the tile in the sender MPI rank.
///
/// @param[in] tag
///     MPI tag, default 0.
///
/// @param[in] life_factor
///     A multiplier for the life count of the broadcasted tile workspace.
///
template <typename scalar_t>
template <Target target>
void BaseMatrix<scalar_t>::tileBcast(
    int64_t i, int64_t j, BaseMatrix<scalar_t> const& B, Layout layout, int tag, int64_t life_factor)
{
    BcastList bcast_list_B;
    bcast_list_B.push_back({i, j, {B}});
    listBcast<target>(bcast_list_B, layout, tag, life_factor);
}

//------------------------------------------------------------------------------
/// Send tile {i, j} of op(A) to all MPI ranks in the list of submatrices
/// bcast_list.
/// Data received must be in 'layout' (ColMajor/RowMajor) major.
///
/// @tparam target
///     Destination to target; either Host (default) or Device.
///
/// @param[in] bcast_list
///     List of submatrices defining the MPI ranks to send to.
///     Usually it is the portion of the matrix to be updated by tile {i, j}.
///
/// @param[in] layout
///     Indicates the Layout (ColMajor/RowMajor) of the broadcasted data.
///     WARNING: must match the layout of the tile in the sender MPI rank.
///
/// @param[in] tag
///     MPI tag, default 0.
///
/// @param[in] life_factor
///     A multiplier for the life count of the broadcasted tile workspace.
/// @param[in] is_shared
///     A flag to get and hold the broadcasted (prefetched) tiles on the
///     devices. This flag prevents any subsequent calls of tileRelease()
///     routine to release these tiles (clear up the devices memories).
///     WARNING: must set unhold these tiles before releasing them to free
///     up the allocated memories.
///
template <typename scalar_t>
template <Target target>
void BaseMatrix<scalar_t>::listBcast(
    BcastList& bcast_list, Layout layout,
    int tag, int64_t life_factor, bool is_shared)
{
    if (target == Target::Devices) {
        assert(num_devices() > 0);
    }

    // It is possible that the same tile, with the same data, is sent twice.
    // This happens, e.g., in the hemm and symm routines, where the same tile
    // is sent once as part of A and once as part of A^T.
    // This cannot be avoided without violating the upper bound on the buffer size
    // used for hosting communicated tiles.
    // Due to dynamic scheduling, the second communication may occur before the
    // first tile has been discarded.
    // If that happens, instead of creating the tile, the life of the existing
    // tile is increased.
    // Also, currently, the message is received to the same buffer.

    std::vector< std::set<ij_tuple> > tile_set(num_devices());
    int mpi_size;
    MPI_Comm_size(mpiComm(), &mpi_size);

    std::vector<MPI_Request> send_requests;

    for (auto bcast : bcast_list) {

        auto i = std::get<0>(bcast);
        auto j = std::get<1>(bcast);
        auto submatrices_list = std::get<2>(bcast);

        int64_t life = 0;
        for (auto submatrix : submatrices_list) {
            life += submatrix.numLocalTiles() * life_factor;
        }

        // Find the set of participating ranks.
        std::set<int> bcast_set;
        bcast_set.insert(tileRank(i, j));       // Insert root.
        for (auto submatrix : submatrices_list) // Insert destinations.
            submatrix.getRanks(&bcast_set);

        // If this rank is in the set.
        if (bcast_set.find(mpi_rank_) != bcast_set.end()) {

            // If receiving the tile.
            storage_->tilePrepareToReceive( globalIndex( i, j ), life, layout_ );

            // Send across MPI ranks.
            // Previous used MPI bcast: tileBcastToSet(i, j, bcast_set);
            // Currently uses 2D hypercube p2p send.
            tileIbcastToSet(i, j, bcast_set, 2, tag, layout, send_requests, target);
        }

        // Copy to devices.
        // TODO: should this be inside above if-then?
        // todo: this may incur extra communication,
        //       tile(i,j) is not necessarily needed on all devices where this matrix resides
        if (target == Target::Devices) {
            std::set<int> dev_set;
            for (auto submatrix : submatrices_list)
                submatrix.getLocalDevices(&dev_set);

            if (mpi_size == 1) {
                for (auto device : dev_set)
                    tile_set[device].insert({i, j});
            }
            else {
                #pragma omp taskgroup
                for (auto device : dev_set) {
                    // note: dev_set structure is released after the if-target block
                    #pragma omp task slate_omp_default_none \
                        firstprivate( i, j, device, is_shared )
                    {
                        if (is_shared) {
                            tileGetAndHold(i, j, device, LayoutConvert::None);
                        }
                        else {
                            tileGetForReading(i, j, device, LayoutConvert::None);
                        }
                    }
                }
            }
        }
    }

    if (target == Target::Devices) {
        if (mpi_size == 1) {
            #pragma omp taskgroup
            for (int d = 0; d < num_devices(); ++d) {
                if (! tile_set[d].empty()) {
                    #pragma omp task slate_omp_default_none \
                        firstprivate( d, is_shared ) shared( tile_set )
                    {
                        if (is_shared) {
                            tileGetAndHold(tile_set[d], d, LayoutConvert::None);
                        }
                        else {
                            tileGetForReading(tile_set[d], d, LayoutConvert::None);
                        }
                    }
                }
            }
        }
    }
    slate_mpi_call(
        MPI_Waitall(send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE));
}

//------------------------------------------------------------------------------
/// Send tile {i, j} of op(A) to all MPI ranks in the list of submatrices
/// bcast_list (using OpenMP tasksloop and multi-threaded MPI).
/// Data received must be in 'layout' (ColMajor/RowMajor) major.
///
/// @tparam target
///     Destination to target; either Host (default) or Device.
///
/// @param[in] bcast_list
///     List of submatrices defining the MPI ranks to send to.
///     Usually it is the portion of the matrix to be updated by tile {i, j}.
///     Each tile {i, j} to be broadcast has a tag in the bcast_list.
///
/// @param[in] layout
///     Indicates the Layout (ColMajor/RowMajor) of the broadcasted data.
///     WARNING: must match the layout of the tile in the sender MPI rank.
///
/// @param[in] life_factor
///     A multiplier for the life count of the broadcasted tile workspace.
///
/// @param[in] is_shared
///     A flag to get and hold the broadcasted (prefetched) tiles on the
///     devices. This flag prevents any subsequent calls of tileRelease()
///     routine to release these tiles (clear up the devices memories).
///     WARNING: must set unhold these tiles before releasing them to free
///     up the allocated memories.
///
template <typename scalar_t>
template <Target target>
void BaseMatrix<scalar_t>::listBcastMT(
    BcastListTag& bcast_list, Layout layout,
    int64_t life_factor, bool is_shared)
{
    if (target == Target::Devices) {
        assert(num_devices() > 0);
    }

    // It is possible that the same tile, with the same data, is sent twice.
    // This happens, e.g., in the hemm and symm routines, where the same tile
    // is sent once as part of A and once as part of A^T.
    // This cannot be avoided without violating the upper bound on the buffer size
    // used for hosting communicated tiles.
    // Due to dynamic scheduling, the second communication may occur before the
    // first tile has been discarded.
    // If that happens, instead of creating the tile, the life of the existing
    // tile is increased.
    // Also, currently, the message is received to the same buffer.

    int mpi_size;
    MPI_Comm_size(mpiComm(), &mpi_size);

    // This uses multiple OMP threads for MPI broadcast communication
    // todo: threads may clash with panel-threads slowing performance
    // for multi-threaded panel routines
    using BcastTag =
        std::tuple< int64_t, int64_t, std::list<BaseMatrix<scalar_t> >, int64_t >;

    #if defined( SLATE_HAVE_MT_BCAST )
        #pragma omp taskloop slate_omp_default_none \
            shared( bcast_list ) \
            firstprivate(life_factor, layout, mpi_size, is_shared)
    #endif
    for (size_t bcastnum = 0; bcastnum < bcast_list.size(); ++bcastnum) {

        BcastTag bcast = bcast_list[bcastnum];
        auto i = std::get<0>(bcast);
        auto j = std::get<1>(bcast);
        auto submatrices_list = std::get<2>(bcast);
        auto tagij = std::get<3>(bcast);
        int tag = int(tagij) % 32768;  // MPI_TAG_UB is at least 32767

        int64_t life = 0;
        for (auto submatrix : submatrices_list) {
            life += submatrix.numLocalTiles() * life_factor;
        }

        {
            trace::Block trace_block(
                std::string("listBcast("+std::to_string(i)+","+std::to_string(j)+")").c_str());

            // Find the set of participating ranks.
            std::set<int> bcast_set;
            bcast_set.insert(tileRank(i, j));       // Insert root.
            for (auto submatrix : submatrices_list) // Insert destinations.
                submatrix.getRanks(&bcast_set);

            // If this rank is in the set.
            if (bcast_set.find(mpi_rank_) != bcast_set.end()) {
                // If receiving the tile.
                storage_->tilePrepareToReceive( globalIndex( i, j ), life, layout_ );

                // Send across MPI ranks.
                // Previous used MPI bcast: tileBcastToSet(i, j, bcast_set);
                // Currently uses radix-D hypercube p2p send.
                int radix = 4; // bcast_set.size(); // 2;
                tileBcastToSet(i, j, bcast_set, radix, tag, layout, target);
            }

            // Copy to devices.
            // todo: should this be inside above if-then?
            // todo: this may incur extra communication,
            //       tile(i,j) is not necessarily needed on all devices where this matrix resides
            if (target == Target::Devices) {
                std::set<int> dev_set;
                for (auto submatrix : submatrices_list)
                    submatrix.getLocalDevices(&dev_set);

                // #pragma omp taskgroup
                for (auto dev : dev_set) {
                    //todo: test #pragma omp task default(none) firstprivate(i,j,dev,is_shared) if (mpi_size == 1)
                    if (is_shared)
                        tileGetAndHold(i, j, dev, LayoutConvert::None);
                    else
                        tileGetForReading(i, j, dev, LayoutConvert::None);
                }
            } // paren added for the trace_block label
        }
    }
}

//------------------------------------------------------------------------------
///
template <typename scalar_t>
template <Target target>
void BaseMatrix<scalar_t>::listReduce(ReduceList& reduce_list, Layout layout, int tag)
{
    for (auto reduce : reduce_list) {

        auto i = std::get<0>(reduce);
        auto j = std::get<1>(reduce);
        auto submatrices_dest = std::get<2>(reduce);
        auto submatrices_list = std::get<3>(reduce);

        // Find the set of participating ranks.
        std::set<int> reduce_set;
        int root_rank = submatrices_dest.tileRank(0, 0);
        for (auto submatrix : submatrices_list) // Insert sources.
            submatrix.getRanks(&reduce_set);

        // If this rank is in the set.
        if (root_rank == mpi_rank_
            || reduce_set.find(mpi_rank_) != reduce_set.end()) {

            // Reduce across MPI ranks.
            // Uses 2D hypercube p2p send.
            tileReduceFromSet(i, j, root_rank, reduce_set, 2, tag, layout);

            // If not the tile owner.
            if (! tileIsLocal(i, j)) {

                // todo: should we check its life count before erasing?
                // Destroy the tile.
                // todo: should it be a tileRelease()?
                if (mpi_rank_ != root_rank)
                    tileErase( i, j, HostNum );
            }
            else if (root_rank == mpi_rank_ && reduce_set.size() > 1) {
                tileModified( i, j );
            }
        }
    }
}

//------------------------------------------------------------------------------
/// @deprecated
/// [internal]
/// Broadcast tile {i, j} to all MPI ranks in the bcast_set.
/// This should be called by all (and only) ranks that are in bcast_set,
/// as either the root sender or a receiver.
/// This implementation creates a subcommunicator and calls MPI broadcast.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] bcast_set
///     Set of MPI ranks to broadcast to.
///
// todo: use the commFromSet() function from slate_internal_comm.cc
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileBcastToSet(
    int64_t i, int64_t j, std::set<int> const& bcast_set)
{
    // this function does not use tags, kept for reference
    assert(false);  // This variant of tileBcastToSet() is obsolete

    // Quit if only root in the broadcast set.
    if (bcast_set.size() == 1)
        return;

    // Convert the set of ranks to a vector.
    std::vector<int> bcast_vec(bcast_set.begin(), bcast_set.end());

    // Create the broadcast group.
    MPI_Group bcast_group;
    #pragma omp critical(slate_mpi)
    slate_mpi_call(
        MPI_Group_incl(mpi_group_, bcast_vec.size(), bcast_vec.data(),
                       &bcast_group));

    // Create a broadcast communicator.
    int tag = 0;
    MPI_Comm bcast_comm;
    #pragma omp critical(slate_mpi)
    {
        trace::Block trace_block("MPI_Comm_create_group");
        slate_mpi_call(
            MPI_Comm_create_group(mpi_comm_, bcast_group, tag, &bcast_comm));
    }
    assert(bcast_comm != MPI_COMM_NULL);

    // Find the broadcast rank.
    int bcast_rank;
    #pragma omp critical(slate_mpi)
    MPI_Comm_rank(bcast_comm, &bcast_rank);

    // Find the broadcast root rank.
    int root_rank = tileRank(i, j);
    int bcast_root;
    #pragma omp critical(slate_mpi)
    slate_mpi_call(
        MPI_Group_translate_ranks(mpi_group_, 1, &root_rank,
                                  bcast_group, &bcast_root));

    // Do the broadcast.
    at(i, j).bcast(bcast_root, bcast_comm);

    // Free the group.
    #pragma omp critical(slate_mpi)
    slate_mpi_call(
        MPI_Group_free(&bcast_group));

    // Free the communicator.
    #pragma omp critical(slate_mpi)
    slate_mpi_call(
        MPI_Comm_free(&bcast_comm));
}

//------------------------------------------------------------------------------
/// [internal]
/// Broadcast tile {i, j} to all MPI ranks in the bcast_set.
/// This should be called by all (and only) ranks that are in bcast_set,
/// as either the root sender or a receiver.
/// This function implements a custom pattern using sends and receives.
/// Data received must be in 'layout' (ColMajor/RowMajor) major.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] bcast_set
///     Set of MPI ranks to broadcast to.
///
/// @param[in] radix
///     Radix of the communication pattern.
///
/// @param[in] tag
///     MPI tag, default 0.
///
/// @param[in] layout
///     Indicates the Layout (ColMajor/RowMajor) of the received data.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileBcastToSet(
    int64_t i, int64_t j, std::set<int> const& bcast_set,
    int radix, int tag, Layout layout, Target target)
{
    std::vector<MPI_Request> requests;
    requests.reserve(radix);

    tileIbcastToSet(i, j, bcast_set, radix, tag, layout, requests, target);
    slate_mpi_call(MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE));
}

//------------------------------------------------------------------------------
/// [internal]
/// Broadcast tile {i, j} to all MPI ranks in the bcast_set.
/// This should be called by all (and only) ranks that are in bcast_set,
/// as either the root sender or a receiver.
/// This function implements a custom pattern using sends and receives.
/// Data received must be in 'layout' (ColMajor/RowMajor) major.
/// Nonblocking sends are used, with requests appended to the provided vector.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] bcast_set
///     Set of MPI ranks to broadcast to.
///
/// @param[in] radix
///     Radix of the communication pattern.
///
/// @param[in] tag
///     MPI tag, default 0.
///
/// @param[in] layout
///     Indicates the Layout (ColMajor/RowMajor) of the received data.
///
/// @param[in,out] send_requests
///     Vector where requests for this bcast are appended.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileIbcastToSet(
    int64_t i, int64_t j, std::set<int> const& bcast_set,
    int radix, int tag, Layout layout,
    std::vector<MPI_Request>& send_requests,
    Target target)
{
    // Quit if only root in the broadcast set.
    if (bcast_set.size() == 1)
        return;

    // Convert the set to a vector.
    std::vector<int> bcast_vec(bcast_set.begin(), bcast_set.end());

    // TODO: std::set is already sorted (it's really an ordered_set), no need to sort again?
    // Sort the ranks.
    std::sort(bcast_vec.begin(), bcast_vec.end());

    // Find root.
    int root_rank = tileRank(i, j);
    auto root_iter = std::find(bcast_vec.begin(), bcast_vec.end(), root_rank);

    // Shift root to position zero.
    std::vector<int> new_vec(root_iter, bcast_vec.end());
    new_vec.insert(new_vec.end(), bcast_vec.begin(), root_iter);

    // Find the new rank.
    auto rank_iter = std::find(new_vec.begin(), new_vec.end(), mpi_rank_);
    int new_rank = std::distance(new_vec.begin(), rank_iter);

    // Get the send/recv pattern.
    std::list<int> recv_from;
    std::list<int> send_to;
    internal::cubeBcastPattern(new_vec.size(), new_rank, radix,
                               recv_from, send_to);

    int device = HostNum;
    if (target == Target::Devices && gpu_aware_mpi()) {
        device = tileDevice( i, j );
    }

    // Receive.
    if (! recv_from.empty()) {
        // read tile
        tileAcquire(i, j, device, layout);

        at(i, j, device).recv(new_vec[recv_from.front()], mpi_comm_, layout, tag);
        tileModified(i, j, device, true);
    }

    if (! send_to.empty()) {
        // read tile
        tileGetForReading(i, j, device, LayoutConvert(layout));

        auto Aij = at(i, j, device);
        // Forward using multiple mpi_isend() calls
        for (int dst : send_to) {
            MPI_Request request;
            Aij.isend(new_vec[dst], mpi_comm_, tag, &request);
            send_requests.push_back(request);
        }
    }
}

//------------------------------------------------------------------------------
/// [internal]
/// WARNING: Sent and Recevied tiles are converted to 'layout' major.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileReduceFromSet(
    int64_t i, int64_t j, int root_rank, std::set<int>& reduce_set,
    int radix, int tag, Layout layout)
{
    const scalar_t one = 1.0;

    // Quit if the reduction set is empty
    if (reduce_set.empty())
        return;

    reduce_set.insert(root_rank);

    // Convert the set to a vector.
    std::vector<int> reduce_vec(reduce_set.begin(), reduce_set.end());

    // Sort the ranks.
    std::sort(reduce_vec.begin(), reduce_vec.end());

    // Find root.
    auto root_iter = std::find(reduce_vec.begin(), reduce_vec.end(), root_rank);

    // Shift root to position zero.
    std::vector<int> new_vec(root_iter, reduce_vec.end());
    new_vec.insert(new_vec.end(), reduce_vec.begin(), root_iter);

    // Find the new rank.
    auto rank_iter = std::find(new_vec.begin(), new_vec.end(), mpi_rank_);
    int new_rank = std::distance(new_vec.begin(), rank_iter);

    // Get the send/recv pattern.
    std::list<int> recv_from;
    std::list<int> send_to;
    internal::cubeReducePattern(new_vec.size(), new_rank, radix,
                                recv_from, send_to);

    if (! (send_to.empty() && recv_from.empty())) {
        // read tile on host memory
        tileGetForReading(i, j, LayoutConvert(layout));

        auto Aij = at(i, j);

        std::vector<scalar_t> data(Aij.mb() * Aij.nb());
        int64_t lda = (Aij.op() == Op::NoTrans ? Aij.mb() : Aij.nb());
        Tile<scalar_t> tile(Aij, &data[0], lda, TileKind::Workspace);

        // Receive, accumulate.
        for (int src : recv_from) {
            // Receive.
            tile.recv(new_vec[src], mpi_comm_, layout, tag);
            tileGetForWriting(i, j, LayoutConvert(layout));
            // Accumulate.
            tile::add( one, tile, Aij );
        }

        // Forward.
        if (! send_to.empty())
            Aij.send(new_vec[send_to.front()], mpi_comm_, tag);
    }
}

//------------------------------------------------------------------------------
/// [internal]
/// Copies data from src_tile into dst_tile and converts into target_layout
/// Handles all cases of:
///     - Square tiles.
///     - Rectangular contiguous tiles.
///     - Rectangular extended tiles.
/// assumes at least one of src_tile and dst_tile is device resident
/// attempts to make layout conversion on the device whenever possible
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileCopyDataLayout(Tile<scalar_t>* src_tile,
                                              Tile<scalar_t>* dst_tile,
                                              Layout target_layout,
                                              bool async)
{
    int64_t mb = src_tile->mb();
    int64_t nb = src_tile->nb();
    bool is_square = mb == nb;

    int src_device = src_tile->device();
    int dst_device = dst_tile->device();
    int work_device = ( dst_device == HostNum ? src_device : dst_device );

    Layout src_layout = src_tile->layout();
    bool need_convert = src_layout != target_layout;

    // A race condition can occur when doing GPU->GPU transfers when async
    // TODO Consider inter-Queue dependencies instead of disabling async
    async &= !(dst_device != HostNum && src_device != HostNum);


    if (is_square || ! need_convert) {
        lapack::Queue* queue = comm_queue( work_device );
        src_tile->copyData( dst_tile, *queue, async );

        if (need_convert) {
            // NB. if dst is a host tile, the queue is ignored
            dst_tile->layoutConvert( *queue, async );
        }
    }
    else {
        if (dst_tile->layout() != target_layout && ! dst_tile->isTransposable()) {
            storage_->tileMakeTransposable( dst_tile );
        }
        dst_tile->setLayout( target_layout );

        scalar_t* work_data = nullptr;
        int64_t work_stride = -1;
        // Look for a spare buffer that can be borrowed
        if (dst_tile->extended() && dst_device != HostNum) {
            work_data = dst_tile->layoutBackData();
            work_stride = dst_tile->layoutBackStride();
        }
        else if (src_tile->extended() && src_device != HostNum) {
            // This is the one device->device case that uses src_device
            work_device = src_device;
            work_data = src_tile->layoutBackData();
            work_stride = src_tile->layoutBackStride();
        }

        lapack::Queue* queue = comm_queue( work_device );
        bool copy_first = ( work_device == dst_device );
        int64_t phys_mb = ( src_layout == Layout::ColMajor ? mb : nb );
        int64_t phys_nb = ( src_layout == Layout::ColMajor ? nb : mb );
        int64_t dst_stride = ( target_layout == Layout::ColMajor ? mb : nb );

        bool need_workspace = work_data == nullptr;
        if (need_workspace) {
            work_data = storage_->allocWorkspaceBuffer( work_device );
            work_stride = ( copy_first ? phys_mb : dst_stride );
        }
        Layout work_layout = ( copy_first ? src_layout : target_layout );
        Tile<scalar_t> work_tile( mb, nb, work_data, work_stride, work_device,
                                  TileKind::Workspace, work_layout);

        if (copy_first) {
            src_tile->copyData( &work_tile, *queue, true );

            device::transpose( false, phys_mb, phys_nb,
                               work_data, work_stride,
                               dst_tile->data(), dst_tile->stride(),
                               *queue );
        }
        else {
            device::transpose( false, phys_mb, phys_nb,
                               src_tile->data(), src_tile->stride(),
                               work_data, work_stride,
                               *queue );
            work_tile.copyData( dst_tile, *queue, true );
        }
        // Above kernels are asynchronous
        if (! async || need_workspace) {
            queue->sync();
        }
        if (need_workspace) {
            storage_->releaseWorkspaceBuffer( work_data, work_device );
        }
    }
}

//------------------------------------------------------------------------------
/// Acquire tile(i, j) on device without copying data if not already exists.
/// This is used when the destination tile's data will be overriden.
/// Converts destination Layout to 'layout' param.
/// Assumes the TileNode(i, j) already exists.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's destination: host or device ID.
///
/// @param[in] layout
///     Indicates the required Layout of the received tile:
///     - ColMajor: column major.
///     - RowMajor: row major.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileAcquire(int64_t i, int64_t j, int device,
                                       Layout layout)
{
    auto tile = storage_->tileInsert( globalIndex(i, j, device),
                                      TileKind::Workspace, layout );

    // Change ColMajor <=> RowMajor if needed.
    if (tile->layout() != layout) {
        if (! tile->isTransposable()) {
            storage_->tileMakeTransposable(tile);
        }
        tile->setLayout( layout );
        // tileLayoutConvert(i, j, device, Layout(layout), false);
    }
}

//------------------------------------------------------------------------------
/// Gets tile(i, j) on device.
/// Will copy-in the tile if it does not exist or its state is MOSI::Invalid.
/// Finds a source tile whose state is valid (Modified|Shared) by
/// looping on existing tile instances.
/// Updates source tile's state to shared if copied-in.
/// If 'modify' param is true, marks the destination tile as MOSI::Modified,
/// and invalidates other instances. Otherwise, sets destination tile state
/// to MOSI::Shared if copied-in.
/// Converts destination Layout based on 'layout' param.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] dst_device
///     Tile's destination: host or device ID.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
/// @param[in] modify
///     true: marks the destination tile as MOSI::Modified and invalidates other instances.
///     false: marks the destination tile as MOSI::Shared.
///
/// @param[in] hold
///     true: marks the destination tile as MOSI::OnHold.
///
/// @param[in] async
///     true: does not synchronize with device stream.
///
// todo: async version
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGet(int64_t i, int64_t j, int dst_device,
                                   LayoutConvert layout, bool modify, bool hold,
                                   bool async)
{
    // todo: need to acquire read access to the TilesMap
    // LockGuard guard2(storage_->getTilesMapLock());

    Tile<scalar_t>* src_tile = nullptr;
    // default value to silence compiler warning will be overridden below
    Layout target_layout = Layout::ColMajor;

    const int invalid_dev = HostNum - 1; // invalid device number
    int src_device = invalid_dev;

    // find tile on destination
    auto& tile_node = storage_->at(globalIndex(i, j));

    // acquire write access to the (i, j) TileNode
    LockGuard guard(tile_node.getLock());

    if ((! tile_node.existsOn(dst_device)) ||
        (  tile_node[dst_device]->state() == MOSI::Invalid)) {

        // find a valid source (Modified/Shared) tile
        for (int d = num_devices()-1; d >= HostNum; --d) {
            // Most current systems have higher GPU->GPU than GPU->CPU
            // TODO Poll hardware topolgy to determine order
            if (d != dst_device && tile_node.existsOn(d)) {
                if (tile_node[d]->state() != MOSI::Invalid) {
                    src_device = d;
                    src_tile = tile_node[d];
                    break;
                }
            }
        }

        if (src_device == invalid_dev) {
            slate_error(std::string("Error copying tile(")
                         + std::to_string(i) + ", " + std::to_string(j)
                         + "), rank(" + std::to_string(this->mpiRank())
                         + "), invalid source " + std::to_string(src_device)
                         + " -> " + std::to_string(dst_device));
        }

        target_layout = layout == LayoutConvert::None ?
                        src_tile->layout() :
                        Layout(layout);
    }

    if (! tile_node.existsOn(dst_device)) {
        // Create a copy on the destination.
        storage_->tileInsert( globalIndex(i, j, dst_device),
                              TileKind::Workspace, target_layout );
    }

    Tile<scalar_t>* dst_tile = tile_node[dst_device];
    if (dst_tile->state() == MOSI::Invalid) {
        // Update the destination tile's data.

        tileCopyDataLayout( src_tile, dst_tile, target_layout, async );

        dst_tile->state(MOSI::Shared);
        if (src_tile->stateOn(MOSI::Modified))
            src_tile->state(MOSI::Shared);
    }
    if (modify) {
        tileModified(i, j, dst_device);
    }
    if (hold) {
        dst_tile->state(MOSI::OnHold);
    }

    // Change ColMajor <=> RowMajor if needed.
    if (layout != LayoutConvert::None && dst_tile->layout() != Layout(layout)) {
        tileLayoutConvert(i, j, dst_device, Layout(layout), false, async);
    }
}

//------------------------------------------------------------------------------
/// Gets a set of tiles on device.
/// If destination device is host, forwards LayoutConvert param to tileGet()
/// otherwise, calls tileLayoutConvert() to process layout conversion in batch
/// mode.
/// @see tileGet
///
/// @param[in] tile_set
///     Set of (i, j) tuples indicating indices of Tiles' to be converted.
///
/// @param[in] device
///     Tile's destination: host or device ID, defaults to host.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
/// @param[in] modify
///     true: marks the destination tiles as MOSI::Modified and invalidates other instances.
///     false: marks the destination tiles as MOSI::Shared.
///
/// @param[in] hold
///     true: marks the destination tiles as MOSI::OnHold.
///
/// @param[in] async
///     if true, does not synchronize with device stream.
///
// todo: async version
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGet(std::set<ij_tuple>& tile_set, int device,
                                   LayoutConvert in_layoutConvert, bool modify, bool hold,
                                   bool async)
{
    if (device != HostNum) {
        LockGuard guard(storage_->getTilesMapLock());

        // find number of already existing tiles on the device
        int64_t existing_tiles = 0;
        for (auto iter = tile_set.begin(); iter != tile_set.end(); ++iter) {
            int64_t i = std::get<0>(*iter);
            int64_t j = std::get<1>(*iter);
            existing_tiles += tileExists(i, j, device);
        }

        // ensure workspace exists for the rest
        if (tile_set.size() > size_t(existing_tiles))
            storage_->ensureDeviceWorkspace(device, tile_set.size() - existing_tiles);
    }

    LayoutConvert layoutConvert = (device == HostNum)
                                  ? in_layoutConvert
                                  : LayoutConvert::None;

    for (auto iter = tile_set.begin(); iter != tile_set.end(); ++iter) {
        int64_t i = std::get<0>(*iter);
        int64_t j = std::get<1>(*iter);
        {
            tileGet(i, j, device, layoutConvert, modify, hold, true);
        }
    }

    // todo: if modify and target is host, batch convert on device first
    if (device != HostNum && in_layoutConvert != LayoutConvert::None) {
        tileLayoutConvert(tile_set, device, Layout(in_layoutConvert));
    }

    if (! async && device != HostNum)
        comm_queue(device)->sync();
}

//------------------------------------------------------------------------------
/// Gets tile(i, j) for reading on device.
/// Will copy-in the tile if it does not exist or its state is MOSI::Invalid.
/// Sets tile state to MOSI::Shared if copied-in.
/// Finds a source tile whose state is valid (Modified|Shared) by
/// looping on existing tile instances.
/// Updates source tile's state to shared if copied-in.
/// Converts destination Layout based on 'layout' param.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's destination: host or device ID, defaults to host.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
// todo: async version
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetForReading(int64_t i, int64_t j, int device,
                                             LayoutConvert layout)
{
    tileGet(i, j, device, layout, false, false, false);
}

//------------------------------------------------------------------------------
/// Gets a set of tiles for reading on device.
/// @see tileGetForReading
///
/// @param[in] tile_set
///     Set of (i, j) tuples indicating indices of Tiles' to be acquired.
///
/// @param[in] device
///     Tile's destination: host or device ID.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
// todo: async version
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetForReading(std::set<ij_tuple>& tile_set,
                                             int device,
                                             LayoutConvert layout)
{
    tileGet(tile_set, device, layout, false, false, false);
}

//------------------------------------------------------------------------------
/// Gets a set of tiles for reading on host from a specific device.
/// @see tileGetForReading
///
/// @param[in] tile_set
///     Set of (i, j) tuples indicating indices of Tiles' to be acquired.
///     Tiles should exist on the specified device.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
/// @param[in] from_device
///     Tiles' source device ID.
///
// todo: async version
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetForReading(std::set<ij_tuple>& tile_set,
                                             LayoutConvert layout, int from_device)
{
    tileGet( tile_set, HostNum, layout, false, false, false );
}

//------------------------------------------------------------------------------
/// Gets tile(i, j) for writing on device.
/// Sets destination tile's state to MOSI::Modified.
/// Will copy-in the tile if it does not exist or its state is MOSI::Invalid.
/// Other instances will be invalidated.
/// Finds a source tile whose state is valid (Modified|Shared) by
/// scanning existing tile instances.
/// Converts destination Layout based on 'layout' param.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's destination: host or device ID, defaults to host.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetForWriting(int64_t i, int64_t j, int device,
                                             LayoutConvert layout)
{
    tileGet(i, j, device, layout, true, false, false);
}

//------------------------------------------------------------------------------
/// Gets a set of tiles for writing on device.
/// @see tileGetForWriting
///
/// @param[in] tile_set
///     Set of (i, j) tuples indicating indices of Tiles' to be acquired.
///
/// @param[in] device
///     Tile's destination: host or device ID, defaults to host.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetForWriting(std::set<ij_tuple>& tile_set,
                                             int device, LayoutConvert layout)
{
    tileGet( tile_set, device, layout, true, false, false );
}

//------------------------------------------------------------------------------
/// Gets tile(i, j) on device for reading and marks it as MOSI::OnHold.
/// Will copy tile in if it does not exist or its state is MOSI::Invalid.
/// Updates the source tile's state to MOSI::Shared if copied-in.
/// Converts destination Layout based on 'layout' param.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's destination: host or device ID, defaults to host.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetAndHold(int64_t i, int64_t j, int device,
                                          LayoutConvert layout)
{
    tileGet(i, j, device, layout, false, true, false);
}

//------------------------------------------------------------------------------
/// Gets a set of tiles for reading on device and marks them as MOSI::OnHold.
/// @see tileGetAndHold
///
/// @param[in] tile_set
///     Set of (i, j) tuples indicating indices of Tiles' to be acquired.
///
/// @param[in] device
///     Tile's destination: host or device ID, defaults to host.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetAndHold(std::set<ij_tuple>& tile_set, int device,
                                          LayoutConvert layout)
{
    tileGet( tile_set, device, layout, false, true, false );
}

//------------------------------------------------------------------------------
/// Gets all local tiles for reading on device.
/// @see tileGetForReading.
///
/// @param[in] device
///     Tile's destination: host or device ID, defaults to host.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
// todo: should this take into consideration the local tiles distribution on devices?
//       currently, this will cram all local tiles into specified device.
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetAllForReading(int device, LayoutConvert layout)
{
    std::set<ij_tuple> tiles_set;
    for (int64_t j = 0; j < nt(); ++j)
        for (int64_t i = 0; i < mt(); ++i)
            // todo: if (tileIsLocal(i, j) && (device == HostNum || device == tileDevice(i, j))) {
            if (tileIsLocal(i, j)) {
                tiles_set.insert({i, j});
            }

    tileGetForReading(tiles_set, device, layout);
}

//------------------------------------------------------------------------------
/// Gets all local tiles for writing on device.
/// @see tileGetForWriting.
///
/// @param[in] device
///     Tile's destination: host or device ID, defaults to host.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
// todo: should this take into consideration the local tiles distribution on devices?
//       currently, this will cram all local tiles into specified device.
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetAllForWriting(int device, LayoutConvert layout)
{
    std::set<ij_tuple> tiles_set;
    for (int64_t j = 0; j < nt(); ++j)
        for (int64_t i = 0; i < mt(); ++i)
            // todo: if (tileIsLocal(i, j) && (device == HostNum || device == tileDevice(i, j))) {
            if (tileIsLocal(i, j)) {
                tiles_set.insert({i, j});
            }

    tileGetForWriting(tiles_set, device, layout);
}

//------------------------------------------------------------------------------
/// Gets all local tiles on device and marks them as MOSI::OnHold.
/// @see tileGetAndHold.
///
/// @param[in] device
///     Tile's destination: host or device ID, defaults to host.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
// todo: should this take into consideration the local tiles distribution on devices?
//       currently, this will cram all local tiles into specified device.
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetAndHoldAll(int device, LayoutConvert layout)
{
    std::set<ij_tuple> tiles_set;
    for (int64_t j = 0; j < nt(); ++j)
        for (int64_t i = 0; i < mt(); ++i)
            // todo: if (tileIsLocal(i, j) && (device == HostNum || device == tileDevice(i, j))) {
            if (tileIsLocal(i, j)) {
                tiles_set.insert({i, j});
            }

    tileGetAndHold( tiles_set, device, layout );
}

//------------------------------------------------------------------------------
/// Gets all local tiles for reading on corresponding devices.
/// @see tileGetForReading.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetAllForReadingOnDevices(LayoutConvert layout)
{
    std::vector< std::set<ij_tuple> > tiles_set(num_devices());
    for (int64_t j = 0; j < nt(); ++j) {
        for (int64_t i = 0; i < mt(); ++i) {
            if (tileIsLocal(i, j)) {
                tiles_set[tileDevice(i, j)].insert({i, j});
            }
        }
    }

    #pragma omp taskgroup
    {
        for (int d = 0; d < num_devices(); ++d) {
            if (! tiles_set[d].empty()) {
                #pragma omp task slate_omp_default_none \
                    firstprivate( d, layout ) shared( tiles_set )
                {
                    tileGetForReading(tiles_set[d], d, layout);
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Gets all local tiles for writing on corresponding devices.
/// @see tileGetForWriting.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetAllForWritingOnDevices(LayoutConvert layout)
{
    std::vector< std::set<ij_tuple> > tiles_set(num_devices());
    for (int64_t j = 0; j < nt(); ++j) {
        for (int64_t i = 0; i < mt(); ++i) {
            if (tileIsLocal(i, j)) {
                tiles_set[tileDevice(i, j)].insert({i, j});
            }
        }
    }

    #pragma omp taskgroup
    {
        for (int d = 0; d < num_devices(); ++d) {
            if (! tiles_set[d].empty()) {
                #pragma omp task slate_omp_default_none \
                    firstprivate( d, layout ) shared( tiles_set )
                {
                    tileGetForWriting(tiles_set[d], d, layout);
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Gets all local tiles on corresponding devices and marks them as MOSI::OnHold.
/// @see tileGetAndHold.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
//
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetAndHoldAllOnDevices(LayoutConvert layout)
{
    std::vector< std::set<ij_tuple> > tiles_set(num_devices());
    for (int64_t j = 0; j < nt(); ++j) {
        for (int64_t i = 0; i < mt(); ++i) {
            if (tileIsLocal(i, j)) {
                tiles_set[tileDevice(i, j)].insert({i, j});
            }
        }
    }

    #pragma omp taskgroup
    {
        for (int d = 0; d < num_devices(); ++d) {
            if (! tiles_set[d].empty()) {
                #pragma omp task slate_omp_default_none \
                    firstprivate( d, layout ) shared( tiles_set )
                {
                    tileGetAndHold(tiles_set[d], d, layout);
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Updates the origin instance of tile(i, j) if MOSI::Invalid
/// tile must be local.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @return Pointer to origin instance of tile(i, j)
///
template <typename scalar_t>
Tile<scalar_t> BaseMatrix<scalar_t>::tileUpdateOrigin(int64_t i, int64_t j)
{
    auto& tile_node = storage_->at(globalIndex(i, j));

    LockGuard guard(tile_node.getLock());

    // find on host
    if (tile_node.existsOn( HostNum )
        && tile_node[ HostNum ]->origin()) {
        if (tile_node[ HostNum ]->stateOn(MOSI::Invalid)) {
            // todo: should this request Layout conversion to this->layout() ?
            tileGetForReading(i, j, LayoutConvert::None);
        }
        return *(tile_node[ HostNum ]);
    }
    else {
        auto device = tileDevice(i, j);
        if (tile_node.existsOn(device) &&
            tile_node[device]->origin()) {
            if (tile_node[device]->stateOn(MOSI::Invalid)) {
                // todo: should this request Layout conversion to this->layout() ?
                tileGetForReading(i, j, device, LayoutConvert::None);
            }
        }
        else
            slate_error( std::string("Origin tile not found! tile(")
                        +std::to_string(i)+","+std::to_string(j)+")");
        return *(tile_node[device]);
    }
}

//------------------------------------------------------------------------------
/// Updates all origin instances of local tiles if MOSI::Invalid.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileUpdateAllOrigin()
{
    std::vector< std::set<ij_tuple> > tiles_set_host(num_devices());
    std::vector< std::set<ij_tuple> > tiles_set_dev(num_devices());
    for (int64_t j = 0; j < this->nt(); ++j) {
        for (int64_t i = 0; i < this->mt(); ++i) {
            if (this->tileIsLocal(i, j)) {
                // this->tileUpdateOrigin(i, j);
                auto& tile_node = storage_->at(globalIndex(i, j));

                // find on host
                if (tile_node.existsOn( HostNum )
                    && tile_node[ HostNum ]->origin()) {
                    if (tile_node[ HostNum ]->stateOn(MOSI::Invalid)) {
                        // tileGetForReading(i, j, LayoutConvert::None);
                        for (int d = 0; d < num_devices(); ++d) {
                            if (tile_node.existsOn(d)
                                && tile_node[d]->state() != MOSI::Invalid)
                            {
                                tiles_set_host[d].insert({i, j});
                                break;
                            }
                        }
                    }
                }
                else {
                    auto device = tileDevice(i, j);
                    if (tile_node.existsOn(device) &&
                        tile_node[device]->origin()) {
                        if (tile_node[device]->stateOn(MOSI::Invalid)) {
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
        for (int d = 0; d < num_devices(); ++d) {
            if (! tiles_set_host[d].empty()) {
                #pragma omp task slate_omp_default_none \
                    firstprivate( d ) shared( tiles_set_host )
                {
                    tileGetForReading(tiles_set_host[d], LayoutConvert::None, d);
                }
            }
            if (! tiles_set_dev[d].empty()) {
                #pragma omp task slate_omp_default_none \
                    firstprivate( d ) shared( tiles_set_dev )
                {
                    tileGetForReading(tiles_set_dev[d], d, LayoutConvert::None);
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Returns whether tile(i, j, device) can be safely transposed.
/// based on its 'TileKind', buffer size, Layout, and stride.
/// Tile instance on 'device' should exist.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's host or device ID, defaults to host.
///
// todo: validate working for sub- and sliced- matrix
template <typename scalar_t>
[[deprecated( "SLATE now manages convertibility internally. Will be removed 2024-10." )]]
bool BaseMatrix<scalar_t>::tileLayoutIsConvertible(int64_t i, int64_t j, int device)
{
    return storage_->at( globalIndex(i, j, device) )->isTransposable();
}

//------------------------------------------------------------------------------
/// Converts tile(i, j, device) into 'layout'.
/// Tile should exist on 'device', will assert otherwise.
/// Tile will be made Convertible if it was not.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's host or device ID.
///
/// @param[in] layout
///     Intended Layout of tile:
///     - Layout::ColMajor or
///     - Layout::RowMajor.
///
/// @param[in] reset
///     Optinally resets the tile extended buffers.
///
/// todo: handle op(A), sub-matrix, and sliced-matrix
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileLayoutConvert(
    int64_t i, int64_t j, int device, Layout layout, bool reset, bool async)
{
    auto& tile_node = storage_->at( globalIndex(i, j) );
    LockGuard guard( tile_node.getLock() );
    auto tile = tile_node[ device ];
    if (tile->layout() != layout) {
        if (! tile->isTransposable()) {
            assert(! reset); // Can't change to ext buffer then reset
            storage_->tileMakeTransposable(tile);
        }

        scalar_t* work_data = nullptr;
        // if rectangular and not extended, need a workspace buffer
        bool need_workspace = tile->mb() != tile->nb() && (! tile->extended());

        if (need_workspace)
            work_data = storage_->allocWorkspaceBuffer(tile->device());

        if (tile->device() == HostNum) {
            tile->layoutConvert(work_data);
        }
        else {
            lapack::Queue* queue = comm_queue(tile->device());
            bool use_async = async && ! need_workspace && ! reset;
            tile->layoutConvert(work_data, *queue, use_async);
        }

        // release the workspace buffer if allocated
        if (need_workspace)
            storage_->releaseWorkspaceBuffer(work_data, tile->device());
    }
    if (reset) {
        assert(tile->layout() == this->layout());
        storage_->tileLayoutReset(tile);
    }
}

//------------------------------------------------------------------------------
/// Converts tiles indicated in 'tile_set' that exist on 'device' into 'layout'
/// if not already in 'layout' major.
/// Tiles should exist on 'device', will throw exception otherwise.
/// Operates in batch mode when tiles are on devices.
/// If device is not Host, will bucket tiles into uniform size and stride
/// batches, then launches each batch transpose.
///
/// @param[in] tile_set
///     Set of (i, j) tuples indicating indices of Tiles' to be converted.
///
/// @param[in] device
///     Tiles' host or device ID.
///
/// @param[in] layout
///     Intended Layout of tiles:
///     - Layout::ColMajor or
///     - Layout::RowMajor.
///
/// @param[in] reset
///     Optinally resets the tiles extended buffers.
///
// todo: async API
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileLayoutConvert(
    std::set<ij_tuple>& tile_set, int device, Layout layout, bool reset)
{
    if (device == HostNum) {
        #pragma omp taskgroup
        for (auto iter = tile_set.begin(); iter != tile_set.end(); ++iter) {
            int64_t i = std::get<0>(*iter);
            int64_t j = std::get<1>(*iter);
            #pragma omp task slate_omp_default_none \
                firstprivate( i, j, device, layout, reset )
            {
                tileLayoutConvert(i, j, device, layout, reset);
            }
        }
    }
    else {
        // todo: this is not an optimal lock,
        // two calls on different devices will be serialized,
        // ideally, two concurrent read accesses to the TilesMap should be allowed
        // but not a concurrent read and write.
        LockGuard guard(storage_->getTilesMapLock());

        // map key tuple: m, n, extended, stride, work_stride
        using mnss_tuple = std::tuple<int64_t, int64_t, bool, int64_t, int64_t>;
        // map value tuple: data and extended data buffers
        using data_tuple = std::pair<std::vector<scalar_t*>, std::vector<scalar_t*>>;

        using BatchedTilesBuckets = std::map< mnss_tuple, data_tuple >;

        BatchedTilesBuckets tilesBuckets;

        for (auto iter = tile_set.begin(); iter != tile_set.end(); ++iter) {
            int64_t i = std::get<0>(*iter);
            int64_t j = std::get<1>(*iter);

            auto tile = storage_->at( globalIndex(i, j, device) );

            // if we need to convert layout
            if (tile->layout() != layout) {
                // make sure tile is transposable
                if (! tile->isTransposable()) {
                    storage_->tileMakeTransposable(tile);
                }
                tile->setLayout( layout );

                int64_t work_stride = layout == Layout::ColMajor
                                      ? tile->nb()
                                      : tile->mb();

                // bucket index
                mnss_tuple mns = {tile->mb(), tile->nb(),
                                  tile->extended(),
                                  tile->stride(),
                                  tile->mb() != tile->nb()
                                    ? (tile->extended() ?
                                        tile->layoutBackStride() : work_stride)
                                    : 0};

                // add this tile's data to the corrsponding bucket of batch array
                tilesBuckets[mns].first.push_back(tile->data());

                // if rectangular, prepare a workspace
                if (tile->mb() != tile->nb()) {
                    if (tile->extended())
                        tilesBuckets[mns].second.push_back(
                            tile->layoutBackData());
                    else
                        tilesBuckets[mns].second.push_back(
                            storage_->allocWorkspaceBuffer(device));
                }
            }
        }

        // Allocate batch arrays, if not done already.
        int64_t batch_count = 0;
        for (auto bucket  = tilesBuckets.begin();
                  bucket != tilesBuckets.end();
                ++bucket) {
            batch_count =
                std::max(batch_count, int64_t(bucket->second.first.size()));
        }

        lapack::Queue* queue = comm_queue(device);

        // for each bucket
        for (auto bucket  = tilesBuckets.begin();
                  bucket != tilesBuckets.end();
                ++bucket) {
            batch_count = bucket->second.first.size();

            scalar_t** array_dev = this->array_device(device);
            scalar_t** work_array_dev = this->array_device(device) + batch_count;

            assert(array_dev      != nullptr);
            assert(work_array_dev != nullptr);

            // mb and nb are the new dimensions
            int64_t mb          = std::get<0>(bucket->first);
            int64_t nb          = std::get<1>(bucket->first);
            int64_t extended    = std::get<2>(bucket->first);
            int64_t stride      = std::get<3>(bucket->first);
            int64_t work_stride = std::get<4>(bucket->first);

            // todo: should we handle batch of size one differently?
            //       will need to store the (i,j) tuple with the batch array!
            // if (batch_count == 1) {
            //     tileLayoutConvert(i, j, device, layout);
            // }
            // else

            blas::device_memcpy<scalar_t*>(
                array_dev, bucket->second.first.data(),
                batch_count, blas::MemcpyKind::HostToDevice, *queue);

            if (mb == nb) {
                // in-place transpose
                device::transpose_batch(false,
                                        nb,
                                        array_dev, stride,
                                        batch_count, *queue);
            }
            else {
                // rectangular tiles: out-of-place transpose
                blas::device_memcpy<scalar_t*>(
                    work_array_dev, bucket->second.second.data(),
                    batch_count, blas::MemcpyKind::HostToDevice, *queue);

                if (! extended) {
                    // copy back to data buffer
                    device::gecopy(layout == Layout::ColMajor ? nb : mb,
                                   layout == Layout::ColMajor ? mb : nb,
                                   array_dev, work_stride,
                                   work_array_dev, work_stride,
                                   batch_count, *queue);
                }

                device::transpose_batch(false,
                                        layout == Layout::ColMajor ? nb : mb,
                                        layout == Layout::ColMajor ? mb : nb,
                                        work_array_dev, work_stride,
                                        array_dev, stride,
                                        batch_count, *queue);
            }

            // release workspace buffer if allocated
            if ((mb != nb) && (! extended)) {
                queue->sync();

                for (auto iter  = bucket->second.second.begin();
                          iter != bucket->second.second.end();
                          ++iter) {
                    storage_->releaseWorkspaceBuffer(*iter, device);
                }
            }
        }

        queue->sync();

        if (reset) {
            for (auto iter = tile_set.begin(); iter != tile_set.end(); ++iter) {
                // #pragma omp task default(none)
                {
                    int64_t i = std::get<0>(*iter);
                    int64_t j = std::get<1>(*iter);
                    auto tile = storage_->at( globalIndex(i, j, device) );
                    storage_->tileLayoutReset(tile);
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Converts all existing tile instances on 'device' into 'layout'
/// Operates in batch mode.
///
/// @param[in] device
///     Tiles' host or device ID.
///
/// @param[in] layout
///     Intended Layout of tiles:
///     - Layout::ColMajor or
///     - Layout::RowMajor.
///
/// @param[in] reset
///     Optinally resets the tiles extended buffers.
///
// todo: override on BaseTrapezoidMatrix and BandMatrix
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileLayoutConvert(int device, Layout layout, bool reset)
{
    std::set<ij_tuple> tiles_set;
    for (int64_t j = 0; j < nt(); ++j) {
        for (int64_t i = 0; i < mt(); ++i) {
            // if (tileIsLocal(i, j) && device == tileDevice(i, j) && tileExists(i, j, device))
            if (tileExists(i, j, device)) {
                tiles_set.insert({i, j});
            }
        }
    }

    tileLayoutConvert(tiles_set, device, layout, reset);
}

//------------------------------------------------------------------------------
/// Converts all existing tile instances on available devices into 'layout'.
/// Host tiles are not affected.
/// Tiles should exist already on devices.
/// Operates in batch mode.
///
/// @param[in] layout
///     Intended Layout of tiles:
///     - Layout::ColMajor or
///     - Layout::RowMajor.
///
/// @param[in] reset
///     Optinally resets the tile extended buffers.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileLayoutConvertOnDevices(Layout layout, bool reset)
{
    std::vector< std::set<ij_tuple> > tiles_set(num_devices());
    for (int64_t j = 0; j < nt(); ++j) {
        for (int64_t i = 0; i < mt(); ++i) {
            for (int d = 0; d < num_devices(); ++d) {
                if (tileExists(i, j, d)) {
                    tiles_set[d].insert({i, j});
                }
            }
        }
    }
    #pragma omp taskgroup
    {
        for (int d = 0; d < num_devices(); ++d) {
            if (! tiles_set[d].empty()) {
                #pragma omp task slate_omp_default_none \
                    firstprivate( d, layout, reset ) shared( tiles_set )
                {
                    tileLayoutConvert(tiles_set[d], d, layout, reset);
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Converts tile(i, j) into current layout and resets its extended buffer.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's device ID; default is HostNum.
///
/// @param[in] layout
///     Intended Layout of tiles:
///     - Layout::ColMajor or
///     - Layout::RowMajor.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileLayoutReset(int64_t i, int64_t j, int device,
                                           Layout layout)
{
    tileLayoutConvert(i, j, device, layout, true);
}

//------------------------------------------------------------------------------
/// Converts set of tiles into current layout and resets their extended buffers.
/// Operates in batch mode.
///
/// @param[in] tile_set
///     Set of (i, j) indices of tiles to be converted and reset.
///
/// @param[in] device
///     Tile's device ID; default is HostNum.
///
/// @param[in] layout
///     Intended Layout of tiles:
///     - Layout::ColMajor or
///     - Layout::RowMajor.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileLayoutReset(std::set<ij_tuple>& tile_set,
                                           int device, Layout layout)
{
    tileLayoutConvert(tile_set, device, layout, true);
}

//------------------------------------------------------------------------------
/// Converts all origin tiles into current matrix-layout.
/// Operates in batch mode.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileLayoutReset()
{
    std::set<ij_tuple> tiles_set_host;
    std::vector< std::set<ij_tuple> > tiles_set_dev(num_devices());

    for (int64_t i = 0; i < mt(); ++i) {
        for (int64_t j = 0; j < nt(); ++j) {
            if (tileIsLocal(i, j)) {
                auto tile = tileUpdateOrigin(i, j);
                if (tile.layout() != this->layout()) {
                    assert(tile.isTransposable());
                }

                if (tile.device() == HostNum) {
                    tiles_set_host.insert({i, j});
                }
                else {
                    tiles_set_dev[tile.device()].insert({i, j});
                }
            }
        }
    }

    #pragma omp taskgroup
    {
        if (! tiles_set_host.empty()) {
            auto layout = this->layout();
            #pragma omp task slate_omp_default_none \
                firstprivate( layout ) shared( tiles_set_host )
            {
                tileLayoutReset( tiles_set_host, HostNum, layout );
            }
        }
        for (int d = 0; d < num_devices(); ++d) {
            if (! tiles_set_dev[d].empty()) {
                auto layout = this->layout();
                #pragma omp task slate_omp_default_none \
                    firstprivate( d, layout ) shared( tiles_set_dev )
                {
                    tileLayoutReset(tiles_set_dev[d], d, layout);
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Puts all MPI ranks that have tiles in the matrix into the set.
///
/// @param[out] bcast_set
///     On output, set of MPI ranks that this sub-matrix has tiles on.
///
// todo: pass bcast_set by reference
template <typename scalar_t>
void BaseMatrix<scalar_t>::getRanks(std::set<int>* bcast_set) const
{
    for (int64_t i = 0; i < mt(); ++i)
        for (int64_t j = 0; j < nt(); ++j)
            bcast_set->insert(tileRank(i, j));
}

//------------------------------------------------------------------------------
/// Returns the origin tile instance of tile(i, j)
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
template <typename scalar_t>
Tile<scalar_t> BaseMatrix<scalar_t>::originTile(int64_t i, int64_t j)
{
    auto& tile_node = storage_->at(globalIndex(i, j));

    // find on host
    if (tile_node.existsOn( HostNum )
        && tile_node[ HostNum ]->origin()) {
        return *(tile_node[ HostNum ]);
    }
    else {
        auto device = tileDevice(i, j);
        if (tile_node.existsOn( device )
            && tile_node[ device ]->origin()) {
            return *(tile_node[ device ]);
        }
        else
            slate_error(std::string("Origin tile not found! tile(")
                        + std::to_string(i) + "," + std::to_string(j) + ")");
    }
}

//------------------------------------------------------------------------------
/// Puts all devices that have local tiles in the matrix into the set.
///
/// @param[out] dev_set
///     On output, set of device IDs that this sub-matrix has local tiles on.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::getLocalDevices(std::set<int>* dev_set) const
{
    for (int64_t i = 0; i < mt(); ++i)
        for (int64_t j = 0; j < nt(); ++j)
            if (tileIsLocal(i, j))
                dev_set->insert(tileDevice(i, j));
}

//------------------------------------------------------------------------------
/// Returns number of local tiles in this matrix.
/// Used for the lifespan of a temporary tile that updates every tile in
/// the matrix.
///
template <typename scalar_t>
int64_t BaseMatrix<scalar_t>::numLocalTiles() const
{
    // Find the tile's lifespan.
    int64_t life = 0;
    for (int64_t i = 0; i < mt(); ++i)
        for (int64_t j = 0; j < nt(); ++j)
            if (tileIsLocal(i, j))
                ++life;

    return life;
}

//------------------------------------------------------------------------------
/// [internal]
/// Returns index {i, j} in global matrix as a tuple,
/// taking into account the local offset and transpose.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
template <typename scalar_t>
std::tuple< int64_t, int64_t >
    BaseMatrix<scalar_t>::globalIndex(int64_t i, int64_t j) const
{
    assert(0 <= i && i < mt());
    assert(0 <= j && j < nt());
    if (op_ == Op::NoTrans)
        return { ioffset_ + i, joffset_ + j };
    else
        return { ioffset_ + j, joffset_ + i };
}

//------------------------------------------------------------------------------
/// [internal]
/// Returns index {i, j, dev} in global matrix as a tuple,
/// taking into account the local offset and transpose.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's device ID.
///
template <typename scalar_t>
std::tuple<int64_t, int64_t, int>
    BaseMatrix<scalar_t>::globalIndex(int64_t i, int64_t j, int device) const
{
    assert(0 <= i && i < mt());
    assert(0 <= j && j < nt());
    // Given AnyDevice = -3, AllDevices = -2, HostNum = -1,
    // GPU devices 0, 1, ..., num_devices-1.
    assert( AnyDevice <= device && device < num_devices_ );
    if (op_ == Op::NoTrans)
        return { ioffset_ + i, joffset_ + j, device };
    else
        return { ioffset_ + j, joffset_ + i, device };
}

//------------------------------------------------------------------------------
/// Erases a given local workspace tile, if not on hold or modified.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::releaseLocalWorkspaceTile(int64_t i, int64_t j)
{
    if (this->tileIsLocal( i, j )) {
        for (int device = HostNum; device < this->num_devices(); ++device) {
            tileRelease( i, j, device );
        }
    }
}

//------------------------------------------------------------------------------
/// Erases all local workspace tiles, if not on hold or modified.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::releaseLocalWorkspace()
{
    for (int64_t j = 0; j < this->nt(); ++j) {
        for (int64_t i = 0; i < this->mt(); ++i) {
            releaseLocalWorkspaceTile( i, j );
        }
    }
}

//------------------------------------------------------------------------------
/// Erases a given set of local workspace tiles
/// from all devices including host, if not on hold or modified.
///
/// @param[in] tile_set
///     Set of (i, j) tuples indicating indices of tiles to be erased.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::releaseLocalWorkspace(
    std::set<ij_tuple>& tile_set)
{
    for (auto ij : tile_set) {
        int64_t i = std::get<0>( ij );
        int64_t j = std::get<1>( ij );
        releaseLocalWorkspaceTile( i, j );
    }
}

//------------------------------------------------------------------------------
/// Erases a given tile that is not local to node from all devices
/// including host, if not on hold or modified.
/// The tile's receive count is decremented. If the receive count
/// reaches zero, the tile is erased. Otherwise, tile is not erased.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::releaseRemoteWorkspaceTile(int64_t i, int64_t j)
{
    if (! tileIsLocal( i, j )) { // erase remote tiles
        // This lock ensures that no other thread is trying to
        // remove this tile from the map of tiles.
        LockGuard guard( storage_->getTilesMapLock() );

        if (tileExists( i, j, AnyDevice )) {
            tileDecrementReceiveCount( i, j );
            if (tileReceiveCount( i, j ) <= 0) {
                tileRelease( i, j, AllDevices );
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Erases tiles that are not local to node from all devices
/// including host, if not on hold or modified.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::releaseRemoteWorkspace()
{
    for (int64_t j = 0; j < nt(); ++j) {
        for (int64_t i = 0; i < mt(); ++i) {
            releaseRemoteWorkspaceTile( i, j );
        }
    }
}

//------------------------------------------------------------------------------
/// Erases a given set of tiles that are not local to node
/// from all devices including host, if not on hold or modified.
///
/// @param[in] tile_set
///     Set of (i, j) tuples indicating indices of tiles to be erased.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::releaseRemoteWorkspace(
    std::set<ij_tuple>& tile_set)
{
    for (auto ij : tile_set) {
        int64_t i = std::get<0>( ij );
        int64_t j = std::get<1>( ij );
        releaseRemoteWorkspaceTile( i, j );
    }
}

//------------------------------------------------------------------------------
template <typename scalar_t>
int BaseMatrix<scalar_t>::num_devices_ = 0;

//------------------------------------------------------------------------------
// from ScaLAPACK's indxg2l
// todo: where to put utilities like this?
inline int64_t indexGlobal2Local(int64_t i, int64_t nb, int num_ranks)
{
    return nb*(i/(nb*num_ranks)) + (i % nb);
}

//------------------------------------------------------------------------------
// from ScaLAPACK's numroc
/// [internal]
/// Computes the number of Rows Or Columns of a distributed
/// matrix owned by the process indicated by IPROC.
///
/// @param[in] n
///     The number of rows/columns in distributed matrix.
///
/// @param[in] nb
///     Block size, size of the blocks the distributed matrix is split into.
///
/// @param[in] iproc
///     The coordinate of the process whose local array row or
///     column is to be determined.
///
/// @param[in] isrcproc
///     The coordinate of the process that possesses the first
///     row or column of the distributed matrix.
///
/// @param[in] nprocs
///     The total number processes over which the matrix is distributed.
///
inline int64_t numberLocalRowOrCol(int64_t n, int64_t nb, int iproc, int isrcproc, int nprocs)
{
    int64_t numroc;
    // Figure PROC's distance from source process
    int mydist = (nprocs+iproc-isrcproc) % nprocs;

    // Figure the total number of whole NB blocks N is split up into
    int nblocks = (int)(n / nb);
    // Figure the minimum number of rows/cols a process can have
    numroc = (int64_t)(nblocks/nprocs) * nb;
    // See if there are any extra blocks
    int extrablks = nblocks % nprocs;

    // If I have an extra block
    if (mydist < extrablks) {
        numroc = numroc + nb;
    }
    // If I have last block, it may be a partial block
    else if (mydist == extrablks) {
        numroc = numroc + n % nb;
    }
    return numroc;
}

} // namespace slate

#endif // SLATE_BASE_MATRIX_HH
