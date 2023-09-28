// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_STORAGE_HH
#define SLATE_STORAGE_HH

#include "slate/internal/Memory.hh"
#include "slate/Tile.hh"
#include "slate/types.hh"
#include "slate/internal/util.hh"

#include "blas.hh"
#include "lapack.hh"
#include "lapack/device.hh"

#include <algorithm>
#include <functional>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "slate/internal/mpi.hh"
#include "slate/internal/openmp.hh"
#include "slate/internal/LockGuard.hh"

namespace slate {

//------------------------------------------------------------------------------
///
template <typename scalar_t>
class TileNode {
private:
    /// vector of tile instances indexed by device id.
    std::vector< Tile<scalar_t>* > tiles_;
    int num_instances_;
    int64_t life_;
    /// number of times a tile is received.
    /// This variable is used for only MPI communications.
    int64_t receive_count_;

    /// OMP lock used to protect operations that modify the Tiles within
    mutable omp_nest_lock_t lock_;

public:
    /// Constructor for TileNode class
    TileNode(int num_devices)
        : num_instances_(0),
          life_(0),
          receive_count_(0)
    {
        slate_assert(num_devices >= 0);
        omp_init_nest_lock(&lock_);
        for (int d = 0; d < num_devices+1; ++d) {
            tiles_.push_back( nullptr );
        }
    }

    /// Destructor for TileNode class
    ~TileNode()
    {
        omp_destroy_nest_lock(&lock_);
        // for debug mode
        assert(num_instances_ == 0);
    }

    //--------------------------------------------------------------------------
    // 2. copy constructor -- not allowed; lock_ and tiles_ are not copyable
    // 3. move constructor -- not allowed; lock_ and tiles_ are not copyable
    // 4. copy assignment  -- not allowed; lock_ and tiles_ are not copyable
    // 5. move assignment  -- not allowed; lock_ and tiles_ are not copyable
    TileNode(TileNode&  orig) = delete;
    TileNode(TileNode&& orig) = delete;
    TileNode& operator = (TileNode&  orig) = delete;
    TileNode& operator = (TileNode&& orig) = delete;

    //--------------------------------------------------------------------------
    /// Return pointer to tile instance OMP lock
    omp_nest_lock_t* getLock()
    {
        return &lock_;
    }

    //--------------------------------------------------------------------------
    /// Inserts a tile instance at device and increments the number of resident instances
    void insertOn(int device, Tile<scalar_t>* tile, MOSI_State state)
    {
        slate_assert(device >= -1 && device+1 < int(tiles_.size()));
        slate_assert(tiles_[device+1] == nullptr);
        tile->state( MOSI(state) );
        tiles_[device+1] = tile;
        ++num_instances_;
    }

    //--------------------------------------------------------------------------
    /// Returns whether a tile instance exists at device
    bool existsOn(int device) const
    {
        slate_assert(device >= -1 && device+1 < int(tiles_.size()));
        return tiles_[device+1] != nullptr;
    }

    //--------------------------------------------------------------------------
    /// Deletes the tile instance at device
    // CAUTION: tile's memory must have been already released to MatrixStorage Memory
    void eraseOn(int device)
    {
        slate_assert(device >= -1 && device+1 < int(tiles_.size()));
        if (tiles_[device+1] != nullptr) {
            tiles_[device+1]->state(MOSI::Invalid);
            delete tiles_[device+1];
            tiles_[device+1] = nullptr;
            --num_instances_;
        }
    }

    //--------------------------------------------------------------------------
    /// Returns a pointer to the tile instance at device
    Tile<scalar_t>* operator[](int device) const
    {
        slate_assert(device >= -1 && device+1 < int(tiles_.size()));
        return tiles_[device+1];
    }

    //--------------------------------------------------------------------------
    /// Returns a pointer to the tile instance at device
    Tile<scalar_t>* at(int dev) const
    {
        slate_assert(dev >= -1 && dev+1 < int(tiles_.size()));
        return tiles_[dev+1];
    }

    int64_t& lives()
    {
        return life_;
    }

    int64_t& receiveCount()
    {
        return receive_count_;
    }

    bool empty() const
    {
        return num_instances_ == 0;
    }
};

//------------------------------------------------------------------------------
/// Slate::MatrixStorage class
/// Used to store the map of distributed tiles.
/// @tparam scalar_t Data type for the elements of the matrix
///
template <typename scalar_t>
class MatrixStorage {
public:
    friend class Debug;

    template <typename T>
    friend class BaseMatrix;

    typedef Tile<scalar_t> Tile_t;
    typedef TileNode<scalar_t> TileNode_t;

    using ijdev_tuple = std::tuple<int64_t, int64_t, int>;
    using ij_tuple    = std::tuple<int64_t, int64_t>;
    using TilesMap = std::map< ij_tuple, std::shared_ptr<TileNode_t> >;

    MatrixStorage( int64_t m, int64_t n, int64_t mb, int64_t nb,
                   GridOrder order, int p, int q, MPI_Comm mpi_comm );

    MatrixStorage(std::function<int64_t (int64_t i)>& inTileMb,
                  std::function<int64_t (int64_t j)>& inTileNb,
                  std::function<int (ij_tuple ij)>& inTileRank,
                  std::function<int (ij_tuple ij)>& inTileDevice,
                  MPI_Comm mpi_comm);


    // 1. destructor
    ~MatrixStorage();

    //--------------------------------------------------------------------------
    // 2. copy constructor -- not allowed; object is shared
    // 3. move constructor -- not allowed; object is shared
    // 4. copy assignment  -- not allowed; object is shared
    // 5. move assignment  -- not allowed; object is shared
    MatrixStorage(MatrixStorage&  orig) = delete;
    MatrixStorage(MatrixStorage&& orig) = delete;
    MatrixStorage& operator = (MatrixStorage&  orig) = delete;
    MatrixStorage& operator = (MatrixStorage&& orig) = delete;

protected:
    // used in constructor and destructor
    void initQueues();
    void destroyQueues();

public:
    //--------------------------------------------------------------------------
    // batch arrays
    void allocateBatchArrays(int64_t batch_size, int64_t num_arrays);
    void clearBatchArrays();

    /// @return currently allocated batch array size
    int64_t batchArraySize() const
    {
        return batch_array_size_;
    }

    //--------------------------------------------------------------------------
    // workspace
    void reserveHostWorkspace(int64_t num_tiles);
    void reserveDeviceWorkspace(int64_t num_tiles);
    void ensureDeviceWorkspace(int device, int64_t num_tiles);
    void clearWorkspace();
    void releaseWorkspace();

    scalar_t* allocWorkspaceBuffer(int device);
    void      releaseWorkspaceBuffer(scalar_t* data, int device);

private:
    // Iterator routines should be called only within a Tiles Map LockGuard.
    // Otherwise, there may be race conditions with the returned iterator.

    //--------------------------------------------------------------------------
    /// @return TileNode(i, j) if it has instance on device, end() otherwise
    typename TilesMap::iterator find(ijdev_tuple ijdev)
    {
        int64_t i  = std::get<0>(ijdev);
        int64_t j  = std::get<1>(ijdev);
        int device = std::get<2>(ijdev);
        auto it = tiles_.find({i, j});
        if (it != tiles_.end() && it->second->existsOn(device))
            return it;
        else
            return tiles_.end();
    }

    //--------------------------------------------------------------------------
    /// @return TileNode(i, j) if found, end() otherwise
    typename TilesMap::iterator find(ij_tuple ij)
    {
        return tiles_.find(ij);
    }

    //--------------------------------------------------------------------------
    /// @return begin iterator of TileNode map
    typename TilesMap::iterator begin()
    {
        return tiles_.begin();
    }

    //--------------------------------------------------------------------------
    /// @return begin iterator of TileNode map
    typename TilesMap::iterator end()
    {
        return tiles_.end();
    }

public:
    //--------------------------------------------------------------------------
    /// @return reference to TileNode(i, j).
    /// Throws exception if entry doesn't exist.
    // at() doesn't create new (null) entries in map as operator[] would
    TileNode_t& at(ij_tuple ij)
    {
        LockGuard guard(getTilesMapLock());
        return *(tiles_.at(ij));
    }

    /// @return pointer to an actual Tile object
    /// Throws exception if entry doesn't exist.
    Tile<scalar_t>* at(ijdev_tuple ijdev)
    {
        int64_t i  = std::get<0>(ijdev);
        int64_t j  = std::get<1>(ijdev);
        int device = std::get<2>(ijdev);
        auto& tile_node = at( {i, j} );

        // TODO ideally, this would be accessed with a lock on the tile node,
        // but that breaks BaseMatrix::tileLayoutConvert(set<ij>, ...)
        return tile_node.at(device);
    }

    void erase(ijdev_tuple ijdev);
    void erase(ij_tuple ij);
    void release(ijdev_tuple ijdev);
private:
    void release(typename TilesMap::iterator iter, int device);
public:
    void freeTileMemory(Tile<scalar_t>* tile);
    void clear();

    //--------------------------------------------------------------------------
    /// @return number of allocated tile nodes (size of tiles map).
    size_t size() const
    {
        LockGuard guard(getTilesMapLock());
        return tiles_.size();
    }

    //--------------------------------------------------------------------------
    /// @return True if map has no tiles.
    bool empty() const { return size() == 0; }

    //--------------------------------------------------------------------------
    /// Return pointer to tiles-map OMP lock
    omp_nest_lock_t* getTilesMapLock()
    {
        return &lock_;
    }

    //--------------------------------------------------------------------------
    std::function<int64_t (int64_t i)> tileMb;
    std::function<int64_t (int64_t j)> tileNb;
    std::function<int (ij_tuple ij)> tileRank;
    std::function<int (ij_tuple ij)> tileDevice;

    //--------------------------------------------------------------------------
    /// @return whether tile {i, j} is local.
    bool tileIsLocal(ij_tuple ij)
    {
        return tileRank(ij) == mpi_rank_;
    }

    Tile<scalar_t>* tileInsert(
        ijdev_tuple ijdev, TileKind, Layout layout=Layout::ColMajor);
    Tile<scalar_t>* tileInsert(
        ijdev_tuple ijdev, scalar_t* data, int64_t lda,
        Layout layout=Layout::ColMajor);

    bool tileExists( ijdev_tuple ijdev )
    {
        LockGuard guard( getTilesMapLock() );
        int64_t i  = std::get<0>(ijdev);
        int64_t j  = std::get<1>(ijdev);
        int device = std::get<2>(ijdev);
        if (device == AnyDevice) {
            return find( {i, j} ) != end();
        }
        else {
            return find( ijdev ) != end();
        }
    }

    void tileMakeTransposable(Tile<scalar_t>* tile);
    void tileLayoutReset(Tile<scalar_t>* tile);

    void tileTick(ij_tuple ij);

    //--------------------------------------------------------------------------
    /// @return tile's life counter.
    int64_t tileLife(ij_tuple ij)
    {
        LockGuard guard( getTilesMapLock() );
        return tiles_.at( ij )->lives();
    }

    //--------------------------------------------------------------------------
    /// Set tile's life counter.
    void tileLife(ij_tuple ij, int64_t life)
    {
        LockGuard guard( getTilesMapLock() );
        tiles_.at( ij )->lives() = life;
    }

    //--------------------------------------------------------------------------
    /// @return tile's receive counter.
    int64_t tileReceiveCount(ij_tuple ij)
    {
        LockGuard guard( getTilesMapLock() );
        return tiles_.at( ij )->receiveCount();
    }

    //--------------------------------------------------------------------------
    /// Increment tile's receive counter.
    void tileIncrementReceiveCount(ij_tuple ij)
    {
        LockGuard guard( getTilesMapLock() );
        tiles_.at( ij )->receiveCount()++;
    }

    //--------------------------------------------------------------------------
    /// Decrement tile's receive counter.
    void tileDecrementReceiveCount(ij_tuple ij)
    {
        LockGuard guard( getTilesMapLock() );
        tiles_.at( ij )->receiveCount()--;
    }

    /// Ensures the tile node exists and increments both the tile life and
    /// recieve count
    void tilePrepareToReceive(ij_tuple ij, int life, Layout layout)
    {
        if (! tileIsLocal(ij)) {
            // Create tile to receive data, with life span.
            // If tile already exists, add to its life span.
            //
            LockGuard guard( getTilesMapLock() );
            int64_t i  = std::get<0>( ij );
            int64_t j  = std::get<1>( ij );

            auto iter = find( ij );

            if (iter == end())
                tileInsert( {i, j, HostNum}, TileKind::Workspace, layout );
            else
                life += tileLife( ij );
            tileLife( ij, life );
            tileIncrementReceiveCount( ij );
        }
    }

    // MOSI management
    /// Gets the state of the given tile
    MOSI tileState(ijdev_tuple ijdev)
    {
        LockGuard guard( getTilesMapLock() );
        auto iter = find( ijdev );
        assert(iter != end());

        int device = std::get<2>(ijdev);
        return iter->second->at(device)->state();
    }

    /// Checks whether the given tile is on hold
    MOSI tileOnHold(ijdev_tuple ijdev)
    {
        LockGuard guard( getTilesMapLock() );
        auto iter = find( ijdev );
        assert(iter != end());

        int device = std::get<2>(ijdev);
        return iter->second->at(device)->stateOn(MOSI::OnHold);
    }

    /// Unsets any hold on the given tile
    void tileUnsetHold(ijdev_tuple ijdev)
    {
        LockGuard guard( getTilesMapLock() );
        auto iter = find( ijdev );
        if (iter != end()) {
            int device = std::get<2>(ijdev);
            iter->second->at(device)->state(~MOSI::OnHold);
        }
    }

private:
    TilesMap tiles_;        ///< map of tiles and associated states
    mutable omp_nest_lock_t lock_;  ///< TilesMap lock
    slate::Memory memory_;  ///< memory allocator
    scalar_t *host_mem;
    std::map< int, std::stack<void*> > allocated_mem_;
    bool own;

    int mpi_rank_;
    static int num_devices_;

    int64_t batch_array_size_;

    // BLAS++ communication queues
    std::vector< lapack::Queue* > comm_queues_;
    // BLAS++ compute queues
    std::vector< std::vector< lapack::Queue* > > compute_queues_;

    // host pointers arrays for batch GEMM
    std::vector< std::vector< scalar_t** > > array_host_;

    // device pointers arrays for batch GEMM
    std::vector< std::vector< scalar_t** > > array_dev_;
};

//------------------------------------------------------------------------------
template <typename scalar_t>
MatrixStorage<scalar_t>::MatrixStorage(
    int64_t m, int64_t n, int64_t mb, int64_t nb,
    GridOrder order, int p, int q, MPI_Comm mpi_comm)
    : tiles_(),
      memory_(sizeof(scalar_t) * mb * nb),  // block size in bytes
      batch_array_size_(0)
{
    slate_mpi_call(
        MPI_Comm_rank(mpi_comm, &mpi_rank_));

    // todo: these are static, but we (re-)initialize with each matrix.
    // todo: similar code in BaseMatrix(...) and MatrixStorage(...)
    num_devices_ = memory_.num_devices_;

    // TODO: these all assume 2D block cyclic with fixed size tiles (mb x nb)
    // lambdas that capture m, n, mb, nb for
    // computing tile's mb (rows) and nb (cols)
    tileMb = [m, mb](int64_t i) { return (i + 1)*mb > m ? m%mb : mb; };
    tileNb = [n, nb](int64_t j) { return (j + 1)*nb > n ? n%nb : nb; };

    // lambda that captures p, q for computing tile's rank,
    // assuming 2D block cyclic
    if (order == GridOrder::Col) {
        tileRank = [p, q]( ij_tuple ij ) {
            int64_t i = std::get<0>( ij );
            int64_t j = std::get<1>( ij );
            return int((i%p) + (j%q)*p);
        };
    }
    else if (order == GridOrder::Row) {
        tileRank = [p, q]( ij_tuple ij ) {
            int64_t i = std::get<0>( ij );
            int64_t j = std::get<1>( ij );
            return int((i%p)*q + (j%q));
        };
    }
    else {
        slate_error( "invalid GridOrder, must be Col or Row" );
    }

    // lambda that captures q, num_devices to distribute local matrix
    // in 1D column block cyclic fashion among devices
    if (num_devices_ > 0) {
        int num_devices = num_devices_;  // local copy to capture
        tileDevice = [q, num_devices](ij_tuple ij) {
            int64_t j = std::get<1>(ij);
            return int(j/q)%num_devices;
        };
    }
    else {
        tileDevice = []( ij_tuple ij ) {
            return HostNum;
        };
    }

    initQueues();
    omp_init_nest_lock(&lock_);
}

//------------------------------------------------------------------------------
/// For memory, assumes tiles of size mb = inTileMb(0) x nb = inTileNb(0).
template <typename scalar_t>
MatrixStorage<scalar_t>::MatrixStorage(
    std::function<int64_t (int64_t i)>& inTileMb,
    std::function<int64_t (int64_t j)>& inTileNb,
    std::function<int (ij_tuple ij)>& inTileRank,
    std::function<int (ij_tuple ij)>& inTileDevice,
    MPI_Comm mpi_comm)
    : tileMb(inTileMb),
      tileNb(inTileNb),
      tileRank(inTileRank),
      tileDevice(inTileDevice),
      tiles_(),
      memory_(sizeof(scalar_t) * inTileMb(0) * inTileNb(0)),  // block size in bytes
      batch_array_size_(0)
{
    slate_mpi_call(
        MPI_Comm_rank(mpi_comm, &mpi_rank_));

    // todo: these are static, but we (re-)initialize with each matrix.
    // todo: similar code in BaseMatrix(...) and MatrixStorage(...)
    num_devices_ = memory_.num_devices_;

    initQueues();
    omp_init_nest_lock(&lock_);
}

//------------------------------------------------------------------------------
/// Destructor deletes all tiles and frees workspace buffers.
///
template <typename scalar_t>
MatrixStorage<scalar_t>::~MatrixStorage()
{
    try {
        clear();
        clearBatchArrays();
        // Clear all host and device memory allocations
        memory_.clearHostBlocks();
        for (int device = 0; device < num_devices_; ++device) {
            blas::Queue* queue = comm_queues_[device];
            memory_.clearDeviceBlocks(device, queue);
        }
        destroyQueues(); // must occur after clearBatchArrays
        omp_destroy_nest_lock(&lock_);
    }
    catch (std::exception const& ex) {
        // If debugging, die on exceptions.
        // Otherwise, ignore errors: destructors should not throw errors!
        assert(false);
    }
}

//------------------------------------------------------------------------------
/// Initializes BLAS++ compute and communcation queues on each device.
/// Also initializes the host and device batch arrays.
/// Called in constructor.
///
template <typename scalar_t>
void MatrixStorage<scalar_t>::initQueues()
{
    comm_queues_   .resize(num_devices_);
    compute_queues_.resize(1);

    compute_queues_.at(0).resize(num_devices_, nullptr);
    for (int device = 0; device < num_devices_; ++device) {
        comm_queues_        [ device ] = new lapack::Queue( device );
        compute_queues_[ 0 ][ device ] = new lapack::Queue( device );
    }

    array_host_.resize(1);
    array_dev_ .resize(1);

    array_host_.at(0).resize(num_devices_, nullptr);
    array_dev_ .at(0).resize(num_devices_, nullptr);
}

//------------------------------------------------------------------------------
/// Destroys BLAS++ compute and communcation queues on each device.
/// As this is called in the destructor, it should NOT throw exceptions.
///
template <typename scalar_t>
void MatrixStorage<scalar_t>::destroyQueues()
{
    int num_queues = int(compute_queues_.size());
    for (int device = 0; device < num_devices_; ++device) {
        delete comm_queues_[device];
               comm_queues_[device] = nullptr;

        for (int queue = 0; queue < num_queues; ++queue) {
            delete compute_queues_.at(queue)[device];
                   compute_queues_.at(queue)[device] = nullptr;
        }
    }
}

//------------------------------------------------------------------------------
/// Allocates batch arrays and BLAS++ queues for all devices.
/// If arrays are already allocated, frees and reallocates the arrays only if
/// batch_size is larger than the existing size.
///
/// @param[in] batch_size
///     Allocate batch arrays as needed so that
///     size of each batch array >= batch_size >= 0.
///
/// @param[in] num_arrays
///     Allocate batch arrays as needed so that
///     number of batch arrays per device >= num_arrays >= 1.
///
// todo: rename resizeBatchArrays?
///
template <typename scalar_t>
void MatrixStorage<scalar_t>::allocateBatchArrays(
    int64_t batch_size, int64_t num_arrays)
{
    assert(batch_size >= 0);
    assert(num_arrays >= 1);
    assert(array_host_.size() ==      array_dev_.size());
    assert(array_host_.size() == compute_queues_.size());

    bool is_resized = false;
    int64_t i_begin = 0;

    if (int64_t(array_host_.size()) < num_arrays) {
        i_begin = array_host_.size();

        array_host_    .resize(num_arrays);
        array_dev_     .resize(num_arrays);
        compute_queues_.resize(num_arrays);

        for (int64_t i = i_begin; i < num_arrays; ++i) {
            array_host_    .at(i).resize(num_devices_, nullptr);
            array_dev_     .at(i).resize(num_devices_, nullptr);
            compute_queues_.at(i).resize(num_devices_, nullptr);
        }
        is_resized = true;
    }

    if ((batch_array_size_ < batch_size) || is_resized) {

        if (batch_array_size_ < batch_size) {
            // Grow all batch arrays, not just new ones.
            i_begin = 0;
        }
        else {
            // Make new batch arrays match old batch arrays.
            batch_size = batch_array_size_;
        }

        assert(int(array_host_.size()) >= num_arrays);

        for (std::size_t i = i_begin; i < array_host_.size(); ++i) {
            for (int device = 0; device < num_devices_; ++device) {

                // Get the original queue for malloc
                blas::Queue* queue = comm_queues_[device];

                // Free host arrays.
                blas::host_free_pinned(array_host_[i][device], *queue);

                // Free device arrays.
                blas::device_free(array_dev_[i][device], *queue);

                // Free queues.
                delete compute_queues_[i][device];

                // Allocate queues.
                compute_queues_[ i ][ device ] = new lapack::Queue( device );

                // Allocate host arrays;
                array_host_[i][device]
                    = blas::host_malloc_pinned<scalar_t*>(batch_size*3, *queue);

                // Allocate device arrays.
                array_dev_[i][device]
                    = blas::device_malloc<scalar_t*>(batch_size*3, *queue);

            }
        }

        batch_array_size_ = batch_size;
    }
}

//------------------------------------------------------------------------------
/// Frees device batch arrays that were allocated by allocateBatchArrays().
///
// todo: rename destroyBatchArrays? freeBatchArrays?
//
template <typename scalar_t>
void MatrixStorage<scalar_t>::clearBatchArrays()
{
    assert(array_host_.size() == array_dev_.size());

    for (std::size_t i = 0; i < array_host_.size(); ++i) {
        for (int device = 0; device < num_devices_; ++device) {

            // Get the original queue for memory allocations
            blas::Queue* queue = comm_queues_[device];

            // Free host arrays.
            if (array_host_[i][device] != nullptr) {
                blas::host_free_pinned(array_host_[i][device], *queue);
                array_host_[i][device] = nullptr;
            }

            // Free device arrays.
            if (array_dev_[i][device] != nullptr) {
                blas::device_free(array_dev_[i][device], *queue);
                array_dev_[i][device] = nullptr;
            }
        }
    }
    batch_array_size_ = 0;
}

//------------------------------------------------------------------------------
/// Reserves num_tiles on host in allocator.
template <typename scalar_t>
void MatrixStorage<scalar_t>::reserveHostWorkspace(int64_t num_tiles)
{
    int64_t n = num_tiles - memory_.capacity( HostNum );
    if (n > 0) {
        memory_.addHostBlocks(n);
        // Usually now capacity == num_tiles, but if multiple
        // threads reserve memory, capacity >= num_tiles.
    }
}

//------------------------------------------------------------------------------
/// Reserves num_tiles on each device in allocator.
template <typename scalar_t>
void MatrixStorage<scalar_t>::reserveDeviceWorkspace(int64_t num_tiles)
{
    for (int device = 0; device < num_devices_; ++device) {
        int64_t n = num_tiles - memory_.capacity(device);
        if (n > 0) {
            blas::Queue* queue = comm_queues_[device];
            memory_.addDeviceBlocks(device, n, queue);
            // Usually now capacity == num_tiles, but if multiple
            // threads reserve memory, capacity >= num_tiles.
        }
    }
}

//------------------------------------------------------------------------------
/// Ensures there is unoccupied workspace for num_tiles on device in allocator.
template <typename scalar_t>
void MatrixStorage<scalar_t>::ensureDeviceWorkspace(int device, int64_t num_tiles)
{
    if (memory_.available(device) < size_t(num_tiles)) {
        // if device==HostNum (-1) use nullptr as queue (not comm_queues_[-1])
        blas::Queue* queue = ( device == HostNum ? nullptr : comm_queues_[device]);
        memory_.addDeviceBlocks(device, num_tiles - memory_.available(device), queue);
    }
}

//------------------------------------------------------------------------------
/// Return tiles allocated memory and extended memory to the memory factory
template <typename scalar_t>
void MatrixStorage<scalar_t>::freeTileMemory(Tile<scalar_t>* tile)
{
    slate_assert(tile != nullptr);
    if (tile->allocated())
        //delete[] tile->data();
        memory_.free(tile->data(), tile->device());
    if (tile->extended())
        memory_.free(tile->extData(), tile->device());
}

//------------------------------------------------------------------------------
/// Clears all host and device workspace tiles.
///
template <typename scalar_t>
void MatrixStorage<scalar_t>::clearWorkspace()
{
    LockGuard guard(getTilesMapLock());
    for (auto iter = begin(); iter != end(); /* incremented below */) {
        auto& tile_node = *(iter->second);
        for (int d = HostNum; d < num_devices_; ++d) {
            if (tile_node.existsOn(d) &&
                tile_node[d]->workspace())
            {
                freeTileMemory(tile_node[d]);
                tile_node.eraseOn(d);
            }
        }
        if (tile_node.empty())
            // Since we can't increment the iterator after deleting the
            // element, use post-fix iter++ to increment it but
            // erase the current value.
            erase((iter++)->first);
        else
            ++iter;
    }
    // Free host & device memory only if there are no unallocated blocks
    // from non-workspace (SlateOwned) tiles.
    if (memory_.allocated( HostNum ) == 0) {
        memory_.clearHostBlocks();
    }

    for (int device = 0; device < num_devices_; ++device) {
        if (memory_.allocated(device) == 0) {
            blas::Queue* queue = comm_queues_[device];
            memory_.clearDeviceBlocks(device, queue);
        }
    }
}

//------------------------------------------------------------------------------
/// Clears all host and device workspace tiles that are not OnHold.
/// For local tiles, it ensures that a valid copy remains.
///
/// Note that local tiles are currently not released if it would leave all
/// remaining tiles invalid, but this behavior may change in the future
/// and should not be relied on.
template <typename scalar_t>
void MatrixStorage<scalar_t>::releaseWorkspace()
{
    LockGuard guard(getTilesMapLock());
    for (auto iter = begin(); iter != end(); /* incremented below */) {
        // Since we can't increment the iterator after deleting the element
        // and release deletes empty nodes, use post-fix iter++ to
        // increment it but pass the current value to release.
        release(iter++, AllDevices);
    }
    // Free host & device memory only if there are no unallocated blocks
    // from non-workspace (SlateOwned) tiles.
    if (memory_.allocated( HostNum ) == 0) {
        memory_.clearHostBlocks();
    }
    for (int device = 0; device < num_devices_; ++device) {
        if (memory_.allocated(device) == 0) {
            blas::Queue* queue = comm_queues_[device];
            memory_.clearDeviceBlocks(device, queue);
        }
    }
}

//------------------------------------------------------------------------------
/// Remove a tile instance from device and delete it unconditionally.
/// If tile node becomes empty, deletes it.
/// If tile's memory was allocated by SLATE, then its memory is freed back
/// to the allocator memory pool.
///
// todo: currently ignores if ijdev doesn't exist; is that right?
template <typename scalar_t>
void MatrixStorage<scalar_t>::erase(ijdev_tuple ijdev)
{
    LockGuard guard(getTilesMapLock());

    auto iter = find(ijdev);
    if (iter != end()) {

        auto& tile_node = *(iter->second);

        int64_t i  = std::get<0>(ijdev);
        int64_t j  = std::get<1>(ijdev);
        int device = std::get<2>(ijdev);

        freeTileMemory(tile_node[device]);
        tile_node.eraseOn(device);

        if (tile_node.empty())
            erase({i, j});
    }
}

//------------------------------------------------------------------------------
/// Remove a tile instance on device and delete it if it is a workspace,
/// not OnHold, and not the last valid instance of a local tile.
/// If tile node becomes empty, deletes it.
/// If tile's memory was allocated by SLATE, then its memory is freed back
/// to the allocator memory pool.
/// device can be AllDevices.
///
/// This is an internal version to share logic between release and
/// releaseWorkspace.
template <typename scalar_t>
void MatrixStorage<scalar_t>::release(
    typename MatrixStorage<scalar_t>::TilesMap::iterator iter,
    int device)
{
    auto& tile_node = *(iter->second);

    int begin = device;
    int end   = device + 1;
    if (device == AllDevices) {
        begin = HostNum;
        end   = num_devices_;
    }

    // Don't release tiles if it'd delete the last valid copy
    // Remote tiles never have the last valid copy
    bool last_valid = tileIsLocal( iter->first );
    for (int dev = HostNum; dev < num_devices_; ++dev) {
        if (tile_node.existsOn( dev )
            && (dev < begin || dev >= end || tile_node[ dev ]->origin())
            && ! tile_node[ dev ]->stateOn( MOSI::Invalid )) {

            last_valid = false;
            break;
        }
    }
    // TODO consider copying to origin when last_valid

    for (int dev = begin; dev < end; ++dev) {
        if (tile_node.existsOn( dev )
            && tile_node[ dev ]->workspace()
            && ! tile_node[ dev ]->stateOn( MOSI::OnHold )
            && (! last_valid || tile_node[ dev ]->stateOn( MOSI::Invalid ))) {

            freeTileMemory( tile_node[ dev ] );
            tile_node.eraseOn( dev );
        }
    }
    if (tile_node.empty())
        erase( iter->first );
}

//------------------------------------------------------------------------------
/// Remove a tile instance on device and delete it
/// if it is a workspace and not OnHold.
/// If tile node becomes empty, deletes it.
/// If tile's memory was allocated by SLATE, then its memory is freed back
/// to the allocator memory pool.
/// For local tiles, it ensures that a valid copy remains.
/// device can be AllDevices.
///
/// Note that local tiles are currently not released if it would leave all
/// remaining tiles invalid, but this behavior may change in the future
/// and should not be relied on.
///
// todo: currently ignores if ijdev doesn't exist; is that right?
template <typename scalar_t>
void MatrixStorage<scalar_t>::release(ijdev_tuple ijdev)
{
    LockGuard guard(getTilesMapLock());

    int64_t i  = std::get<0>(ijdev);
    int64_t j  = std::get<1>(ijdev);
    int device = std::get<2>(ijdev);
    auto iter = find( { i, j } ); // not device, to allow AllDevices
    if (iter != end()) {
        release(iter, device);
    }
}

//------------------------------------------------------------------------------
/// Remove a tile with all instances on all devices from map and delete it
/// unconditionally.
/// If tile's memory was allocated by SLATE, then its memory is freed back
/// to the allocator memory pool.
///
// todo: currently ignores if ijdev doesn't exist; is that right?
template <typename scalar_t>
void MatrixStorage<scalar_t>::erase(ij_tuple ij)
{
    LockGuard guard(getTilesMapLock());

    auto iter = tiles_.find(ij);
    if (iter != tiles_.end()) {

        auto& tile_node = iter->second;

        for (int d = HostNum; (! tile_node->empty()) && d < num_devices_; ++d) {
            if (tile_node->existsOn(d)) {
                freeTileMemory(tile_node->at(d));
                tile_node->eraseOn(d);
            }
        }
        tiles_.erase(ij);
    }
}

//------------------------------------------------------------------------------
/// Delete all tiles.
template <typename scalar_t>
void MatrixStorage<scalar_t>::clear()
{
    LockGuard guard(getTilesMapLock());

    for (auto iter = begin(); iter != end(); /* incremented below */) {
        // erasing the element invalidates the iterator,
        // so use iter++ to erase the current value but increment it first.
        erase((iter++)->first); // todo: in-efficient
    }

    // todo: what if some tiles were not erased
    slate_assert(tiles_.size() == 0);  // should be empty now
}

//------------------------------------------------------------------------------
/// Allocates a memory block on device to be used as a workspace buffer,
/// to be released with call to releaseWorkspaceBuffer()
/// @return pointer to memory block on device
///
/// @param[in] device
///     Device ID (GPU or Host) where the memory block is needed.
///
template <typename scalar_t>
scalar_t* MatrixStorage<scalar_t>::allocWorkspaceBuffer(int device)
{
    int64_t mb = tileMb(0);
    int64_t nb = tileNb(0);
    // if device==HostNum (-1) use nullptr as queue (not comm_queues_[-1])
    blas::Queue* queue = ( device == HostNum ? nullptr : comm_queues_[device]);
    scalar_t* data = (scalar_t*) memory_.alloc(device, sizeof(scalar_t) * mb * nb, queue);
    return data;
}

//------------------------------------------------------------------------------
/// Release the memory block indicated by data on device to the memory manager
///
/// @param[in] data
///     Pointer to memory block to be released.
///
/// @param[in] device
///     Device ID (GPU or Host) where the memory block is.
///
template <typename scalar_t>
void MatrixStorage<scalar_t>::releaseWorkspaceBuffer(scalar_t* data, int device)
{
    memory_.free(data, device);
}

//------------------------------------------------------------------------------
/// Inserts tile {i, j} on given device, which can be host,
/// allocating new memory for it.
/// Creates TileNode(i, j) if not already exists (Tile's life is set 0).
/// Tile kind should be either TileKind::Workspace or TileKind::SlateOwned.
///
/// @return Pointer to newly inserted Tile.
///
template <typename scalar_t>
Tile<scalar_t>* MatrixStorage<scalar_t>::tileInsert(
    ijdev_tuple ijdev, TileKind kind, Layout layout)
{
    assert(kind == TileKind::Workspace ||
           kind == TileKind::SlateOwned);
    int64_t i  = std::get<0>(ijdev);
    int64_t j  = std::get<1>(ijdev);
    int device = std::get<2>(ijdev);

    LockGuard tiles_guard(getTilesMapLock());

    // find the tileNode
    // if not found, insert new-entry in TilesMap
    if (find({i, j}) == end()) {
        tiles_[{i, j}] = std::make_shared<TileNode_t>( num_devices_ );
    }
    auto& tile_node = this->at({i, j});

    // if tile instance does not exist, insert new instance
    if (! tile_node.existsOn(device)) {
        int64_t mb = tileMb(i);
        int64_t nb = tileNb(j);
        // if device==HostNum (-1) use nullptr as queue (not comm_queues_[-1])
        blas::Queue* queue = ( device == HostNum ? nullptr : comm_queues_[device]);
        scalar_t* data = (scalar_t*) memory_.alloc(device, sizeof(scalar_t) * mb * nb, queue);
        int64_t stride = layout == Layout::ColMajor ? mb : nb;
        Tile<scalar_t>* tile
            = new Tile<scalar_t>(mb, nb, data, stride, device, kind, layout);
        tile_node.insertOn(device, tile, kind == TileKind::Workspace ?
                                         MOSI::Invalid :
                                         MOSI::Shared);
    }
    return tile_node[device];
}

//------------------------------------------------------------------------------
/// This is intended for inserting the original matrix.
/// Inserts tile {i, j} on given device, which can be host,
/// wrapping existing memory for it.
/// Sets tile kind = TileKind::UserOwned.
/// This will be the origin tile, thus TileNode(i, j) should not pre-exist.
/// @return Pointer to newly inserted Tile.
///
template <typename scalar_t>
Tile<scalar_t>* MatrixStorage<scalar_t>::tileInsert(
    ijdev_tuple ijdev, scalar_t* data, int64_t lda, Layout layout)
{
    int64_t i  = std::get<0>(ijdev);
    int64_t j  = std::get<1>(ijdev);
    int device = std::get<2>(ijdev);
    slate_assert( HostNum <= device && device < num_devices_ );

    LockGuard guard(getTilesMapLock());

    assert(find({i, j}) == end());
    // insert new-entry in map
    tiles_[{i, j}] = std::make_shared<TileNode_t>( num_devices_ );

    auto& tile_node = this->at({i, j});

    // if tile instance does not exist, insert new instance
    if (! tile_node.existsOn(device)) {
        int64_t mb = tileMb(i);
        int64_t nb = tileNb(j);
        Tile<scalar_t>* tile
            = new Tile<scalar_t>(
                  mb, nb, data, lda, device, TileKind::UserOwned, layout);
        tile_node.insertOn(device, tile, MOSI::Shared);
    }
    return tile_node[device];
}

//------------------------------------------------------------------------------
/// Makes tile layout convertible by extending its data buffer.
/// Attaches an auxiliary buffer to hold the transposed data when needed.
///
/// @param[in,out] tile
///     Pointer to tile to extend its data buffer.
///
template <typename scalar_t>
void MatrixStorage<scalar_t>::tileMakeTransposable(Tile<scalar_t>* tile)
{
    // quick return
    if (tile->isTransposable())
        return;

    int device = tile->device();
    int64_t mb = tileMb(0);
    int64_t nb = tileNb(0);
    // if device==HostNum (-1) use nullptr as queue (not comm_queues_[-1])
    blas::Queue* queue = ( device == HostNum ? nullptr : comm_queues_[device]);
    scalar_t* data = (scalar_t*) memory_.alloc(device, sizeof(scalar_t) * mb * nb, queue);
    tile->makeTransposable(data);
}

//------------------------------------------------------------------------------
/// Resets the extended tile.
/// Frees the extended buffer and returns to memory manager
/// then resets the tile's extended member fields
///
/// @param[in,out] tile
///     Pointer to extended tile.
///
template <typename scalar_t>
void MatrixStorage<scalar_t>::tileLayoutReset(Tile<scalar_t>* tile)
{
    if (tile->extended()) {
        memory_.free(tile->extData(), tile->device());
        tile->layoutReset();
    }
}

//------------------------------------------------------------------------------
/// If tile {i, j} is a workspace tile (i.e., not local),
/// decrement its life counter by 1;
/// if life reaches 0, erase tile on the host and all devices.
///
template <typename scalar_t>
void MatrixStorage<scalar_t>::tileTick(ij_tuple ij)
{
    if (! tileIsLocal(ij)) {
        LockGuard guard(getTilesMapLock());
        int64_t life = --(tiles_.at(ij)->lives());
        if (life == 0) {
            erase(ij);
        }
    }
}

//------------------------------------------------------------------------------
template <typename scalar_t>
int MatrixStorage<scalar_t>::num_devices_ = 0;

} // namespace slate

#endif // SLATE_STORAGE_HH
