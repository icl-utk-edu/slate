//------------------------------------------------------------------------------
// Copyright (c) 2017, University of Tennessee
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the University of Tennessee nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF TENNESSEE BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//------------------------------------------------------------------------------
// This research was supported by the Exascale Computing Project (17-SC-20-SC),
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
//------------------------------------------------------------------------------
// For assistance with SLATE, email <slate-user@icl.utk.edu>.
// You can also join the "SLATE User" Google group by going to
// https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user,
// signing in with your Google credentials, and then clicking "Join group".
//------------------------------------------------------------------------------

#ifndef SLATE_STORAGE_HH
#define SLATE_STORAGE_HH

#include "slate/internal/Memory.hh"
#include "slate/Tile.hh"
#include "slate/types.hh"
#include "slate/internal/util.hh"

#include "lapack.hh"

#include <algorithm>
#include <functional>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "slate/internal/cuda.hh"
#include "slate/internal/cublas.hh"
#include "slate/internal/mpi.hh"
#include "slate/internal/openmp.hh"
#include "slate/internal/LockGuard.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Type-safe wrapper for cudaMalloc. Throws errors.
template<typename value_type>
void slateCudaMalloc(value_type** ptr, size_t nelements)
{
    slate_cuda_call(
        cudaMalloc((void**) ptr, nelements * sizeof(value_type)));
}

template<typename value_type>
void slateCudaMallocHost(value_type** ptr, size_t nelements)
{
    slate_cuda_call(
        cudaMallocHost((void**) ptr, nelements * sizeof(value_type)));
}

//------------------------------------------------------------------------------
/// A tile state in the MOSI coherency protocol
enum MOSI {
    Modified = 0x100,   ///< tile data is modified, other instances should be Invalid, cannot be purged
    OnHold = 0x1000,  ///< a hold is placed on this tile instance, cannot be purged
    Shared = 0x010,   ///< tile data is up-to-date, other instances may be Shared, or Invalid, may be purged
    Invalid = 0x001,   ///< tile data is obsolete, other instances may be Modified, Shared, or Invalid, may be purged
};
typedef short MOSI_State;

//------------------------------------------------------------------------------
///
template <typename scalar_t>
class TileInstance {
private:
    Tile<scalar_t>* tile_;
    MOSI_State state_;
    mutable omp_nest_lock_t lock_;

public:
    TileInstance()
        : tile_(nullptr),
          state_(MOSI::Invalid)
    {
        omp_init_nest_lock(&lock_);
    }

    /// Destructor for TileInstance class
    ~TileInstance()
    {
        omp_destroy_nest_lock(&lock_);
        assert(tile_ == nullptr);
    }

    //--------------------------------------------------------------------------
    // 2. copy constructor -- not allowed; lock_ is not copyable
    // 3. move constructor -- not allowed; lock_ is not copyable
    // 4. copy assignment  -- not allowed; lock_ is not copyable
    // 5. move assignment  -- not allowed; lock_ is not copyable
    TileInstance(TileInstance&  orig) = delete;
    TileInstance(TileInstance&& orig) = delete;
    TileInstance& operator = (TileInstance&  orig) = delete;
    TileInstance& operator = (TileInstance&& orig) = delete;

    //--------------------------------------------------------------------------
    /// Get and Set tile pointer
    Tile<scalar_t>* tile() const { return tile_;}
    void tile(Tile<scalar_t>* tile) { tile_ = tile; }

    //--------------------------------------------------------------------------
    /// Returns whether this tile instance is valid (Tile instance exists)
    bool valid() const { return tile_ != nullptr;}

    //--------------------------------------------------------------------------
    /// Initialize tile pointer and MOSI state
    void init(Tile<scalar_t>* tile, MOSI_State state)
    {
        slate_assert(tile_ == nullptr);
        slate_assert(tile  != nullptr);
        tile_ = tile;
        state_ = state;
    }

    //--------------------------------------------------------------------------
    /// Retrun pointer to tile instance OMP lock
    omp_nest_lock_t* getLock()
    {
        return &lock_;
    }

    //--------------------------------------------------------------------------
    /// Set the state to: Modified/Shared/Invalid, or set On/Off the OnHold flag
    void setState(MOSI_State stateIn)
    {
        switch (stateIn) {
            case MOSI::Modified:
            case MOSI::Shared:
            case MOSI::Invalid:
                state_ = (state_ & MOSI::OnHold) | stateIn;
                break;
            case MOSI::OnHold:
                state_ |= stateIn;
                break;
            case ~MOSI::OnHold:
                state_ &= stateIn;
                break;
            default:
                assert(false);  // Unknown state
                break;
        }
    }

    /// Returns the current MOSI state (Modified/Shared/Invalid)
    /// to check the OnHold flag use stateOn
    MOSI getState() const
    {
        return MOSI(state_ & MOSI_State(~MOSI::OnHold));
    }

    /// returns whether the Modified/Shared/Invalid state or the OnHold flag is On
    bool stateOn(MOSI_State stateIn) const
    {
        switch (stateIn) {
            case MOSI::Modified:
            case MOSI::Shared:
            case MOSI::Invalid:
                return (state_ & ~MOSI::OnHold) == stateIn;
                break;
            case MOSI::OnHold:
                return (state_ & MOSI::OnHold) == stateIn;
                break;
            default:
                assert(false);  // Unknown state
                break;
        }
        return false;
    }
};

//------------------------------------------------------------------------------
///
template <typename scalar_t>
class TileNode {
private:
    typedef TileInstance<scalar_t> TileInstance_t;
    /// vector of tile instances indexed by device id.
    using TileInstances = std::vector< std::unique_ptr<TileInstance_t> >;

    TileInstances tile_instances_;
    int num_instances_;
    int64_t life_;

    /// OMP lock used to protect operations that modify the TileInstances within
    mutable omp_nest_lock_t lock_;

public:
    /// Constructor for TileNode class
    TileNode(int num_devices)
        : num_instances_(0),
          life_(0)
    {
        slate_assert(num_devices >= 0);
        omp_init_nest_lock(&lock_);
        for (int d = 0; d < num_devices+1; ++d) {
            tile_instances_.push_back(
                std::unique_ptr<TileInstance_t>( new TileInstance_t() ));
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
    // 2. copy constructor -- not allowed; lock_ and tile_instances_ are not copyable
    // 3. move constructor -- not allowed; lock_ and tile_instances_ are not copyable
    // 4. copy assignment  -- not allowed; lock_ and tile_instances_ are not copyable
    // 5. move assignment  -- not allowed; lock_ and tile_instances_ are not copyable
    TileNode(TileNode&  orig) = delete;
    TileNode(TileNode&& orig) = delete;
    TileNode& operator = (TileNode&  orig) = delete;
    TileNode& operator = (TileNode&& orig) = delete;

    //--------------------------------------------------------------------------
    /// Retrun pointer to tile instance OMP lock
    omp_nest_lock_t* getLock()
    {
        return &lock_;
    }

    //--------------------------------------------------------------------------
    /// Inserts a tile instance at device and increments the number of resident instances
    void insertOn(int device, Tile<scalar_t>* tile, MOSI_State state)
    {
        slate_assert(device >= -1 && device+1 < int(tile_instances_.size()));
        slate_assert(! tile_instances_[device+1]->valid());
        tile_instances_[device+1]->init(tile, state);
        ++num_instances_;
    }

    //--------------------------------------------------------------------------
    /// Returns whether a tile instance exists at device
    bool existsOn(int device) const
    {
        slate_assert(device >= -1 && device+1 < int(tile_instances_.size()));
        return tile_instances_[device+1]->valid();
    }

    //--------------------------------------------------------------------------
    /// Deletes the tile instance at device
    // CAUTION: tile's memory must have been already released to MatrixStorage Memory
    void eraseOn(int device)
    {
        slate_assert(device >= -1 && device+1 < int(tile_instances_.size()));
        if (tile_instances_[device+1]->valid()) {
            tile_instances_[device+1]->setState(MOSI::Invalid);
            delete tile_instances_[device+1]->tile();
            tile_instances_[device+1]->tile(nullptr);
            --num_instances_;
        }
    }

    //--------------------------------------------------------------------------
    /// Returns a reference to the tile instance at device
    TileInstance_t& operator[](int device) const
    {
        slate_assert(device >= -1 && device+1 < int(tile_instances_.size()));
        return *(tile_instances_[device+1]);
    }

    //--------------------------------------------------------------------------
    /// Returns a reference to the tile instance at device
    TileInstance_t& at(int dev) const
    {
        slate_assert(dev >= -1 && dev+1 < int(tile_instances_.size()));
        return *(tile_instances_[dev+1]);
    }

    int numInstances() const
    {
        return num_instances_;
    }

    int64_t& lives()
    {
        return life_;
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

    typedef TileInstance<scalar_t> TileInstance_t;
    typedef TileNode<scalar_t> TileNode_t;

    using ijdev_tuple = std::tuple<int64_t, int64_t, int>;
    using ij_tuple    = std::tuple<int64_t, int64_t>;
    using TilesMap = std::map< ij_tuple, std::unique_ptr<TileNode_t> >;

    MatrixStorage(int64_t m, int64_t n, int64_t mb, int64_t nb,
                  int p, int q, MPI_Comm mpi_comm);

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
    void initCudaStreams();
    void destroyCudaStreams();

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

    //--------------------------------------------------------------------------
    /// @return TileNode(i, j) if it has instance on device, end() otherwise
    typename TilesMap::iterator find(ijdev_tuple ijdev)
    {
        LockGuard guard(getTilesMapLock());
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
        LockGuard guard(getTilesMapLock());
        return tiles_.find(ij);
    }

    //--------------------------------------------------------------------------
    /// @return begin iterator of TileNode map
    typename TilesMap::iterator begin()
    {
        LockGuard guard(getTilesMapLock());
        return tiles_.begin();
    }

    //--------------------------------------------------------------------------
    /// @return begin iterator of TileNode map
    typename TilesMap::iterator end()
    {
        LockGuard guard(getTilesMapLock());
        return tiles_.end();
    }

    //--------------------------------------------------------------------------
    /// @return reference to single tile instance.
    /// Throws exception if instance doesn't exist.
    // at() doesn't create new (null) entries in map as operator[] would
    TileInstance_t& at(ijdev_tuple ijdev)
    {
        LockGuard guard(getTilesMapLock());
        int64_t i  = std::get<0>(ijdev);
        int64_t j  = std::get<1>(ijdev);
        int device = std::get<2>(ijdev);
        auto& tile_node = tiles_.at({i, j});
        slate_assert(tile_node->existsOn(device));
        return tile_node->at(device);
    }

    //--------------------------------------------------------------------------
    /// @return reference to TileNode(i, j).
    /// Throws exception if entry doesn't exist.
    // at() doesn't create new (null) entries in map as operator[] would
    TileNode_t& at(ij_tuple ij)
    {
        LockGuard guard(getTilesMapLock());
        return *(tiles_.at(ij));
    }

    void erase(ijdev_tuple ijdev);
    void erase(ij_tuple ij);
    void release(ijdev_tuple ijdev);
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
    /// Retrun pointer to tiles-map OMP lock
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

    TileInstance<scalar_t>& tileInsert(
        ijdev_tuple ijdev, TileKind, Layout layout=Layout::ColMajor);
    TileInstance<scalar_t>& tileInsert(
        ijdev_tuple ijdev, scalar_t* data, int64_t lda,
        Layout layout=Layout::ColMajor);
    TileInstance<scalar_t>& tileAcquire(ijdev_tuple ijdev, Layout layout);

    void tileMakeTransposable(Tile<scalar_t>* tile);
    void tileLayoutReset(Tile<scalar_t>* tile);

    void tileTick(ij_tuple ij);

    //--------------------------------------------------------------------------
    /// @return tile's life counter.
    int64_t tileLife(ij_tuple ij)
    {
        LockGuard guard(getTilesMapLock());
        return tiles_.at(ij)->lives();
    }

    //--------------------------------------------------------------------------
    /// Set tile's life counter.
    void tileLife(ij_tuple ij, int64_t life)
    {
        LockGuard guard(getTilesMapLock());
        tiles_.at(ij)->lives() = life;
    }

private:
    TilesMap tiles_;        ///< map of tiles and associated states
    mutable omp_nest_lock_t lock_;  ///< TilesMap lock
    slate::Memory memory_;  ///< memory allocator
    scalar_t *host_mem;
    std::map< int, std::stack<void*> > allocated_mem_;
    bool own;

    int mpi_rank_;
    static int host_num_;
    static int num_devices_;

    int64_t batch_array_size_;

    // CUDA streams and cuBLAS handles
    std::vector<cudaStream_t> compute_streams_;
    std::vector<cudaStream_t> comm_streams_;
    std::vector<cublasHandle_t> cublas_handles_;

    // host pointers arrays for batch GEMM
    std::vector< std::vector< scalar_t** > > array_host_;

    // device pointers arrays for batch GEMM
    std::vector< std::vector< scalar_t** > > array_dev_;
};

//------------------------------------------------------------------------------
template <typename scalar_t>
MatrixStorage<scalar_t>::MatrixStorage(
    int64_t m, int64_t n, int64_t mb, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
    : tiles_(),
      memory_(sizeof(scalar_t) * mb * nb),  // block size in bytes
      batch_array_size_(0)
{
    slate_mpi_call(
        MPI_Comm_rank(mpi_comm, &mpi_rank_));

    // todo: these are static, but we (re-)initialize with each matrix.
    // todo: similar code in BaseMatrix(...) and MatrixStorage(...)
    host_num_    = memory_.host_num_;
    num_devices_ = memory_.num_devices_;

    // TODO: these all assume 2D block cyclic with fixed size tiles (mb x nb)
    // lambdas that capture m, n, mb, nb for
    // computing tile's mb (rows) and nb (cols)
    tileMb = [m, mb](int64_t i) { return (i + 1)*mb > m ? m%mb : mb; };
    tileNb = [n, nb](int64_t j) { return (j + 1)*nb > n ? n%nb : nb; };

    // lambda that captures p, q for computing tile's rank,
    // assuming 2D block cyclic
    tileRank = [p, q](ij_tuple ij) {
        int64_t i = std::get<0>(ij);
        int64_t j = std::get<1>(ij);
        return int(i%p + (j%q)*p);
    };

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
        int host_num = host_num_;  // local copy to capture
        tileDevice = [host_num](ij_tuple ij) {
            return host_num;
        };
    }

    array_host_.resize(1);
    array_dev_.resize(1);

    array_host_.at(0).resize(num_devices_, nullptr);
    array_dev_.at(0).resize(num_devices_, nullptr);

    initCudaStreams();

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
    host_num_    = memory_.host_num_;
    num_devices_ = memory_.num_devices_;

    array_host_.resize(1);
    array_dev_.resize(1);

    // todo: factor out this duplicated code.
    array_host_.at(0).resize(num_devices_, nullptr);
    array_dev_.at(0).resize(num_devices_, nullptr);

    initCudaStreams();

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
        destroyCudaStreams();
        clearBatchArrays();
        omp_destroy_nest_lock(&lock_);
    }
    catch (std::exception const& ex) {
        // If debugging, die on exceptions.
        // Otherwise, ignore errors: destructors should not throw errors!
        assert(false);
    }
}

//------------------------------------------------------------------------------
/// Initializes CUDA streams and cuBLAS handles on each device.
/// Called in constructor.
///
template <typename scalar_t>
void MatrixStorage<scalar_t>::initCudaStreams()
{
    compute_streams_.resize(num_devices_);
    comm_streams_   .resize(num_devices_);
    cublas_handles_ .resize(num_devices_);

    for (int device = 0; device < num_devices_; ++device) {
        slate_cuda_call(
            cudaSetDevice(device));
        slate_cuda_call(
            cudaStreamCreate(&compute_streams_[device]));
        // todo: need to have seperate in/out streams (at least), or multiple streams
        slate_cuda_call(
            cudaStreamCreate(&comm_streams_[device]));

        // create cuBLAS handles, associated with compute_streams_
        slate_cublas_call(
            cublasCreate(&cublas_handles_[device]));

        slate_cublas_call(
            cublasSetStream(cublas_handles_[device],
                                          compute_streams_[device]));
    }
}

//------------------------------------------------------------------------------
/// Destroys CUDA streams and cuBLAS handles on each device.
/// As this is called in the destructor, it should NOT throw exceptions.
///
template <typename scalar_t>
void MatrixStorage<scalar_t>::destroyCudaStreams()
{
    for (int device = 0; device < num_devices_; ++device) {
        slate_cuda_call(
            cudaSetDevice(device));

        // destroy cuBLAS handles, associated with compute_streams_
        slate_cublas_call(
            cublasDestroy(cublas_handles_[device]));
        cublas_handles_[device] = nullptr;

        // destroy CUDA streams
        slate_cuda_call(
            cudaStreamDestroy(compute_streams_[device]));
        compute_streams_[device] = nullptr;

        slate_cuda_call(
            cudaStreamDestroy(comm_streams_[device]));
        comm_streams_[device] = nullptr;
    }
}

//------------------------------------------------------------------------------
/// Allocates CUDA batch arrays.
/// If arrays are already allocated, frees and reallocates the arrays only if
/// batch_size is larger than the existing size.
///
/// @param[in] batch_size
///     On exit, size of each batch array >= batch_size >= 0.
///
// todo: rename resizeBatchArrays?
///
template <typename scalar_t>
void MatrixStorage<scalar_t>::allocateBatchArrays(
    int64_t batch_size, int64_t num_arrays)
{
    assert(batch_size >= 0);
    assert(num_arrays >= 1);

    bool is_resized = false;
    int64_t i_begin = 0;

    if (int64_t(array_host_.size()) < num_arrays) {
        // const int64_t temp = batch_array_size_;
        // clearBatchArrays();
        // batch_array_size_ = temp;

        i_begin = array_host_.size();

        array_host_.resize(num_arrays);
        array_dev_.resize(num_arrays);

        for (int64_t i = i_begin; i < num_arrays; ++i) {
            std::vector< scalar_t** >& array_host = array_host_.at(i);
            std::vector< scalar_t** >& array_dev  = array_dev_.at(i);
            array_host.resize(num_devices_, nullptr);
            array_dev.resize(num_devices_, nullptr);
        }
        is_resized = true;
    }

    if ((batch_array_size_ < batch_size) || is_resized) {

        if (batch_array_size_ < batch_size) {
            i_begin = 0;
        }
        else {
            batch_size = batch_array_size_;
        }

        assert(int(array_host_.size()) >= num_arrays);

        for (std::size_t i = i_begin; i < array_host_.size(); ++i) {
            std::vector< scalar_t** >& array_host = array_host_.at(i);
            std::vector< scalar_t** >& array_dev  = array_dev_.at(i);

            assert(int(array_host.size()) == num_devices_);

            for (int device = 0; device < num_devices_; ++device) {
                slate_cuda_call(cudaSetDevice(device));

                // Free host arrays.
                slate_cuda_call(cudaFreeHost(array_host[device]));

                // Free device arrays.
                slate_cuda_call(cudaFree(array_dev[device]));

                // Allocate host arrays.
                slateCudaMallocHost(&array_host[device], batch_size*3);

                // Allocate device arrays.
                slateCudaMalloc(&array_dev[device], batch_size*3);
            }
        }

        batch_array_size_ = batch_size;
    }
}

//------------------------------------------------------------------------------
/// Frees CUDA batch arrays that were allocated by allocateBatchArrays().
///
// todo: rename destroyBatchArrays? freeBatchArrays?
//
template <typename scalar_t>
void MatrixStorage<scalar_t>::clearBatchArrays()
{
    for (std::size_t i = 0; i < array_host_.size(); ++i) {
        std::vector< scalar_t** >& array_host = array_host_.at(i);
        std::vector< scalar_t** >& array_dev  = array_dev_.at(i);

        assert(int(array_host.size()) == num_devices_);
        for (int device = 0; device < num_devices_; ++device) {
            slate_cuda_call(cudaSetDevice(device));

            // Free host arrays.
            slate_cuda_call(cudaFreeHost(array_host[device]));

            // Free device arrays.
            slate_cuda_call(cudaFree(array_dev[device]));

            array_host[device] = nullptr;

            array_dev[device] = nullptr;
        }
    }
    batch_array_size_ = 0;
}

//------------------------------------------------------------------------------
/// Reserves num_tiles on host in allocator.
template <typename scalar_t>
void MatrixStorage<scalar_t>::reserveHostWorkspace(int64_t num_tiles)
{
    memory_.addHostBlocks(num_tiles);
}

//------------------------------------------------------------------------------
/// Reserves num_tiles on each device in allocator.
template <typename scalar_t>
void MatrixStorage<scalar_t>::reserveDeviceWorkspace(int64_t num_tiles)
{
    for (int device = 0; device < num_devices_; ++device)
        memory_.addDeviceBlocks(device, num_tiles);
}

//------------------------------------------------------------------------------
/// Ensures there is unoccupied workspace for num_tiles on device in allocator.
template <typename scalar_t>
void MatrixStorage<scalar_t>::ensureDeviceWorkspace(int device, int64_t num_tiles)
{
    if (memory_.available(device) < size_t(num_tiles))
        memory_.addDeviceBlocks(device, num_tiles - memory_.available(device));
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
    // incremented below
    for (auto iter = begin(); iter != end();) {
        auto& tile_node = *(iter->second);
        for (int d = host_num_; d < num_devices_; ++d) {
            if (tile_node.existsOn(d) &&
                tile_node[d].tile()->workspace())
            {
                freeTileMemory(tile_node[d].tile());
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
    if (memory_.allocated(host_num_) == 0) {
        memory_.clearHostBlocks();
    }

    for (int device = 0; device < num_devices_; ++device) {
        if (memory_.allocated(device) == 0) {
            memory_.clearDeviceBlocks(device);
        }
    }
}

//------------------------------------------------------------------------------
/// Clears all host and device workspace tiles that are not OnHold nor Modified.
///
template <typename scalar_t>
void MatrixStorage<scalar_t>::releaseWorkspace()
{
    LockGuard guard(getTilesMapLock());
    // incremented below
    for (auto iter = begin(); iter != end();) {
        auto& tile_node = *(iter->second);
        for (int d = host_num_; d < num_devices_; ++d) {
            if (tile_node.existsOn(d) &&
                tile_node[d].tile()->workspace() &&
                ! (tile_node[d].stateOn(MOSI::OnHold) ||
                   tile_node[d].stateOn(MOSI::Modified))
                )
            {
                freeTileMemory(tile_node[d].tile());
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
    if (memory_.allocated(host_num_) == 0) {
        memory_.clearHostBlocks();
    }
    for (int device = 0; device < num_devices_; ++device) {
        if (memory_.allocated(device) == 0) {
            memory_.clearDeviceBlocks(device);
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

        freeTileMemory(tile_node[device].tile());
        tile_node.eraseOn(device);

        if (tile_node.empty())
            erase({i, j});
    }
}

//------------------------------------------------------------------------------
/// Remove a tile instance on device and delete it
/// if it is a workspace and not OnHold nor Modified.
/// If tile node becomes empty, deletes it.
/// If tile's memory was allocated by SLATE, then its memory is freed back
/// to the allocator memory pool.
///
// todo: currently ignores if ijdev doesn't exist; is that right?
template <typename scalar_t>
void MatrixStorage<scalar_t>::release(ijdev_tuple ijdev)
{
    LockGuard guard(getTilesMapLock());

    auto iter = find(ijdev);
    if (iter != end()) {

        auto& tile_node = *(iter->second);

        int64_t i  = std::get<0>(ijdev);
        int64_t j  = std::get<1>(ijdev);
        int device = std::get<2>(ijdev);

        if (tile_node[device].tile()->workspace() &&
            ! (tile_node[device].stateOn(MOSI::OnHold) ||
               tile_node[device].stateOn(MOSI::Modified))
            ) {
            freeTileMemory(tile_node[device].tile());
            tile_node.eraseOn(device);
        }
        if (tile_node.empty())
            erase({i, j});
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

        for (int d = host_num_; (! tile_node->empty()) && d < num_devices_; ++d) {
            if (tile_node->existsOn(d)) {
                freeTileMemory(tile_node->at(d).tile());
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

    // incremented below
    for (auto iter = begin(); iter != end();) {
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
    scalar_t* data = (scalar_t*) memory_.alloc(device, sizeof(scalar_t) * mb * nb);
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
/// Acquires tile {i, j} on given device, which can be host,
/// allocating new memory for it.
/// TileNode(i, j) is assumed to pre-exist (equivalently the origin tile),
/// thus, tile kind is set to TileKind::Workspace,
///
/// @return Reference to newly inserted TileInstance.
///
template <typename scalar_t>
TileInstance<scalar_t>& MatrixStorage<scalar_t>::tileAcquire(
    ijdev_tuple ijdev, Layout layout)
{
    int64_t i  = std::get<0>(ijdev);
    int64_t j  = std::get<1>(ijdev);
    int device = std::get<2>(ijdev);

    LockGuard tiles_guard(getTilesMapLock());

    // find the tileNode
    // if not found, insert new-entry in TilesMap
    // todo: is this needed?
    if (find({i, j}) == end()) {
        tiles_[{i, j}] = std::unique_ptr<TileNode_t>( new TileNode_t( num_devices_ ) );
    }

    auto& tile_node = this->at({i, j});

    // if tile instance does not exist, insert new instance
    if (! tile_node.existsOn(device)) {
        int64_t mb = tileMb(i);
        int64_t nb = tileNb(j);
        scalar_t* data = (scalar_t*) memory_.alloc(device, sizeof(scalar_t) * mb * nb);
        int64_t stride = layout == Layout::ColMajor ? mb : nb;
        Tile<scalar_t>* tile
            = new Tile<scalar_t>(
                  mb, nb, data, stride, device, TileKind::Workspace, layout);
        tile_node.insertOn(device, tile, MOSI::Invalid);
    }
    return tile_node[device];
}

//------------------------------------------------------------------------------
/// Inserts tile {i, j} on given device, which can be host,
/// allocating new memory for it.
/// Creates TileNode(i, j) if not already exists (Tile's life is set 0).
/// Tile kind should be either TileKind::Workspace or TileKind::SlateOwned.
///
/// @return Reference to newly inserted TileInstance.
///
template <typename scalar_t>
TileInstance<scalar_t>& MatrixStorage<scalar_t>::tileInsert(
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
        tiles_[{i, j}] = std::unique_ptr<TileNode_t>( new TileNode_t( num_devices_ ) );
    }
    auto& tile_node = this->at({i, j});

    // if tile instance does not exist, insert new instance
    if (! tile_node.existsOn(device)) {
        int64_t mb = tileMb(i);
        int64_t nb = tileNb(j);
        scalar_t* data = (scalar_t*) memory_.alloc(device, sizeof(scalar_t) * mb * nb);
        //scalar_t* data = new scalar_t[mb*nb];
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
/// @return Pointer to newly inserted TileInstance.
///
template <typename scalar_t>
TileInstance<scalar_t>& MatrixStorage<scalar_t>::tileInsert(
    ijdev_tuple ijdev, scalar_t* data, int64_t lda, Layout layout)
{
    int64_t i  = std::get<0>(ijdev);
    int64_t j  = std::get<1>(ijdev);
    int device = std::get<2>(ijdev);
    slate_assert(host_num_ <= device && device < num_devices_);

    LockGuard guard(getTilesMapLock());

    assert(find({i, j}) == end());
    // insert new-entry in map
    tiles_[{i, j}] = std::unique_ptr<TileNode_t>( new TileNode_t( num_devices_ ) );

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
    //printf("\n tileInsert2 \n");
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
    if (tile->isTransposable())
        // early return
        return;

    int device = tile->device();
    int64_t mb = tileMb(0);
    int64_t nb = tileNb(0);
    scalar_t* data = (scalar_t*) memory_.alloc(device, sizeof(scalar_t) * mb * nb);

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
int MatrixStorage<scalar_t>::host_num_ = HostNum;

template <typename scalar_t>
int MatrixStorage<scalar_t>::num_devices_ = 0;

} // namespace slate

#endif // SLATE_STORAGE_HH
