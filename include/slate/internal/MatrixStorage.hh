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

#include "slate/internal/Map.hh"
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
enum MOSI
{
    Modified = 0x100,   ///< tile data is modified, other instances should be Invalid, cannot be purged
    OnHold = 0x1000,  ///< a hold is placed on this tile instance, cannot be purged
    Shared = 0x010,   ///< tile data is up-to-date, other instances may be Shared, or Invalid, may be purged
    Invalid = 0x001,   ///< tile data is obsolete, other instances may be Modified, Shared, or Invalid, may be purged
};

//------------------------------------------------------------------------------
///
template <typename scalar_t>
struct TileEntry
{
    Tile<scalar_t>* tile_;
    short state_;
    omp_nest_lock_t lock_;

    /// Constructor for TileEntry class
    TileEntry(Tile<scalar_t>* tile,
              short state)
              : tile_(tile), state_(state)
    {
        omp_init_nest_lock(&lock_);
    }
    TileEntry() : tile_(nullptr), state_(MOSI::Invalid)
    {
        omp_init_nest_lock(&lock_);
    }

    /// Destructor for TileEntry class
    ~TileEntry()
    {
        omp_destroy_nest_lock(&lock_);
    }

    omp_nest_lock_t* get_lock()
    {
        return &lock_;
    }

    void setState(short stateIn)
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
                assert("Unkown state!");
                break;
        }
    }

    MOSI getState()
    {
        return MOSI(state_ & short(~MOSI::OnHold));
    }

    bool stateOn(short stateIn)
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
                assert("Unkown state!");
                break;
        }
        return false;
    }
};

//------------------------------------------------------------------------------
/// @brief Slate::MatrixStorage class
/// @details Used to store the map of distributed tiles.
/// @tparam scalar_t Data type for the elements of the matrix
///
template <typename scalar_t>
class MatrixStorage {
public:
    friend class Debug;

    template <typename T>
    friend class BaseMatrix;

    using ijdev_tuple = std::tuple<int64_t, int64_t, int>;
    using ij_tuple    = std::tuple<int64_t, int64_t>;

    typedef TileEntry<scalar_t> TileEntry_t;

    /// Map of tiles and states indexed by {i, j, device}.
    using TilesMap = slate::Map< ijdev_tuple, TileEntry_t >;

    /// Map of lives is indexed by {i, j}. The life applies to all devices.
    using LivesMap = slate::Map<ij_tuple, int64_t>;

    MatrixStorage(int64_t m, int64_t n, int64_t nb,
                  int p, int q, MPI_Comm mpi_comm);
    ~MatrixStorage();

protected:
    // used in constructor and destructor
    void initCudaStreams();
    void destroyCudaStreams();

public:
    void allocateBatchArrays(int64_t max_batch_size);
    void clearBatchArrays();
    void reserveHostWorkspace(int64_t num_tiles);
    void reserveDeviceWorkspace(int64_t num_tiles);
    void clearWorkspace();
    void releaseWorkspace();

    scalar_t* allocWorkspaceBuffer(int device);
    void      releaseWorkspaceBuffer(scalar_t* data, int device);

    //--------------------------------------------------------------------------
    // 2. copy constructor -- not allowed; object is shared
    // 3. move constructor -- not allowed; object is shared
    // 4. copy assignment  -- not allowed; object is shared
    // 5. move assignment  -- not allowed; object is shared
    MatrixStorage(MatrixStorage&  orig) = delete;
    MatrixStorage(MatrixStorage&& orig) = delete;
    MatrixStorage& operator = (MatrixStorage&  orig) = delete;
    MatrixStorage& operator = (MatrixStorage&& orig) = delete;

    //--------------------------------------------------------------------------
    /// @return
    typename TilesMap::iterator find(ijdev_tuple ijdev)
    {
        return tiles_.find(ijdev);
    }

    //--------------------------------------------------------------------------
    /// @return
    typename TilesMap::iterator begin()
    {
        return tiles_.begin();
    }

    //--------------------------------------------------------------------------
    /// @return
    typename TilesMap::iterator end()
    {
        return tiles_.end();
    }

    //--------------------------------------------------------------------------
    /// @return pointer to single tile entry. Throws exception if entry doesn't exist.
    // at() doesn't create new (null) entries in map as operator[] would
    // Tile<scalar_t>* at(ijdev_tuple ijdev)
    TileEntry_t& at(ijdev_tuple ijdev)
    {
        // return tiles_.at(ijdev).tile_;
        return tiles_.at(ijdev);
    }

    void erase(ijdev_tuple ijdev);
    void clear();

    //--------------------------------------------------------------------------
    /// @return number of allocated tiles (size of tiles map).
    size_t size() const { return tiles_.size(); }

    /// @return True if Tile is empty, False otherwise.
    bool empty() const { return size() == 0; }

    //--------------------------------------------------------------------------
    std::function <int (ij_tuple ij)> tileRank;
    std::function <int (ij_tuple ij)> tileDevice;
    std::function <int64_t (int64_t i)> tileMb;
    std::function <int64_t (int64_t j)> tileNb;

    //--------------------------------------------------------------------------
    /// @return whether tile {i, j} is local.
    bool tileIsLocal(ij_tuple ij)
    {
        return tileRank(ij) == mpi_rank_;
    }

    TileEntry<scalar_t>& tileInsert(ijdev_tuple ijdev, TileKind, Layout layout=Layout::ColMajor);
    TileEntry<scalar_t>& tileInsert(ijdev_tuple ijdev, scalar_t* data, int64_t lda, Layout layout=Layout::ColMajor);

    void tileMakeTransposable(Tile<scalar_t>* tile);
    void tileLayoutReset(Tile<scalar_t>* tile);

    void tileTick(ij_tuple ij);

    //--------------------------------------------------------------------------
    /// @return tile's life counter.
    // todo: logically, this is const, but if ij doesn't exist in lives_,
    // it is added and returns 0, so that makes it non-const.
    // Could use at() instead, but then would throw errors.
    int64_t tileLife(ij_tuple ij)
    {
        return lives_[ij];
    }

    //--------------------------------------------------------------------------
    /// Set tile's life counter.
    void tileLife(ij_tuple ij, int64_t life)
    {
        lives_[ij] = life;
    }

    //--------------------------------------------------------------------------
    /// Return p, q of 2D block-cyclic distribution.
    /// These will eventually disappear when distributions are generalized.
    int p() const { return p_; }
    int q() const { return q_; }

    //--------------------------------------------------------------------------
    /// @return currently allocated batch array size
    int64_t batchArraySize()
    {
        return batch_array_size;
    }

private:
    int64_t m_;
    int64_t n_;
    int64_t mt_;
    int64_t nt_;
    int64_t nb_;
    int p_, q_;

    TilesMap tiles_;        ///< map of tiles and associated states
    LivesMap lives_;        ///< map of tiles' lives
    slate::Memory memory_;  ///< memory allocator

    int mpi_rank_;
    static int host_num_;
    static int num_devices_;

    int64_t batch_array_size;

    // CUDA streams and cuBLAS handles
    std::vector<cudaStream_t> compute_streams_;
    std::vector<cudaStream_t> comm_streams_;
    std::vector<cublasHandle_t> cublas_handles_;

    // host pointers arrays for batch GEMM
    std::vector<scalar_t**> a_array_host_;
    std::vector<scalar_t**> b_array_host_;
    std::vector<scalar_t**> c_array_host_;

    // device pointers arrays for batch GEMM
    std::vector<scalar_t**> a_array_dev_;
    std::vector<scalar_t**> b_array_dev_;
    std::vector<scalar_t**> c_array_dev_;
};

//------------------------------------------------------------------------------
template <typename scalar_t>
MatrixStorage<scalar_t>::MatrixStorage(
    int64_t m, int64_t n, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
    : m_(m),
      n_(n),
      mt_(ceildiv(m, nb)),
      nt_(ceildiv(n, nb)),
      nb_(nb),
      p_(p),
      q_(q),
      tiles_(),
      lives_(),
      memory_(sizeof(scalar_t) * nb * nb),  // block size in bytes
      batch_array_size(0)
{
    slate_mpi_call(
        MPI_Comm_rank(mpi_comm, &mpi_rank_));

    // todo: these are static, but we (re-)initialize with each matrix.
    // todo: similar code in BaseMatrix(...) and MatrixStorage(...)
    host_num_    = memory_.host_num_;
    num_devices_ = memory_.num_devices_;

    // TODO: these all assume 2D block cyclic with fixed size tiles (nb)
    // lambdas that capture m, n, nb for
    // computing tile's mb (rows) and nb (cols)
    tileMb = [=](int64_t i) { return (i + 1)*nb > m ? m%nb : nb; };
    tileNb = [=](int64_t j) { return (j + 1)*nb > n ? n%nb : nb; };

    // lambda that captures p, q for computing tile's rank,
    // assuming 2D block cyclic
    tileRank = [=](ij_tuple ij) {
        int64_t i = std::get<0>(ij);
        int64_t j = std::get<1>(ij);
        return int(i%p + (j%q)*p);
    };

    // lambda that captures q, num_devices_ to distribute local matrix
    // in 1D column block cyclic fashion among devices
    if (num_devices_ > 0) {
        tileDevice = [=](ij_tuple ij) {
            int64_t j = std::get<1>(ij);
            return int(j/q)%num_devices_;
        };
    }
    else {
        tileDevice = [=](ij_tuple ij) {
            return host_num_;
        };
    }

    initCudaStreams();
}

//------------------------------------------------------------------------------
// 1. destructor: deletes all tiles and frees workspace buffers
template <typename scalar_t>
MatrixStorage<scalar_t>::~MatrixStorage()
{
    clear();
    destroyCudaStreams();
    clearBatchArrays();
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
        // todo: should this be a non-blocking stream
        slate_cuda_call(
            cudaStreamCreate(&compute_streams_[device]));
        // todo: should this be a non-blocking stream
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
/// Allocates CUDA batch arrays
// todo: resizeBatchArrays?
///
template <typename scalar_t>
void MatrixStorage<scalar_t>::allocateBatchArrays(int64_t max_batch_size)
{
    assert(max_batch_size >= 0);

    batch_array_size = max_batch_size;

    a_array_host_.resize(num_devices_);
    b_array_host_.resize(num_devices_);
    c_array_host_.resize(num_devices_);

    a_array_dev_.resize(num_devices_);
    b_array_dev_.resize(num_devices_);
    c_array_dev_.resize(num_devices_);

    for (int device = 0; device < num_devices_; ++device) {
        // Allocate host arrays.
        slateCudaMallocHost(&a_array_host_[device], max_batch_size);
        slateCudaMallocHost(&b_array_host_[device], max_batch_size);
        slateCudaMallocHost(&c_array_host_[device], max_batch_size);

        // Set the device.
        slate_cuda_call(
            cudaSetDevice(device));

        // Allocate device arrays.
        slateCudaMalloc(&a_array_dev_[device], max_batch_size);
        slateCudaMalloc(&b_array_dev_[device], max_batch_size);
        slateCudaMalloc(&c_array_dev_[device], max_batch_size);
    }
}

//------------------------------------------------------------------------------
/// Frees CUDA batch arrays that were allocated by allocateBatchArrays().
/// As this is called in the destructor, it should NOT throw exceptions.
///
// todo: destructor can't throw; how to deal with errors?
template <typename scalar_t>
void MatrixStorage<scalar_t>::clearBatchArrays()
{
    // if allocateBatchArrays() has been called, size is num_devices_,
    // otherwise it's zero and there's nothing to do.
    int size = (int) a_array_host_.size();
    assert(size == 0 || size == num_devices_);
    for (int device = 0; device < size; ++device) {
        // Free host arrays.
        slate_cuda_call(
            cudaFreeHost(a_array_host_[device]));
        slate_cuda_call(
            cudaFreeHost(b_array_host_[device]));
        slate_cuda_call(
            cudaFreeHost(c_array_host_[device]));

        // Set the device.
        slate_cuda_call(
            cudaSetDevice(device));

        // Free device arrays.
        slate_cuda_call(
            cudaFree(a_array_dev_[device]));
        slate_cuda_call(
            cudaFree(b_array_dev_[device]));
        slate_cuda_call(
            cudaFree(c_array_dev_[device]));
    }

    a_array_host_.clear();
    b_array_host_.clear();
    c_array_host_.clear();

    a_array_dev_.clear();
    b_array_dev_.clear();
    c_array_dev_.clear();
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
/// Clears all host and device workspace tiles.
/// Also clears life.
///
template <typename scalar_t>
void MatrixStorage<scalar_t>::clearWorkspace()
{
    LockGuard(tiles_.get_lock());
    // incremented below
    for (auto iter = tiles_.begin(); iter != tiles_.end();) {
        if (iter->second.tile_->workspace()) {
            // Since we can't increment the iterator after deleting the
            // element, use post-fix iter++ to increment it but
            // erase the current value.
            erase((iter++)->first);
        }
        else {
            ++iter;
        }
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
/// Also clears life.
///
template <typename scalar_t>
void MatrixStorage<scalar_t>::releaseWorkspace()
{
    LockGuard(tiles_.get_lock());
    // incremented below
    for (auto iter = tiles_.begin(); iter != tiles_.end();) {
        if (iter->second.tile_->workspace()) {
            // Since we can't increment the iterator after deleting the
            // element, use post-fix iter++ to increment it but
            // erase the current value.
            if (iter->second.stateOn(MOSI::OnHold) || iter->second.stateOn(MOSI::Modified))
                iter++;
            else
                erase((iter++)->first);
        }
        else {
            ++iter;
        }
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
/// Remove a tile from the map and delete it.
/// If tile's memory was allocated by SLATE, then its memory is freed back
/// to the allocator memory pool.
/// Doesn't delete life; see tileTick for deleting life.
///
// todo: currently ignores if ijdev doesn't exist; is that right?
template <typename scalar_t>
void MatrixStorage<scalar_t>::erase(ijdev_tuple ijdev)
{
    LockGuard(tiles_.get_lock());
    auto iter = tiles_.find(ijdev);
    if (iter != tiles_.end()) {
        Tile<scalar_t>* tile = iter->second.tile_;
        if (tile->allocated())
            memory_.free(tile->data(), tile->device());
        if (tile->extended())
            memory_.free(tile->extData(), tile->device());
        delete tile;
        tiles_.erase(ijdev);
    }
}

//------------------------------------------------------------------------------
/// Delete all tiles.
template <typename scalar_t>
void MatrixStorage<scalar_t>::clear()
{
    // incremented below
    for (auto iter = tiles_.begin(); iter != tiles_.end();) {
        // erasing the element invalidates the iterator,
        // so use iter++ to erase the current value but increment it first.
        erase((iter++)->first); // todo: in-efficient
    }
    // todo: what if some tiles were not erased
    assert(tiles_.size() == 0);  // should be empty now
    lives_.clear();
}

//------------------------------------------------------------------------------
/// Allocates a memory block on device to be used as a workspace buffer,
///     to be released with call to releaseWorkspaceBuffer()
/// @returns pointer to memory block on device
///
/// @param[in] device
///     Device ID (GPU or Host) where the memory block is needed.
///
template <typename scalar_t>
scalar_t* MatrixStorage<scalar_t>::allocWorkspaceBuffer(int device)
{
    scalar_t* data = (scalar_t*) memory_.alloc(device);
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
/// Tile kind should be either TileKind::Workspace or TileKind::SlateOwned.
/// Does not set tile's life.
/// @return Pointer to newly inserted TileEntry.
///
template <typename scalar_t>
TileEntry<scalar_t>& MatrixStorage<scalar_t>::tileInsert(
    ijdev_tuple ijdev, TileKind kind, Layout layout)
{
    assert(kind == TileKind::Workspace ||
           kind == TileKind::SlateOwned);
    assert(tiles_.find(ijdev) == tiles_.end());  // doesn't exist yet
    int64_t i  = std::get<0>(ijdev);
    int64_t j  = std::get<1>(ijdev);
    int device = std::get<2>(ijdev);
    scalar_t* data = (scalar_t*) memory_.alloc(device);
    int64_t mb = tileMb(i);
    int64_t nb = tileNb(j);
    int64_t stride = layout == Layout::ColMajor ? mb : nb;
    Tile<scalar_t>* tile
        = new Tile<scalar_t>(mb, nb, data, stride, device, kind, layout);
    tiles_[ijdev] = TileEntry<scalar_t>(tile, kind == TileKind::Workspace ? MOSI::Invalid : MOSI::Shared);
    return tiles_[ijdev];
}

//------------------------------------------------------------------------------
/// This is intended for inserting the original matrix.
/// Inserts tile {i, j} on given device, which can be host,
/// wrapping existing memory for it.
/// Sets tile kind = TileKind::UserOwned.
/// Does not set tile's life.
/// @return Pointer to newly inserted TileEntry.
///
template <typename scalar_t>
TileEntry<scalar_t>& MatrixStorage<scalar_t>::tileInsert(
    ijdev_tuple ijdev, scalar_t* data, int64_t lda, Layout layout)
{
    assert(tiles_.find(ijdev) == tiles_.end());  // doesn't exist yet
    int64_t i  = std::get<0>(ijdev);
    int64_t j  = std::get<1>(ijdev);
    int device = std::get<2>(ijdev);
    int64_t mb = tileMb(i);
    int64_t nb = tileNb(j);
    Tile<scalar_t>* tile
        = new Tile<scalar_t>(mb, nb, data, lda, device, TileKind::UserOwned, layout);
    tiles_[ijdev] = TileEntry<scalar_t>(tile, MOSI::Shared);
    return tiles_[ijdev];
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
    scalar_t* data = (scalar_t*) memory_.alloc(device);

    tile->makeTransposable(data);
}

//------------------------------------------------------------------------------
/// Resets the extended tile.
/// Frees the extended buffer and returns to memory manager
///     then resets the tile's extended member fields
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
        LockGuard(lives_.get_lock());
        int64_t life = --lives_.at(ij);
        if (life == 0) {
            int64_t i = std::get<0>(ij);
            int64_t j = std::get<1>(ij);
            erase({i, j, host_num_});
            for (int device = 0; device < num_devices_; ++device) {
                erase({i, j, device});
            }
            lives_.erase(ij);
        }
    }
}

//------------------------------------------------------------------------------
template <typename scalar_t>
int MatrixStorage<scalar_t>::host_num_ = -1;

template <typename scalar_t>
int MatrixStorage<scalar_t>::num_devices_ = 0;

} // namespace slate

#endif // SLATE_STORAGE_HH
