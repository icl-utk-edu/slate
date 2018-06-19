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
// Need assistance with the SLATE software? Join the "SLATE User" Google group
// by going to https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user
// and clicking "Apply to join group". Upon acceptance, email your questions and
// comments to <slate-user@icl.utk.edu>.
//------------------------------------------------------------------------------

#ifndef SLATE_STORAGE_HH
#define SLATE_STORAGE_HH

#include "slate_Map.hh"
#include "slate_Memory.hh"
#include "slate_Tile.hh"
#include "slate_types.hh"

#include "lapack.hh"

#include <algorithm>
#include <functional>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "slate_cuda.hh"
#include "slate_cublas.hh"
#include "slate_mpi.hh"
#include "slate_openmp.hh"

namespace slate {

//------------------------------------------------------------------------------
/// @return ceil(a / b), for integer type T.
template<typename T>
T ceildiv(T a, T b)
{
    return ((a + b - 1) / b);
}

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

    /// Map of tiles is indexed by {i, j, device}.
    using TilesMap = slate::Map< ijdev_tuple, Tile<scalar_t>* >;

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
    /// @return pointer to single tile. Throws exception if tile doesn't exist.
    // at() doesn't create new (null) entries in map as operator[] would
    Tile<scalar_t>* at(ijdev_tuple ijdev)
    {
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

    Tile<scalar_t>* tileInsert(ijdev_tuple ijdev);
    Tile<scalar_t>* tileInsert(ijdev_tuple ijdev, scalar_t* data, int64_t lda);
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

private:
    int64_t m_;
    int64_t n_;
    int64_t mt_;
    int64_t nt_;
    int64_t nb_;

    TilesMap tiles_;        ///< map of tiles
    LivesMap lives_;        ///< map of tiles' lives
    slate::Memory memory_;  ///< memory allocator

    int mpi_rank_;
    static int host_num_;
    static int num_devices_;

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
      tiles_(),
      lives_(),
      memory_(sizeof(scalar_t) * nb * nb)  // block size in bytes
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
        slate_cuda_call(
            cudaStreamCreate(&compute_streams_[device]));
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
        if (! iter->second->origin()) {
            // Since we can't increment the iterator after deleting the
            // element, use post-fix iter++ to increment it but
            // erase the current value.
            erase((iter++)->first);
        }
        else
            ++iter;
    }
    memory_.clearHostBlocks();
    for (int device = 0; device < num_devices_; ++device)
        memory_.clearDeviceBlocks(device);
}

//------------------------------------------------------------------------------
/// Remove a tile from the map and delete it.
/// If tile is workspace, i.e., not origin, then its memory is freed back
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
        Tile<scalar_t>* tile = tiles_.at(ijdev);
        if (! tile->origin())
            memory_.free(tile->data(), tile->device());
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
        erase((iter++)->first);
    }
    assert(tiles_.size() == 0);  // should be empty now
    lives_.clear();
}

//------------------------------------------------------------------------------
/// Inserts workspace tile {i, j} on given device, which can be host,
/// allocating new memory for it.
/// Sets tile origin = false.
/// Does not set tile's life.
/// @return Pointer to newly inserted Tile.
///
template <typename scalar_t>
Tile<scalar_t>* MatrixStorage<scalar_t>::tileInsert(ijdev_tuple ijdev)
{
    assert(tiles_.find(ijdev) == tiles_.end());  // doesn't exist yet
    int64_t i  = std::get<0>(ijdev);
    int64_t j  = std::get<1>(ijdev);
    int device = std::get<2>(ijdev);
    scalar_t* data = (scalar_t*) memory_.alloc(device);
    int64_t mb = tileMb(i);
    int64_t nb = tileNb(j);
    Tile<scalar_t>* tile = new Tile<scalar_t>(mb, nb, data, mb,
                                              device, false);
    tiles_[ijdev] = tile;
    return tile;
}

//------------------------------------------------------------------------------
/// This is intended for inserting the original matrix.
/// Inserts tile {i, j} on given device, which can be host,
/// wrapping existing memory for it.
/// Sets tile origin = true.
/// Does not set tile's life.
/// @return Pointer to newly inserted Tile.
///
template <typename scalar_t>
Tile<scalar_t>* MatrixStorage<scalar_t>::tileInsert(
    ijdev_tuple ijdev, scalar_t* data, int64_t lda)
{
    assert(tiles_.find(ijdev) == tiles_.end());  // doesn't exist yet
    int64_t i  = std::get<0>(ijdev);
    int64_t j  = std::get<1>(ijdev);
    int device = std::get<2>(ijdev);
    int64_t mb = tileMb(i);
    int64_t nb = tileNb(j);
    Tile<scalar_t>* tile = new Tile<scalar_t>(mb, nb, data, lda,
                                              device, true);
    tiles_[ijdev] = tile;
    return tile;
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
