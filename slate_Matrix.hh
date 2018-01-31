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

#ifndef SLATE_MATRIX_HH
#define SLATE_MATRIX_HH

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

#ifdef SLATE_WITH_CUDA
    #include <cublas_v2.h>
    #include <cuda_runtime.h>
#else
    #include "slate_NoCuda.hh"
    #include "slate_NoCublas.hh"
#endif

#ifdef SLATE_WITH_MPI
    #include <mpi.h>
#else
    #include "slate_NoMpi.hh"
#endif

#ifdef _OPENMP
    #include <omp.h>
#else
    #include "slate_NoOpenmp.hh"
#endif

namespace slate {

///-----------------------------------------------------------------------------
/// \class
/// \brief
///
template <typename FloatType>
class Matrix {
public:
    friend class Debug;

    Matrix(int64_t m, int64_t n,
            FloatType *a, int64_t lda, int64_t nb,
            MPI_Comm mpi_comm, int64_t p, int64_t q);

    Matrix(const Matrix &a,
           int64_t m1, int64_t m2,
           int64_t n1, int64_t n2);

    ~Matrix() {
        // NOTE: no destruction of auxiliary CUDA/cuBLAS structures
        //       Perhaps this could be handled in BBLAS++.
    }

    //---------------------------
    // initialization and cleanup
    void init(FloatType *a, int64_t lda);
    void gather(FloatType *a, int64_t lda);
    void clean();

// TODO: Friend global computational functions.
// private:
    //-----------------------------
    // tile and submatrix operators
    Tile<FloatType>* &operator()(int64_t i, int64_t j)
    {
        return (*tiles_)[{it_+i, jt_+j, host_num_}];
    }
    Tile<FloatType>* &operator()(int64_t i, int64_t j) const
    {
        return (*tiles_)[{it_+i, jt_+j, host_num_}];
    }
    Tile<FloatType>* &operator()(int64_t i, int64_t j, int device)
    {
        return (*tiles_)[{it_+i, jt_+j, device}];
    }
    Tile<FloatType>* &operator()(int64_t i, int64_t j, int device) const
    {
        return (*tiles_)[{it_+i, jt_+j, device}];
    }
    Matrix<FloatType> operator()(int64_t i1, int64_t i2, int64_t j1, int64_t j2)
    {
        return Matrix(*this, i1, i2, j1, j2);
    }

    //-----------------------------
    // mapping to ranks and devices
    int64_t tileRank(int64_t i, int64_t j)
    {
        return tileRankFunc(it_+i, jt_+j);
    }
    int64_t tileDevice(int64_t i, int64_t j)
    {
        return tileDeviceFunc(it_+i, jt_+j);
    }
    bool tileIsLocal(int64_t i, int64_t j)
    {
        return tileRank(i, j) == mpi_rank_;
    }

    //-----------
    // tile sizes
    int64_t tileMb(int64_t i) { return tileMbFunc(it_+i); }
    int64_t tileNb(int64_t j) { return tileNbFunc(jt_+j); }

    //------------------------------
    // node-level memory consistency
    void tileCopyToDevice(int64_t i, int64_t j, int dst_device);
    void tileCopyToHost(int64_t i, int64_t j, int src_device);
    void tileMoveToDevice(int64_t i, int64_t j, int dst_device);
    void tileMoveToHost(int64_t i, int64_t j, int src_device);
    void tileErase(int64_t i, int64_t j, int device);
    void tileTick(int64_t i, int64_t j);

    //---------------------------------
    // distributed memory communication
    template <Target target = Target::Host>
    void tileSend(int64_t i, int64_t j, Matrix &&a);

    template <Target target = Target::Host>
    void tileSend(int64_t i, int64_t j, Matrix &&a1, Matrix &&a2);

    void tileSend(int64_t i, int64_t j, std::set<int> &bcast_set);

    void tileSendFindRanks(Matrix &a, std::set<int> *bcast_set);
    int64_t tileSendFindLife(Matrix &a);

    //-------------------------
    // auxiliary CUDA functions
    void initCudaStreams();
    void initCublasHandles();
    void initBatchArrays();

    //----------------------------
    // memory management functions
    int64_t getMaxHostTiles();
    int64_t getMaxDeviceTiles(int device);

    //-------------------------------
    // submatrix functions prototypes 
    #include "slate_Matrix.inc"

    int64_t it_; ///< first row of tiles
    int64_t jt_; ///< first column of tiles
    int64_t mt_; ///< number of tile rows
    int64_t nt_; ///< number of tile columns

    std::function <int64_t (int64_t i, int64_t j)> tileRankFunc;
    std::function <int64_t (int64_t i, int64_t j)> tileDeviceFunc;
    std::function <int64_t (int64_t i)> tileMbFunc;
    std::function <int64_t (int64_t j)> tileNbFunc;

    typedef Map<std::tuple<int64_t, int64_t, int>, Tile<FloatType>*> TilesMap;
    typedef Map<std::tuple<int64_t, int64_t>, int64_t> LivesMap;

    std::shared_ptr<TilesMap> tiles_; ///< map of tiles
    std::shared_ptr<LivesMap> lives_; ///< map of tiles' lives
    std::shared_ptr<Memory> memory_;  ///< memory allocator

    static int host_num_; ///< host ID
    int num_devices_;     ///< number of devices

    MPI_Comm mpi_comm_;
    MPI_Group mpi_group_;
    int mpi_size_;
    int mpi_rank_;

    // CUDA streams and cuBLAS handles
    std::vector<cudaStream_t> gemm_stream_;
    std::vector<cudaStream_t> comm_stream_;
    std::vector<cublasHandle_t> cublas_handle_;

    // host pointers arrays for batch GEMM
    std::vector<const FloatType**> a_array_h_;
    std::vector<const FloatType**> b_array_h_;
    std::vector<FloatType**> c_array_h_;

    // device pointers arrays for batch GEMM
    std::vector<const FloatType**> a_array_d_;
    std::vector<const FloatType**> b_array_d_;
    std::vector<FloatType**> c_array_d_;
};

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
Matrix<FloatType>::Matrix(int64_t m, int64_t n, FloatType *a, int64_t lda,
                          int64_t nb, MPI_Comm mpi_comm, int64_t p, int64_t q)
{
    tiles_ = std::make_shared<TilesMap>();
    lives_ = std::make_shared<LivesMap>();

    it_ = 0;
    jt_ = 0;
    mt_ = m % nb == 0 ? m/nb : m/nb+1;
    nt_ = n % nb == 0 ? n/nb : n/nb+1;

    mpi_comm_ = mpi_comm;
    assert(MPI_Comm_rank(mpi_comm_, &mpi_rank_) == MPI_SUCCESS);
    assert(MPI_Comm_size(mpi_comm_, &mpi_size_) == MPI_SUCCESS);
    assert(MPI_Comm_group(mpi_comm_, &mpi_group_) == MPI_SUCCESS);

    host_num_ = omp_get_initial_device();
#ifdef SLATE_WITH_CUDA
    num_devices_ = omp_get_num_devices();
#else
    num_devices_ = 0;
#endif

    tileMbFunc = [=] (int64_t i) { return i*nb > m ? m%nb : nb; };
    tileNbFunc = [=] (int64_t j) { return j*nb > n ? n%nb : nb; };

    tileRankFunc = [=] (int64_t i, int64_t j) { return i%p + (j%q)*p; };

    if (num_devices_ > 0) {
        tileDeviceFunc = [=] (int64_t i, int64_t j)
            { return j/q%num_devices_; };
    }
    else {
        tileDeviceFunc = [=] (int64_t i, int64_t j)
            { return host_num_; };
    }

    initCudaStreams();
    initCublasHandles();
    initBatchArrays();

    memory_ = std::make_shared<Memory>(sizeof(FloatType)*nb*nb);
    memory_->addHostBlocks(getMaxHostTiles());

    init(a, lda);
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
Matrix<FloatType>::Matrix(const Matrix &a,
                          int64_t m1, int64_t m2,
                          int64_t n1, int64_t n2)
{
    assert(m1 <= m2);
    assert(n1 <= n2);

    assert(m2 < a.mt_);
    assert(n2 < a.nt_);

    *this = a;
    it_ += m1;
    jt_ += n1;
    mt_ = m2-m1+1;
    nt_ = n2-n1+1;
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
void Matrix<FloatType>::init(FloatType *a, int64_t lda)
{
    int64_t m = 0;
    for (int64_t i = 0; i < mt_; ++i) {
        int64_t n = 0;
        for (int64_t j = 0; j <= i; ++j) {
            if (tileIsLocal(i, j)) {
                Tile<FloatType> *tile =
                    new Tile<FloatType>(tileMb(i), tileNb(j),
                                        &a[(size_t)lda*n+m], lda,
                                        memory_, mpi_comm_);
                tile->origin_ = true;
                (*this)(i, j) = tile;
            }
            n += tileNb(j);
        }
        m += tileMb(i);
    }
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
void Matrix<FloatType>::gather(FloatType *a, int64_t lda)
{
    int64_t m = 0;
    for (int64_t i = 0; i < mt_; ++i) {
        int64_t n = 0;
        for (int64_t j = 0; j <= i && j < nt_; ++j) {
            if (mpi_rank_ == 0) {
                if (!tileIsLocal(i, j)) {

                   (*this)(i, j) =
                        new Tile<FloatType>(tileMb(i), tileNb(j),
                                            &a[(size_t)lda*n+m], lda,
                                            memory_, mpi_comm_);

                    (*this)(i, j)->recv(tileRank(i, j));
                }
            }
            else {
                if (tileIsLocal(i, j))
                    (*this)(i, j)->send(0);
            }
            n += tileNb(j);
        }
        m += tileMb(i);
    }
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
void Matrix<FloatType>::clean()
{
    for (auto it = tiles_->begin(); it != tiles_->end(); ++it) {
        Tile<FloatType> *tile = it->second;
        if (tile->origin_ == false) {
            delete tile;
            tiles_->erase(it);
        }
    }
}

///-----------------------------------------------------------------------------
/// \brief Copy a tile to a device.
///
/// If the tile is not on the device, copy the tile to the device.
/// If the tile is on the device, but is not valid, update the tile's data.
/// Do not invalidate the source tile.
///
template <typename FloatType>
void Matrix<FloatType>::tileCopyToDevice(int64_t i, int64_t j, int dst_device)
{
    // If the tile is not on the device.
    auto it = tiles_->find({it_+i, jt_+j, dst_device});
    if (it == tiles_->end()) {

        // Create a copy on the device.
        Tile<FloatType> *src_tile = (*tiles_)[{it_+i, jt_+j, host_num_}];
        Tile<FloatType> *dst_tile =
            src_tile->copyToDevice(dst_device, comm_stream_[dst_device]);

        (*tiles_)[{it_+i, jt_+j, dst_device}] = dst_tile;
    }
    else {
        // If the tile on the device is not valid.
        Tile<FloatType> *dst_tile = it->second;
        if (dst_tile->valid_ == false) {

            // Update the device tile's data.
            Tile<FloatType> *src_tile = (*tiles_)[{it_+i, jt_+j, host_num_}];
            src_tile->copyDataToDevice(dst_tile, comm_stream_[dst_device]);
            dst_tile->valid_ = true;
        }
    }
}

///-----------------------------------------------------------------------------
/// \brief Copy a tile to the host.
///
/// If the tile not on the host, copy the tile to the host.
/// If the tile is on the host, but is not valid, update the tile's data.
/// Do not invalidate the source tile.
///
template <typename FloatType>
void Matrix<FloatType>::tileCopyToHost(int64_t i, int64_t j, int src_device)
{
    // If the tile is not on the host.
    auto it = tiles_->find({it_+i, jt_+j, host_num_});
    if (it == tiles_->end()) {

        // Create a copy on the host.
        Tile<FloatType> *src_tile = (*tiles_)[{it_+i, jt_+j, src_device}];
        Tile<FloatType> *dst_tile =
            src_tile->copyToHost(comm_stream_[src_device]);

        (*tiles_)[{it_+i, jt_+j, host_num_}] = dst_tile;
    }
    else {
        // If the tile on the host is not valid.
        Tile<FloatType> *dst_tile = it->second;
        if (dst_tile->valid_ == false) {

            // Update the host tile's data.
            Tile<FloatType> *src_tile = (*tiles_)[{it_+i, jt_+j, src_device}];
            src_tile->copyDataToHost(dst_tile, comm_stream_[src_device]);
            dst_tile->valid_ = true;
        }
    }
}

///-----------------------------------------------------------------------------
/// \brief Move a tile to a device.
///
/// If the tile is not on the device, copy the tile to the device.
/// If the tile is on the device, but is not valid, update the tile's data.
/// Invalidate the source tile.
///
template <typename FloatType>
void Matrix<FloatType>::tileMoveToDevice(int64_t i, int64_t j, int dst_device)
{
    // Copy the tile to the device.
    tileCopyToDevice(i, j, dst_device);

    // If the host tile exists, invalidate it.
    auto it = tiles_->find({it_+i, jt_+j, host_num_});
    if (it != tiles_->end())
        it->second->valid_ = false;
}

///-----------------------------------------------------------------------------
/// \brief Move a tile to the host.
///
/// If the tile is not on the host, copy the tile to the host.
/// If the tile is on the host, but is not valid, update the tile's data.
/// Invalidate the source tile.
///
template <typename FloatType>
void Matrix<FloatType>::tileMoveToHost(int64_t i, int64_t j, int src_device)
{
    // If source is not the host.
    if (src_device != host_num_) {

        // Copy the tile to the host.
        tileCopyToHost(i, j, src_device);

        // If the device tile exists, invalidate it.
        auto it = tiles_->find({it_+i, jt_+j, src_device});
        if (it != tiles_->end())
            it->second->valid_ = false;
    }
}

///-----------------------------------------------------------------------------
/// \brief Erase a tile.
///
/// If the tile exists and is not the origin, delete the tile
/// and erase it from the map.
///
template <typename FloatType>
void Matrix<FloatType>::tileErase(int64_t i, int64_t j, int device)
{
    // If the tile exists in the specified location.
    auto it = tiles_->find({it_+i, jt_+j, device});
    if (it != tiles_->end()) {

        // If the tile is not the origin.
        Tile<FloatType> *tile = it->second;
        if (tile->origin_ == false) {

            // Delete and erase the tile.
            delete (*tiles_)[{it_+i, jt_+j, device}];
            tiles_->erase({it_+i, jt_+j, device});
        }
    }
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
void Matrix<FloatType>::tileTick(int64_t i, int64_t j)
{
    if (!tileIsLocal(i, j)) {
        int64_t life = --(*lives_)[{it_+i, jt_+j}];
        if (life == 0) {
            tileErase(i, j, host_num_);
            for (int device = 0; device < num_devices_; ++device)
                tileErase(i, j, device);
            lives_->erase({it_+i, jt_+j});
        }
    }
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
template <Target target>
void Matrix<FloatType>::tileSend(int64_t i, int64_t j, Matrix &&a)
{
    // Find the set of participating ranks.
    std::set<int> bcast_set;
    bcast_set.insert(tileRank(i, j));
    tileSendFindRanks(a, &bcast_set);

    // If contained in the set.
    if (bcast_set.find(mpi_rank_) != bcast_set.end()) {

        // If receiving the tile.
        if (!tileIsLocal(i, j)) {

            // Create the tile.
            Tile<FloatType> *tile;
            tile = new Tile<FloatType>(tileMb(i), tileNb(j),
                                       memory_, mpi_comm_);
            (*this)(i, j) = tile;

            // Find the tile's life.
            (*lives_)[{it_+i, jt_+j}] = tileSendFindLife(a);
        }
        // Send across MPI ranks.
        tileSend(i, j, bcast_set);

        // Copy to devices.
        if (target == Target::Devices)
            for (int device = 0; device < num_devices_; ++device)
                tileCopyToDevice(i, j, device);
    }
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
template <Target target>
void Matrix<FloatType>::tileSend(int64_t i, int64_t j, Matrix &&a1, Matrix &&a2)
{
    // Find the set of participating ranks.
    std::set<int> bcast_set;
    bcast_set.insert(tileRank(i, j));
    tileSendFindRanks(a1, &bcast_set);
    tileSendFindRanks(a2, &bcast_set);

    // If contained in the set.
    if (bcast_set.find(mpi_rank_) != bcast_set.end()) {

        // If receiving the tile.
        if (!tileIsLocal(i, j)) {

            // Create the tile.
            Tile<FloatType> *tile;
            tile = new Tile<FloatType>(tileMb(i), tileNb(j),
                                       memory_, mpi_comm_);
            (*this)(i, j) = tile;

            // Find the tile's life.
            (*lives_)[{it_+i, jt_+j}]  = tileSendFindLife(a1);
            (*lives_)[{it_+i, jt_+j}] += tileSendFindLife(a2);
        }
        // Send across MPI ranks.
        tileSend(i, j, bcast_set);

        // Copy to devices.
        if (target == Target::Devices)
            for (int device = 0; device < num_devices_; ++device)
                tileCopyToDevice(i, j, device);
    }
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
void Matrix<FloatType>::tileSend(int64_t i, int64_t j, std::set<int> &bcast_set)
{
    // Quit if only root in the broadcast set.
    if (bcast_set.size() == 1)
        return;

    // Convert the set of ranks to a vector.
    std::vector<int> bcast_vec(bcast_set.begin(), bcast_set.end());

    // Create the broadcast group.
    MPI_Group bcast_group;
    int retval;
    #pragma omp critical(slate_mpi)
    retval = MPI_Group_incl(mpi_group_, bcast_vec.size(), bcast_vec.data(),
                            &bcast_group);
    assert(retval == MPI_SUCCESS);

    // Create a broadcast communicator.
    int tag = 0;
    MPI_Comm bcast_comm;
    trace_cpu_start();
    #pragma omp critical(slate_mpi)
    retval = MPI_Comm_create_group(mpi_comm_, bcast_group, tag, &bcast_comm);
    assert(retval == MPI_SUCCESS);
    assert(bcast_comm != MPI_COMM_NULL);
    trace_cpu_stop("Crimson");

    // Find the broadcast rank.
    int bcast_rank;
    #pragma omp critical(slate_mpi)
    MPI_Comm_rank(bcast_comm, &bcast_rank);

    // Find the broadcast root rank.
    int root_rank = tileRank(i, j);
    int bcast_root;
    #pragma omp critical(slate_mpi)
    retval = MPI_Group_translate_ranks(mpi_group_, 1, &root_rank,
                                       bcast_group, &bcast_root);
    assert(retval == MPI_SUCCESS);

    // Do the broadcast.
    (*this)(i, j)->bcast(bcast_root, bcast_comm);

    // Free the group.
    #pragma omp critical(slate_mpi)
    retval = MPI_Group_free(&bcast_group);
    assert(retval == MPI_SUCCESS);

    // Free the communicator.
    #pragma omp critical(slate_mpi)
    retval = MPI_Comm_free(&bcast_comm);
    assert(retval == MPI_SUCCESS);
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
void Matrix<FloatType>::tileSendFindRanks(Matrix &a, std::set<int> *bcast_set)
{
    // Find the set of participating ranks.
    for (int64_t i = 0; i < a.mt_; ++i)
        for (int64_t j = 0; j < a.nt_; ++j)
            bcast_set->insert(a.tileRank(i, j));
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
int64_t Matrix<FloatType>::tileSendFindLife(Matrix &a)
{
    // Find the tile's lifespan.
    int64_t life = 0;
    for (int64_t i = 0; i < a.mt_; ++i)
        for (int64_t j = 0; j < a.nt_; ++j)
            if (a.tileIsLocal(i, j))
                ++life;

    return life;
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
void Matrix<FloatType>::initCudaStreams()
{
    gemm_stream_.resize(num_devices_);
    comm_stream_.resize(num_devices_);

    for (int device = 0; device < num_devices_; ++device) {

        cudaError_t error;
        error = cudaSetDevice(device);
        assert(error == cudaSuccess);

        error = cudaStreamCreate(&gemm_stream_[device]);
        // error = cudaStreamCreateWithFlags(&gemm_stream_[device],
        //                                   cudaStreamNonBlocking);
        assert(error == cudaSuccess);

        error = cudaStreamCreate(&comm_stream_[device]);
        // error = cudaStreamCreateWithFlags(&gemm_stream_[device],
        //                                   cudaStreamNonBlocking);
        assert(error == cudaSuccess);
    }
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
void Matrix<FloatType>::initCublasHandles()
{
    cublas_handle_.resize(num_devices_);

    for (int device = 0; device < num_devices_; ++device) {

        cudaError_t error;
        error = cudaSetDevice(device);
        assert(error == cudaSuccess);

        cublasStatus_t status;
        status = cublasCreate(&cublas_handle_[device]);
        assert(status == CUBLAS_STATUS_SUCCESS);

        status = cublasSetStream(cublas_handle_[device], gemm_stream_[device]);
        assert(status == CUBLAS_STATUS_SUCCESS);
    }
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
void Matrix<FloatType>::initBatchArrays()
{
    a_array_h_.resize(num_devices_);
    b_array_h_.resize(num_devices_);
    c_array_h_.resize(num_devices_);

    a_array_d_.resize(num_devices_);
    b_array_d_.resize(num_devices_);
    c_array_d_.resize(num_devices_);

    for (int device = 0; device < num_devices_; ++device) {

        int64_t max_batch_size = getMaxDeviceTiles(device);
        cudaError_t error;

        // Allocate host arrays.
        error = cudaMallocHost((void**)(&a_array_h_[device]),
                               sizeof(FloatType*)*max_batch_size);
        assert(error == cudaSuccess);
        error = cudaMallocHost((void**)(&b_array_h_[device]),
                               sizeof(FloatType*)*max_batch_size);
        assert(error == cudaSuccess);
        error = cudaMallocHost((void**)(&c_array_h_[device]),
                               sizeof(FloatType*)*max_batch_size);
        assert(error == cudaSuccess);

        // Set the device.
        error = cudaSetDevice(device);
        assert(error == cudaSuccess);

        // Allocate device arrays.
        error = cudaMalloc((void**)(&a_array_d_[device]),
                           sizeof(FloatType*)*max_batch_size);
        assert(error == cudaSuccess);
        error = cudaMalloc((void**)(&b_array_d_[device]),
                           sizeof(FloatType*)*max_batch_size);
        assert(error == cudaSuccess);
        error = cudaMalloc((void**)(&c_array_d_[device]),
                           sizeof(FloatType*)*max_batch_size);
        assert(error == cudaSuccess);
    }
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
int64_t Matrix<FloatType>::getMaxHostTiles()
{
    int64_t max_batch_size = 0;
    for (int64_t i = 0; i < mt_; ++i)
        for (int64_t j = 0; j <= i; ++j)
            if (tileIsLocal(i, j))
                ++max_batch_size;

    return max_batch_size;
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
int64_t Matrix<FloatType>::getMaxDeviceTiles(int device)
{
    int64_t max_batch_size = 0;
    for (int64_t i = 0; i < mt_; ++i)
        for (int64_t j = 0; j <= i; ++j)
            if (tileIsLocal(i, j) && tileDevice(i, j) == device)
                ++max_batch_size;

    return max_batch_size;
}

} // namespace slate

#endif // SLATE_MATRIX_HH
