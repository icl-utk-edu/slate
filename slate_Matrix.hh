
#ifndef SLATE_MATRIX_HH
#define SLATE_MATRIX_HH

#include "slate_Tile.hh"
#include "slate_Memory.hh"

#include "lapack.hh"

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>
#include <iostream>

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

#ifdef SLATE_WITH_OPENMP
    #include <omp.h>
#else
    #include "slate_NoOpenmp.hh"
#endif

namespace slate {

enum class Target {Devices, Host, HostTask, HostNest, HostBatch};

//------------------------------------------------------------------------------
template <typename FloatType>
class Matrix {
public:
    Matrix(int64_t m, int64_t n, FloatType *a, int64_t lda,
           int64_t nb, MPI_Comm mpi_comm, int64_t p, int64_t q);

    Matrix(const Matrix &a, int64_t m1, int64_t m2, int64_t n1, int64_t n2);

    ~Matrix() {
        // only if not a submatrix
        // assert(cublasDestroy(cublas_handle_) == CUBLAS_STATUS_SUCCESS);
    }

    void random();
    void copyTo(FloatType *a, int64_t lda);
    void copyFrom(FloatType *a, int64_t lda);
    void copyFromFull(FloatType *a, int64_t lda);
    void gather();

    void trsm(blas::Side side, blas::Uplo uplo,
              blas::Op trans, blas::Diag diag,
              FloatType alpha, const Matrix &a);

    template <Target target = Target::HostTask>
    void potrf(blas::Uplo uplo, int64_t lookahead = 0);

private:
    template <Target> class TargetType {};

    struct TileHash {
        size_t operator()(const std::tuple<int64_t, int64_t, int> &key) const
        {
            size_t hash_i = std::hash<int64_t>()(std::get<0>(key));
            size_t hash_j = std::hash<int64_t>()(std::get<1>(key));
            size_t hash_k = std::hash<int>    ()(std::get<2>(key));
            return (hash_i*31 + hash_j)*31 + hash_k;
        }
    };

    Tile<FloatType>* &operator()(int64_t i, int64_t j)
    {
        omp_set_lock(tiles_lock_);
        Tile<FloatType>* &tile = (*tiles_)[{it_+i, jt_+j, host_num_}];
        omp_unset_lock(tiles_lock_);
        return tile;
    }
    Tile<FloatType>* &operator()(int64_t i, int64_t j) const
    {
        omp_set_lock(tiles_lock_);
        Tile<FloatType>* &tile = (*tiles_)[{it_+i, jt_+j, host_num_}];
        omp_unset_lock(tiles_lock_);
        return tile;
    }
    Tile<FloatType>* &operator()(int64_t i, int64_t j, int device)
    {
        omp_set_lock(tiles_lock_);
        Tile<FloatType>* &tile = (*tiles_)[{it_+i, jt_+j, device}];
        omp_unset_lock(tiles_lock_);
        return tile;
    }
    Tile<FloatType>* &operator()(int64_t i, int64_t j, int device) const
    {
        omp_set_lock(tiles_lock_);
        Tile<FloatType>* &tile = (*tiles_)[{it_+i, jt_+j, device}];
        omp_unset_lock(tiles_lock_);
        return tile;
    }

    int64_t tileRank(int64_t i, int64_t j) {
        return tileRankFunc(it_+i, jt_+j);
    }
    int64_t tileDevice(int64_t i, int64_t j) {
        return tileDeviceFunc(it_+i, jt_+j);
    }
    int64_t tileMb(int64_t i) { return tileMbFunc(it_+i); }
    int64_t tileNb(int64_t j) { return tileNbFunc(jt_+j); }

    bool tileIsLocal(int64_t i, int64_t j) {
        return tileRank(i, j) == mpi_rank_;
    }

    //----------------------
    template <Target target>
    void potrf_impl(TargetType<target>,
                    blas::Uplo uplo, int64_t lookahead);

    void potrf_impl(TargetType<Target::Devices>,
                    blas::Uplo uplo, int64_t lookahead);

    //------------------------------------------
    template <Target target = Target::HostTask>
    void syrk(blas::Uplo uplo, blas::Op trans,
              FloatType alpha, const Matrix &a, FloatType beta);

    void syrk_impl(TargetType<Target::HostTask>,
                   blas::Uplo uplo, blas::Op trans,
                   FloatType alpha, const Matrix &a, FloatType beta);

    void syrk_impl(TargetType<Target::HostNest>,
                   blas::Uplo uplo, blas::Op trans,
                   FloatType alpha, const Matrix &a, FloatType beta);

    void syrk_impl(TargetType<Target::HostBatch>,
                   blas::Uplo uplo, blas::Op trans,
                   FloatType alpha, const Matrix &a, FloatType beta);

    void syrk_impl(TargetType<Target::Devices>,
                   blas::Uplo uplo, blas::Op trans,
                   FloatType alpha, const Matrix &a, FloatType beta);

    //--------------------------------------------
    void tileSend(int64_t i, int64_t j, int dest);
    void tileRecv(int64_t i, int64_t j, int src);

    template <Target target = Target::Host>
    void tileSend(int64_t m, int64_t n,
                  std::array<int64_t, 4> range);

    template <Target target = Target::Host>
    void tileSend(int64_t m, int64_t n,
                  std::array<int64_t, 4> range1,
                  std::array<int64_t, 4> range2);

    void tileSendFindRanks(int64_t i, int64_t j,
                           std::array<int64_t, 4> range,
                           std::set<int> *bcast_set);

    int64_t tileSendFindLife(int64_t i, int64_t j,
                             std::array<int64_t, 4> range);

    void tileSend(int64_t i, int64_t j, std::set<int> &bcast_set);

    //----------------------------------------------------------
    void tileCopyToDevice(int64_t i, int64_t j, int dst_device);
    void tileMoveToDevice(int64_t i, int64_t j, int dst_device);
    void tileMoveToHost(int64_t i, int64_t j, int src_device);
    void tileErase(int64_t i, int64_t j, int device);

    void initCudaStreams();
    void initCublasHandles();
    void initBatchArrays();

    int64_t getMaxHostTiles();
    int64_t getMaxDeviceTiles(int device);

    void checkLife();
    void printLife();

    int64_t it_; ///< first row of tiles
    int64_t jt_; ///< first column of tiles
    int64_t mt_; ///< number of tile rows
    int64_t nt_; ///< number of tile columns

    std::function <int64_t (int64_t i, int64_t j)> tileRankFunc;
    std::function <int64_t (int64_t i, int64_t j)> tileDeviceFunc;
    std::function <int64_t (int64_t i)> tileMbFunc;
    std::function <int64_t (int64_t j)> tileNbFunc;

    std::map<std::tuple<int64_t, int64_t, int>, Tile<FloatType>*> *tiles_;
    // std::unordered_map<std::tuple<int64_t, int64_t, int>,
    //                    Tile<FloatType>*,
    //                    TileHash> *tiles_;

    omp_lock_t *tiles_lock_ = new omp_lock_t();

    MPI_Comm mpi_comm_;
    MPI_Group mpi_group_;
    int mpi_size_;
    int mpi_rank_;

    static int host_num_; ///< host ID
    int num_devices_;     ///< number of devices
    Memory *memory_;      ///< memory allocator

    // CUDA streams and cuBLAS handles
    std::vector<cudaStream_t> gemm_stream_;
    std::vector<cudaStream_t> comm_stream_;
    std::vector<cublasHandle_t> cublas_handle_;

    // host pointer arrays for batch GEMM
    std::vector<const FloatType**> a_array_h_;
    std::vector<const FloatType**> b_array_h_;
    std::vector<FloatType**> c_array_h_;

    // device pointer arrays for batch GEMM
    std::vector<const FloatType**> a_array_d_;
    std::vector<const FloatType**> b_array_d_;
    std::vector<FloatType**> c_array_d_;
};

//------------------------------------------------------------------------------
// @brief Copy the tile to the device, if not already there.
//        If it's already been copied, it won't be copied again.
//
template <typename FloatType>
void Matrix<FloatType>::tileCopyToDevice(int64_t i, int64_t j, int dst_device)
{
    omp_set_lock(tiles_lock_);
    // If the tile not on the device.
    if (tiles_->find({it_+i, jt_+j, dst_device}) == tiles_->end()) {
        // Copy the tile to the device.
        Tile<FloatType> *src_tile = (*tiles_)[{it_+i, jt_+j, host_num_}];
        omp_unset_lock(tiles_lock_);
        Tile<FloatType> *tile =
            new Tile<FloatType>(src_tile, dst_device, comm_stream_[dst_device]);
        omp_set_lock(tiles_lock_);
        (*tiles_)[{it_+i, jt_+j, dst_device}] = tile;
    }
    omp_unset_lock(tiles_lock_);
}

//------------------------------------------------------------------------------
// @brief Move the tile to the device, if not already there.
//        If it's already been moved, it won't be moved again.
//
template <typename FloatType>
void Matrix<FloatType>::tileMoveToDevice(int64_t i, int64_t j, int dst_device)
{
    omp_set_lock(tiles_lock_);
    // If the tile not on the device.
    if (tiles_->find({it_+i, jt_+j, dst_device}) == tiles_->end()) {
        // Move the tile to the device.
        Tile<FloatType> *src_tile = (*tiles_)[{it_+i, jt_+j, host_num_}];
        omp_unset_lock(tiles_lock_);
        Tile<FloatType> *tile =
            new Tile<FloatType>(src_tile, dst_device, comm_stream_[dst_device]);
        omp_set_lock(tiles_lock_);
        (*tiles_)[{it_+i, jt_+j, dst_device}] = tile;
        delete (*tiles_)[{it_+i, jt_+j, host_num_}];
        tiles_->erase({it_+i, jt_+j, host_num_});
    }
    omp_unset_lock(tiles_lock_);
}

//------------------------------------------------------------------------------
// @brief Move the tile to the host, if not already there.
//        If it's already been moved, it won't be moved again.
//
template <typename FloatType>
void Matrix<FloatType>::tileMoveToHost(int64_t i, int64_t j, int src_device)
{
    omp_set_lock(tiles_lock_);
    // If the tile not on the host.
    if (tiles_->find({it_+i, jt_+j, host_num_}) == tiles_->end()) {
        // Move the tile to the host.
        Tile<FloatType> *src_tile = (*tiles_)[{it_+i, jt_+j, src_device}];
        omp_unset_lock(tiles_lock_);
        Tile<FloatType> *tile =
            new Tile<FloatType>(src_tile, host_num_, comm_stream_[src_device]);
        omp_set_lock(tiles_lock_);
        (*tiles_)[{it_+i, jt_+j, host_num_}] = tile;
        delete (*tiles_)[{it_+i, jt_+j, src_device}];
        tiles_->erase({it_+i, jt_+j, src_device});
    }
    omp_unset_lock(tiles_lock_);
}

//------------------------------------------------------------------------------
// @brief Erase the tile, if it exists in the specified location.
//        Don't try to erase tiles that have already been erased.
//
template <typename FloatType>
void Matrix<FloatType>::tileErase(int64_t i, int64_t j, int device)
{
    omp_set_lock(tiles_lock_);
    // If the tile exists in the specified location.
    if (tiles_->find({it_+i, jt_+j, device}) != tiles_->end()) {
        // Erase the tile.
        delete (*tiles_)[{it_+i, jt_+j, device}];
        tiles_->erase({it_+i, jt_+j, device});
    }
    omp_unset_lock(tiles_lock_);
}

//------------------------------------------------------------------------------
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

//------------------------------------------------------------------------------
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

//------------------------------------------------------------------------------
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

//------------------------------------------------------------------------------
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

//------------------------------------------------------------------------------
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

//------------------------------------------------------------------------------
template <typename FloatType>
Matrix<FloatType>::Matrix(int64_t m, int64_t n, FloatType *a, int64_t lda,
                          int64_t nb, MPI_Comm mpi_comm, int64_t p, int64_t q)
{
    tiles_ = new std::map<std::tuple<int64_t, int64_t, int>, Tile<FloatType>*>;
    // tiles_ = new std::unordered_map<std::tuple<int64_t, int64_t, int>,
    //                                 Tile<FloatType>*,
    //                                 TileHash>;
    omp_init_lock(tiles_lock_);

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

    memory_ = new Memory(sizeof(FloatType)*nb*nb);
    memory_->addHostBlocks(getMaxHostTiles());

    if (a != nullptr)
        copyTo(a, lda);
    else
        random();
}

//------------------------------------------------------------------------------
template <typename FloatType>
Matrix<FloatType>::Matrix(const Matrix &a, int64_t m1, int64_t m2,
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

//------------------------------------------------------------------------------
template <typename FloatType>
void Matrix<FloatType>::random()
{
    for (int64_t i = 0; i < mt_; ++i) {
        for (int64_t j = 0; j <= i; ++j) {
            if (tileIsLocal(i, j))
            {
                Tile<FloatType> *tile =
                    new Tile<FloatType>(tileMb(i), tileNb(j), memory_);

                int iseed[4];
                iseed[0] = i & 0x0FFF;
                iseed[1] = j & 0x0FFF;
                iseed[2] = ((i >> 12) + (j >> 12)) & 0x0FFF;
                iseed[3] = 1;
                int nb = tileNb(0);
                lapack::larnv(1, iseed, nb*nb, tile->data_);

                if (i == j) {
                    for (int64_t k = 0; k < nb; ++k)
                    tile->data_[k*nb+k] += nb*nt_;
                }
                (*this)(i, j) = tile;
            }
        }
    }
}

//------------------------------------------------------------------------------
template <typename FloatType>
void Matrix<FloatType>::copyTo(FloatType *a, int64_t lda)
{
    int64_t m = 0;
    for (int64_t i = 0; i < mt_; ++i) {
        int64_t n = 0;
        for (int64_t j = 0; j <= i; ++j) {
            if (tileIsLocal(i, j))
                (*this)(i, j) =
                    new Tile<FloatType>(tileMb(i), tileNb(j),
                                        &a[(size_t)lda*n+m], lda, memory_);
            n += tileNb(j);
        }
        m += tileMb(i);
    }
}

//------------------------------------------------------------------------------
template <typename FloatType>
void Matrix<FloatType>::copyFrom(FloatType *a, int64_t lda)
{
    int64_t m = 0;
    for (int64_t i = 0; i < mt_; ++i) {
        int64_t n = 0;
        for (int64_t j = 0; j <= i; ++j) {
            if (tileIsLocal(i, j)) {
                (*this)(i, j)->copyFrom(&a[(size_t)lda*m+n], lda);
            }
            n += tileNb(j);
        }
        m += tileMb(i);
    }
}

//------------------------------------------------------------------------------
template <typename FloatType>
void Matrix<FloatType>::copyFromFull(FloatType *a, int64_t lda)
{
    int64_t m = 0;
    for (int64_t i = 0; i < mt_; ++i) {
        int64_t n = 0;
        for (int64_t j = 0; j <= i; ++j) {
            (*this)(i, j)->copyFrom(&a[(size_t)lda*n+m], lda);
            n += tileNb(j);
        }
        m += tileMb(i);
    }
}

//------------------------------------------------------------------------------
template <typename FloatType>
void Matrix<FloatType>::gather()
{
    for (int64_t i = 0; i < mt_; ++i) {
        for (int64_t j = 0; j <= i && j < nt_; ++j) {
            if (mpi_rank_ == 0) {
                if (!tileIsLocal(i, j))
                    tileRecv(i, j, tileRank(i, j));
            }
            else {
                if (tileIsLocal(i, j))
                    tileSend(i, j, 0);
            }
        }
    }
}

//------------------------------------------------------------------------------
template <typename FloatType>
void Matrix<FloatType>::tileSend(int64_t i, int64_t j, int dest)
{
    Tile<FloatType> *tile = (*this)(i, j);
    int count = tile->mb_*tile->nb_;
    int tag = 0;
    int retval;
    #pragma omp critical
    retval = MPI_Send(tile->data_, count, MPI_DOUBLE, dest, tag, mpi_comm_);
    assert(retval == MPI_SUCCESS);
}

//------------------------------------------------------------------------------
template <typename FloatType>
void Matrix<FloatType>::tileRecv(int64_t i, int64_t j, int src)
{
    Tile<FloatType> *tile = new Tile<FloatType>(tileMb(i), tileNb(j), memory_);
    (*this)(i, j) = tile;
    int count = tile->mb_*tile->nb_;
    int tag = 0;
    int retval;
    #pragma omp critical
    retval = MPI_Recv(tile->data_, count, MPI_DOUBLE, src, tag, mpi_comm_,
                      MPI_STATUS_IGNORE);
    assert(retval == MPI_SUCCESS);
}

//------------------------------------------------------------------------------
template <typename FloatType>
template <Target target>
void Matrix<FloatType>::tileSend(int64_t i, int64_t j,
                                 std::array<int64_t, 4> range)
{
    // Find the set of participating ranks.
    std::set<int> bcast_set;
    bcast_set.insert(tileRank(i, j));
    tileSendFindRanks(i, j, range, &bcast_set);

    // If contained in the set.
    if (bcast_set.find(mpi_rank_) != bcast_set.end()) {

        // If receiving the tile.
        if (!tileIsLocal(i, j)) {

            // Create the tile.
            Tile<FloatType> *tile;
            tile = new Tile<FloatType>(tileMb(i), tileNb(j), memory_);
            (*this)(i, j) = tile;

            // Find the tile's life.
            tile->local_ = false;
            tile->life_ = tileSendFindLife(i, j, range);
        }
        // Send across MPI ranks.
        tileSend(i, j, bcast_set);

        // Copy to devices.
        if (target == Target::Devices)
            for (int device = 0; device < num_devices_; ++device)
                tileCopyToDevice(i, j, device);
    }
}

//------------------------------------------------------------------------------
template <typename FloatType>
template <Target target>
void Matrix<FloatType>::tileSend(int64_t i, int64_t j,
                                 std::array<int64_t, 4> range1,
                                 std::array<int64_t, 4> range2)
{
    // Find the set of participating ranks.
    std::set<int> bcast_set;
    bcast_set.insert(tileRank(i, j));
    tileSendFindRanks(i, j, range1, &bcast_set);
    tileSendFindRanks(i, j, range2, &bcast_set);

    // If contained in the set.
    if (bcast_set.find(mpi_rank_) != bcast_set.end()) {

        // If receiving the tile.
        if (!tileIsLocal(i, j)) {

            // Create the tile.
            Tile<FloatType> *tile;
            tile = new Tile<FloatType>(tileMb(i), tileNb(j), memory_);
            (*this)(i, j) = tile;

            // Find the tile's life.
            tile->local_ = false;
            tile->life_  = tileSendFindLife(i, j, range1);
            tile->life_ += tileSendFindLife(i, j, range2);
        }
        // Send across MPI ranks.
        tileSend(i, j, bcast_set);

        // Copy to devices.
        if (target == Target::Devices)
            for (int device = 0; device < num_devices_; ++device)
                tileCopyToDevice(i, j, device);
    }
}

//------------------------------------------------------------------------------
template <typename FloatType>
void Matrix<FloatType>::tileSendFindRanks(int64_t i, int64_t j,
                                          std::array<int64_t, 4> range,
                                          std::set<int> *bcast_set)
{
    int64_t i1 = range[0];
    int64_t i2 = range[1];
    int64_t j1 = range[2];
    int64_t j2 = range[3];

    // Find the set of participating ranks.
    for (int64_t i = i1; i <= i2; ++i)
        for (int64_t j = j1; j <= j2; ++j)
            bcast_set->insert(tileRank(i, j));
}

//------------------------------------------------------------------------------
template <typename FloatType>
int64_t Matrix<FloatType>::tileSendFindLife(int64_t i, int64_t j,
                                            std::array<int64_t, 4> range)
{
    int64_t i1 = range[0];
    int64_t i2 = range[1];
    int64_t j1 = range[2];
    int64_t j2 = range[3];

    // Find the tile's lifespan.
    int64_t life = 0;
    for (int64_t i = i1; i <= i2; ++i)
        for (int64_t j = j1; j <= j2; ++j)
            if (tileIsLocal(i, j))
                ++life;

    return life;
}

//------------------------------------------------------------------------------
template <typename FloatType>
void Matrix<FloatType>::tileSend(int64_t i, int64_t j, std::set<int> &bcast_set)
{
    // Quit if only root in the broadcast set.
    if (bcast_set.size() == 1)
        return;

    // Convert the set of ranks to a vector.
    std::vector<int> bcast_vec(bcast_set.begin(), bcast_set.end());

    // Create the broadcast group.
    Tile<FloatType> *tile = (*this)(i, j);
    int retval;
    #pragma omp critical
    retval = MPI_Group_incl(mpi_group_, bcast_vec.size(), bcast_vec.data(),
                            &tile->bcast_group_);
    assert(retval == MPI_SUCCESS);

    // Create a broadcast communicator.
    int tag = 0;
    trace_cpu_start();
    #pragma omp critical
    retval = MPI_Comm_create_group(mpi_comm_, tile->bcast_group_, tag,
                                   &tile->bcast_comm_);
    assert(retval == MPI_SUCCESS);
    assert(tile->bcast_comm_ != MPI_COMM_NULL);
    trace_cpu_stop("Crimson");

    // Find the broadcast rank.
    int bcast_rank;
    #pragma omp critical
    MPI_Comm_rank(tile->bcast_comm_, &bcast_rank);

    // Find the broadcast root rank.
    int root_rank = tileRank(i, j);
    int bcast_root;
    #pragma omp critical
    retval = MPI_Group_translate_ranks(mpi_group_, 1, &root_rank,
                                       tile->bcast_group_, &bcast_root);
    assert(retval == MPI_SUCCESS);

    // Do the broadcast.
    int count = tile->mb_*tile->nb_;
    #pragma omp critical
    retval = MPI_Bcast(tile->data_, count, MPI_DOUBLE, bcast_root,
                       tile->bcast_comm_);
    assert(retval == MPI_SUCCESS);

    // Free the group.
    #pragma omp critical
    retval = MPI_Group_free(&tile->bcast_group_);
    assert(retval == MPI_SUCCESS);

    // Free the communicator.
    #pragma omp critical
    retval = MPI_Comm_free(&tile->bcast_comm_);
    assert(retval == MPI_SUCCESS);
}

//------------------------------------------------------------------------------
template <typename FloatType>
void Matrix<FloatType>::checkLife()
{
    for (auto it = tiles_->begin(); it != tiles_->end(); ++it) {
        if (!tileIsLocal(std::get<0>(it->first), std::get<1>(it->first)))
            if (it->second->life_ != 0 || it->second->data_ != nullptr)
                std::cout << "P" << mpi_rank_
                          << " TILE " << std::get<0>(it->first)
                          << " " << std::get<1>(it->first)
                          << " LIFE " << it->second->life_
                          << " data_ " << it->second->data_ 
                          << " DEV " << std::get<2>(it->first) << std::endl;
    }
}

//------------------------------------------------------------------------------
template <typename FloatType>
void Matrix<FloatType>::printLife()
{
    if (mpi_rank_ == 0) {
        for (int64_t i = 0; i < mt_; ++i) {
            for (int64_t j = 0; j < nt_; j++) {
                if (tiles_->find({i, j, host_num_}) == tiles_->end())
                    printf("  .");
                else
                    printf("%3ld", (*tiles_)[{i, j, host_num_}]->life_);
            }
            printf("\n");
        }
    }
}

} // namespace slate

#endif // SLATE_MATRIX_HH
