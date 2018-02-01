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

#ifndef SLATE_TILE_HH
#define SLATE_TILE_HH

#include "slate_Memory.hh"
#include "slate_Trace.hh"

#include <blas.hh>
#include <lapack.hh>

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <memory>

#ifdef SLATE_WITH_CUDA
    #include <cuda_runtime.h>
#else
    #include "slate_NoCuda.hh"
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
template <typename scalar_t>
class Tile {
public:
    Tile()
    {}

    Tile(int64_t mb, int64_t nb,
         std::weak_ptr<Memory> memory,
         MPI_Comm mpi_comm);

    Tile(int64_t mb, int64_t nb,
         scalar_t *a, int64_t lda,
         std::weak_ptr<Memory> memory,
         MPI_Comm mpi_comm);

    Tile(const Tile<scalar_t> *src_tile, int dst_device_num);

    // todo:
    // Tile doesn't own data; shouldn't deallocate it.
    // MatrixStorage should deallocate it when it removes tile from map.
    // ~Tile()
    // {
    //     if (! origin) {
    //         deallocate();
    //     }
    // }

    Tile<scalar_t>* copyToHost(cudaStream_t stream);
    Tile<scalar_t>* copyToDevice(int device_num, cudaStream_t stream);

    void copyDataToHost(Tile<scalar_t> *dst_tile, cudaStream_t stream);
    void copyDataToDevice(Tile<scalar_t> *dst_tile, cudaStream_t stream);

    // todo: send/recv take mpi_comm?
    void send(int dst);
    void recv(int src);
    void bcast(int bcast_root, MPI_Comm bcast_comm);

    // todo: remove tile BLAS from class    
    /// static void gemm(blas::Op transa, blas::Op transb,
    ///                  scalar_t alpha, Tile<scalar_t> *a,
    ///                                  Tile<scalar_t> *b,
    ///                  scalar_t beta,  Tile<scalar_t> *c);
    /// 
    /// static void potrf(blas::Uplo uplo, Tile<scalar_t> *a);
    /// 
    /// static void syrk(blas::Uplo uplo, blas::Op trans,
    ///                  scalar_t alpha, Tile<scalar_t> *a,
    ///                  scalar_t beta,  Tile<scalar_t> *c);
    /// 
    /// static void trsm(blas::Side side, blas::Uplo uplo,
    ///                  blas::Op transa, blas::Diag diag,
    ///                  scalar_t alpha, Tile<scalar_t> *a,
    ///                                  Tile<scalar_t> *b);

    // Tiles and Matrices use same transpose functions ;)
    /// Returns shallow copy of tile that is transposed.
    template< typename TileType >
    friend TileType transpose( TileType& A );

    /// Returns shallow copy of tile that is conjugate-transposed.
    template< typename TileType >
    friend TileType conj_transpose( TileType& A );
    
    /// m is rows of op(A)
    int64_t m()      const { return (op_ == blas::Op::NoTrans ? mb_ : nb_); }
    
    /// n is cols of op(A)
    int64_t n()      const { return (op_ == blas::Op::NoTrans ? nb_ : mb_); }
    
    /// column stride of A
    int64_t stride() const { return stride_; }
    
    /// data pointer
    scalar_t const* data() const { return data_; }
    scalar_t*       data()       { return data_; }
    
    /// returns op(A)_{i, j}.
    /// If op() is ConjTrans, data is NOT conjugated,
    /// because a reference is returned.
    scalar_t& operator() ( int64_t i, int64_t j )
    {
        assert( 0 <= i && i < m() );
        assert( 0 <= j && j < n() );
        if (op_ == blas::Op::NoTrans) {
            return data_[ i + j*stride_ ];
        }
        else {
            return data_[ j + i*stride_ ];
        }
    }
    
    /// sets/gets whether this tile is valid (cache coherency protocol)
    bool valid() const { return valid_; }
    void valid( bool val ) { valid_ = val; }
    
    /// whether this tile was originally given by the user (true),
    /// or is a workspace buffer.
    bool origin() const { return origin_; }

    // todo: in C++ std, size is # elements; add bytes() for actual size
    /// number of bytes; but NOT consecutive if stride != m.
    size_t bytes() { return sizeof(scalar_t) * size2(); }
    
    /// number of elements; but elements are NOT consecutive if stride != m.
    size_t size2() { return (size_t) mb_ * nb_; }

protected:
    void allocate();
    void deallocate();
    
    int64_t mb_;
    int64_t nb_;
    int64_t stride_;
    blas::Op op_;
    blas::Uplo uplo_;

    scalar_t *data_;

    bool valid_;
    bool origin_;

    static int host_num_;
    int device_num_;

    MPI_Comm mpi_comm_;  // todo: remove?
    std::weak_ptr<Memory> memory_;  // todo: remove?
};

///-----------------------------------------------------------------------------
/// \brief
/// Creates new tile in host memory from workspace allocator.
/// Data is uninitialized.
template <typename scalar_t>
Tile<scalar_t>::Tile(int64_t mb, int64_t nb,
                      std::weak_ptr<Memory> memory,
                      MPI_Comm mpi_comm)
    : mb_(mb),
      nb_(nb),
      stride_(mb),
      data_(nullptr),
      valid_(true),
      origin_(false),
      device_num_(host_num_),
      mpi_comm_(mpi_comm),
      memory_(memory)
{
    allocate();
}

///-----------------------------------------------------------------------------
/// \brief
/// Creates tile that wraps existing memory buffer.
/// TODO: currently assumes data is on host.
/// Sets origin = true.
template <typename scalar_t>
Tile<scalar_t>::Tile(int64_t mb, int64_t nb,
                      scalar_t *a, int64_t lda,
                      std::weak_ptr<Memory> memory,
                      MPI_Comm mpi_comm)
    : mb_(mb),
      nb_(nb),
      stride_(lda),
      data_(a),
      valid_(true),
      origin_(true),
      device_num_(host_num_),  // todo: take device_num
      mpi_comm_(mpi_comm),
      memory_(memory)
{}

///-----------------------------------------------------------------------------
/// \brief
/// Creates new tile on given device from existing tile.
// todo: use ref instead of pointer?
template <typename scalar_t>
Tile<scalar_t>::Tile(const Tile<scalar_t> *src_tile, int dst_device_num)
{
    *this = *src_tile;
    this->origin_ = false;
    this->stride_ = this->mb_;
    this->device_num_ = dst_device_num;
    allocate();
}

///-----------------------------------------------------------------------------
/// \brief
/// Creates new tile on host and copies data from device to host.
// todo: could these all implicitly figure out ToHost / ToDevice from direction from the device_nums?
template <typename scalar_t>
Tile<scalar_t>*
Tile<scalar_t>::copyToHost(cudaStream_t stream)
{
    Tile<scalar_t> *dst_tile = new Tile<scalar_t>(this, this->host_num_);
    this->copyDataToHost(dst_tile, stream);
    return dst_tile;
}

///-----------------------------------------------------------------------------
/// \brief
/// Creates new tile on device and copies data from host to device.
template <typename scalar_t>
Tile<scalar_t>*
Tile<scalar_t>::copyToDevice(int device_num, cudaStream_t stream)
{
    Tile<scalar_t> *dst_tile = new Tile<scalar_t>(this, device_num);
    this->copyDataToDevice(dst_tile, stream);
    return dst_tile;
}

///-----------------------------------------------------------------------------
/// \brief
/// Copies data from this tile on device to dst_tile on host.
// todo: tile shouldn't be const
template <typename scalar_t>
void Tile<scalar_t>::copyDataToHost(
    Tile<scalar_t> *dst_tile, cudaStream_t stream)
{
    trace::Block trace_block(trace::Color::Gray);

    cudaError_t error;
    error = cudaSetDevice(device_num_);
    assert(error == cudaSuccess);

    // If no stride on both sides.
    if (stride_ == mb_ &&
        dst_tile->stride_ == dst_tile->mb_) {

        // Use simple copy.
        error = cudaMemcpyAsync(
            dst_tile->data_, data_, bytes(),
            cudaMemcpyDeviceToHost, stream);
        assert(error == cudaSuccess);
    }
    else {
        // Otherwise, use 2D copy.
        void* dst = dst_tile->data_;
        const void* src = data_;
        size_t dpitch = sizeof(scalar_t)*dst_tile->stride_;
        size_t spitch = sizeof(scalar_t)*stride_;
        size_t width = sizeof(scalar_t)*mb_;
        size_t height = nb_;

        error = cudaMemcpy2DAsync(
            dst, dpitch,
            src, spitch,
            width, height,
            cudaMemcpyDeviceToHost, stream);
        assert(error == cudaSuccess);
    }

    error = cudaStreamSynchronize(stream);
    assert(error == cudaSuccess);
}

///-----------------------------------------------------------------------------
/// \brief
/// Copies data from this tile on host to tile on device.
template <typename scalar_t>
void Tile<scalar_t>::copyDataToDevice(
    Tile<scalar_t> *dst_tile, cudaStream_t stream)
{
    trace::Block trace_block(trace::Color::LightGray);

    cudaError_t error;
    error = cudaSetDevice(dst_tile->device_num_);
    assert(error == cudaSuccess);

    // If no stride on both sides.
    if (stride_ == mb_ &&
        dst_tile->stride_ == dst_tile->mb_) {

        // Use simple copy.
        error = cudaMemcpyAsync(
            dst_tile->data_, data_, bytes(),
            cudaMemcpyHostToDevice, stream);
        assert(error == cudaSuccess);
    }
    else {
        // Otherwise, use 2D copy.
        void* dst = dst_tile->data_;
        const void* src = data_;
        size_t dpitch = sizeof(scalar_t)*dst_tile->stride_;
        size_t spitch = sizeof(scalar_t)*stride_;
        size_t width = sizeof(scalar_t)*mb_;
        size_t height = nb_;

        error = cudaMemcpy2DAsync(
            dst, dpitch,
            src, spitch,
            width, height,
            cudaMemcpyHostToDevice, stream);
        assert(error == cudaSuccess);
    }

    error = cudaStreamSynchronize(stream);
    assert(error == cudaSuccess);
}

///-----------------------------------------------------------------------------
/// \brief
/// Sends tile to MPI rank dst.
// todo: take communicator?
template <typename scalar_t>
void Tile<scalar_t>::send(int dst)
{
    // If no stride.
    if (stride_ == mb_) {
        // Use simple send.
        int count = mb_*nb_;
        int tag = 0;
        int retval;

        #pragma omp critical(slate_mpi)
        retval = MPI_Send(data_, count, MPI_DOUBLE, dst, tag, mpi_comm_);
        assert(retval == MPI_SUCCESS);
    }
    else {
        // Otherwise, use strided send.
        int count = nb_;
        int blocklength = mb_;
        int stride = stride_;
        MPI_Datatype newtype;
        int tag = 0;
        int retval;

        #pragma omp critical(slate_mpi)
        retval = MPI_Type_vector(
            count, blocklength, stride, MPI_DOUBLE, &newtype);
        assert(retval == MPI_SUCCESS);

        #pragma omp critical(slate_mpi)
        retval = MPI_Type_commit(&newtype);
        assert(retval == MPI_SUCCESS);

        #pragma omp critical(slate_mpi)
        retval = MPI_Send(data_, 1, newtype, dst, tag, mpi_comm_);
        assert(retval == MPI_SUCCESS);

        #pragma omp critical(slate_mpi)
        retval = MPI_Type_free(&newtype);
        assert(retval == MPI_SUCCESS);
    }
}

///-----------------------------------------------------------------------------
/// \brief
/// Receives tile from MPI rank src.
// todo: take communicator?
template <typename scalar_t>
void Tile<scalar_t>::recv(int src)
{
    // If no stride.
    if (stride_ == mb_) {
        // Use simple recv.
        int count = mb_*nb_;
        int tag = 0;
        int retval;

        #pragma omp critical(slate_mpi)
        retval = MPI_Recv(
            data_, count, MPI_DOUBLE, src, tag, mpi_comm_, MPI_STATUS_IGNORE);
        assert(retval == MPI_SUCCESS);
    }
    else {
        // Otherwise, use strided recv. 
        int count = nb_;
        int blocklength = mb_;
        int stride = stride_;
        MPI_Datatype newtype;
        int retval;

        #pragma omp critical(slate_mpi)
        retval = MPI_Type_vector(
            count, blocklength, stride, MPI_DOUBLE, &newtype);
        assert(retval == MPI_SUCCESS);

        #pragma omp critical(slate_mpi)
        retval = MPI_Type_commit(&newtype);
        assert(retval == MPI_SUCCESS);

        int tag = 0;
        #pragma omp critical(slate_mpi)
        retval = MPI_Recv(
            data_, 1, newtype, src, tag, mpi_comm_, MPI_STATUS_IGNORE);
        assert(retval == MPI_SUCCESS);

        #pragma omp critical(slate_mpi)
        retval = MPI_Type_free(&newtype);
        assert(retval == MPI_SUCCESS);
    }
}

///-----------------------------------------------------------------------------
/// \brief
/// Broadcasts tile from MPI rank bcast_root, using given communicator.
template <typename scalar_t>
void Tile<scalar_t>::bcast(int bcast_root, MPI_Comm bcast_comm)
{
    trace::Block trace_block(trace::Color::Crimson);

    // If no stride.
    if (stride_ == mb_) {
        // Use simple bcast.
        int count = mb_*nb_;
        int retval;

        #pragma omp critical(slate_mpi)
        retval = MPI_Bcast(data_, count, MPI_DOUBLE, bcast_root, bcast_comm);
        assert(retval == MPI_SUCCESS);
    }
    else {
        // Otherwise, use strided bcast.
        int count = nb_;
        int blocklength = mb_;
        int stride = stride_;
        MPI_Datatype newtype;
        int retval;

        #pragma omp critical(slate_mpi)
        retval = MPI_Type_vector(
            count, blocklength, stride, MPI_DOUBLE, &newtype);
        assert(retval == MPI_SUCCESS);

        #pragma omp critical(slate_mpi)
        retval = MPI_Type_commit(&newtype);
        assert(retval == MPI_SUCCESS);

        #pragma omp critical(slate_mpi)
        retval = MPI_Bcast(data_, 1, newtype, bcast_root, bcast_comm);
        assert(retval == MPI_SUCCESS);

        #pragma omp critical(slate_mpi)
        retval = MPI_Type_free(&newtype);
        assert(retval == MPI_SUCCESS);
    }
}

///=============================================================================
// Tile BLAS

///-----------------------------------------------------------------------------
/// \brief
/// General matrix multiply: $C = \alpha op(A) op(B) + \beta C$.
/// Use transpose/conj_transpose to set $op(A)$ and $op(B)$.
template <typename scalar_t>
void gemm(
    scalar_t alpha, Tile<scalar_t> const& A,
                    Tile<scalar_t> const& B,
    scalar_t beta,  Tile<scalar_t>& C )
{
    trace::Block trace_block(trace::Color::MediumAquamarine);

    assert( A.uplo() == blas::Uplo::General );
    assert( B.uplo() == blas::Uplo::General );
    assert( C.uplo() == blas::Uplo::General );
    assert( C.op() == blas::Op::NoTrans );  // todo: row-major
    assert( C.m() == A.m() );  // m
    assert( C.n() == B.n() );  // n
    assert( A.n() == B.m() );  // k
    blas::gemm( blas::Layout::ColMajor,
                A.op(), B.op(),
                C.m(), C.n(), A.n(),
                alpha, A.data(), A.stride(),
                       B.data(), B.stride(),
                beta,  C.data(), C.stride() );
}

///-----------------------------------------------------------------------------
/// \brief
/// Cholesky factorization of tile: $L L^H = A$ or $U^H U = A$.
/// uplo is set in the tile.
template <typename scalar_t>
void potrf( Tile<scalar_t>& A )
{
    trace::Block trace_block(trace::Color::RosyBrown);

    assert( A.op() == blas::Op::NoTrans );  // todo: row-major
    lapack::potrf( A.uplo(),
                   A.n(),
                   A.data(), A.stride() );
}

///-----------------------------------------------------------------------------
/// \brief
/// Symmetric rank-k update: $C = \alpha op(A) op(A)^T + \beta C$.
/// Use transpose/conj_transpose to set $op(A)$.
template <typename scalar_t>
void syrk(
    scalar_t alpha, Tile<scalar_t> const& A,
    scalar_t beta,  Tile<scalar_t>& C )
{
    trace::Block trace_block(trace::Color::CornflowerBlue);

    assert( A.uplo() == blas::Uplo::General );
    assert( C.m() == C.n() );  // square
    assert( C.m() == A.m() );  // n
    assert( C.op() == blas::Op::NoTrans );  // todo: row-major
    blas::syrk( blas::Layout::ColMajor,
                C.uplo(), A.op(),
                C.n(), A.n(),
                alpha, A.data(), A.stride(),
                beta,  C.data(), C.stride() );
}

///-----------------------------------------------------------------------------
/// \brief
/// Triangular solve: $B = \alpha op(A)^{-1} B$ or $B = \alpha B op(A)^{-1}$.
/// Use transpose/conj_transpose to set op(A). uplo is set in the tile.
template <typename scalar_t>
void trsm(
    blas::Side side, blas::Diag diag,
    scalar_t alpha, Tile<scalar_t> const& A,
                    Tile<scalar_t>& B )
{
    trace::Block trace_block(trace::Color::MediumPurple);

    assert( B.uplo() == blas::Uplo::General );
    assert( B.op() == blas::Op::NoTrans );  // todo: row-major
    assert( A.m() == A.n() );  // square
    assert( side == blas::Side::Left ? A.m() == B.m()     // m
                                     : A.m() == B.n() );  // n
    blas::trsm( blas::Layout::ColMajor,
                side, A.uplo(), A.op(), diag,
                B.m(), B.n(),
                alpha, A.data(), A.stride(),
                       B.data(), B.stride() );
}

///=============================================================================
// Tile methods

///-----------------------------------------------------------------------------
/// \brief
/// Allocates a tile in host or device memory using workspace allocator.
template <typename scalar_t>
void Tile<scalar_t>::allocate()
{
    trace::Block trace_block(trace::Color::Aqua);
    data_ = (scalar_t*)memory_.lock()->alloc(device_num_);
}

///-----------------------------------------------------------------------------
/// \brief
/// Frees a tile in host or device memory using workspace allocator.
template <typename scalar_t>
void Tile<scalar_t>::deallocate()
{
    trace::Block trace_block(trace::Color::Aquamarine);
    memory_.lock()->free(data_, device_num_);
    data_ = nullptr;
}

} // namespace slate

#endif // SLATE_TILE_HH
