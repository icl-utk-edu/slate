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
#include "slate_trace_Trace.hh"
#include "slate_types.hh"

#include <blas.hh>
#include <lapack.hh>

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <memory>

#include "slate_cuda.hh"
#include "slate_mpi.hh"
#include "slate_openmp.hh"

namespace slate {

///-----------------------------------------------------------------------------
/// transpose returns Tile, Matrix, SymmetricMatrix, etc.
/// Making a template avoids repeating the code ad nauseum in each class.
/// Tile and BaseMatrix make this a friend, to change op.
template< typename MatrixType >
MatrixType transpose(MatrixType& A)
{
    MatrixType AT = A;
    if (AT.op_ == Op::NoTrans) {
        AT.op_ = Op::Trans;
    }
    else if (AT.op_ == Op::Trans || A.is_real) {
        AT.op_ = Op::NoTrans;
    }
    else {
        throw std::exception();  // todo: op_ = Op::Conj doesn't exist
    }
    return AT;
}

/// @see transpose()
template< typename MatrixType >
MatrixType conj_transpose(MatrixType& A)
{
    MatrixType AT = A;
    if (AT.op_ == Op::NoTrans) {
        AT.op_ = Op::ConjTrans;
    }
    else if (AT.op_ == Op::ConjTrans || A.is_real) {
        AT.op_ = Op::NoTrans;
    }
    else {
        throw std::exception();  // todo: op_ = Op::Conj doesn't exist
    }
    return AT;
}

//------------------------------------------------------------------------------
/// Tile holding an mb-by-nb matrix, with leading dimension (stride).
template <typename scalar_t>
class Tile {
public:
    static constexpr bool is_complex = is_complex< scalar_t >::value;
    static constexpr bool is_real    = ! is_complex;

    Tile();

    Tile(int64_t mb, int64_t nb,
         scalar_t* A, int64_t lda, int device, bool origin=true);

    // defaults okay (tile doesn't own data, doesn't allocate/deallocate data)
    // 1. destructor
    // 2. copy & move constructors
    // 3. copy & move assignment

    void copyDataToHost(Tile<scalar_t>* dst_tile, cudaStream_t stream) const;
    void copyDataToDevice(Tile<scalar_t>* dst_tile, cudaStream_t stream) const;

    void send(int dst, MPI_Comm mpi_comm) const;
    void recv(int src, MPI_Comm mpi_comm);
    void bcast(int bcast_root, MPI_Comm mpi_comm);

    /// Returns shallow copy of tile that is transposed.
    template< typename TileType >
    friend TileType transpose(TileType& A);

    /// Returns shallow copy of tile that is conjugate-transposed.
    template< typename TileType >
    friend TileType conj_transpose(TileType& A);

    /// Returns number of rows of op(A), where A is this tile
    int64_t mb() const { return (op_ == Op::NoTrans ? mb_ : nb_); }

    /// Returns number of cols of op(A), where A is this tile
    int64_t nb() const { return (op_ == Op::NoTrans ? nb_ : mb_); }

    /// Returns column stride of this tile
    int64_t stride() const { return stride_; }

    /// Returns const pointer to data, i.e., A(0,0), where A is this tile
    scalar_t const* data() const { return data_; }

    /// Returns pointer to data, i.e., A(0,0), where A is this tile
    scalar_t*       data()       { return data_; }

    scalar_t operator() (int64_t i, int64_t j) const;
    scalar_t const& at(int64_t i, int64_t j) const;
    scalar_t&       at(int64_t i, int64_t j);

    /// Returns whether this tile is valid (cache coherency protocol).
    bool valid() const { return valid_; }

    /// Sets whether this tile is valid (cache coherency protocol).
    void valid(bool val) { valid_ = val; }  // todo: protected?

    /// Returns whether this is a local tile, originally given by the user (true),
    /// or is a workspace buffer.
    bool origin() const { return origin_; }

    /// Returns number of bytes; but NOT consecutive if stride != mb_.
    size_t bytes() const { return sizeof(scalar_t) * size(); }

    /// Returns number of elements; but NOT consecutive if stride != mb_.
    size_t size()  const { return (size_t) mb_ * nb_; }

    /// Returns upper, lower, or general storage flag.
    Uplo uplo() const { return uplo_; }
    Uplo uplo_logical() const;

    /// Sets upper, lower, or general storage flag.
    void uplo(Uplo uplo) { uplo_ = uplo; }  // todo: protected?

    /// Returns transposition operation.
    Op op() const { return op_; }

    /// Sets transposition operation.
    void op(Op op) { op_ = op; }  // todo: protected?

    /// Returns which host or GPU device tile's data is located on.
    int device() const { return device_; }

protected:
    int64_t mb_;
    int64_t nb_;
    int64_t stride_;
    Op op_;
    Uplo uplo_;

    scalar_t* data_;

    bool valid_;
    bool origin_;

    int device_;
};

//------------------------------------------------------------------------------
/// Create empty tile.
template <typename scalar_t>
Tile<scalar_t>::Tile()
    : mb_(0),
      nb_(0),
      stride_(0),
      op_(Op::NoTrans),
      uplo_(Uplo::General),
      data_(nullptr),
      valid_(false),
      origin_(true),
      device_(-1)  // todo: host_num
{}

//------------------------------------------------------------------------------
/// Create tile that wraps existing memory buffer.
///
/// @param[in] mb
///     Number of rows of the tile. mb >= 0.
///
/// @param[in] nb
///     Number of columns of the tile. nb >= 0.
///
/// @param[in,out] A
///     The mb-by-nb tile A, stored in an lda-by-nb array.
///
/// @param[in] lda
///     Leading dimension of the array A. lda >= mb.
///
/// @param[in] device
///     Tile's device ID.
///
/// @param[in] origin
///     Whether tile is a local tile, originally given by the user (true),
///     or is a workspace tile.
template <typename scalar_t>
Tile<scalar_t>::Tile(
    int64_t mb, int64_t nb,
    scalar_t* A, int64_t lda, int device, bool origin)
    : mb_(mb),
      nb_(nb),
      stride_(lda),
      op_(Op::NoTrans),
      uplo_(Uplo::General),
      data_(A),
      valid_(true),
      origin_(origin),
      device_(device)
{
    assert( mb >= 0 );
    assert( nb >= 0 );
    assert( A != nullptr );
    assert( lda >= mb );
}

//------------------------------------------------------------------------------
/// Returns element {i, j} of op(A).
/// The actual value is returned, not a reference. Use at() to get a reference.
/// If op() is ConjTrans, data IS conjugated, unlike with at().
///
/// @param[in] i
///     Row index. 0 <= i < mb.
///
/// @param[in] j
///     Column index. 0 <= j < nb.
template <typename scalar_t>
scalar_t Tile<scalar_t>::operator() (int64_t i, int64_t j) const
{
    using blas::conj;
    assert(0 <= i && i < mb());
    assert(0 <= j && j < nb());
    if (op_ == Op::NoTrans) {
        return data_[ i + j*stride_ ];
    }
    else if (op_ == Op::Trans) {
        return data_[ j + i*stride_ ];
    }
    else {
        assert(op_ == Op::ConjTrans);
        return conj( data_[ j + i*stride_ ] );
    }
}

//------------------------------------------------------------------------------
/// Returns a const reference to element {i, j} of op(A).
/// If op() is ConjTrans, data is NOT conjugated,
/// because a reference is returned.
/// Use operator() to get the actual value, conjugated if need be.
///
/// @param[in] i
///     Row index. 0 <= i < mb.
///
/// @param[in] j
///     Column index. 0 <= j < nb.
template <typename scalar_t>
scalar_t const& Tile<scalar_t>::at(int64_t i, int64_t j) const
{
    assert(0 <= i && i < mb());
    assert(0 <= j && j < nb());
    if (op_ == Op::NoTrans) {
        return data_[ i + j*stride_ ];
    }
    else {
        assert(op_ == Op::Trans || op_ == Op::ConjTrans);
        return data_[ j + i*stride_ ];
    }
}

//------------------------------------------------------------------------------
/// Returns a reference to element {i, j} of op(A).
/// If op() is ConjTrans, data is NOT conjugated,
/// because a reference is returned.
/// Use operator() to get the actual value, conjugated if need be.
///
/// @param[in] i
///     Row index. 0 <= i < mb.
///
/// @param[in] j
///     Column index. 0 <= j < nb.
template <typename scalar_t>
scalar_t& Tile<scalar_t>::at(int64_t i, int64_t j)
{
    // forward to const at() version
    return const_cast<scalar_t&>( static_cast<const Tile>(*this).at(i, j) );
}

//--------------------------------------------------------------------------
/// Returns whether op(A) is logically Upper, Lower, or General storage,
///         taking the transposition operation into account.
/// @see uplo()
template <typename scalar_t>
Uplo Tile<scalar_t>::uplo_logical() const
{
    if (this->uplo() == Uplo::General) {
        return Uplo::General;
    }
    else if ((this->uplo() == Uplo::Lower && this->op() == Op::NoTrans) ||
             (this->uplo() == Uplo::Upper && this->op() != Op::NoTrans))
    {
        return Uplo::Lower;
    }
    else {
        return Uplo::Upper;
    }
}

//------------------------------------------------------------------------------
/// Copies data from this tile on device to dst_tile on host.
///
/// @param[in] dst_tile
///     Destination tile, assumed to be on host.
///
/// @param[in] stream
///     CUDA stream for copy.
//
// todo need to copy or verify metadata (sizes, op, uplo, ...)
template <typename scalar_t>
void Tile<scalar_t>::copyDataToHost(
    Tile<scalar_t>* dst_tile, cudaStream_t stream) const
{
    cudaError_t error;
    error = cudaSetDevice(device_);
    assert(error == cudaSuccess);

    // If no stride on both sides.
    if (stride_ == mb_ &&
        dst_tile->stride_ == dst_tile->mb_) {

        // Use simple copy.
        trace::Block trace_block("cudaMemcpyAsync");

        error = cudaMemcpyAsync(
            dst_tile->data_, data_, bytes(),
            cudaMemcpyDeviceToHost, stream);
        assert(error == cudaSuccess);

        error = cudaStreamSynchronize(stream);
        assert(error == cudaSuccess);
    }
    else {
        // Otherwise, use 2D copy.
        trace::Block trace_block("cudaMemcpy2DAsync");

        void* dst = dst_tile->data_;
        const void* src = data_;
        size_t dpitch = sizeof(scalar_t)*dst_tile->stride_;
        size_t spitch = sizeof(scalar_t)*stride_;
        size_t width  = sizeof(scalar_t)*mb_;
        size_t height = nb_;

        error = cudaMemcpy2DAsync(
            dst, dpitch,
            src, spitch,
            width, height,
            cudaMemcpyDeviceToHost, stream);
        assert(error == cudaSuccess);

        error = cudaStreamSynchronize(stream);
        assert(error == cudaSuccess);
    }
}

//------------------------------------------------------------------------------
/// Copies data from this tile on host to dst_tile on device.
///
/// @param[in] dst_tile
///     Destination tile, assumed to be on device.
///
/// @param[in] stream
///     CUDA stream for copy.
//
// todo: need to copy or verify metadata (sizes, op, uplo, ...)
template <typename scalar_t>
void Tile<scalar_t>::copyDataToDevice(
    Tile<scalar_t>* dst_tile, cudaStream_t stream) const
{
    cudaError_t error;
    error = cudaSetDevice(dst_tile->device_);
    assert(error == cudaSuccess);

    // If no stride on both sides.
    if (stride_ == mb_ &&
        dst_tile->stride_ == dst_tile->mb_) {

        // Use simple copy.
        trace::Block trace_block("cudaMemcpyAsync");

        error = cudaMemcpyAsync(
            dst_tile->data_, data_, bytes(),
            cudaMemcpyHostToDevice, stream);
        assert(error == cudaSuccess);

        error = cudaStreamSynchronize(stream);
        assert(error == cudaSuccess);
    }
    else {
        // Otherwise, use 2D copy.
        trace::Block trace_block("cudaMemcpy2DAsync");

        void* dst = dst_tile->data_;
        const void* src = data_;
        size_t dpitch = sizeof(scalar_t)*dst_tile->stride_;
        size_t spitch = sizeof(scalar_t)*stride_;
        size_t width  = sizeof(scalar_t)*mb_;
        size_t height = nb_;

        error = cudaMemcpy2DAsync(
            dst, dpitch,
            src, spitch,
            width, height,
            cudaMemcpyHostToDevice, stream);
        assert(error == cudaSuccess);

        error = cudaStreamSynchronize(stream);
        assert(error == cudaSuccess);
    }
}

//------------------------------------------------------------------------------
/// Sends tile to MPI rank dst.
///
/// @param[in] dst
///     Destination MPI rank in mpi_comm.
///
/// @param[in] mpi_comm
///     MPI communicator.
//
// todo need to copy or verify metadata (sizes, op, uplo, ...)
template <typename scalar_t>
void Tile<scalar_t>::send(int dst, MPI_Comm mpi_comm) const
{
    // If no stride.
    if (stride_ == mb_) {
        // Use simple send.
        int count = mb_*nb_;
        int tag = 0;
        int retval;

        #pragma omp critical(slate_mpi)
        {
            retval = MPI_Send(data_, count, mpi_type<scalar_t>::value, dst, tag, mpi_comm);
        }
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
        {
            retval = MPI_Type_vector(
                count, blocklength, stride, mpi_type<scalar_t>::value, &newtype);
        }
        assert(retval == MPI_SUCCESS);

        #pragma omp critical(slate_mpi)
        {
            retval = MPI_Type_commit(&newtype);
        }
        assert(retval == MPI_SUCCESS);

        #pragma omp critical(slate_mpi)
        {
            retval = MPI_Send(data_, 1, newtype, dst, tag, mpi_comm);
        }
        assert(retval == MPI_SUCCESS);

        #pragma omp critical(slate_mpi)
        {
            retval = MPI_Type_free(&newtype);
        }
        assert(retval == MPI_SUCCESS);
    }
}

//------------------------------------------------------------------------------
/// Receives tile from MPI rank src.
///
/// @param[in] src
///     Source MPI rank in mpi_comm.
///
/// @param[in] mpi_comm
///     MPI communicator.
//
// todo need to copy or verify metadata (sizes, op, uplo, ...)
template <typename scalar_t>
void Tile<scalar_t>::recv(int src, MPI_Comm mpi_comm)
{
    // If no stride.
    if (stride_ == mb_) {
        // Use simple recv.
        int count = mb_*nb_;
        int tag = 0;
        int retval;

        #pragma omp critical(slate_mpi)
        {
            retval = MPI_Recv(
                data_, count, mpi_type<scalar_t>::value, src, tag, mpi_comm,
                MPI_STATUS_IGNORE);
        }
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
        {
            retval = MPI_Type_vector(
                count, blocklength, stride, mpi_type<scalar_t>::value, &newtype);
        }
        assert(retval == MPI_SUCCESS);

        #pragma omp critical(slate_mpi)
        {
            retval = MPI_Type_commit(&newtype);
        }
        assert(retval == MPI_SUCCESS);

        int tag = 0;
        #pragma omp critical(slate_mpi)
        {
            retval = MPI_Recv(
                data_, 1, newtype, src, tag, mpi_comm, MPI_STATUS_IGNORE);
        }
        assert(retval == MPI_SUCCESS);

        #pragma omp critical(slate_mpi)
        {
            retval = MPI_Type_free(&newtype);
        }
        assert(retval == MPI_SUCCESS);
    }
}

//------------------------------------------------------------------------------
/// Broadcasts tile from MPI rank bcast_root, using given communicator.
///
/// @param[in] bcast_root
///     Root (source) MPI rank in mpi_comm.
///
/// @param[in] mpi_comm
///     MPI communicator.
//
// todo: OpenMPI MPI_Bcast seems to have a bug such that either all ranks must
// use the simple case, or all ranks use vector case, even though the type
// signatures match.
//
// todo need to copy or verify metadata (sizes, op, uplo, ...)
template <typename scalar_t>
void Tile<scalar_t>::bcast(int bcast_root, MPI_Comm mpi_comm)
{
    // If no stride.
    //if (stride_ == mb_) {
    //    // Use simple bcast.
    //    int count = mb_*nb_;
    //    int retval;
    //
    //    #pragma omp critical(slate_mpi)
    //    retval = MPI_Bcast(data_, count, mpi_type<scalar_t>::value,
    //        bcast_root, mpi_comm);
    //    assert(retval == MPI_SUCCESS);
    //}
    //else
    {
        // Otherwise, use strided bcast.
        trace::Block trace_block("MPI_Bcast");

        int count = nb_;
        int blocklength = mb_;
        int stride = stride_;
        MPI_Datatype newtype;
        int retval;

        #pragma omp critical(slate_mpi)
        {
            retval = MPI_Type_vector(
                count, blocklength, stride, mpi_type<scalar_t>::value, &newtype);
        }
        assert(retval == MPI_SUCCESS);

        #pragma omp critical(slate_mpi)
        {
            retval = MPI_Type_commit(&newtype);
        }
        assert(retval == MPI_SUCCESS);

        #pragma omp critical(slate_mpi)
        {
            retval = MPI_Bcast(data_, 1, newtype, bcast_root, mpi_comm);
        }
        assert(retval == MPI_SUCCESS);

        #pragma omp critical(slate_mpi)
        {
            retval = MPI_Type_free(&newtype);
        }
        assert(retval == MPI_SUCCESS);
    }
}

} // namespace slate

#endif // SLATE_TILE_HH
