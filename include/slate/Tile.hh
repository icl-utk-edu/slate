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

#ifndef SLATE_TILE_HH
#define SLATE_TILE_HH

#include "slate/internal/Memory.hh"
#include "slate/internal/Trace.hh"
#include "slate/types.hh"
#include "slate/Exception.hh"

#include <blas.hh>
#include <lapack.hh>

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <memory>

#include "slate/internal/cuda.hh"
#include "slate/internal/mpi.hh"
#include "slate/internal/openmp.hh"

namespace slate {

///-----------------------------------------------------------------------------
/// Transpose returns Tile, Matrix, SymmetricMatrix, etc.
/// Making a template avoids repeating the code ad nauseum in each class.
/// Tile and BaseMatrix make this a friend, to change op.
///
template<typename MatrixType>
MatrixType transpose(MatrixType& A)
{
    MatrixType AT = A;
    if (AT.op_ == Op::NoTrans)
        AT.op_ = Op::Trans;
    else if (AT.op_ == Op::Trans || A.is_real)
        AT.op_ = Op::NoTrans;
    else
        slate_error("unsupported operation, results in conjugate-no-transpose");

    return AT;
}

///-------------------------------------
/// Converts rvalue refs to lvalue refs.
template<typename MatrixType>
MatrixType transpose(MatrixType&& A)
{
    return transpose(A);
}

/// @see transpose()
template<typename MatrixType>
MatrixType conj_transpose(MatrixType& A)
{
    MatrixType AT = A;
    if (AT.op_ == Op::NoTrans)
        AT.op_ = Op::ConjTrans;
    else if (AT.op_ == Op::ConjTrans || A.is_real)
        AT.op_ = Op::NoTrans;
    else
        slate_error("unsupported operation, results in conjugate-no-transpose");

    return AT;
}

///-------------------------------------
/// Converts rvalue refs to lvalue refs.
template<typename MatrixType>
MatrixType conj_transpose(MatrixType&& A)
{
    return conj_transpose(A);
}

//------------------------------------------------------------------------------
/// Whether a tile is workspace or origin (local non-workspace),
/// and who owns (allocated, deallocates) the data.
enum class TileKind
{
    Workspace,   ///< SLATE allocated workspace tile
    SlateOwned,  ///< SLATE allocated origin tile
    UserOwned,   ///< User owned origin tile
};

//------------------------------------------------------------------------------
/// Tile holding an mb-by-nb matrix, with leading dimension (stride).
template <typename scalar_t>
class Tile {
public:
    static constexpr bool is_complex = slate::is_complex<scalar_t>::value;
    static constexpr bool is_real    = ! is_complex;

    Tile();

    Tile(int64_t mb, int64_t nb,
         scalar_t* A, int64_t lda, int device, TileKind kind, Layout layout=Layout::ColMajor);

    // defaults okay (tile doesn't own data, doesn't allocate/deallocate data)
    // 1. destructor
    // 2. copy & move constructors
    // 3. copy & move assignment

    void copyDataToHost(Tile<scalar_t>* dst_tile, cudaStream_t stream) const;
    void copyDataToDevice(Tile<scalar_t>* dst_tile, cudaStream_t stream) const;

    void send(int dst, MPI_Comm mpi_comm, int tag = 0) const;
    void recv(int src, MPI_Comm mpi_comm, int tag = 0);
    void bcast(int bcast_root, MPI_Comm mpi_comm);

    /// Returns shallow copy of tile that is transposed.
    template <typename TileType>
    friend TileType transpose(TileType& A);

    /// Returns shallow copy of tile that is conjugate-transposed.
    template <typename TileType>
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

    scalar_t operator()(int64_t i, int64_t j) const;
    scalar_t const& at(int64_t i, int64_t j) const;
    scalar_t&       at(int64_t i, int64_t j);

    /// Returns whether this tile is valid (cache coherency protocol).
    bool valid() const { return valid_; }

    /// Sets whether this tile is valid (cache coherency protocol).
    void valid(bool val) { valid_ = val; }  // todo: protected?

    /// Returns true if this is an origin (local non-workspace) tile.
    bool origin() const { return ! workspace(); }

    /// Returns true if this is a workspace tile.
    bool workspace() const { return kind_ == TileKind::Workspace; }

    /// Returns true if SLATE allocated this tile's memory,
    /// false if the user provided the tile's memory,
    /// e.g., via a fromScaLAPACK constructor.
    bool allocated() const { return kind_ != TileKind::UserOwned; }

    /// Returns number of bytes; but NOT consecutive if stride != mb_.
    size_t bytes() const { return sizeof(scalar_t) * size(); }

    /// Returns number of elements; but NOT consecutive if stride != mb_.
    size_t size()  const { return (size_t) mb_ * nb_; }

    /// Returns whether op(A) is logically Lower, Upper, or General storage.
    Uplo uplo() const { return uploLogical(); }
    Uplo uploLogical() const;
    Uplo uploPhysical() const;
    Uplo uplo_logical() const { return uploLogical(); }  ///< @deprecated

    /// Sets upper, lower, or general storage flag.
    void uplo(Uplo uplo) { uplo_ = uplo; }  // todo: protected?

    /// Returns transposition operation.
    Op op() const { return op_; }

    /// Sets transposition operation.
    void op(Op op) { op_ = op; }  // todo: protected?

    /// Returns which host or GPU device tile's data is located on.
    int device() const { return device_; }

    Layout layout() const { return layout_; }
    void   layout(Layout in_layout) { layout_ = in_layout; }

    void set(scalar_t alpha);
    void set(scalar_t alpha, scalar_t beta);

    /// Returns whether this tile can be safely transposed
    /// based on its 'TileKind', buffer size, and stride.
    bool isTransposable ()
    {
        return ! (kind_ == TileKind::UserOwned
               && mb_ != nb_
               && stride_ != mb_);
    }

protected:
    // BaseMatrix sets mb, nb, offset.
    template <typename T>
    friend class BaseMatrix;

    void mb(int64_t in_mb);
    void nb(int64_t in_nb);
    void offset(int64_t i, int64_t j);

    int64_t mb_;
    int64_t nb_;
    int64_t stride_;
    Op op_;
    Uplo uplo_;

    scalar_t* data_;

    bool valid_;
    TileKind kind_;
    /// layout_: The physical ordering of elements in the data buffer:
    ///          - ColMajor: elements of a column are 1-strided
    ///          - RowMajor: elements of a row are 1-strided
    Layout layout_;

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
      kind_(TileKind::UserOwned),
      layout_(Layout::ColMajor),
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
/// @param[in] kind
///     The kind of tile:
///     - Workspace:  temporary tile, allocated by SLATE
///     - SlateOwned: origin tile, allocated by SLATE
///     - UserOwned:  origin tile, allocated by user
///
/// @param[in] layout
///     The physical ordering of elements in the data buffer:
///     - ColMajor: elements of a column are 1-strided
///     - RowMajor: elements of a row are 1-strided
///
template <typename scalar_t>
Tile<scalar_t>::Tile(
    int64_t mb, int64_t nb,
    scalar_t* A, int64_t lda, int device, TileKind kind, Layout layout)
    : mb_(mb),
      nb_(nb),
      stride_(lda),
      op_(Op::NoTrans),
      uplo_(Uplo::General),
      data_(A),
      valid_(true),
      kind_(kind),
      layout_(layout),
      device_(device)
{
    slate_assert(mb >= 0);
    slate_assert(nb >= 0);
    slate_assert(A != nullptr);
    slate_assert(lda >= mb);
}

//------------------------------------------------------------------------------
/// Returns element {i, j} of op(A).
/// The actual value is returned, not a reference. Use at() to get a reference.
/// If op() is ConjTrans, data IS conjugated, unlike with at().
/// This takes column-major / row-major layout into account.
///
/// @param[in] i
///     Row index. 0 <= i < mb.
///
/// @param[in] j
///     Column index. 0 <= j < nb.
///
template <typename scalar_t>
scalar_t Tile<scalar_t>::operator()(int64_t i, int64_t j) const
{
    using blas::conj;
    assert(0 <= i && i < mb());
    assert(0 <= j && j < nb());
    if (op_ == Op::ConjTrans) {
        if (layout_ == Layout::ColMajor)
            return conj(data_[ j + i*stride_ ]);
        else
            return conj(data_[ i + j*stride_ ]);
    }
    else if ((op_ == Op::NoTrans) == (layout_ == Layout::ColMajor)) {
        // (NoTrans && ColMajor) ||
        // (Trans   && RowMajor)
        return data_[ i + j*stride_ ];
    }
    else {
        // (NoTrans && RowMajor) ||
        // (Trans   && ColMajor)
        return data_[ j + i*stride_ ];
    }
}

//------------------------------------------------------------------------------
/// Returns a const reference to element {i, j} of op(A).
/// If op() is ConjTrans, data is NOT conjugated,
/// because a reference is returned.
/// Use operator() to get the actual value, conjugated if need be.
/// This takes column-major / row-major layout into account.
///
/// @param[in] i
///     Row index. 0 <= i < mb.
///
/// @param[in] j
///     Column index. 0 <= j < nb.
///
template <typename scalar_t>
scalar_t const& Tile<scalar_t>::at(int64_t i, int64_t j) const
{
    assert(0 <= i && i < mb());
    assert(0 <= j && j < nb());
    if ((op_ == Op::NoTrans) == (layout_ == Layout::ColMajor)) {
        // (NoTrans && ColMajor) ||
        // (Trans   && RowMajor)
        return data_[ i + j*stride_ ];
    }
    else {
        // (NoTrans && RowMajor) ||
        // (Trans   && ColMajor)
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
///
template <typename scalar_t>
scalar_t& Tile<scalar_t>::at(int64_t i, int64_t j)
{
    // forward to const at() version
    return const_cast<scalar_t&>(static_cast<const Tile>(*this).at(i, j));
}

//------------------------------------------------------------------------------
/// Returns whether op(A) is logically Upper, Lower, or General storage,
/// taking the transposition operation into account. Same as uplo().
/// @see uploPhysical()
///
template <typename scalar_t>
Uplo Tile<scalar_t>::uploLogical() const
{
    if (this->uplo_ == Uplo::General)
        return Uplo::General;
    else if ((this->uplo_ == Uplo::Lower) == (this->op_ == Op::NoTrans))
        return Uplo::Lower;
    else
        return Uplo::Upper;
}

//------------------------------------------------------------------------------
/// Returns whether A is Upper, Lower, or General storage,
/// ignoring the transposition operation.
/// @see uplo()
/// @see uploLogical()
///
template <typename scalar_t>
Uplo Tile<scalar_t>::uploPhysical() const
{
   return this->uplo_;
}

//------------------------------------------------------------------------------
/// Sets number of rows of op(A), where A is this tile.
///
/// @param[in] in_mb
///     New number of rows. 0 <= in_mb <= mb.
///
template <typename scalar_t>
void Tile<scalar_t>::mb(int64_t in_mb)
{
    assert(0 <= in_mb && in_mb <= mb());
    if (op_ == Op::NoTrans)
        mb_ = in_mb;
    else
        nb_ = in_mb;
}

//------------------------------------------------------------------------------
/// Sets number of cols of op(A), where A is this tile.
///
/// @param[in] in_nb
///     New number of cols. 0 <= in_nb <= nb.
///
template <typename scalar_t>
void Tile<scalar_t>::nb(int64_t in_nb)
{
    assert(0 <= in_nb && in_nb <= nb());
    if (op_ == Op::NoTrans)
        nb_ = in_nb;
    else
        mb_ = in_nb;
}

//------------------------------------------------------------------------------
/// Offsets data pointer to op(A)(i, j), where A is this tile.
///
/// @param[in] i
///     Row offset. 0 <= i < mb.
///
/// @param[in] j
///     Col offset. 0 <= j < nb.
///
template <typename scalar_t>
void Tile<scalar_t>::offset(int64_t i, int64_t j)
{
    assert(0 <= i && i < mb());
    assert(0 <= j && j < nb());
    if (op_ == Op::NoTrans)
        data_ = &data_[ i + j*stride_ ];
    else
        data_ = &data_[ j + i*stride_ ];
}

//------------------------------------------------------------------------------
/// Copies data from this tile on device to dst_tile on host.
///
/// @param[in] dst_tile
///     Destination tile, assumed to be on host.
///
/// @param[in] stream
///     CUDA stream for copy.
///
// todo need to copy or verify metadata (sizes, op, uplo, ...)
template <typename scalar_t>
void Tile<scalar_t>::copyDataToHost(
    Tile<scalar_t>* dst_tile, cudaStream_t stream) const
{
    slate_cuda_call(
        cudaSetDevice(device_));

    // If no stride on both sides.
    if (stride_ == mb_ &&
        dst_tile->stride_ == dst_tile->mb_) {

        // Use simple copy.
        trace::Block trace_block("cudaMemcpyAsync");

        slate_cuda_call(
            cudaMemcpyAsync(
                    dst_tile->data_, data_, bytes(),
                    cudaMemcpyDeviceToHost, stream));

        slate_cuda_call(
            cudaStreamSynchronize(stream));
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

        slate_cuda_call(
            cudaMemcpy2DAsync(
                    dst, dpitch,
                    src, spitch,
                    width, height,
                    cudaMemcpyDeviceToHost, stream));

        slate_cuda_call(
            cudaStreamSynchronize(stream));
    }
    dst_tile->layout(this->layout());
}

//------------------------------------------------------------------------------
/// Copies data from this tile on host to dst_tile on device.
///
/// @param[in] dst_tile
///     Destination tile, assumed to be on device.
///
/// @param[in] stream
///     CUDA stream for copy.
///
// todo: need to copy or verify metadata (sizes, op, uplo, ...)
template <typename scalar_t>
void Tile<scalar_t>::copyDataToDevice(
    Tile<scalar_t>* dst_tile, cudaStream_t stream) const
{
    slate_cuda_call(
        cudaSetDevice(dst_tile->device_));

    // If no stride on both sides.
    if (stride_ == mb_ &&
        dst_tile->stride_ == dst_tile->mb_) {

        // Use simple copy.
        trace::Block trace_block("cudaMemcpyAsync");

        slate_cuda_call(
            cudaMemcpyAsync(
                    dst_tile->data_, data_, bytes(),
                    cudaMemcpyHostToDevice, stream));

        slate_cuda_call(
            cudaStreamSynchronize(stream));
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

        slate_cuda_call(
            cudaMemcpy2DAsync(
                    dst, dpitch,
                    src, spitch,
                    width, height,
                    cudaMemcpyHostToDevice, stream));

        slate_cuda_call(
            cudaStreamSynchronize(stream));
    }
    dst_tile->layout(this->layout());
}

//------------------------------------------------------------------------------
/// Sends tile to MPI rank dst.
///
/// @param[in] dst
///     Destination MPI rank in mpi_comm.
///
/// @param[in] mpi_comm
///     MPI communicator.
///
// todo need to copy or verify metadata (sizes, op, uplo, ...)
template <typename scalar_t>
void Tile<scalar_t>::send(int dst, MPI_Comm mpi_comm, int tag) const
{
    trace::Block trace_block("MPI_Send");

    // If no stride.
    if (stride_ == mb_) {
        // Use simple send.
        int count = mb_*nb_;

        slate_mpi_call(
            MPI_Send(data_, count, mpi_type<scalar_t>::value, dst, tag,
                     mpi_comm));
    }
    else {
        // Otherwise, use strided send.
        int count = nb_;
        int blocklength = mb_;
        int stride = stride_;
        MPI_Datatype newtype;

        slate_mpi_call(
            MPI_Type_vector(count, blocklength, stride,
                            mpi_type<scalar_t>::value, &newtype));

        slate_mpi_call(MPI_Type_commit(&newtype));
        slate_mpi_call(MPI_Send(data_, 1, newtype, dst, tag, mpi_comm));
        slate_mpi_call(MPI_Type_free(&newtype));
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
///
// todo need to copy or verify metadata (sizes, op, uplo, ...)
template <typename scalar_t>
void Tile<scalar_t>::recv(int src, MPI_Comm mpi_comm, int tag)
{
    trace::Block trace_block("MPI_Recv");

    // If no stride.
    if (stride_ == mb_) {
        // Use simple recv.
        int count = mb_*nb_;

        slate_mpi_call(
            MPI_Recv(data_, count, mpi_type<scalar_t>::value, src, tag,
                     mpi_comm, MPI_STATUS_IGNORE));
    }
    else {
        // Otherwise, use strided recv.
        int count = nb_;
        int blocklength = mb_;
        int stride = stride_;
        MPI_Datatype newtype;

        slate_mpi_call(
            MPI_Type_vector(
                count, blocklength, stride, mpi_type<scalar_t>::value,
                &newtype));

        slate_mpi_call(MPI_Type_commit(&newtype));

        slate_mpi_call(
            MPI_Recv(data_, 1, newtype, src, tag, mpi_comm,
                     MPI_STATUS_IGNORE));

        slate_mpi_call(MPI_Type_free(&newtype));
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
///
// todo: OpenMPI MPI_Bcast seems to have a bug such that either all ranks must
// use the simple case, or all ranks use vector case, even though the type
// signatures match.
// Bug confirmed with OpenMPI folks. All nodes need to make same decision
// about using a pipelined bcast algorithm (in the contiguous case), and can't.
//
// todo need to copy or verify metadata (sizes, op, uplo, ...)
template <typename scalar_t>
void Tile<scalar_t>::bcast(int bcast_root, MPI_Comm mpi_comm)
{
    // If no stride.
    //if (stride_ == mb_) {
    //    // Use simple bcast.
    //    int count = mb_*nb_;
    //
    //    #pragma omp critical(slate_mpi)
    //    slate_mpi_call(
    //        MPI_Bcast(data_, count, mpi_type<scalar_t>::value,
    //                  bcast_root, mpi_comm));
    //}
    //else
    {
        // Otherwise, use strided bcast.
        trace::Block trace_block("MPI_Bcast");

        int count = nb_;
        int blocklength = mb_;
        int stride = stride_;
        MPI_Datatype newtype;

        #pragma omp critical(slate_mpi)
        {
            slate_mpi_call(
                MPI_Type_vector(
                    count, blocklength, stride, mpi_type<scalar_t>::value,
                    &newtype));
        }

        #pragma omp critical(slate_mpi)
        {
            slate_mpi_call(
                MPI_Type_commit(&newtype));
        }

        #pragma omp critical(slate_mpi)
        {
            slate_mpi_call(
                MPI_Bcast(data_, 1, newtype, bcast_root, mpi_comm));
        }

        #pragma omp critical(slate_mpi)
        {
            slate_mpi_call(
                MPI_Type_free(&newtype));
        }
    }
}

//------------------------------------------------------------------------------
/// Set tile data
///
/// @param[in] alpha
///     value set on diagonals.
///
/// @param[in] beta
///     value set on off-diagonals.
///
template <typename scalar_t>
void Tile<scalar_t>::set(scalar_t alpha, scalar_t beta)
{
    lapack::MatrixType mt = (lapack::MatrixType)uplo_;// TODO is this safe?
    lapack::laset(mt, mb(), nb(),
                  alpha, beta,
                  data(), stride());
}

//------------------------------------------------------------------------------
/// Set tile data
///
/// @param[in] alpha
///     value for both diagonals and off-diagonals
///
template <typename scalar_t>
void Tile<scalar_t>::set(scalar_t alpha)
{
    set(alpha, alpha);
}


} // namespace slate

#endif // SLATE_TILE_HH
