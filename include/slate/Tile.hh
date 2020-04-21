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
#include "slate/internal/device.hh"
#include "slate/types.hh"
#include "slate/Exception.hh"
#include "slate/Tile_aux.hh"

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

//------------------------------------------------------------------------------
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

//--------------------------------------
/// Converts rvalue refs to lvalue refs.
template<typename MatrixType>
MatrixType transpose(MatrixType&& A)
{
    return transpose(A);
}

/// @see transpose()
template<typename MatrixType>
MatrixType conjTranspose(MatrixType& A)
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

//--------------------------------------
/// Converts rvalue refs to lvalue refs.
template<typename MatrixType>
MatrixType conjTranspose(MatrixType&& A)
{
    return conjTranspose(A);
}

//--------------------------------------
/// @deprecated
template<typename MatrixType>
MatrixType conj_transpose(MatrixType& A)
{
    return conjTranspose(A);
}

//--------------------------------------
/// @deprecated
template<typename MatrixType>
MatrixType conj_transpose(MatrixType&& A)
{
    return conjTranspose(A);
}

//------------------------------------------------------------------------------
/// Whether a tile is workspace or origin (local non-workspace),
/// and who owns (allocated, deallocates) the data.
/// @ingroup enum
///
enum class TileKind {
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
    void copyData(Tile<scalar_t>* dst_tile, cudaStream_t stream,
                  bool async = false) const;

    /// copies this tile's data to dst_tile data, both assumed on host
    void copyData(Tile<scalar_t>* dst_tile) const
    {
        assert(this->device_ < 0);
        assert(dst_tile->device() < 0);
        // forward
        copyData(dst_tile, nullptr, false);
    }

    void send(int dst, MPI_Comm mpi_comm, int tag = 0) const;
    void isend(int dst, MPI_Comm mpi_comm, int tag, MPI_Request *req); // const;
    void recv(int src, MPI_Comm mpi_comm, Layout layout, int tag = 0);
    void bcast(int bcast_root, MPI_Comm mpi_comm);

    /// Returns shallow copy of tile that is transposed.
    template <typename TileType>
    friend TileType transpose(TileType& A);

    /// Returns shallow copy of tile that is conjugate-transposed.
    template <typename TileType>
    friend TileType conjTranspose(TileType& A);

    /// @deprecated
    template <typename TileType>
    friend TileType conj_transpose(TileType& A);

    /// Returns number of rows of op(A), where A is this tile
    int64_t mb() const { return (op_ == Op::NoTrans ? mb_ : nb_); }

    /// Returns number of cols of op(A), where A is this tile
    int64_t nb() const { return (op_ == Op::NoTrans ? nb_ : mb_); }

    /// Returns column stride of this tile
    int64_t stride() const { return stride_; }

    /// Sets column stride of this tile
    void stride(int64_t in_stride) { stride_ = in_stride; }

    /// Returns const pointer to data, i.e., A(0,0), where A is this tile
    scalar_t const* data() const { return data_; }

    /// Returns pointer to data, i.e., A(0,0), where A is this tile
    scalar_t*       data()       { return data_; }

    scalar_t operator()(int64_t i, int64_t j) const;
    scalar_t const& at(int64_t i, int64_t j) const;
    scalar_t&       at(int64_t i, int64_t j);

    // Return the number of elements to increment to get to the next
    // element in the row (col), accouting for row-major/col-major and
    // transposed tiles.
    int64_t rowIncrement() const {
        if ( (op_ == Op::NoTrans) == (layout_ == Layout::ColMajor) )
            // (NoTrans && ColMajor) || (Trans   && RowMajor)
            return stride_;
        else return 1;
    }
    int64_t colIncrement() const {
        if ( (op_ == Op::NoTrans) == (layout_ == Layout::ColMajor) )
            // (NoTrans && ColMajor) || (Trans   && RowMajor)
            return 1;
        else return stride_;
    }

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
    // todo: set front buffer inside layout(...)
    void   layout(Layout in_layout) { layout_ = in_layout; }
    Layout userLayout() const { return user_layout_; }

    void set(scalar_t alpha);
    void set(scalar_t alpha, scalar_t beta);

    /// @return Whether the front memory buffer is contiguous
    bool isContiguous() const
    {
        return (layout_ == Layout::ColMajor && stride_ == mb_)
            || (layout_ == Layout::RowMajor && stride_ == nb_);
    }

    /// Returns whether this tile can safely store its data in transposed form
    /// based on its 'TileKind', buffer size, Layout, and stride.
    /// todo: validate and handle sliced-matrix
    bool isTransposable()
    {
        return    extended()                    // already extended buffer
               || mb_ == nb_                    // square tile
               || kind_ != TileKind::UserOwned  // SLATE allocated
               || isContiguous();               // contiguous
    }

    void makeTransposable(scalar_t* data);
    void layoutReset();

    /// @return Whether this tile has extended buffer
    bool extended() const { return ext_data_ != nullptr; }

    /// @return Pointer to the extended buffer
    scalar_t* extData() { return ext_data_; }

    /// @return Pointer to the user allocated buffer
    scalar_t* userData() { return user_data_; }

    void layoutSetFrontDataExt(bool front = true);

    /// @return Pointer to the back buffer
    scalar_t* layoutBackData()
    {
        if (data_ == user_data_)
            return ext_data_;
        else
            return user_data_;
    }
    /// @return Stride of the back buffer
    int64_t layoutBackStride() const
    {
        if (data_ == user_data_)
            return stride_; // todo: validate
        else
            return user_stride_;
    }

    void layoutConvert( scalar_t* work_data,
                        cudaStream_t stream = nullptr,
                        bool async = false);

    /// Convert layout of this tile (assuming no workspace is needed)
    /// CUDA stream must be provided if conversion is to happen on device
    void layoutConvert( cudaStream_t stream = nullptr,
                        bool async = false)
    {
        assert(mb() == nb() || extended());
        layoutConvert(nullptr, stream, async);
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
    int64_t user_stride_; // Temporarily store user-provided-memory's stride

    Op op_;
    Uplo uplo_;

    scalar_t* data_;
    scalar_t* user_data_; // Temporarily point to user-provided memory buffer.
    scalar_t* ext_data_; // Points to auxiliary buffer.

    TileKind kind_;
    /// layout_: The physical ordering of elements in the data buffer:
    ///          - ColMajor: elements of a column are 1-strided
    ///          - RowMajor: elements of a row are 1-strided
    Layout layout_;
    Layout user_layout_; // Temporarily store user-provided-memory's layout

    int device_;
};

//------------------------------------------------------------------------------
/// Create empty tile.
template <typename scalar_t>
Tile<scalar_t>::Tile()
    : mb_(0),
      nb_(0),
      stride_(0),
      user_stride_(0),
      op_(Op::NoTrans),
      uplo_(Uplo::General),
      data_(nullptr),
      user_data_(nullptr),
      ext_data_(nullptr),
      kind_(TileKind::UserOwned),
      layout_(Layout::ColMajor),
      user_layout_(Layout::ColMajor),
      device_(HostNum)
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
///     The mb-by-nb tile A, stored in an
///     lda-by-nb array if ColMajor, or
///     lda-by-mb array if RowMajor.
///
/// @param[in] lda
///     Leading dimension of the array A.
///     lda >= mb if ColMajor.
///     lda >= nb if RowMajor.
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
      user_stride_(lda),
      op_(Op::NoTrans),
      uplo_(Uplo::General),
      data_(A),
      user_data_(nullptr),
      ext_data_(nullptr),
      kind_(kind),
      layout_(layout),
      user_layout_(layout),
      device_(device)
{
    slate_assert(mb >= 0);
    slate_assert(nb >= 0);
    slate_assert(A != nullptr);
    slate_assert( (layout == Layout::ColMajor && lda >= mb)
               || (layout == Layout::RowMajor && lda >= nb));
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
    slate_assert(0 <= i && i < mb());
    slate_assert(0 <= j && j < nb());
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
    slate_assert(0 <= i && i < mb());
    slate_assert(0 <= j && j < nb());
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
    slate_assert(0 <= in_mb && in_mb <= mb());
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
    slate_assert(0 <= in_nb && in_nb <= nb());
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
    slate_assert(0 <= i && i < mb());
    slate_assert(0 <= j && j < nb());
    if (op_ == Op::NoTrans)
        data_ = &data_[ i + j*stride_ ];
    else
        data_ = &data_[ j + i*stride_ ];
}

//------------------------------------------------------------------------------
/// Attaches the new_data buffer to this tile as an extended buffer
/// extended buffer to be used to hold the transposed data of rectangular tiles
/// Marks the tile as extended
/// NOTE: does not set the front buffer to be the extended one
/// NOTE: throws error if not already transposable.
///
template <typename scalar_t>
void Tile<scalar_t>::makeTransposable(scalar_t* new_data)
{
    slate_assert(! isTransposable());

    // preserve currrent data pointer and stride
    user_data_ = data_;
    user_stride_ = stride_;
    user_layout_ = layout_;
    ext_data_ = new_data;
}

//------------------------------------------------------------------------------
/// Sets the front buffer of the extended tile,
/// and adjusts stride accordingly.
/// NOTE: tile should be already extended, throws error otherwise.
///
template <typename scalar_t>
void Tile<scalar_t>::layoutSetFrontDataExt(bool front)
{
    slate_assert(extended());

    if (front) {
        data_ = ext_data_;
        stride_ = user_layout_ == Layout::RowMajor ?
                  mb_ : nb_;
    }
    else {
        data_ = user_data_;
        stride_ = user_stride_;
        layout_ = user_layout_;
    }
}

//------------------------------------------------------------------------------
/// Resets the tile's member fields related to being extended.
/// WARNING: should be called within MatrixStorage::tileLayoutReset() only.
/// NOTE: the front buffer should be already swapped to be the user buffer,
///       throws error otherwise.
///
template <typename scalar_t>
void Tile<scalar_t>::layoutReset()
{
    slate_assert(data_ == user_data_);
    user_data_ = nullptr;
    ext_data_ = nullptr;
}

//------------------------------------------------------------------------------
/// Convert layout (Column / Row major) of this tile.
/// Performs:
///     - In-place conversion for square tiles
///     - In-place conversion for contiguous rectangular tiles, using a workspace.
///     - Out-of-place conversion if extended tile, swaps front buffer accordingly.
///
/// Tile must be transposable already, should call makeTransposable() if not.
/// A CUDA stream should be provided if tile instance is on a GPU device.
///
/// @param[in] work_data
///     Pointer to a workspace buffer, needed for out-of-place transpose.
///
/// @param[in] stream
///     CUDA stream to run the kernels on if conversion is carried on the device.
///
/// @ingroup convert_tile
///
template <typename scalar_t>
void Tile<scalar_t>::layoutConvert(scalar_t* work_data, cudaStream_t stream,
    bool async)
{
    slate_assert(device_ == HostNum || stream != nullptr);
    slate_assert(isTransposable());

    trace::Block trace_block("slate::convertLayout");
    // square tile
    if (mb() == nb()) {
        // in-place convert
        if (device_ == HostNum)
            transpose(nb(), data_, stride_);
        else {
            slate_cuda_call(
                cudaSetDevice(device_));

            device::transpose(mb(), data(), stride(), stream);

            if (! async)
                slate_cuda_call(
                    cudaStreamSynchronize(stream));
        }
    }
    // rectangular tile
    else {
        // if tile made Convertible
        if (extended()) {
            // out-of-place convert
            scalar_t* src_data;
            int64_t src_stride = 0;

            if (user_data_ == data_) {
                // need to convert into ext_data_
                data_ = ext_data_;
                src_data = user_data_;
                src_stride = user_stride_;
                stride_ = user_layout_ == Layout::RowMajor ?
                          mb_ : nb_;
            }
            else if (ext_data_ == data_) {
                // need to convert into user_data_
                data_ = user_data_;
                src_data = ext_data_;
                src_stride = stride_;
                stride_ = user_stride_;
            }
            else {
                assert(0);
            }
            if (device_ == HostNum)
                transpose(layout() == Layout::ColMajor ? mb_ : nb_,
                          layout() == Layout::ColMajor ? nb_ : mb_,
                          src_data, src_stride,
                          data_, stride_);
            else {
                slate_cuda_call(
                    cudaSetDevice(device_));

                device::transpose(layout() == Layout::ColMajor ? mb_ : nb_,
                                  layout() == Layout::ColMajor ? nb_ : mb_,
                                  src_data, src_stride,
                                  data_, stride_,
                                  stream);
                if (! async)
                    slate_cuda_call(
                        cudaStreamSynchronize(stream));
            }
        }
        else {
            // tile already Convertible
            slate_assert(isContiguous());
            // need a workspace buffer
            slate_assert(work_data != nullptr);

            // out-of-place convert
            int64_t work_stride = layout() == Layout::ColMajor ?
                                  nb() : mb();
            if (device_ == HostNum) {
                transpose(layout() == Layout::ColMajor ? mb_ : nb_,
                          layout() == Layout::ColMajor ? nb_ : mb_,
                          data_, stride_,
                          work_data, work_stride);
                std::memcpy(data_, work_data, bytes());
            }
            else {
                slate_cuda_call(
                    cudaSetDevice(device_));

                device::transpose(layout() == Layout::ColMajor ? mb_ : nb_,
                                  layout() == Layout::ColMajor ? nb_ : mb_,
                                  data_, stride_,
                                  work_data, work_stride,
                                  stream);
                slate_cuda_call(
                    cudaMemcpyAsync(
                            data_, work_data, bytes(),
                            cudaMemcpyDeviceToDevice, stream));

                if (! async)
                    slate_cuda_call(
                        cudaStreamSynchronize(stream));
            }
            stride_ = work_stride;
        }
    }
    layout(layout() == Layout::RowMajor ? Layout::ColMajor : Layout::RowMajor);
}


//------------------------------------------------------------------------------
/// @deprecated
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
    // sizes has to match
    slate_assert(mb_ == dst_tile->mb_);
    slate_assert(nb_ == dst_tile->nb_);

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
/// @deprecated
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
    // sizes has to match
    slate_assert(mb_ == dst_tile->mb_);
    slate_assert(nb_ == dst_tile->nb_);

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
/// Copies data from this tile to dst_tile.
/// Figures out the direction of copy and the source and destination devices
/// from the destination tile and this tile.
/// WARNING: device ID set in device_ of both tiles should be properly set.
///
/// @param[in] dst_tile
///     Destination tile.
///
/// @param[in] stream
///     CUDA stream for copy if needed.
///
// todo: need to copy or verify metadata (sizes, op, uplo, ...)
template <typename scalar_t>
void Tile<scalar_t>::copyData(
    Tile<scalar_t>* dst_tile, cudaStream_t stream,
    bool async) const
{
    // sizes has to match
    slate_assert(mb_ == dst_tile->mb_);
    slate_assert(nb_ == dst_tile->nb_);

    int device;
    cudaMemcpyKind memcpy_kind;

    // figure out copy direction and device
    if (this->device_ >= 0 && dst_tile->device() == HostNum) {
        // device to host copy
        device = this->device_;
        memcpy_kind = cudaMemcpyDeviceToHost;
    }
    else if (this->device_ == HostNum && dst_tile->device() >= 0) {
        // host to device copy
        device = dst_tile->device();
        memcpy_kind = cudaMemcpyHostToDevice;
    }
    else if (this->device_ == HostNum && dst_tile->device() == HostNum) {
        // host to host copy
        device = -1;
        memcpy_kind = cudaMemcpyHostToHost;
    }
    else if (this->device_ >= 0 && dst_tile->device() >= 0) {
        // device to device copy
        device = this->device_;
        memcpy_kind = cudaMemcpyDeviceToDevice;
        // todo: handle peer to peer copy
        if (this->device_ != dst_tile->device())
            assert(0);
    }
    else {
        device = HostNum; // silence a compiler warning
        memcpy_kind = cudaMemcpyHostToHost; // silence a compiler warning
        slate_error("illegal combination of source and destination devices");
    }

    if (device >= 0) {
        slate_cuda_call(
            cudaSetDevice(device));
    }

    // adjust stride if not UserOwned
    if (dst_tile->kind_ != TileKind::UserOwned) {
        dst_tile->stride_ = this->layout() == Layout::ColMajor ? mb_ : nb_;
    }

    if (memcpy_kind == cudaMemcpyHostToHost) {
        gecopy(*this, *dst_tile);
    }
    else {
        assert(stream != nullptr);

        // If no stride on both sides.
        if (this->isContiguous() &&
            dst_tile->isContiguous()) {

            // Use simple copy.
            trace::Block trace_block("cudaMemcpyAsync");

            slate_cuda_call(
                cudaMemcpyAsync(
                        dst_tile->data_, data_, bytes(),
                        memcpy_kind, stream));

            if (! async)
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
            size_t width  = sizeof(scalar_t)*(this->layout() == Layout::ColMajor ? mb_ : nb_);
            size_t height = (this->layout() == Layout::ColMajor ? nb_ : mb_);

            slate_cuda_call(
                cudaMemcpy2DAsync(
                        dst, dpitch,
                        src, spitch,
                        width, height,
                        memcpy_kind, stream));

            if (! async)
                slate_cuda_call(
                    cudaStreamSynchronize(stream));
        }
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
    if (this->isContiguous()) {
        // Use simple send.
        int count = mb_*nb_;

        slate_mpi_call(
            MPI_Send(data_, count, mpi_type<scalar_t>::value, dst, tag,
                     mpi_comm));
    }
    else {
        // Otherwise, use strided send.
        int count = layout_ == Layout::ColMajor ? nb_ : mb_;
        int blocklength = layout_ == Layout::ColMajor ? mb_ : nb_;
        int stride = stride_;
        MPI_Datatype newtype;

        slate_mpi_call(
            MPI_Type_vector(count, blocklength, stride,
                            mpi_type<scalar_t>::value, &newtype));

        slate_mpi_call(MPI_Type_commit(&newtype));
        slate_mpi_call(MPI_Send(data_, 1, newtype, dst, tag, mpi_comm));
        slate_mpi_call(MPI_Type_free(&newtype));
    }
    // todo: would specializing to Triangular / Band tiles improve performance
    // by receiving less / compacted data
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
void Tile<scalar_t>::isend(int dst, MPI_Comm mpi_comm, int tag, MPI_Request *req) // const
{
    trace::Block trace_block("MPI_Isend");

    // If no stride.
    if (this->isContiguous()) {
        // Use simple send.
        int count = mb_*nb_;

        slate_mpi_call(
            MPI_Isend(data_, count, mpi_type<scalar_t>::value, dst, tag,
                      mpi_comm, req));
    }
    else {
        // Otherwise, use strided send.
        int count = layout_ == Layout::ColMajor ? nb_ : mb_;
        int blocklength = layout_ == Layout::ColMajor ? mb_ : nb_;
        int stride = stride_;
        MPI_Datatype newtype;

        slate_mpi_call(
            MPI_Type_vector(count, blocklength, stride,
                            mpi_type<scalar_t>::value, &newtype));

        slate_mpi_call(MPI_Type_commit(&newtype));
        slate_mpi_call(MPI_Isend(data_, 1, newtype, dst, tag, mpi_comm, req));
        slate_mpi_call(MPI_Type_free(&newtype));
    }
    // todo: would specializing to Triangular / Band tiles improve performance
    // by receiving less / compacted data
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
/// @param[in] layout
///     Indicates the Layout (ColMajor/RowMajor) of the received data.
///     WARNING: need to call tileLayout(...) to properly set the layout of the
///              origin matrix tile afterwards.
///
// todo need to copy or verify metadata (sizes, op, uplo, ...)
template <typename scalar_t>
void Tile<scalar_t>::recv(int src, MPI_Comm mpi_comm, Layout layout, int tag)
{
    trace::Block trace_block("MPI_Recv");

    // If no stride.
    if (this->isContiguous()) {
        // Use simple recv.
        int count = mb_*nb_;

        slate_mpi_call(
            MPI_Recv(data_, count, mpi_type<scalar_t>::value, src, tag,
                     mpi_comm, MPI_STATUS_IGNORE));
    }
    else {
        // Otherwise, use strided recv.
        int count = layout_ == Layout::ColMajor ? nb_ : mb_;
        int blocklength = layout_ == Layout::ColMajor ? mb_ : nb_;
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
    // set this tile layout to match the received data layout
    this->layout(layout);
    // todo: would specializing to Triangular / Band tiles improve performance
    // by receiving less / compacted data
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
        // todo: layout
        int count = layout_ == Layout::ColMajor ? nb_ : mb_;
        int blocklength = layout_ == Layout::ColMajor ? mb_ : nb_;
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
/// Set tile data to constants.
///
/// @param[in] offdiag_value
///     Value set on off-diagonal elements.
///
/// @param[in] diag_value
///     Value set on diagonal elements.
///
template <typename scalar_t>
void Tile<scalar_t>::set(scalar_t offdiag_value, scalar_t diag_value)
{
    lapack::MatrixType mtype = (lapack::MatrixType)uplo_;// TODO is this safe?
    lapack::laset(mtype, mb(), nb(),
                  offdiag_value, diag_value,
                  data(), stride());
}

//------------------------------------------------------------------------------
/// Set tile data to constant.
///
/// @param[in] value
///     Value set on both diagonal and off-diagonal elements.
///
template <typename scalar_t>
void Tile<scalar_t>::set(scalar_t value)
{
    set(value, value);
}

} // namespace slate

#endif // SLATE_TILE_HH
