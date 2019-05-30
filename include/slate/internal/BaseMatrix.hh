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

#ifndef SLATE_BASE_MATRIX_HH
#define SLATE_BASE_MATRIX_HH

#include "slate/internal/comm.hh"
#include "slate/internal/Map.hh"
#include "slate/internal/Memory.hh"
#include "slate/internal/MatrixStorage.hh"
#include "slate/Tile.hh"
#include "slate/Tile_blas.hh"
// #include "slate/Tile_aux.hh"
#include "slate/types.hh"

#include "lapack.hh"

#include <algorithm>
#include <memory>
#include <set>
#include <list>
#include <tuple>
#include <utility>
#include <vector>

#include "slate/internal/cuda.hh"
#include "slate/internal/cublas.hh"
#include "slate/internal/mpi.hh"
#include "slate/internal/openmp.hh"

namespace slate {

//==============================================================================
/// Base class for all SLATE distributed, tiled matrices.
/// In general, the documentation refers to the current matrix object as op(A),
/// to emphasize that it may be transposed with respect to its parent matrix.
///
template <typename scalar_t>
class BaseMatrix {
public:
    using BcastList =
        std::list<std::tuple<int64_t, int64_t,
                             std::list<BaseMatrix<scalar_t> > > >;

    using ReduceList =
        std::list<std::tuple<int64_t, int64_t,
                             std::list<BaseMatrix<scalar_t> > > >;

    using ij_tuple = std::tuple<int64_t, int64_t>;

    friend class Debug;

    static constexpr bool is_complex = slate::is_complex<scalar_t>::value;
    static constexpr bool is_real    = ! is_complex;

    typedef scalar_t value_type;

protected:
    BaseMatrix();

    // Defaults okay for:
    // 1. destructor
    // 2. copy constructor
    // 3. move constructor
    // 4. copy assignment
    // 5. move assignment

    BaseMatrix(int64_t m, int64_t n, int64_t nb, int p, int q,
               MPI_Comm mpi_comm);

    BaseMatrix(BaseMatrix& orig,
               int64_t i1, int64_t i2,
               int64_t j1, int64_t j2);

    // internal utility class to differentiate slicing (row/col indices)
    // from sub-matrix (tile indices)
    class Slice {
    public:
        Slice( int64_t in_row1, int64_t in_row2,
               int64_t in_col1, int64_t in_col2 )
        : row1( in_row1 ),
          row2( in_row2 ),
          col1( in_col1 ),
          col2( in_col2 )
        {}
        int64_t row1, row2, col1, col2;
    };

    BaseMatrix(BaseMatrix& orig, Slice slice);

private:
    void initSubmatrix(
        int64_t i1, int64_t i2,
        int64_t j1, int64_t j2);

    void initSlice(int64_t row0_offset, int64_t col0_offset,
                   int64_t last_mb, int64_t last_nb);

public:
    /// Returns shallow copy of op(A) that is transposed.
    /// @see conj_transpose
    template<typename MatrixType>
    friend MatrixType transpose(MatrixType& A);

    /// Returns shallow copy of op(A) that is conjugate-transpose.
    /// @see transpose
    template<typename MatrixType>
    friend MatrixType conj_transpose(MatrixType& A);

    template <typename T>
    friend void swap(BaseMatrix<T>& A, BaseMatrix<T>& B);

    Tile<scalar_t> operator()(int64_t i, int64_t j, int device=host_num_);

    /// Alias of operator().
    Tile<scalar_t> at(int64_t i, int64_t j, int device=host_num_)
    {
        return (*this)(i, j, device);
    }

    /// Returns number of devices (per MPI process) to distribute matrix to.
    int num_devices() const { return num_devices_; }

    int64_t m() const;
    int64_t n() const;

    /// Returns number of block rows in op(A).
    int64_t mt() const { return (op_ == Op::NoTrans ? mt_ : nt_); }

    /// Returns number of block cols in op(A).
    int64_t nt() const { return (op_ == Op::NoTrans ? nt_ : mt_); }

    /// Returns transposition operation op(A) as NoTrans, Trans, or ConjTrans.
    Op op() const { return op_; }

    /// returns true if tile exists on specified device
    bool tileExists(int64_t i, int64_t j, int device=host_num_)
    {
        return storage_->find(globalIndex(i, j, device)) != storage_->end();
    }

    /// Returns MPI rank of tile {i, j} of op(A).
    int tileRank(int64_t i, int64_t j) const
    {
        return storage_->tileRank(globalIndex(i, j));
    }

    /// Returns device of tile {i, j} of op(A).
    int tileDevice(int64_t i, int64_t j) const
    {
        return storage_->tileDevice(globalIndex(i, j));
    }

    /// Returns whether tile {i, j} of op(A) is local.
    bool tileIsLocal(int64_t i, int64_t j) const
    {
        return storage_->tileIsLocal(globalIndex(i, j));
    }

    /// Returns whether op(A) is logically Lower, Upper, or General.
    Uplo uplo() const { return uploLogical(); }
    Uplo uploLogical() const;
    Uplo uploPhysical() const;

    /// Returns the origin tile instance of tile(i, j)
    Tile<scalar_t>* originTile(int64_t i, int64_t j);

    int64_t tileMb(int64_t i) const;
    int64_t tileNb(int64_t j) const;
private:
    int64_t tileMbInternal(int64_t i) const;
    int64_t tileNbInternal(int64_t j) const;

public:
    Tile<scalar_t>* tileInsert(int64_t i, int64_t j, int device=host_num_);
    Tile<scalar_t>* tileInsert(int64_t i, int64_t j, int device,
                               scalar_t* A, int64_t ld);

    /// Insert tile with default device=host_num. @see tileInsert.
    Tile<scalar_t>* tileInsert(int64_t i, int64_t j,
                               scalar_t* A, int64_t ld)
    {
        return tileInsert(i, j, host_num_, A, ld);
    }

    TileEntry<scalar_t>& tileInsertWorkspace(int64_t i, int64_t j, int device, Layout layout);
    TileEntry<scalar_t>& tileInsertWorkspace(int64_t i, int64_t j, int device)
    {
        return tileInsertWorkspace(i, j, device, layout_);
    }
    TileEntry<scalar_t>& tileInsertWorkspace(int64_t i, int64_t j, Layout layout)
    {
        return tileInsertWorkspace(i, j, host_num_, layout);
    }
    TileEntry<scalar_t>& tileInsertWorkspace(int64_t i, int64_t j)
    {
        return tileInsertWorkspace(i, j, host_num_, layout_);
    }

    //--------------------------------------------------------------------------
    // MOSI
private:
    void tileGet(int64_t i, int64_t j, int dst_device,
                 LayoutConvert layout, bool modify, bool hold,
                 bool async);

    void tileGet(std::set<ij_tuple>& tile_set, int dst_device,
                 LayoutConvert layout, bool modify, bool hold,
                 bool async);

public:

    MOSI tileState(int64_t i, int64_t j, int device=host_num_);

    void tileState(int64_t i, int64_t j, int device, MOSI mosi);

    /// Sets tile(i, j)'s state on host.
    void tileState(int64_t i, int64_t j, MOSI mosi)
    {
        tileState(i, j, host_num_, mosi);
    }

    bool tileOnHold(int64_t i, int64_t j, int device=host_num_);

    void tileUnsetHold(int64_t i, int64_t j, int device=host_num_);

    void tileUnsetHoldAll(int device=host_num_);

    void tileUnsetHoldAllOnDevices();

    void tileModified(int64_t i, int64_t j, int device=host_num_, bool permissive=false);

    void tileGetForReading(int64_t i, int64_t j, int device, LayoutConvert layout);

    void tileGetForReading(std::set<ij_tuple>& tile_set, int device, LayoutConvert layout);

    /// Gets tile(i, j) for reading on host.
    /// @see tileGetForReading
    void tileGetForReading(int64_t i, int64_t j, LayoutConvert layout)
    {
        tileGetForReading(i, j, host_num_, layout);
    }

    /// Gets a set of tiles for reading on host.
    /// @see tileGetForReading
    void tileGetForReading(std::set<ij_tuple>& tile_set, LayoutConvert layout)
    {
        tileGetForReading(tile_set, host_num_, layout);
    }

    void tileGetAllForReading(int device, LayoutConvert layout);

    void tileGetAllForReadingOnDevices(LayoutConvert layout);

    void tileGetForWriting(int64_t i, int64_t j, int device, LayoutConvert layout);

    void tileGetForWriting(std::set<ij_tuple>& tile_set, int device, LayoutConvert layout);

    /// Gets tile(i, j) for writing on host.
    /// @see tileGetForWriting
    void tileGetForWriting(int64_t i, int64_t j, LayoutConvert layout)
    {
        tileGetForWriting(i, j, host_num_, layout);
    }

    /// Gets a set of tiles for writing on host.
    /// @see tileGetForWriting
    void tileGetForWriting(std::set<ij_tuple>& tile_set, LayoutConvert layout)
    {
        tileGetForWriting(tile_set, host_num_, layout);
    }

    void tileGetAllForWriting(int device, LayoutConvert layout);

    void tileGetAllForWritingOnDevices(LayoutConvert layout);

    void tileGetAndHold(int64_t i, int64_t j, int device, LayoutConvert layout);

    /// Gets tile(i, j) on host for reading and marks it as MOSI::OnHold.
    /// @see tileGetAndHold
    void tileGetAndHold(int64_t i, int64_t j, LayoutConvert layout)
    {
        tileGetAndHold(i, j, host_num_, layout);
    }

    void tileGetAndHold(std::set<ij_tuple>& tile_set, int device, LayoutConvert layout);

    /// Gets a set of tiles for reading on host and marks them as MOSI::OnHold.
    /// @see tileGetAndHold
    void tileGetAndHold(std::set<ij_tuple>& tile_set, LayoutConvert layout)
    {
        tileGetAndHold(tile_set, host_num_, layout);
    }

    void tileGetAndHoldAll(int device, LayoutConvert layout);

    void tileGetAndHoldAllOnDevices(LayoutConvert layout);

    /// Updates the origin instance of tile(i, j) if MOSI::Invalid.
    /// @return Pointer to origin instance of tile(i, j).
    Tile<scalar_t>* tileUpdateOrigin(int64_t i, int64_t j);

    /// Updates all origin instances of tiles if MOSI::Invalid.
    void tileUpdateAllOrigin();

    /// Returns life counter of tile {i, j} of op(A).
    int64_t tileLife(int64_t i, int64_t j) const
    {
        return storage_->tileLife(globalIndex(i, j));
    }

    /// Set life counter of tile {i, j} of op(A).
    void tileLife(int64_t i, int64_t j, int64_t life)
    {
        storage_->tileLife(globalIndex(i, j), life);
    }

    /// Decrements life counter of workspace tile {i, j} of op(A).
    /// Then, if life reaches 0, deletes tile on all devices.
    /// For local, non-workspace tiles, does nothing.
    void tileTick(int64_t i, int64_t j)
    {
        storage_->tileTick(globalIndex(i, j));
    }

    void tileErase(int64_t i, int64_t j, int device=host_num_);

    /// Deletes the tile(i, j)'s instance on device if it is a workspace tile
    /// that is not modified and no hold is set on it.
    void tileRelease(int64_t i, int64_t j, int device=host_num_);

    //--------------------------------------------------------------------------
    template <Target target = Target::Host>
    void tileSend(int64_t i, int64_t j, int dst_rank, int tag = 0);

    template <Target target = Target::Host>
    void tileRecv(int64_t i, int64_t j, int dst_rank,
                  Layout layout, int tag = 0);

    template <Target target = Target::Host>
    void tileBcast(int64_t i, int64_t j, BaseMatrix const& B,
                   Layout layout, int tag = 0, int64_t life_factor = 1);

    template <Target target = Target::Host>
    void listBcast(BcastList& bcast_list,
                   Layout layout, int tag = 0, int64_t life_factor = 1);

    template <Target target = Target::Host>
    void listReduce(ReduceList& reduce_list, Layout layout, int tag = 0);

    //--------------------------------------------------------------------------
    // LAYOUT
public:
    /// Returns matrix layout flag
    Layout layout() const { return layout_; }

    /// Returns Layout of tile(i, j, device)
    Layout tileLayout(int64_t i, int64_t j, int device=host_num_)
    {
        return storage_->at(globalIndex(i, j, device)).tile_->layout();
    }

    /// Sets Layout of tile(i, j, device)
    void tileLayout(int64_t i, int64_t j, int device, Layout layout)
    {
        storage_->at(globalIndex(i, j, device)).tile_->layout(layout);
    }

    /// Sets Layout of tile(i, j, host)
    void tileLayout(int64_t i, int64_t j, Layout layout)
    {
        storage_->at(globalIndex(i, j, host_num_)).tile_->layout(layout);
    }

    bool tileLayoutIsConvertible(int64_t i, int64_t j, int device=host_num_);

    void tileLayoutConvert(int64_t i, int64_t j, int device, Layout layout, bool reset = false);
    /// Convert layout of tile(i, j) to layout on host, optionally reset
    void tileLayoutConvert(int64_t i, int64_t j, Layout layout, bool reset = false)
    {
        tileLayoutConvert(i, j, host_num_, layout, reset);
    }
    void tileLayoutConvert(std::set<ij_tuple>& tile_set, int device, Layout layout, bool reset = false);
    /// Convert layout of a set of tiles to layout on host, optionally reset
    void tileLayoutConvert(std::set<ij_tuple>& tile_set, Layout layout, bool reset = false)
    {
        tileLayoutConvert(tile_set, host_num_, layout, reset);
    }
    void tileLayoutConvert(int device, Layout layout, bool reset = false);
    void tileLayoutConvertOnDevices(Layout layout, bool reset = false);

    //--------------------------------------------------------------------------
protected:
    void tileBcastToSet(int64_t i, int64_t j, std::set<int> const& bcast_set);
    void tileBcastToSet(int64_t i, int64_t j, std::set<int> const& bcast_set,
                        int radix, int tag, Layout layout);

    // todo: should this be private?
    void tileReduceFromSet(int64_t i, int64_t j,
                           std::set<int> const& reduce_set, int radix, int tag,
                           Layout layout);

public:

    void getRanks(std::set<int>* bcast_set) const;
    void getLocalDevices(std::set<int>* dev_set) const;
    int64_t numLocalTiles() const;
    MPI_Comm  mpiComm()  const { return mpi_comm_; }
    int       mpiRank()  const { return mpi_rank_; }
    MPI_Group mpiGroup() const { return mpi_group_; }
    int       hostNum()  const { return host_num_; }

    /// Removes all tiles from matrix.
    /// WARNING: currently this clears the entire parent matrix,
    /// not just a sub-matrix.
    void clear()
    {
        storage_->clear();
    }

    void releaseWorkspace()
    {
        storage_->releaseWorkspace();
    }

    /// Removes all temporary host and device workspace tiles from matrix.
    /// WARNING: currently this clears the entire parent matrix,
    /// not just a sub-matrix.
    void clearWorkspace()
    {
        storage_->clearWorkspace();
    }

    /// Removes batch arrays from matrix for all devices.
    /// WARNING: currently this clears the entire parent matrix,
    /// not just a sub-matrix.
    void clearBatchArrays()
    {
        storage_->clearBatchArrays();
    }

    /// @return currently allocated batch array size
    int64_t batchArraySize()
    {
        return storage_->batchArraySize();
    }

    //--------------------------------------------------------------------------
    /// @return batch arrays for the A, B, or C matrices,
    /// on host, to send to device
    /// Throws error if arrays were not allocated with allocateBatchArrays.
    scalar_t** a_array_host(int device)
    {
        return storage_->a_array_host_.at(device);
    }
    scalar_t** b_array_host(int device)
    {
        return storage_->b_array_host_.at(device);
    }
    scalar_t** c_array_host(int device)
    {
        return storage_->c_array_host_.at(device);
    }

    //--------------------------------------------------------------------------
    /// @return batch arrays for the A, B, or C matrices, on device
    /// Throws error if arrays were not allocated with allocateBatchArrays.
    scalar_t** a_array_device(int device)
    {
        return storage_->a_array_dev_.at(device);
    }
    scalar_t** b_array_device(int device)
    {
        return storage_->b_array_dev_.at(device);
    }
    scalar_t** c_array_device(int device)
    {
        return storage_->c_array_dev_.at(device);
    }

    //--------------------------------------------------------------------------
    /// @return CUDA streams and cuBLAS handles
    cublasHandle_t cublas_handle(int device)
    {
        return storage_->cublas_handles_.at(device);
    }
    cudaStream_t compute_stream(int device)
    {
        return storage_->compute_streams_.at(device);
    }
    cudaStream_t comm_stream(int device)
    {
        return storage_->comm_streams_.at(device);
    }

protected:
    std::tuple<int64_t, int64_t>
        globalIndex(int64_t i, int64_t j) const;

    std::tuple<int64_t, int64_t, int>
        globalIndex(int64_t i, int64_t j, int device) const;

    /// row offset of first block row.
    int64_t row0_offset() const { return row0_offset_; }

    /// col offset of first block col.
    int64_t col0_offset() const { return col0_offset_; }

    /// rows in last block row.
    int64_t last_mb() const { return last_mb_; }

    /// cols in last block col.
    int64_t last_nb() const { return last_nb_; }

    /// block row offset with respect to original matrix
    int64_t ioffset() const { return ioffset_; }

    /// block col offset with respect to original matrix
    int64_t joffset() const { return joffset_; }

private:
    ///-------------------------------------------------------------------------
    int64_t row0_offset_;  ///< row offset in first block row
    int64_t col0_offset_;  ///< col offset in first block col
    int64_t last_mb_;      ///< size of last block row
    int64_t last_nb_;      ///< size of last block col
    int64_t ioffset_;   ///< block row offset with respect to original matrix
    int64_t joffset_;   ///< block col offset with respect to original matrix
    int64_t mt_;        ///< number of local block rows in this view
    int64_t nt_;        ///< number of local block cols in this view

protected:
    Uplo uplo_;         ///< upper or lower storage
    Op op_;             ///< transpose operation with respect to original matrix

    /// shared storage of tiles and buffers
    std::shared_ptr< MatrixStorage<scalar_t> > storage_;

    // ----- consider where to put these, here or in MatrixStorage
    static int host_num_;
    static int num_devices_;

    MPI_Comm  mpi_comm_;
    MPI_Group mpi_group_;
    int mpi_rank_;

    /// intended layout of the matrix. defaults to ColMajor.
    Layout layout_;
};

//------------------------------------------------------------------------------
/// [internal]
/// Default constructor creates an empty matrix.
/// Does not allocate any memory.
///
template <typename scalar_t>
BaseMatrix<scalar_t>::BaseMatrix()
    : row0_offset_(0),
      col0_offset_(0),
      last_mb_(0),
      last_nb_(0),
      ioffset_(0),
      joffset_(0),
      mt_(0),
      nt_(0),
      uplo_(Uplo::General),
      op_(Op::NoTrans),
      storage_(nullptr),
      layout_(Layout::ColMajor)
{}

//------------------------------------------------------------------------------
/// [internal]
/// Construct matrix with
/// mt = ceil( m / nb ) block rows and
/// nt = ceil( n / nb ) block columns.
/// No tiles are allocated. Creates empty matrix storage.
///
/// @param[in] m
///     Number of rows of the matrix. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix. n >= 0
///
/// @param[in] nb
///     Block size in 2D block-cyclic distribution. n > 0.
///
/// @param[in] p
///     Number of block rows in 2D block-cyclic distribution. p > 0.
///
/// @param[in] q
///     Number of block columns of 2D block-cyclic distribution. q > 0.
///
/// @param[in] mpi_comm
///     MPI communicator to distribute matrix across.
///     p*q == MPI_Comm_size(mpi_comm).
///
template <typename scalar_t>
BaseMatrix<scalar_t>::BaseMatrix(
    int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)
    : row0_offset_(0),
      col0_offset_(0),
      last_mb_(m % nb == 0 ? nb : m % nb),
      last_nb_(n % nb == 0 ? nb : n % nb),
      ioffset_(0),
      joffset_(0),
      mt_(ceildiv(m, nb)),
      nt_(ceildiv(n, nb)),
      uplo_(Uplo::General),
      op_(Op::NoTrans),
      storage_(std::make_shared< MatrixStorage< scalar_t > >(
          m, n, nb, p, q, mpi_comm)),
      mpi_comm_(mpi_comm),
      layout_(Layout::ColMajor)
{
    slate_mpi_call(
        MPI_Comm_rank(mpi_comm_, &mpi_rank_));
    slate_mpi_call(
        MPI_Comm_group(mpi_comm_, &mpi_group_));

    // todo: these are static, but we (re-)initialize with each matrix.
    // todo: similar code in BaseMatrix(...) and MatrixStorage(...)
    host_num_    = storage_->host_num_;
    num_devices_ = storage_->num_devices_;
}

//------------------------------------------------------------------------------
/// [internal]
/// Sub-matrix constructor
/// creates shallow copy view of parent matrix, B[ i1:i2, j1:j2 ].
/// B[ 0:mt-1, 0:nt-1 ] is the entire B matrix.
/// If i2 < i1 or j2 < j1, produces empty matrix.
/// See Matrix::sub().
///
/// @param[in,out] B
///     Parent matrix B.
///
/// @param[in] i1
///     Starting block row index. 0 <= i1 < mt.
///
/// @param[in] i2
///     Ending block row index (inclusive). i2 < mt.
///
/// @param[in] j1
///     Starting block column index. 0 <= j1 < nt.
///
/// @param[in] j2
///     Ending block column index (inclusive). j2 < nt.
///
template <typename scalar_t>
BaseMatrix<scalar_t>::BaseMatrix(
    BaseMatrix<scalar_t>& B,
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
    : BaseMatrix(B)
{
    initSubmatrix(i1, i2, j1, j2);
}

//------------------------------------------------------------------------------
/// [private]
/// Sub-matrix constructor implementation used by
/// BaseMatrix( orig, i1, i2, j1, j2 ) and
/// BaseMatrix( orig, Slice( row1, row2, col1, col2 ) ).
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::initSubmatrix(
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
{
    assert((0 <= i1 && i1 < mt()) || (i1 > i2));  // todo: remove || ... clause?
    assert((0 <= i2 && i2 < mt()) || (i1 > i2));
    assert((0 <= j1 && j1 < nt()) || (j1 > j2));  // todo: remove || ... clause?
    assert((0 <= j2 && j2 < nt()) || (j1 > j2));

    // adjust i2, j2 for empty matrix
    i2 = std::max(i2, i1 - 1);
    j2 = std::max(j2, j1 - 1);

    if (op_ == Op::NoTrans) {
        last_mb_ = tileMb(i2);
        last_nb_ = tileNb(j2);
        ioffset_ += i1;
        joffset_ += j1;
        mt_ = i2 - i1 + 1;
        nt_ = j2 - j1 + 1;
    }
    else {
        last_nb_ = tileMb(i2);
        last_mb_ = tileNb(j2);
        ioffset_ += j1;
        joffset_ += i1;
        mt_ = j2 - j1 + 1;
        nt_ = i2 - i1 + 1;
    }
}

//------------------------------------------------------------------------------
/// [internal]
/// Sliced matrix constructor
/// creates shallow copy view of parent matrix, B[ row1:row2, col1:col2 ].
/// This takes row & col indices instead of block row & block col indices.
/// B[ 0:m-1, 0:n-1 ] is the entire B matrix.
/// If row2 < row1 or col2 < col1, produces empty matrix.
/// See Matrix::slice().
///
/// @param[in,out] B
///     Parent matrix B.
///
/// @param[in] slice
///     Start and end row and column indices:
///
///     - slice.row1: Starting row index. 0 <= row1 < m.
///
///     - slice.row2: Ending row index (inclusive). row2 < m.
///
///     - slice.col1: Starting column index. 0 <= col1 < n.
///
///     - slice.col2: Ending column index (inclusive). col2 < n.
///
template <typename scalar_t>
BaseMatrix<scalar_t>::BaseMatrix(
    BaseMatrix<scalar_t>& B, Slice slice)
    : BaseMatrix(B)
{
    int64_t row1 = slice.row1;
    int64_t row2 = slice.row2;
    int64_t col1 = slice.col1;
    int64_t col2 = slice.col2;
    assert(0 <= row1 && row1 < m());
    assert(0 <= col1 && col1 < n());
    assert(row2 < m());
    assert(col2 < n());

    // Map row indices (row1, row2) => block-row indices (i1, i2).
    int64_t i1 = 0;
    int64_t i2;
    int64_t row = tileMb( i1 );  // past end of tile i1
    while (row <= row1) {
        ++i1;
        row += tileMb( i1 );
    }
    int64_t new_row0_offset = row1 - (row - tileMb( i1 ));  // beginning of tile i1
    i2 = i1;
    while (row <= row2) {
        ++i2;
        row += tileMb( i2 );
    }
    int64_t new_last_mb = row2 - (row - tileMb( i2 )) + 1;
    if (i1 == i2)
        new_last_mb -= new_row0_offset;

    // Map col indices (col1, col2) => block-col indices (j1, j2).
    int64_t j1 = 0;
    int64_t j2;
    int64_t col = tileNb( j1 );  // past end of tile j1
    while (col <= col1) {
        ++j1;
        col += tileNb( j1 );
    }
    int64_t new_col0_offset = col1 - (col - tileNb( j1 ));  // beginning of tile j1
    j2 = j1;
    while (col <= col2) {
        ++j2;
        col += tileNb( j2 );
    }
    int64_t new_last_nb = col2 - (col - tileNb( j2 )) + 1;
    if (j1 == j2)
        new_last_nb -= new_col0_offset;

    // Create sub-matrix of tiles A[ i1:i2, j1:j2 ]
    initSubmatrix(i1, i2, j1, j2);

    // Adjust offsets & sizes of first & last block rows & columns.
    if (op_ == Op::NoTrans) {
        if (i1 == 0)
            new_row0_offset += row0_offset_;
        if (j1 == 0)
            new_col0_offset += col0_offset_;
        initSlice(new_row0_offset, new_col0_offset, new_last_mb, new_last_nb);
    }
    else {
        if (i1 == 0)
            new_row0_offset += col0_offset_;
        if (j1 == 0)
            new_col0_offset += row0_offset_;
        initSlice(new_col0_offset, new_row0_offset, new_last_nb, new_last_mb);
    }
}

//------------------------------------------------------------------------------
/// [private]
/// Used by BaseMatrix( orig, Slice( row1, row2, col1, col2 ) ) constructor to set
/// offsets and sizes for the first and last block row and block col.
/// This ignores transposition, which is handled in the constructor.
///
/// @param[in] row0_offset
///     Row offset in first block-row of A.
///
/// @param[in] col0_offset
///     Col offset in first block-col of A.
///
/// @param[in] last_mb
///     Size of last block-row of A.
///
/// @param[in] last_nb
///     Size of last block-col of A.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::initSlice(
    int64_t row0_offset, int64_t col0_offset,
    int64_t last_mb, int64_t last_nb)
{
    row0_offset_ = row0_offset;
    col0_offset_ = col0_offset;
    last_mb_ = last_mb;
    last_nb_ = last_nb;
}

//------------------------------------------------------------------------------
/// Swap contents of matrices A and B.
template <typename scalar_t>
void swap(BaseMatrix<scalar_t>& A, BaseMatrix<scalar_t>& B)
{
    using std::swap;
    swap(A.ioffset_, B.ioffset_);
    swap(A.joffset_, B.joffset_);
    swap(A.mt_,      B.mt_);
    swap(A.nt_,      B.nt_);
    swap(A.uplo_,    B.uplo_);
    swap(A.op_,      B.op_);
    swap(A.storage_, B.storage_);
}

//------------------------------------------------------------------------------
/// Get shallow copy of tile {i, j} of op(A) on given device,
/// with the tile's op flag set to match the matrix's.
///
/// @param[in] i
///     Tile's block row index. 0 <= i1 < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's device ID; default is host_num.
///
/// @return Tile {i, j, device}.
///
template <typename scalar_t>
Tile<scalar_t> BaseMatrix<scalar_t>::operator()(
    int64_t i, int64_t j, int device)
{
    auto tile = *(storage_->at(globalIndex(i, j, device)).tile_);

    // Set op first, before setting offset, mb, nb!
    tile.op(op_);

    // Set row & col offset within first block-row & block-col; before mb, nb!
    if (op_ == Op::NoTrans) {
        tile.offset(i == 0 ? row0_offset_ : 0,
                    j == 0 ? col0_offset_ : 0);
    }
    else {
        tile.offset(i == 0 ? col0_offset_ : 0,
                    j == 0 ? row0_offset_ : 0);
    }

    // Set tile size.
    tile.mb(tileMb(i));
    tile.nb(tileNb(j));

    // Set uplo on diagonal tile (off-diagonal tiles are always general).
    if (i == j)
        tile.uplo(uplo_);

    return tile;
}

//------------------------------------------------------------------------------
/// Returns number of rows in op(A).
template <typename scalar_t>
int64_t BaseMatrix<scalar_t>::m() const
{
    int64_t sum = 0;
    for (int64_t i = 0; i < mt(); ++i)
        sum += tileMb(i);
    return sum;
}

//------------------------------------------------------------------------------
/// Returns number of columns in op(A).
template <typename scalar_t>
int64_t BaseMatrix<scalar_t>::n() const
{
    int64_t sum = 0;
    for (int64_t j = 0; j < nt(); ++j)
        sum += tileNb(j);
    return sum;
}

//------------------------------------------------------------------------------
/// Returns number of rows (mb) in block row i of op(A).
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
template <typename scalar_t>
int64_t BaseMatrix<scalar_t>::tileMb(int64_t i) const
{
    if (op_ == Op::NoTrans)
        return tileMbInternal(i);
    else
        return tileNbInternal(i);
}

//------------------------------------------------------------------------------
/// Returns whether op(A) is logically Lower, Upper, or General storage,
/// taking the transposition operation into account.
///
template <typename scalar_t>
Uplo BaseMatrix<scalar_t>::uploLogical() const
{
    if (uplo_ == Uplo::General)
        return uplo_;
    else if ((uplo_ == Uplo::Lower) == (op_ == Op::NoTrans))
        return Uplo::Lower;
    else
        return Uplo::Upper;
}

//------------------------------------------------------------------------------
/// Returns whether A is physically Lower, Upper, or General storage,
/// ignoring the transposition operation.
///
template <typename scalar_t>
Uplo BaseMatrix<scalar_t>::uploPhysical() const
{
    return uplo_;
}

//------------------------------------------------------------------------------
/// Returns number of cols (nb) in block col j of op(A).
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
template <typename scalar_t>
int64_t BaseMatrix<scalar_t>::tileNb(int64_t j) const
{
    if (op_ == Op::NoTrans)
        return tileNbInternal(j);
    else
        return tileMbInternal(j);
}

//------------------------------------------------------------------------------
/// [internal]
/// Returns number of rows (mb) in block row i of A,
/// ignoring transposition, which is handled in tileMb().
/// This handles the first and last block rows for slicing.
///
/// @param[in] i
///     Tile's block row index.
///
template <typename scalar_t>
int64_t BaseMatrix<scalar_t>::tileMbInternal(int64_t i) const
{
    if (i == mt_ - 1)
        return last_mb_;
    else if (i == 0)
        return storage_->tileMb(ioffset_ + i) - row0_offset_;
    else
        return storage_->tileMb(ioffset_ + i);
}

//------------------------------------------------------------------------------
/// [internal]
/// Returns number of cols (nb) in block col j of A,
/// ignoring transposition, which is handled in tileNb().
/// This handles the first and last block columns for slicing.
///
/// @param[in] j
///     Tile's block column index.
///
template <typename scalar_t>
int64_t BaseMatrix<scalar_t>::tileNbInternal(int64_t j) const
{
    if (j == nt_ - 1)
        return last_nb_;
    else if (j == 0)
        return storage_->tileNb(joffset_ + j) - col0_offset_;
    else
        return storage_->tileNb(joffset_ + j);
}

//------------------------------------------------------------------------------
/// Insert tile {i, j} of op(A) and allocate its data.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's device ID; default is host_num.
///
/// @return Pointer to new tile.
///
template <typename scalar_t>
Tile<scalar_t>* BaseMatrix<scalar_t>::tileInsert(
    int64_t i, int64_t j, int device)
{
    auto index = globalIndex(i, j, device);
    auto tileEntry = storage_->tileInsert(index, TileKind::SlateOwned, layout_);
    return tileEntry.tile_;
}

//------------------------------------------------------------------------------
/// Insert a workspace tile {i, j} of op(A) and allocate its data.
/// The tile will be freed
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's device ID; default is host_num.
///
/// @return Pointer to new tile.
///
template <typename scalar_t>
TileEntry<scalar_t>& BaseMatrix<scalar_t>::tileInsertWorkspace(
    int64_t i, int64_t j, int device, Layout layout)
{
    auto index = globalIndex(i, j, device);
    return storage_->tileInsert(index, TileKind::Workspace, layout);
}

//------------------------------------------------------------------------------
/// Insert tile {i, j} of op(A) with existing data.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's device ID; default is host_num (provided by overloaded function).
///
/// @param[in,out] data
///     Tile's data. The matrix uses this pointer directly, it does not copy
///     the data, so the data must remain valid while the matrix exists.
///
/// @param[in] ld
///     Leading dimension of data; column stride. ld >= tileMb(i).
///
/// @return Pointer to new tile.
///
template <typename scalar_t>
Tile<scalar_t>* BaseMatrix<scalar_t>::tileInsert(
    int64_t i, int64_t j, int device, scalar_t* data, int64_t ld)
{
    auto index = globalIndex(i, j, device);
    // tile layout must match the matrix layout
    auto tileEntry = storage_->tileInsert(index, data, ld, layout_); // TileKind::UserOwned
    return tileEntry.tile_;
}

//------------------------------------------------------------------------------
/// Erase tile {i, j} of op(A).
/// If tile's memory was allocated by SLATE,
/// via tileInsert(i, j, dev) or tileInsertWorkspace(i, j, dev),
/// then the memory is released to the allocator pool.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's device ID.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileErase(int64_t i, int64_t j, int device)
{
    // todo: erase only workspace tiles? if so, rename with "Workspace"?
    storage_->erase(globalIndex(i, j, device));
}


//------------------------------------------------------------------------------
/// Erase the tile {i, j}'s instance on device if it is a workspace tile
/// that is not modified and no hold is set on it.
/// If tile's memory was allocated by SLATE,
/// via tileInsert(i, j, dev) or tileInsertWorkspace(i, j, dev),
/// then the memory is released to the allocator pool.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's device ID.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileRelease(int64_t i, int64_t j, int device)
{
    auto iter = storage_->find(globalIndex(i, j, device));
    if (iter != storage_->end() && iter->second.tile_->workspace()) {
        // auto tileEntry = storage_->at(globalIndex(i, j, device));
        if ( iter->second.stateOn(MOSI::OnHold) || iter->second.stateOn(MOSI::Modified) )
            return;
        else
            // todo: erase only workspace tiles? if so, rename with "Workspace"?
            storage_->erase(globalIndex(i, j, device));
    }
}


//------------------------------------------------------------------------------
/// Returns tile(i, j)'s state on device (defaults to host).
/// Asserts if tile does not exist.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's device ID.
///
template <typename scalar_t>
MOSI BaseMatrix<scalar_t>::tileState(int64_t i, int64_t j, int device)
{
    auto iter = storage_->find(globalIndex(i, j, device));
    assert(iter != storage_->end());

    return iter->second.getState();
}

//------------------------------------------------------------------------------
/// Sets tile(i, j)'s state on device.
/// Asserts if tile does not exist.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's device ID.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileState(int64_t i, int64_t j, int device, MOSI mosi)
{
    auto tileIter = storage_->find(globalIndex(i, j, device));
    assert(tileIter != storage_->end());

    tileIter->second.setState(mosi);
}

//------------------------------------------------------------------------------
/// Returns whether tile(i, j) is OnHold on device (defaults to host).
/// Asserts if tile does not exist.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's device ID.
///
template <typename scalar_t>
bool BaseMatrix<scalar_t>::tileOnHold(int64_t i, int64_t j, int device)
{
    auto iter = storage_->find(globalIndex(i, j, device));
    assert(iter != storage_->end());

    return iter->second.stateOn(MOSI::OnHold);
}

//------------------------------------------------------------------------------
/// Unsets the hold of tile(i, j) on device (defaults to host) if it was OnHold.
/// Asserts if tile does not exist.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's device ID.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileUnsetHold(int64_t i, int64_t j, int device)
{
    auto iter = storage_->find(globalIndex(i, j, device));
    assert(iter != storage_->end());

    iter->second.setState(~MOSI::OnHold);
}

//------------------------------------------------------------------------------
/// Unsets all local tiles' hold on device.
///
/// @param[in] device
///     Tile's device ID.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileUnsetHoldAll(int device)
{
    for (int64_t j = 0; j < nt(); ++j)
        for (int64_t i = 0; i < mt(); ++i)
            if (tileIsLocal(i, j))
                tileUnsetHold(i, j, device);
}

//------------------------------------------------------------------------------
/// Unsets all local tiles' hold on all devices.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileUnsetHoldAllOnDevices()
{
    for (int64_t j = 0; j < nt(); ++j)
        for (int64_t i = 0; i < mt(); ++i)
            if (tileIsLocal(i, j))
                tileUnsetHold(i, j, tileDevice(i, j));
}


//------------------------------------------------------------------------------
/// Marks tile(i, j) as Modified on device.
/// Other instances will be invalidated.
/// Unless permissive, asserts if other instances are in Modified state.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's device ID, defaults to host.
///
/// @param[in] permissive
///     Defaults to false.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileModified(int64_t i, int64_t j, int device, bool permissive)
{
    auto tileEntry = &storage_->at(globalIndex(i, j, device));
    // auto tileIter = storage_->find(globalIndex(i, j, device));
    // assert(tileIter != storage_->end());

    if (tileEntry->stateOn(MOSI::Modified))
        // no need to update
        return;
    tileEntry->setState(MOSI::Modified);

    // set all other instances to Invalid
    if (device != host_num_) {
        auto otherIter = storage_->find(globalIndex(i, j, host_num_));
        if (otherIter != storage_->end()) {
            if (! permissive)
                assert(otherIter->second.stateOn(MOSI::Modified) == false);
            otherIter->second.setState(MOSI::Invalid);
        }
    }
    for (int d = 0; d < num_devices(); ++d) {
        if (d == device) continue;

        auto otherIter = storage_->find(globalIndex(i, j, d));
        if (otherIter != storage_->end()) {
            if (! permissive)
                assert(otherIter->second.stateOn(MOSI::Modified) == false);
            otherIter->second.setState(MOSI::Invalid);
        }
    }
}

//------------------------------------------------------------------------------
/// Send tile {i, j} of op(A) to the given MPI rank.
/// Destination rank must call tileRecv().
///
/// @tparam target
///     Destination to target; either Host (default) or Device.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] dst_rank
///     Destination MPI rank. If dst_rank == mpiRank, this is a no-op.
///
/// @param[in] tag
///     MPI tag, default 0.
///
template <typename scalar_t>
template <Target target>
void BaseMatrix<scalar_t>::tileSend(
    int64_t i, int64_t j, int dst_rank, int tag)
{
    if (dst_rank != mpiRank()) {
        tileGetForReading(i, j, LayoutConvert::None);
        at(i, j).send(dst_rank, mpiComm(), tag);
    }
}

//------------------------------------------------------------------------------
/// Receive tile {i, j} of op(A) to the given MPI rank.
/// Tile is allocated as workspace with life = 1 if it doesn't yet exist,
/// or 1 is added to life if it does exist.
/// Source rank must call tileSend().
/// Data received must be in 'layout' (ColMajor/RowMajor) major.
///
/// @tparam target
///     Destination to target; either Host (default) or Device.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] src_rank
///     Source MPI rank. If src_rank == mpiRank, this is a no-op.
///
/// @param[in] layout
///     Indicates the Layout (ColMajor/RowMajor) of the received data.
///     WARNING: must match the layout of the tile in the sender MPI rank.
///
/// @param[in] tag
///     MPI tag, default 0.
///
template <typename scalar_t>
template <Target target>
void BaseMatrix<scalar_t>::tileRecv(
    int64_t i, int64_t j, int src_rank, Layout layout, int tag)
{
    if (src_rank != mpiRank()) {
        if (! tileIsLocal(i, j)) {
            // Create tile to receive data, with life span.
            // If tile already exists, add to its life span.
            LockGuard(storage_->tiles_.get_lock());
            auto iter = storage_->find(globalIndex(i, j, host_num_));

            int64_t life = 1;
            if (iter == storage_->end())
                tileInsertWorkspace(i, j, host_num_);
            else
                life += tileLife(i, j);
            tileLife(i, j, life);
        }
        else {
            // todo: tileAquire()
            tileGetForReading(i, j, LayoutConvert::None);
        }

        // Receive data.
        at(i, j).recv(src_rank, mpiComm(), layout, tag);

        tileLayout(i, j, layout);
        tileModified(i, j, hostNum(), true);

        // Copy to devices.
        if (target == Target::Devices) {
            #pragma omp task
            {
                tileGetForReading(i, j, tileDevice(i, j), LayoutConvert::None);
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Send tile {i, j} of op(A) to all MPI ranks in matrix B.
/// If target is Devices, also copies tile to all devices on each MPI rank.
/// This should be called by at least all ranks with local tiles in B;
/// ones that do not have any local tiles are excluded from the broadcast.
/// Data received must be in 'layout' (ColMajor/RowMajor) major.
///
/// @tparam target
///     Destination to target; either Host (default) or Device.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] B
///     Sub-matrix B defines the MPI ranks to send to.
///     Usually it is the portion of the matrix to be updated by tile {i, j}.
///
/// @param[in] layout
///     Indicates the Layout (ColMajor/RowMajor) of the broadcasted data.
///     WARNING: must match the layout of the tile in the sender MPI rank.
///
/// @param[in] tag
///     MPI tag, default 0.
///
/// @param[in] life_factor
///     A multiplier for the life count of the broadcasted tile workspace.
///
template <typename scalar_t>
template <Target target>
void BaseMatrix<scalar_t>::tileBcast(
    int64_t i, int64_t j, BaseMatrix<scalar_t> const& B, Layout layout, int tag, int64_t life_factor)
{
    BcastList bcast_list_B;
    bcast_list_B.push_back({i, j, {B}});
    listBcast<target>(bcast_list_B, layout, tag, life_factor);
}

//------------------------------------------------------------------------------
/// Send tile {i, j} of op(A) to all MPI ranks in the list of submatrices
/// bcast_list.
/// Data received must be in 'layout' (ColMajor/RowMajor) major.
///
/// @tparam target
///     Destination to target; either Host (default) or Device.
///
/// @param[in] bcast_list
///     List of submatrices defining the MPI ranks to send to.
///     Usually it is the portion of the matrix to be updated by tile {i, j}.
///
/// @param[in] layout
///     Indicates the Layout (ColMajor/RowMajor) of the broadcasted data.
///     WARNING: must match the layout of the tile in the sender MPI rank.
///
/// @param[in] tag
///     MPI tag, default 0.
///
/// @param[in] life_factor
///     A multiplier for the life count of the broadcasted tile workspace.
///
template <typename scalar_t>
template <Target target>
void BaseMatrix<scalar_t>::listBcast(
    BcastList& bcast_list, Layout layout, int tag, int64_t life_factor)
{
    // It is possible that the same tile, with the same data, is sent twice.
    // This happens, e.g., in the hemm and symm routines, where the same tile
    // is sent once as part of A and once as part of A^T.
    // This cannot be avoided without violating the upper bound on the buffer size
    // used for hosting communicated tiles.
    // Due to dynamic scheduling, the second communication may occur before the
    // first tile has been discarded.
    // If that happens, instead of creating the tile, the life of the existing
    // tile is increased.
    // Also, currently, the message is received to the same buffer.

    for (auto bcast : bcast_list) {

        auto i = std::get<0>(bcast);
        auto j = std::get<1>(bcast);
        auto submatrices_list = std::get<2>(bcast);

        // Find the set of participating ranks.
        std::set<int> bcast_set;
        bcast_set.insert(tileRank(i, j));       // Insert root.
        for (auto submatrix : submatrices_list) // Insert destinations.
            submatrix.getRanks(&bcast_set);

        // If this rank is in the set.
        if (bcast_set.find(mpi_rank_) != bcast_set.end()) {

            // If receiving the tile.
            if (! tileIsLocal(i, j)) {

                // Create tile to receive data, with life span.
                // If tile already exists, add to its life span.
                LockGuard(storage_->tiles_.get_lock());
                auto iter = storage_->find(globalIndex(i, j, host_num_));

                int64_t life = 0;
                for (auto submatrix : submatrices_list)
                    life += submatrix.numLocalTiles() * life_factor;

                if (iter == storage_->end())
                    tileInsertWorkspace(i, j, host_num_);
                else
                    life += tileLife(i, j); // todo: use temp tile to receive
                tileLife(i, j, life);
            }

            // Send across MPI ranks.
            // Previous used MPI bcast: tileBcastToSet(i, j, bcast_set);
            // Currently uses 2D hypercube p2p send.
            tileBcastToSet(i, j, bcast_set, 2, tag, layout);
        }

        // Copy to devices.
        // TODO: should this be inside above if-then?
        // todo: this may incur extra communication,
        //       tile(i,j) is not necessarily needed on all devices where this matrix resides
        if (target == Target::Devices) {
            // If receiving the tile.
            if (! tileIsLocal(i, j)) {
                std::set<int> dev_set;
                for (auto submatrix : submatrices_list)
                    submatrix.getLocalDevices(&dev_set);

                // todo: should each read be an omp task instead?
                #pragma omp task
                {
                    for (auto device : dev_set)
                        tileGetForReading(i, j, device, LayoutConvert::None);
                }
            }
        }
    }

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
///
template <typename scalar_t>
template <Target target>
void BaseMatrix<scalar_t>::listReduce(ReduceList& reduce_list, Layout layout, int tag)
{
    for (auto reduce : reduce_list) {

        auto i = std::get<0>(reduce);
        auto j = std::get<1>(reduce);
        auto submatrices_list = std::get<2>(reduce);

        // Find the set of participating ranks.
        std::set<int> reduce_set;
        reduce_set.insert(tileRank(i, j));      // Insert root.
        for (auto submatrix : submatrices_list) // Insert sources.
            submatrix.getRanks(&reduce_set);

        // If this rank is in the set.
        if (reduce_set.find(mpi_rank_) != reduce_set.end()) {

            // Reduce across MPI ranks.
            // Uses 2D hypercube p2p send.
            tileReduceFromSet(i, j, reduce_set, 2, tag, layout);

            // If not the tile owner.
            if (! tileIsLocal(i, j)) {

                // todo: should we check its life count before erasing?
                // Destroy the tile.
                LockGuard(storage_->tiles_.get_lock());// todo is this needed here?
                tileErase(i, j, host_num_);// todo: should it be a tileRelease()?
            }
            else {
                tileModified(i, j);
            }
        }
    }

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
/// [internal]
/// Broadcast tile {i, j} to all MPI ranks in the bcast_set.
/// This should be called by all (and only) ranks that are in bcast_set,
/// as either the root sender or a receiver.
/// This implementation creates a subcommunicator and calls MPI broadcast.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] bcast_set
///     Set of MPI ranks to broadcast to.
///
// todo: use the commFromSet() function from slate_internal_comm.cc
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileBcastToSet(
    int64_t i, int64_t j, std::set<int> const& bcast_set)
{
    // this function does not use tags, kept for reference
    assert("This variant of tileBcastToSet() is deprecated");

    // Quit if only root in the broadcast set.
    if (bcast_set.size() == 1)
        return;

    // Convert the set of ranks to a vector.
    std::vector<int> bcast_vec(bcast_set.begin(), bcast_set.end());

    // Create the broadcast group.
    MPI_Group bcast_group;
    #pragma omp critical(slate_mpi)
    slate_mpi_call(
        MPI_Group_incl(mpi_group_, bcast_vec.size(), bcast_vec.data(),
                       &bcast_group));

    // Create a broadcast communicator.
    int tag = 0;
    MPI_Comm bcast_comm;
    #pragma omp critical(slate_mpi)
    {
        trace::Block trace_block("MPI_Comm_create_group");
        slate_mpi_call(
            MPI_Comm_create_group(mpi_comm_, bcast_group, tag, &bcast_comm));
    }
    assert(bcast_comm != MPI_COMM_NULL);

    // Find the broadcast rank.
    int bcast_rank;
    #pragma omp critical(slate_mpi)
    MPI_Comm_rank(bcast_comm, &bcast_rank);

    // Find the broadcast root rank.
    int root_rank = tileRank(i, j);
    int bcast_root;
    #pragma omp critical(slate_mpi)
    slate_mpi_call(
        MPI_Group_translate_ranks(mpi_group_, 1, &root_rank,
                                  bcast_group, &bcast_root));

    // Do the broadcast.
    at(i, j).bcast(bcast_root, bcast_comm);

    // Free the group.
    #pragma omp critical(slate_mpi)
    slate_mpi_call(
        MPI_Group_free(&bcast_group));

    // Free the communicator.
    #pragma omp critical(slate_mpi)
    slate_mpi_call(
        MPI_Comm_free(&bcast_comm));
}

//------------------------------------------------------------------------------
/// [internal]
/// Broadcast tile {i, j} to all MPI ranks in the bcast_set.
/// This should be called by all (and only) ranks that are in bcast_set,
/// as either the root sender or a receiver.
/// This function implements a custom pattern using sends and receives.
/// Data received must be in 'layout' (ColMajor/RowMajor) major.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] bcast_set
///     Set of MPI ranks to broadcast to.
///
/// @param[in] radix
///     Radix of the communication pattern.
///
/// @param[in] tag
///     MPI tag, default 0.
///
/// @param[in] layout
///     Indicates the Layout (ColMajor/RowMajor) of the received data.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileBcastToSet(
    int64_t i, int64_t j, std::set<int> const& bcast_set,
    int radix, int tag, Layout layout)
{
    // Quit if only root in the broadcast set.
    if (bcast_set.size() == 1)
        return;

    // Convert the set to a vector.
    std::vector<int> bcast_vec(bcast_set.begin(), bcast_set.end());

    // TODO: std::set is already sorted (it's really an ordered_set), no need to sort again?
    // Sort the ranks.
    std::sort(bcast_vec.begin(), bcast_vec.end());

    // Find root.
    int root_rank = tileRank(i, j);
    auto root_iter = std::find(bcast_vec.begin(), bcast_vec.end(), root_rank);

    // Shift root to position zero.
    std::vector<int> new_vec(root_iter, bcast_vec.end());
    new_vec.insert(new_vec.end(), bcast_vec.begin(), root_iter);

    // Find the new rank.
    auto rank_iter = std::find(new_vec.begin(), new_vec.end(), mpi_rank_);
    int new_rank = std::distance(new_vec.begin(), rank_iter);

    // Get the send/recv pattern.
    std::list<int> recv_from;
    std::list<int> send_to;
    internal::cubeBcastPattern(new_vec.size(), new_rank, radix,
                               recv_from, send_to);

    // Receive.
    if (! recv_from.empty()) {
        // read tile on host memory
        // todo: tileAquire()
        // todo: this fixed the device origin but corrupted host origin,
        // really need tileAquire()
        // disable temporarily
        // tileGetForReading(i, j);

        at(i, j).recv(new_vec[recv_from.front()], mpi_comm_, layout, tag);
        tileLayout(i, j, layout);
        tileModified(i, j);
    }

    if (! send_to.empty()) {
        // read tile on host memory
        tileGetForReading(i, j, LayoutConvert(layout));
        // Forward.
        for (int dst : send_to)
            at(i, j).send(new_vec[dst], mpi_comm_, tag);
    }
}

//------------------------------------------------------------------------------
/// [internal]
/// WARNING: Sent and Recevied tiles are converted to 'layout' major.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileReduceFromSet(
    int64_t i, int64_t j, std::set<int> const& reduce_set, int radix, int tag,
    Layout layout)
{
    // Quit if only root in the reduction set.
    if (reduce_set.size() == 1)
        return;

    // Convert the set to a vector.
    std::vector<int> reduce_vec(reduce_set.begin(), reduce_set.end());

    // Sort the ranks.
    std::sort(reduce_vec.begin(), reduce_vec.end());

    // Find root.
    int root_rank = tileRank(i, j);
    auto root_iter = std::find(reduce_vec.begin(), reduce_vec.end(), root_rank);

    // Shift root to position zero.
    std::vector<int> new_vec(root_iter, reduce_vec.end());
    new_vec.insert(new_vec.end(), reduce_vec.begin(), root_iter);

    // Find the new rank.
    auto rank_iter = std::find(new_vec.begin(), new_vec.end(), mpi_rank_);
    int new_rank = std::distance(new_vec.begin(), rank_iter);

    // Get the send/recv pattern.
    std::list<int> recv_from;
    std::list<int> send_to;
    internal::cubeReducePattern(new_vec.size(), new_rank, radix,
                                recv_from, send_to);

    if (! (send_to.empty() && recv_from.empty()) ) {
        // read tile on host memory
        tileGetForReading(i, j, LayoutConvert(layout));
    }

    std::vector<scalar_t> data(tileMb(i)*tileNb(j));
    Tile<scalar_t> tile(tileMb(i), tileNb(j), &data[0], tileMb(i), host_num_, TileKind::Workspace);

    // Receive, accumulate.
    for (int src : recv_from) {
        // Receive.
        tile.recv(new_vec[src], mpi_comm_, layout, tag);
        // Accumulate.
        axpy(scalar_t(1.0), tile, at(i, j));
    }

    // Forward.
    if (! send_to.empty())
        at(i, j).send(new_vec[send_to.front()], mpi_comm_, tag);
}

//------------------------------------------------------------------------------
/// Gets tile(i, j) for on device.
/// Will copy-in the tile if it does not exist or its state is MOSI::Invalid.
/// Finds a source tile whose state is valid (Modified|Shared) by
///     looping on existing tile instances.
/// Updates source tile's state to shared if copied-in.
/// If 'modify' param is true, marks the destination tile as MOSI::Modified,
///     and invalidates other instances. Otherwise, sets destination tile state
///     to MOSI::Shared if copied-in.
/// Converts destination Layout based on 'layout' param.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] dst_device
///     Tile's destination: host or device ID.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
/// @param[in] modify
///     true: marks the destination tile as MOSI::Modified and invalidates other instances.
///     false: marks the destination tile as MOSI::Shared.
///
/// @param[in] hold
///     true: marks the destination tile as MOSI::OnHold.
///
/// @param[in] async
///     true: does not synchronize with device stream.
///
// todo: async version
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGet(int64_t i, int64_t j, int dst_device,
                                   LayoutConvert layout, bool modify, bool hold,
                                   bool async)
{
    TileEntry<scalar_t> *dst_tileEntry = nullptr, *src_tileEntry = nullptr;

    // find tile on destination
    auto dst_iter = storage_->find(globalIndex(i, j, dst_device));
    if (dst_iter == storage_->end()) {
        // Create a copy on the destination.
        tileInsertWorkspace(i, j, dst_device);
        dst_iter = storage_->find(globalIndex(i, j, dst_device));
    }
    dst_tileEntry = &dst_iter->second;

    if (dst_tileEntry->getState() == MOSI::Invalid) {

        // find source tile
        // find a valid source (Modified/Shared) device
        const int invalid_dev = host_num_-1; // invalid device number
        int src_device = invalid_dev;
        {
            // check host
            if (dst_device != host_num_) {
                auto iter = storage_->find(globalIndex(i, j, host_num_));
                if (iter != storage_->end()) {
                    if (iter->second.getState() != MOSI::Invalid) {
                        src_device = host_num_;
                        src_tileEntry = &iter->second;
                    }
                }
            }
            // check assigned device
            if (src_device == invalid_dev) {
                auto iter = storage_->find(globalIndex(i, j, tileDevice(i, j)));
                if (iter != storage_->end()) {
                    if (iter->second.getState() != MOSI::Invalid) {
                        src_device = tileDevice(i, j);
                        src_tileEntry = &iter->second;
                    }
                }
            }
            // check other devices
            for (int d = 0; src_device == invalid_dev && d < num_devices(); ++d) {
                if (dst_device == d || tileDevice(i, j) == d) continue;

                auto iter = storage_->find(globalIndex(i, j, d));
                if (iter != storage_->end()) {
                    if (iter->second.getState() != MOSI::Invalid) {
                        src_device = d;
                        src_tileEntry = &iter->second;
                    }
                }
            }
            // todo: find the shortest path / closest source
            // including possibility of device peer-to-peer copy
        }
        if(src_device == invalid_dev){
            slate_error(std::string("Error copying tile(")
                         + std::to_string(i) + ", " + std::to_string(j)
                         + "), invalid source " + std::to_string(src_device)
                         + " -> " + std::to_string(dst_device) );
        }

        // Update the destination tile's data.
        if (dst_device != host_num_ && src_device != host_num_) {
            // todo: device to device copy
            TileEntry<scalar_t> *host_tileEntry;
            auto host_iter = storage_->find(globalIndex(i, j, host_num_));
            if (host_iter == storage_->end()) {
                // Create a copy on the host.
                tileInsertWorkspace(i, j, host_num_);
                host_iter = storage_->find(globalIndex(i, j, host_num_));
            }
            host_tileEntry = &host_iter->second;

            src_tileEntry->tile_->copyDataToHost(host_tileEntry->tile_, comm_stream(src_device));
            host_tileEntry->tile_->copyDataToDevice(dst_tileEntry->tile_, comm_stream(dst_device));
            host_tileEntry->setState(MOSI::Shared);
        }
        else
        if (dst_device == host_num_)
            src_tileEntry->tile_->copyDataToHost(dst_tileEntry->tile_, comm_stream(src_device));
        else
            src_tileEntry->tile_->copyDataToDevice(dst_tileEntry->tile_, comm_stream(dst_device));

        dst_tileEntry->setState(MOSI::Shared);
        if (src_tileEntry->stateOn(MOSI::Modified))
            src_tileEntry->setState(MOSI::Shared);
    }
    if (modify) {
        tileModified(i, j, dst_device);
    }
    if (hold) {
        dst_tileEntry->setState(MOSI::OnHold);
    }

    // Change ColMajor <=> RowMajor if needed.
    if (layout != LayoutConvert::None &&
        dst_tileEntry->tile_->layout() != Layout(layout)) {
        if (dst_device == host_num_) {
            convert_layout(dst_tileEntry->tile_);
        }
        else {
            convert_layout(dst_tileEntry->tile_, compute_stream(dst_device));
        }
    }
}

//------------------------------------------------------------------------------
/// Gets a set of tiles on device.
/// If destination device is host, forwards LayoutConvert param to tileGet()
///     otherwise, calls tileLayoutConvert() to process layout conversion in batch mode.
/// @see tileGet
///
/// @param[in] tile_set
///     Set of (i, j) tuples indicating indices of Tiles' to be converted.
///
/// @param[in] device
///     Tile's destination: host or device ID, defaults to host.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
/// @param[in] modify
///     true: marks the destination tiles as MOSI::Modified and invalidates other instances.
///     false: marks the destination tiles as MOSI::Shared.
///
/// @param[in] hold
///     true: marks the destination tiles as MOSI::OnHold.
///
/// @param[in] async
///     if true, does not synchronize with device stream.
///
// todo: async version
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGet(std::set<ij_tuple>& tile_set, int device,
                                   LayoutConvert in_layoutConvert, bool modify, bool hold,
                                   bool async)
{
    LayoutConvert layoutConvert = (device == host_num_) ? in_layoutConvert : LayoutConvert::None;

    for (auto iter = tile_set.begin(); iter != tile_set.end(); iter++) {
        int64_t i = std::get<0>(*iter);
        int64_t j = std::get<1>(*iter);
        tileGet(i, j, device, layoutConvert, modify, hold, async);
    }

    if (device != host_num_ && in_layoutConvert != LayoutConvert::None) {
        tileLayoutConvert(tile_set, device, Layout(in_layoutConvert));
    }
}

//------------------------------------------------------------------------------
/// Gets tile(i, j) for reading on device.
/// Will copy-in the tile if it does not exist or its state is MOSI::Invalid.
/// Sets tile state to MOSI::Shared if copied-in.
/// Finds a source tile whose state is valid (Modified|Shared) by
/// looping on existing tile instances.
/// Updates source tile's state to shared if copied-in.
/// Converts destination Layout based on 'layout' param.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's destination: host or device ID, defaults to host.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
// todo: async version
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetForReading(int64_t i, int64_t j, int device,
                                             LayoutConvert layout)
{
    tileGet(i, j, device, layout, false, false, false);
}

//------------------------------------------------------------------------------
/// Gets a set of tiles for reading on device.
/// @see tileGetForReading
///
/// @param[in] tile_set
///     Set of (i, j) tuples indicating indices of Tiles' to be acquired.
///
/// @param[in] device
///     Tile's destination: host or device ID.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
// todo: async version
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetForReading(std::set<ij_tuple>& tile_set, int device,
                                             LayoutConvert layout)
{
    tileGet(tile_set, device, layout, false, false, false);
}

//------------------------------------------------------------------------------
/// Gets tile(i, j) for writing on device.
/// Sets destination tile's state to MOSI::Modified.
/// Will copy-in the tile if it does not exist or its state is MOSI::Invalid.
/// Other instances will be invalidated.
/// Finds a source tile whose state is valid (Modified|Shared) by
///     scanning existing tile instances.
/// Converts destination Layout based on 'layout' param.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's destination: host or device ID, defaults to host.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetForWriting(int64_t i, int64_t j, int device,
                                             LayoutConvert layout)
{
    tileGet(i, j, device, layout, true, false, false);
}

//------------------------------------------------------------------------------
/// Gets a set of tiles for writing on device.
/// @see tileGetForWriting
///
/// @param[in] tile_set
///     Set of (i, j) tuples indicating indices of Tiles' to be acquired.
///
/// @param[in] device
///     Tile's destination: host or device ID, defaults to host.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetForWriting(std::set<ij_tuple>& tile_set,
                                             int device, LayoutConvert layout)
{
    tileGet(tile_set, device, layout, true, false, false);
}

//------------------------------------------------------------------------------
/// Gets tile(i, j) on device for reading and marks it as MOSI::OnHold.
/// Will copy tile in if it does not exist or its state is MOSI::Invalid.
/// Updates the source tile's state to MOSI::Shared if copied-in.
/// Converts destination Layout based on 'layout' param.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's destination: host or device ID, defaults to host.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetAndHold(int64_t i, int64_t j, int device,
                                          LayoutConvert layout)
{
    tileGet(i, j, device, layout, false, true, false);
}

//------------------------------------------------------------------------------
/// Gets a set of tiles for reading on device and marks them as MOSI::OnHold.
/// @see tileGetAndHold
///
/// @param[in] tile_set
///     Set of (i, j) tuples indicating indices of Tiles' to be acquired.
///
/// @param[in] device
///     Tile's destination: host or device ID, defaults to host.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetAndHold(std::set<ij_tuple>& tile_set, int device,
                                          LayoutConvert layout)
{
    tileGet(tile_set, device, layout, false, true, false);
}

//------------------------------------------------------------------------------
/// Gets all local tiles for reading on device.
/// @see tileGetForReading.
///
/// @param[in] device
///     Tile's destination: host or device ID, defaults to host.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
// todo: should this take into consideration the local tiles distribution on devices?
//       currently, this will cram all local tiles into specified device.
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetAllForReading(int device, LayoutConvert layout)
{
    std::set<ij_tuple> tiles_set;
    for (int64_t j = 0; j < nt(); ++j)
        for (int64_t i = 0; i < mt(); ++i)
            // todo: if (tileIsLocal(i, j) && (device == host_num_ || device == tileDevice(i, j))) {
            if (tileIsLocal(i, j)) {
                tiles_set.insert({i, j});
            }

    tileGetForReading(tiles_set, device, layout);
}

//------------------------------------------------------------------------------
/// Gets all local tiles for writing on device.
/// @see tileGetForWriting.
///
/// @param[in] device
///     Tile's destination: host or device ID, defaults to host.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
// todo: should this take into consideration the local tiles distribution on devices?
//       currently, this will cram all local tiles into specified device.
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetAllForWriting(int device, LayoutConvert layout)
{
    std::set<ij_tuple> tiles_set;
    for (int64_t j = 0; j < nt(); ++j)
        for (int64_t i = 0; i < mt(); ++i)
            // todo: if (tileIsLocal(i, j) && (device == host_num_ || device == tileDevice(i, j))) {
            if (tileIsLocal(i, j)) {
                tiles_set.insert({i, j});
            }

    tileGetForWriting(tiles_set, device, layout);
}

//------------------------------------------------------------------------------
/// Gets all local tiles on device and marks them as MOSI::OnHold.
/// @see tileGetAndHold.
///
/// @param[in] device
///     Tile's destination: host or device ID, defaults to host.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
// todo: should this take into consideration the local tiles distribution on devices?
//       currently, this will cram all local tiles into specified device.
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetAndHoldAll(int device, LayoutConvert layout)
{
    std::set<ij_tuple> tiles_set;
    for (int64_t j = 0; j < nt(); ++j)
        for (int64_t i = 0; i < mt(); ++i)
            // todo: if (tileIsLocal(i, j) && (device == host_num_ || device == tileDevice(i, j))) {
            if (tileIsLocal(i, j)) {
                tiles_set.insert({i, j});
            }

    tileGetAndHold(tiles_set, layout, device);
}

//------------------------------------------------------------------------------
/// Gets all local tiles for reading on corresponding devices.
/// @see tileGetForReading.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetAllForReadingOnDevices(LayoutConvert layout)
{
    std::vector< std::set<ij_tuple> > tiles_set(num_devices());
    for (int64_t j = 0; j < nt(); ++j)
        for (int64_t i = 0; i < mt(); ++i)
            if (tileIsLocal(i, j)) {
                tiles_set[tileDevice(i, j)].insert({i, j});
            }

    // todo: omp tasks?
    for (int d = 0; d < num_devices(); ++d) {
        tileGetForReading(tiles_set[d], d, layout);
    }
}

//------------------------------------------------------------------------------
/// Gets all local tiles for writing on corresponding devices.
/// @see tileGetForWriting.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetAllForWritingOnDevices(LayoutConvert layout)
{
    std::vector< std::set<ij_tuple> > tiles_set(num_devices());
    for (int64_t j = 0; j < nt(); ++j)
        for (int64_t i = 0; i < mt(); ++i)
            if (tileIsLocal(i, j)) {
                tiles_set[tileDevice(i, j)].insert({i, j});
            }

    // todo: omp tasks?
    for (int d = 0; d < num_devices(); ++d) {
        tileGetForWriting(tiles_set[d], d, layout);
    }
}

//------------------------------------------------------------------------------
/// Gets all local tiles on corresponding devices and marks them as MOSI::OnHold.
/// @see tileGetAndHold.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
//
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetAndHoldAllOnDevices(LayoutConvert layout)
{
    std::vector< std::set<ij_tuple> > tiles_set(num_devices());
    for (int64_t j = 0; j < nt(); ++j)
        for (int64_t i = 0; i < mt(); ++i)
            if (tileIsLocal(i, j)) {
                tiles_set[tileDevice(i, j)].insert({i, j});
            }

    // todo: omp tasks?
    for (int d = 0; d < num_devices(); ++d) {
        tileGetAndHold(tiles_set[d], d, layout);
    }
}

//------------------------------------------------------------------------------
/// Updates the origin instance of tile(i, j) if MOSI::Invalid
/// tile must be local.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @return Pointer to origin instance of tile(i, j)
///
template <typename scalar_t>
Tile<scalar_t>* BaseMatrix<scalar_t>::tileUpdateOrigin(int64_t i, int64_t j)
{
    // find on host
    auto iter = storage_->find(globalIndex(i, j, host_num_));
    if (iter != storage_->end() && iter->second.tile_->origin()) {
        if ( iter->second.stateOn(MOSI::Invalid) )
            tileGetForReading(i, j, LayoutConvert::None);
    }
    else {
        iter = storage_->find(globalIndex(i, j, tileDevice(i, j)));
        if (iter != storage_->end() && iter->second.tile_->origin()) {
            if ( iter->second.stateOn(MOSI::Invalid) )
                tileGetForReading(i, j, tileDevice(i, j), LayoutConvert::None);
        }
        else
            slate_error( std::string("Origin tile not found! tile(")
                        +std::to_string(i)+","+std::to_string(j)+")");
    }
    return iter->second.tile_;
}

//------------------------------------------------------------------------------
/// Updates all origin instances of local tiles if MOSI::Invalid.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileUpdateAllOrigin()
{
    for (int64_t j = 0; j < this->nt(); ++j)
        for (int64_t i = 0; i < this->mt(); ++i)
            if (this->tileIsLocal(i, j))
                this->tileUpdateOrigin(i, j);
}

//------------------------------------------------------------------------------
/// Returns whether tile(i, j, device) can be safely transposed.
/// based on its 'TileKind', buffer size, Layout, and stride.
/// Tile instance on 'device' should exist.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's host or device ID, defaults to host.
///
// todo: validate working for sub- and sliced- matrix
template <typename scalar_t>
bool BaseMatrix<scalar_t>::tileLayoutIsConvertible(int64_t i, int64_t j, int device)
{
    return storage_->at(globalIndex(i, j, device)).tile_->isTransposable();
}


//------------------------------------------------------------------------------
/// Converts tile(i, j, device) into 'layout'.
/// Tile should exist on 'device', will assert otherwise.
/// Tile will be made Convertible if it was not.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's host or device ID.
///
/// @param[in] layout
///     Intended Layout of tile:
///     - Layout::ColMajor or
///     - Layout::RowMajor.
///
/// todo: handle op(A), sub-matrix, and sliced-matrix
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileLayoutConvert(int64_t i, int64_t j, int device,
                                             Layout layout, bool reset)
{
    auto tile = storage_->at(globalIndex(i, j, device)).tile_;
    if (tile->layout() != layout ) {
        if (! tile->isTransposable() ) {
            assert(! reset);
            storage_->tileMakeTransposable(tile);
        }
        scalar_t* work_data = nullptr;
        // if rectangular and not extended, need a workspace buffer
        if (tile->mb() != tile->nb() && (! tile->extended()))
            work_data = storage_->allocWorkspaceBuffer(tile->device());

        tile->layoutConvert(work_data,
                            tile->device() == host_num_ ?
                                              nullptr :
                                              compute_stream(tile->device()));

        // release the workspace buffer if allocated
        if (tile->mb() != tile->nb() && (! tile->extended()))
            storage_->releaseWorkspaceBuffer(work_data, tile->device());
    }
    if (reset) {
        storage_->tileLayoutReset(tile);
    }
}

//------------------------------------------------------------------------------
/// Converts tiles indicated in 'tile_set' that exist on 'device' into 'layout' if
///     not alread in 'layout' major.
/// Tiles should exist on 'device', will assert otherwise.
/// Operates in batch mode when tiles are on devices.
/// If device is not Host, will bucket tiles into uniform size and stride batches,
///     then launches each batch transpose.
///
/// @param[in] tile_set
///     Set of (i, j) tuples indicating indices of Tiles' to be converted.
///
/// @param[in] device
///     Tiles' host or device ID.
///
/// @param[in] layout
///     Intended Layout of tiles:
///     - Layout::ColMajor or
///     - Layout::RowMajor.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileLayoutConvert(std::set<ij_tuple>& tile_set,
                                             int device, Layout layout,
                                             bool reset)
{
    if (device == host_num_) {
        for (auto iter = tile_set.begin(); iter != tile_set.end(); iter++) {
            // #pragma omp task
            {
                int64_t i = std::get<0>(*iter);
                int64_t j = std::get<1>(*iter);
                tileLayoutConvert(i, j, device, layout, reset);
            }
        }
        // #pragma omp taskwait
    }
    else {

        // map key tuple: m, n, extended, stride, work_stride
        using mnss_tuple = std::tuple<int64_t, int64_t, bool, int64_t, int64_t>;
        // map value tuple: data and extended data buffers
        using data_tuple = std::pair<std::vector<scalar_t*>, std::vector<scalar_t*>>;

        using BatchedTilesBuckets = slate::Map< mnss_tuple, data_tuple >;

        BatchedTilesBuckets tilesBuckets;

        for (auto iter = tile_set.begin(); iter != tile_set.end(); iter++) {
            int64_t i = std::get<0>(*iter);
            int64_t j = std::get<1>(*iter);

            auto tile = storage_->at(globalIndex(i, j, device)).tile_;

            // if we need to convert layout
            if ( tile->layout() != layout ) {
                // make sure tile is transposable
                if (! tile->isTransposable() ) {
                    storage_->tileMakeTransposable(tile);
                }

                // prepare tile for batch conversion
                if (tile->extended()) {
                    tile->layoutSetFrontDataExt(layout != tile->userLayout());
                }
                int64_t target_stride = layout == Layout::ColMajor ?
                                        tile->mb() :
                                        tile->nb();

                // bucket index
                mnss_tuple mns = {tile->mb(), tile->nb(),
                                  tile->extended(),
                                  tile->stride(),
                                  tile->mb() != tile->nb() ?
                                    (tile->extended() ?
                                        tile->layoutBackStride() :
                                        target_stride) :
                                    0};

                // add this tile's data to the corrsponding bucket of batch array
                tilesBuckets[mns].first.push_back(tile->data());

                // if rectangular, prepare a workspace
                if (tile->mb() != tile->nb()) {
                    if (tile->extended())
                        tilesBuckets[mns].second.push_back(tile->layoutBackData());
                    else
                        tilesBuckets[mns].second.push_back(storage_->allocWorkspaceBuffer(device));
                }

                // adjust stride if need be
                if (! tile->extended()) {
                    tile->stride(target_stride);
                }

                // adjust layout
                tile->layout(layout);
            }
        }

        cudaStream_t stream = compute_stream(device);
        slate_cuda_call(
            cudaSetDevice(device));

        // for each bucket
        for (auto bucket = tilesBuckets.begin(); bucket != tilesBuckets.end(); bucket++) {

            scalar_t** array_dev  = this->a_array_device(device);
            scalar_t** work_array_dev  = this->b_array_device(device);

            int64_t batch_count = bucket->second.first.size();
            assert(batch_count <=  this->batchArraySize());

            int64_t mb       = std::get<0>(bucket->first);
            int64_t nb       = std::get<1>(bucket->first);
            int64_t extended = std::get<2>(bucket->first);
            int64_t stride   = std::get<3>(bucket->first);
            int64_t work_stride = std::get<4>(bucket->first);

            // todo: should we handle batch of size one differently?
            //       will need to store the (i,j) tuple with the batch array!
            // if (batch_count == 1) {
            //     tileLayoutConvert(i, j, device, layout);
            // }
            // else
            {
                slate_cuda_call(
                    cudaMemcpyAsync(array_dev, bucket->second.first.data(),
                                    sizeof(scalar_t*)*batch_count,
                                    cudaMemcpyHostToDevice,
                                    stream));

                if (mb == nb) {
                    // in-place transpose
                    device::transpose_batch(nb,
                                            array_dev, stride,
                                            batch_count, stream);
                }
                else {
                    // rectangular tiles: out-of-place transpose
                    slate_cuda_call(
                        cudaMemcpyAsync(work_array_dev, bucket->second.second.data(),
                                        sizeof(scalar_t*)*batch_count,
                                        cudaMemcpyHostToDevice,
                                        stream));

                    device::transpose_batch(layout == Layout::ColMajor ? nb : mb,
                                            layout == Layout::ColMajor ? mb : nb,
                                            array_dev, stride,
                                            work_array_dev, work_stride,
                                            batch_count, stream);

                    if (! extended) {
                        // copy back to data buffer
                        device::gecopy( layout == Layout::ColMajor ? mb : nb,
                                        layout == Layout::ColMajor ? nb : mb,
                                        work_array_dev, work_stride,
                                        array_dev, work_stride,
                                        batch_count, stream);
                    }
                }
            }

            // release workspace buffer if allocated
            if ((mb != nb) && (! extended)) {
                slate_cuda_call(
                    cudaStreamSynchronize(stream));
                for (auto iter = bucket->second.second.begin(); iter != bucket->second.second.end(); iter++) {
                    storage_->releaseWorkspaceBuffer(*iter, device);
                }
            }
        }
        slate_cuda_call(
            cudaStreamSynchronize(stream));
    }
}

//------------------------------------------------------------------------------
/// Converts all existing tile instances on 'device' into 'layout'
/// Operates in batch mode.
///
/// @param[in] device
///     Tiles' host or device ID.
///
/// @param[in] layout
///     Intended Layout of tiles:
///     - Layout::ColMajor or
///     - Layout::RowMajor.
///
// todo: override on BaseTrapezoidMatrix and BandMatrix
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileLayoutConvert(int device, Layout layout, bool reset)
{
    std::set<ij_tuple> tiles_set;
    for (int64_t j = 0; j < nt(); ++j) {
        for (int64_t i = 0; i < mt(); ++i) {
            // if ( tileIsLocal(i, j) && device == tileDevice(i, j) && tileExists(i, j, device))
            if ( tileExists(i, j, device)) {
                tiles_set.insert({i, j});
            }
        }
    }

    tileLayoutConvert(tiles_set, device, layout, reset);
}

//------------------------------------------------------------------------------
/// Converts all existing tile instances on available devices into 'layout'.
/// Host tiles are not affected.
/// Tiles should exist already on devices.
/// Operates in batch mode.
///
/// @param[in] layout
///     Intended Layout of tiles:
///     - Layout::ColMajor or
///     - Layout::RowMajor.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileLayoutConvertOnDevices(Layout layout, bool reset)
{
    std::vector< std::set<ij_tuple> > tiles_set(num_devices());
    for (int64_t j = 0; j < nt(); ++j) {
        for (int64_t i = 0; i < mt(); ++i) {
            for (int d = 0; d < num_devices(); ++d) {
                if (tileExists(i, j, d)) {
                    tiles_set[d].insert({i, j});
                }
            }
        }
    }
    // todo: omp tasks?
    for (int d = 0; d < num_devices(); ++d) {
        if (! tiles_set[d].empty())
            tileLayoutConvert(tiles_set[d], d, layout, reset);
    }
}

//------------------------------------------------------------------------------
/// Converts all origin tiles into current matrix-layout.
/// Operates in batch mode.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::resetTilesLayout()
{
    std::set<ij_tuple> tiles_set_host;
    std::vector< std::set<ij_tuple> > tiles_set_dev(num_devices());

    for (int64_t i = 0; i < mt(); ++i) {
        for (int64_t j = 0; j < nt(); ++j) {
            if ( tileIsLocal(i, j) ) {

                auto tile = tileUpdateOrigin(i, j);
                if (tile->layout() != this->layout() ) {
                    if (! tile->isTransposable() ) {
                        // todo: make transposable
                    }
                    if (tile->device() == host_num_) {
                        tiles_set_host.insert({i, j});
                    }
                    else{
                        tiles_set_dev[tile->device()].insert({i, j});
                    }
                }
            }
        }
    }

    if (! tiles_set_host.empty()) {
        tileConvertLayout(tiles_set_host, host_num_, this->layout());
    }
    // todo: omp tasks?
    for (int d = 0; d < num_devices(); ++d) {
        if (! tiles_set_dev[d].empty()) {
            tileConvertLayout(tiles_set_dev[d], d, this->layout());
        }
    }
}

//------------------------------------------------------------------------------
/// Puts all MPI ranks that have tiles in the matrix into the set.
///
/// @param[out] bcast_set
///     On output, set of MPI ranks that this sub-matrix has tiles on.
///
// todo: pass bcast_set by reference
template <typename scalar_t>
void BaseMatrix<scalar_t>::getRanks(std::set<int>* bcast_set) const
{
    for (int64_t i = 0; i < mt(); ++i)
        for (int64_t j = 0; j < nt(); ++j)
            bcast_set->insert(tileRank(i, j));
}

//------------------------------------------------------------------------------
/// Returns the origin tile instance of tile(i, j)
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
template <typename scalar_t>
Tile<scalar_t>* BaseMatrix<scalar_t>::originTile(int64_t i, int64_t j)
{
    // find on host
    auto iter = storage_->find(globalIndex(i, j, host_num_));
    if (iter != storage_->end() && iter->second.tile_->origin() ){
        return iter->second.tile_;
    }
    else {
        iter = storage_->find(globalIndex(i, j, tileDevice(i, j)));
        if (iter != storage_->end() && iter->second.tile_->origin() ){
            return iter->second.tile_;
        }
    }
}

//------------------------------------------------------------------------------
/// Puts all devices that have local tiles in the matrix into the set.
///
/// @param[out] dev_set
///     On output, set of device IDs that this sub-matrix has local tiles on.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::getLocalDevices(std::set<int>* dev_set) const
{
    for (int64_t i = 0; i < mt(); ++i)
        for (int64_t j = 0; j < nt(); ++j)
            if (tileIsLocal(i, j))
                dev_set->insert(tileDevice(i, j));
}

//------------------------------------------------------------------------------
/// Returns number of local tiles in this matrix.
/// Used for the lifespan of a temporary tile that updates every tile in
/// the matrix.
///
template <typename scalar_t>
int64_t BaseMatrix<scalar_t>::numLocalTiles() const
{
    // Find the tile's lifespan.
    int64_t life = 0;
    for (int64_t i = 0; i < mt(); ++i)
        for (int64_t j = 0; j < nt(); ++j)
            if (tileIsLocal(i, j))
                ++life;

    return life;
}

//------------------------------------------------------------------------------
/// [internal]
/// Returns index {i, j} in global matrix as a tuple,
/// taking into account the local offset and transpose.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
template <typename scalar_t>
std::tuple< int64_t, int64_t >
    BaseMatrix<scalar_t>::globalIndex(int64_t i, int64_t j) const
{
    assert(0 <= i && i < mt());
    assert(0 <= j && j < nt());
    if (op_ == Op::NoTrans)
        return { ioffset_ + i, joffset_ + j };
    else
        return { ioffset_ + j, joffset_ + i };
}

//------------------------------------------------------------------------------
/// [internal]
/// Returns index {i, j, dev} in global matrix as a tuple,
/// taking into account the local offset and transpose.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's device ID.
///
template <typename scalar_t>
std::tuple<int64_t, int64_t, int>
    BaseMatrix<scalar_t>::globalIndex(int64_t i, int64_t j, int device) const
{
    assert(0 <= i && i < mt());
    assert(0 <= j && j < nt());
    assert(device == host_num_ || (0 <= device && device < num_devices_));
    if (op_ == Op::NoTrans)
        return { ioffset_ + i, joffset_ + j, device };
    else
        return { ioffset_ + j, joffset_ + i, device };
}

//------------------------------------------------------------------------------
template <typename scalar_t>
int BaseMatrix<scalar_t>::host_num_ = -1;

template <typename scalar_t>
int BaseMatrix<scalar_t>::num_devices_ = 0;

//------------------------------------------------------------------------------
// from ScaLAPACK's indxg2l
// todo: where to put utilities like this?
inline int64_t indexGlobal2Local(int64_t i, int64_t nb, int num_ranks)
{
    return nb*(i/(nb*num_ranks)) + (i % nb);
}

} // namespace slate

#endif // SLATE_BASE_MATRIX_HH
