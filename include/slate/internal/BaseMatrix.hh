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

    TileEntry<scalar_t>& tileInsertWorkspace(int64_t i, int64_t j, int device=host_num_);

    // Returns tile(i, j)'s state on device (defaults to host).
    MOSI tileState(int64_t i, int64_t j, int device=host_num_);

    // Sets tile(i, j)'s state on device.
    void tileState(int64_t i, int64_t j, int device, MOSI mosi);

    // Sets tile(i, j)'s state on host.
    void tileState(int64_t i, int64_t j, MOSI mosi)
    {
        tileState(i, j, host_num_, mosi);
    }

    // Returns whether tile(i, j) is on hold on device (defaults to host).
    bool tileOnHold(int64_t i, int64_t j, int device=host_num_);

    /// Unsets tile(i, j)'s hold on device.
    void tileUnsetHold(int64_t i, int64_t j, int device=host_num_);

    /// Marks tile(i, j) as Modified on device.
    /// Other instances will be invalidated.
    /// Unless permissive, asserts if other instances are in Modified state.
    void tileModified(int64_t i, int64_t j, int device=host_num_, bool permissive=false);

    /// Gets tile(i, j) for reading on device.
    /// Will copy-in the tile if it does not exist or its state is Invalid.
    /// Sets tile state to Shared if copied-in.
    /// Updates source tile's state to shared if copied-in.
    void tileGetForReading(int64_t i, int64_t j, int device=host_num_,
                            LayoutConvert layout=LayoutConvert::ColMajor);

    /// Gets all local tiles for reading on device.
    void tileGetAllForReading(int device=host_num_);

    /// Gets all local tiles for reading on corresponding devices.
    void tileGetAllForReadingOnDevices();

    /// Gets tile(i, j) for writing on device.
    /// Sets state to Modified.
    /// Will copy tile in if not exists or state is Invalid.
    /// Other instances will be invalidated.
    void tileGetForWriting(int64_t i, int64_t j, int device=host_num_,
                            LayoutConvert layout=LayoutConvert::ColMajor);

    /// Gets all local tiles for writing on device.
    void tileGetAllForWriting(int device=host_num_);

    /// Gets all local tiles for writing on corresponding devices.
    void tileGetAllForWritingOnDevices();

    /// Gets tile(i, j) on device and marks it as OnHold.
    /// Will copy tile in if it does not exist or its state is Invalid.
    /// Updates the source tile's state to Shared if copied-in.
    void tileGetAndHold(int64_t i, int64_t j, int device=host_num_,
                            LayoutConvert layout=LayoutConvert::ColMajor);

    /// Gets all local tiles on device and marks them as OnHold.
    void tileGetAndHoldAll(int device=host_num_);

    /// Gets all local tiles on corresponding devices and marks them as OnHold.
    void tileGetAndHoldAllOnDevices();

    /// Updates the origin instance of tile(i, j) if not MOSI::Shared
    void tileUpdateOrigin(int64_t i, int64_t j);

    /// Updates all origin instances of tiles if not MOSI::Shared
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

    template <Target target = Target::Host>
    void tileSend(int64_t i, int64_t j, int dst_rank, int tag = 0);

    template <Target target = Target::Host>
    void tileRecv(int64_t i, int64_t j, int dst_rank, int tag = 0,
                  Layout layout = Layout::ColMajor);

    template <Target target = Target::Host>
    void tileBcast(int64_t i, int64_t j, BaseMatrix const& B, int tag = 0, int life_factor = 1,
                   Layout layout = Layout::ColMajor);

    template <Target target = Target::Host>
    void listBcast(BcastList& bcast_list, int tag = 0,
                   Layout layout = Layout::ColMajor, int64_t life_factor = 1);

    template <Target target = Target::Host>
    void listReduce(ReduceList& reduce_list, int tag = 0);

protected:
    void tileBcastToSet(int64_t i, int64_t j, std::set<int> const& bcast_set);
    void tileBcastToSet(int64_t i, int64_t j, std::set<int> const& bcast_set,
                        int radix, int tag);

    // todo: should this be private?
    void tileReduceFromSet(int64_t i, int64_t j,
                           std::set<int> const& reduce_set, int radix, int tag);

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
      storage_(nullptr)
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
      mpi_comm_(mpi_comm)
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
    auto tileEntry = storage_->tileInsert(index, TileKind::SlateOwned);
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
    int64_t i, int64_t j, int device)
{
    auto index = globalIndex(i, j, device);
    return storage_->tileInsert(index, TileKind::Workspace);
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
    auto tileEntry = storage_->tileInsert(index, data, ld); // TileKind::UserOwned
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
    if (iter != storage_->end()) {
        // auto tileEntry = storage_->at(globalIndex(i, j, device));
        if (iter->second.tile_->workspace() &&
            (iter->second.stateOn(MOSI::OnHold) || iter->second.stateOn(MOSI::Modified)) )
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
/// Sets tile(i, j)'s state on device (defaults to host).
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
        tileGetForReading(i, j);
        at(i, j).send(dst_rank, mpiComm(), tag);
    }
}

//------------------------------------------------------------------------------
/// Receive tile {i, j} of op(A) to the given MPI rank.
/// Tile is allocated as workspace with life = 1 if it doesn't yet exist,
/// or 1 is added to life if it does exist.
/// Source rank must call tileSend().
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
/// @param[in] tag
///     MPI tag, default 0.
///
/// @param[in] layout
///     Layout of final tile.
///     - Layout::ColMajor (default) or
///     - Layout::RowMajor.
///
template <typename scalar_t>
template <Target target>
void BaseMatrix<scalar_t>::tileRecv(
    int64_t i, int64_t j, int src_rank, int tag, Layout layout)
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

        // Receive data.
        at(i, j).recv(src_rank, mpiComm(), tag);

        tileModified(i, j, hostNum(), true);

        // Copy to devices.
        if (target == Target::Devices) {
            #pragma omp task
            {
                tileGetForReading(i, j, tileDevice(i, j), LayoutConvert(layout));
                // todo: handle layout
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Send tile {i, j} of op(A) to all MPI ranks in matrix B.
/// If target is Devices, also copies tile to all devices on each MPI rank.
/// This should be called by at least all ranks with local tiles in B;
/// ones that do not have any local tiles are excluded from the broadcast.
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
/// @param[in] tag
///     MPI tag, default 0.
///
/// @param[in] layout
///     Layout of final tile.
///     - Layout::ColMajor (default) or
///     - Layout::RowMajor.
///
template <typename scalar_t>
template <Target target>
void BaseMatrix<scalar_t>::tileBcast(
    int64_t i, int64_t j, BaseMatrix<scalar_t> const& B, int tag, int life_factor, Layout layout)
{
    BcastList bcast_list_B;
    bcast_list_B.push_back({i, j, {B}});
    listBcast<target>(bcast_list_B, tag, layout, life_factor);
}

//------------------------------------------------------------------------------
/// Send tile {i, j} of op(A) to all MPI ranks in the list of submatrices
/// bcast_list.
///
/// @tparam target
///     Destination to target; either Host (default) or Device.
///
/// @param[in] bcast_list
///     List of submatrices defining the MPI ranks to send to.
///     Usually it is the portion of the matrix to be updated by tile {i, j}.
///
/// @param[in] tag
///     MPI tag, default 0.
///
/// @param[in] layout
///     Layout of final tile.
///     - Layout::ColMajor (default) or
///     - Layout::RowMajor.
///
template <typename scalar_t>
template <Target target>
void BaseMatrix<scalar_t>::listBcast(
    BcastList& bcast_list, int tag, Layout layout, int64_t life_factor)
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
            tileBcastToSet(i, j, bcast_set, 2, tag);
        }

        // Copy to devices.
        // TODO: should this be inside above if-then?
        if (target == Target::Devices) {
            std::set<int> dev_set;
            for (auto submatrix : submatrices_list)
                submatrix.getLocalDevices(&dev_set);

            #pragma omp task
            {
                for (auto device : dev_set)
                    tileGetForReading(i, j, device, LayoutConvert(layout));
                    // todo: handle layout
            }
        }
    }

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
///
template <typename scalar_t>
template <Target target>
void BaseMatrix<scalar_t>::listReduce(ReduceList& reduce_list, int tag)
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
            tileReduceFromSet(i, j, reduce_set, 2, tag);

            // If not the tile owner.
            if (! tileIsLocal(i, j)) {

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
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileBcastToSet(
    int64_t i, int64_t j, std::set<int> const& bcast_set, int radix, int tag)
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
        at(i, j).recv(new_vec[recv_from.front()], mpi_comm_, tag);
        tileModified(i, j);
    }

    // Forward.
    for (int dst : send_to)
        at(i, j).send(new_vec[dst], mpi_comm_, tag);
}

//------------------------------------------------------------------------------
/// [internal]
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileReduceFromSet(
    int64_t i, int64_t j, std::set<int> const& reduce_set, int radix, int tag)
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

    std::vector<scalar_t> data(tileMb(i)*tileNb(j));
    Tile<scalar_t> tile(tileMb(i), tileNb(j), &data[0], tileMb(i), host_num_, TileKind::Workspace);

    // Receive, accumulate.
    for (int src : recv_from) {
        // Receive.
        tile.recv(new_vec[src], mpi_comm_, tag);
        // Accumulate.
        axpy(scalar_t(1.0), tile, at(i, j));
    }

    // Forward.
    if (! send_to.empty())
        at(i, j).send(new_vec[send_to.front()], mpi_comm_, tag);
}

//------------------------------------------------------------------------------
/// Gets tile(i, j) for reading on device.
/// Will copy-in the tile if it does not exist or its state is Invalid.
/// Sets tile state to Shared if copied-in.
/// Finds a source tile whose state is valid (Modified|Shared) by
/// looping on existing tile instances.
/// Updates source tile's state to shared if copied-in.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] dst_device
///     Tile's destination: host or device ID, defaults to host.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetForReading(int64_t i, int64_t j, int dst_device,
                                             LayoutConvert layout)
{
    TileEntry<scalar_t> *dst_tileEntry = nullptr, *src_tileEntry = nullptr;
    do {
        // find tile on destination
        auto dst_iter = storage_->find(globalIndex(i, j, dst_device));
        if (dst_iter == storage_->end()) {
            // Create a copy on the destination.
            tileInsertWorkspace(i, j, dst_device);
            dst_iter = storage_->find(globalIndex(i, j, dst_device));
        }
        dst_tileEntry = &dst_iter->second;

        if (dst_tileEntry->getState() != MOSI::Invalid)
            break;

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
        assert(src_device != invalid_dev);

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

        // Change ColMajor <=> RowMajor if needed.
        if (layout != LayoutConvert::None &&
            dst_tileEntry->tile_->layout() != Layout(layout)) {
            if (dst_device == host_num_) {
                convert_layout(dst_tileEntry->tile_);
            }
            else {
                convert_layout(dst_tileEntry->tile_, comm_stream(dst_device));
            }
        }
    } while(0);
}


//------------------------------------------------------------------------------
/// Gets tile(i, j) for writing on device.
/// Sets state to Modified.
/// Will copy-in the tile if it does not exist or its state is Invalid.
/// Other instances will be invalidated.
/// Finds a source tile whose state is valid (Modified|Shared) by
/// scanning existing tile instances.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] dst_device
///     Tile's destination: host or device ID, defaults to host.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetForWriting(int64_t i, int64_t j, int device, LayoutConvert layout)
{
    tileGetForReading(i, j, device, layout);
    tileModified(i, j, device);
}


//------------------------------------------------------------------------------
/// Gets tile(i, j) on device and marks it as OnHold.
/// Will copy tile in if it does not exist or its state is Invalid.
/// Updates the source tile's state to Shared if copied-in.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] dst_device
///     Tile's destination: host or device ID, defaults to host.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetAndHold(int64_t i, int64_t j, int device, LayoutConvert layout)
{
    tileGetForReading(i, j, device, layout);

    auto tileIter = storage_->find(globalIndex(i, j, device));
    assert(tileIter != storage_->end());

    tileIter->second.setState(MOSI::OnHold);
}

//------------------------------------------------------------------------------
/// Gets all local tiles for reading on device.
///
/// @param[in] device
///     Tile's destination: host or device ID, defaults to host.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetAllForReading(int device)
{
    for (int64_t j = 0; j < nt(); ++j)
        for (int64_t i = 0; i < mt(); ++i)
            if (tileIsLocal(i, j))
                tileGetForReading(i, j, device);
}

//------------------------------------------------------------------------------
/// Gets all local tiles for reading on device.
///
/// @param[in] device
///     Tile's destination: host or device ID, defaults to host.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetAllForWriting(int device)
{
    for (int64_t j = 0; j < nt(); ++j)
        for (int64_t i = 0; i < mt(); ++i)
            if (tileIsLocal(i, j))
                tileGetForWriting(i, j, device);
}

//------------------------------------------------------------------------------
/// Gets all local tiles on device and marks them as OnHold.
///
/// @param[in] device
///     Tile's destination: host or device ID, defaults to host.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetAndHoldAll(int device)
{
    for (int64_t j = 0; j < nt(); ++j)
        for (int64_t i = 0; i < mt(); ++i)
            if (tileIsLocal(i, j))
                tileGetAndHold(i, j, device);
}

//------------------------------------------------------------------------------
/// Gets all local tiles for reading on corresponding devices.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetAllForReadingOnDevices()
{
    for (int64_t j = 0; j < nt(); ++j)
        for (int64_t i = 0; i < mt(); ++i)
            if (tileIsLocal(i, j))
                tileGetForReading(i, j, tileDevice(i, j));
}

//------------------------------------------------------------------------------
/// Gets all local tiles for reading on corresponding devices.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetAllForWritingOnDevices()
{
    for (int64_t j = 0; j < nt(); ++j)
        for (int64_t i = 0; i < mt(); ++i)
            if (tileIsLocal(i, j))
                tileGetForWriting(i, j, tileDevice(i, j));
}

//------------------------------------------------------------------------------
/// Gets all local tiles on corresponding devices and marks them as OnHold.
//
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetAndHoldAllOnDevices()
{
    for (int64_t j = 0; j < nt(); ++j)
        for (int64_t i = 0; i < mt(); ++i)
            if (tileIsLocal(i, j))
                tileGetAndHold(i, j, tileDevice(i, j));
}

//------------------------------------------------------------------------------
/// Updates the origin instance of tile(i, j) if not MOSI::Shared
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileUpdateOrigin(int64_t i, int64_t j)
{
    // find on host
    auto iter = storage_->find(globalIndex(i, j, host_num_));
    if (iter != storage_->end() && iter->second.tile_->origin()) {
        if ( iter->second.stateOn(MOSI::Invalid) )
            tileGetForReading(i, j);
    }
    else {
        iter = storage_->find(globalIndex(i, j, tileDevice(i, j)));
        if (iter != storage_->end() && iter->second.tile_->origin()) {
            if ( iter->second.stateOn(MOSI::Invalid) )
                tileGetForReading(i, j, tileDevice(i, j));
        }
    }
}

//------------------------------------------------------------------------------
/// Updates all origin instances of local tiles if not MOSI::Shared
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
