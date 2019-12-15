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
#include "slate/internal/Memory.hh"
#include "slate/internal/device.hh"
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

    using ij_tuple = typename MatrixStorage<scalar_t>::ij_tuple;

    friend class Debug;

    // Make every class BaseMatrix<T2> a friend of BaseMatrix<scalar_t>.
    template <typename T2>
    friend class BaseMatrix;

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

    BaseMatrix(int64_t m, int64_t n,
               std::function<int64_t (int64_t i)>& inTileMb,
               std::function<int64_t (int64_t j)>& inTileNb,
               std::function<int (ij_tuple ij)>& inTileRank,
               std::function<int (ij_tuple ij)>& inTileDevice,
               MPI_Comm mpi_comm);

    BaseMatrix(int64_t m, int64_t n, int64_t mb, int64_t nb,
               int p, int q, MPI_Comm mpi_comm);

    /// With mb = nb.
    BaseMatrix(int64_t m, int64_t n, int64_t nb,
               int p, int q, MPI_Comm mpi_comm)
        : BaseMatrix(m, n, nb, nb, p, q, mpi_comm)
    {}

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

    template <typename out_scalar_t>
    BaseMatrix<out_scalar_t> baseEmptyLike(int64_t mb, int64_t nb,
                                           Op deepOp);

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

    Tile<scalar_t>* tileInsertWorkspace(int64_t i, int64_t j, int device, Layout layout);
    Tile<scalar_t>* tileInsertWorkspace(int64_t i, int64_t j, int device)
    {
        return tileInsertWorkspace(i, j, device, layout_);
    }
    Tile<scalar_t>* tileInsertWorkspace(int64_t i, int64_t j, Layout layout)
    {
        return tileInsertWorkspace(i, j, host_num_, layout);
    }
    Tile<scalar_t>* tileInsertWorkspace(int64_t i, int64_t j)
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

    void tileCopyDataLayout(Tile<scalar_t>* src_tile,
                            Tile<scalar_t>* dst_tile,
                            Layout target_layout,
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

    void tileAcquire(int64_t i, int64_t j, int device, Layout layout);

    void tileAcquire(int64_t i, int64_t j, Layout layout)
    {
        tileAcquire(i, j, host_num_, layout);
    }

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
    void tileGetForReading(std::set<ij_tuple>& tile_set, LayoutConvert layout,
                            int from_device);

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

    Tile<scalar_t>* tileUpdateOrigin(int64_t i, int64_t j);

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
        return storage_->at(globalIndex(i, j, device)).tile()->layout();
    }

    /// Sets Layout of tile(i, j, device)
    void tileLayout(int64_t i, int64_t j, int device, Layout layout)
    {
        storage_->at(globalIndex(i, j, device)).tile()->layout(layout);
    }

    /// Sets Layout of tile(i, j, host)
    void tileLayout(int64_t i, int64_t j, Layout layout)
    {
        storage_->at(globalIndex(i, j, host_num_)).tile()->layout(layout);
    }

    bool tileLayoutIsConvertible(int64_t i, int64_t j, int device=host_num_);

    void tileLayoutConvert(int64_t i, int64_t j, int device, Layout layout,
                           bool reset = false, bool async = false);
    /// Convert layout of tile(i, j) to layout on host, optionally reset
    void tileLayoutConvert(int64_t i, int64_t j, Layout layout,
                           bool reset = false, bool async = false)
    {
        tileLayoutConvert(i, j, host_num_, layout, reset, async);
    }
    void tileLayoutConvert(std::set<ij_tuple>& tile_set, int device,
                           Layout layout, bool reset = false);
    /// Convert layout of a set of tiles to layout on host, optionally reset
    void tileLayoutConvert(std::set<ij_tuple>& tile_set, Layout layout, bool reset = false)
    {
        tileLayoutConvert(tile_set, host_num_, layout, reset);
    }
    void tileLayoutConvert(int device, Layout layout, bool reset = false);
    void tileLayoutConvertOnDevices(Layout layout, bool reset = false);

    void tileLayoutReset(int64_t i, int64_t j, int device, Layout layout);
    void tileLayoutReset(int64_t i, int64_t j, Layout layout)
    {
        tileLayoutReset(i, j, host_num_, layout);
    }
    void tileLayoutReset(std::set<ij_tuple>& tile_set, int device, Layout layout);
    void tileLayoutReset(std::set<ij_tuple>& tile_set, Layout layout)
    {
        tileLayoutReset(tile_set, host_num_, layout);;
    }
    void tileLayoutReset();

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

    /// Allocates batch arrays.
    /// Matrix classes override this with versions that can also allocate based
    /// on the number of local tiles.
    ///
    /// @param[in] batch_size
    ///     On exit, size of batch arrays >= batch_size >= 0.
    ///
    void allocateBatchArrays(int64_t batch_size=0, int64_t num_arrays=1)
    {
        storage_->allocateBatchArrays(batch_size, num_arrays);
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
    scalar_t** array_host(int device, int64_t i=0)
    {
        assert(i >= 0);
        std::vector< scalar_t** >& array = storage_->array_host_.at(i);
        return array.at(device);
    }

    //--------------------------------------------------------------------------
    /// @return batch arrays for the A, B, or C matrices, on device
    scalar_t** array_device(int device, int64_t i=0)
    {
        assert(i >= 0);
        std::vector< scalar_t** >& array = storage_->array_dev_.at(i);
        return array.at(device);
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
    //--------------------------------------------------------------------------
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
/// Construct matrix with mt block rows and nt block columns, such that
///     sum_{i = 0}^{mt-1} tileMb(i) >= m,
///     sum_{j = 0}^{nt-1} tileNb(j) >= n,
/// where tileMb, tileNb, tileRank, tileDevice are given as functions.
/// No tiles are allocated. Creates empty matrix storage.
///
/// @param[in] m
///     Number of rows of the matrix. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix. n >= 0
///
/// @param[in] inTileMb
///     Function that takes block-row index, returns block-row size.
///
/// @param[in] inTileNb
///     Function that takes block-col index, returns block-col size.
///
/// @param[in] inTileRank
///     Function that takes tuple of { block-row, block-col } indices,
///     returns MPI rank for that tile.
///
/// @param[in] inTileDevice
///     Function that takes tuple of { block-row, block-col } indices,
///     returns local GPU device ID for that tile.
///
/// @param[in] mpi_comm
///     MPI communicator to distribute matrix across.
///     p*q == MPI_Comm_size(mpi_comm).
///
template <typename scalar_t>
BaseMatrix<scalar_t>::BaseMatrix(
    int64_t m, int64_t n,
    std::function<int64_t (int64_t i)>& inTileMb,
    std::function<int64_t (int64_t j)>& inTileNb,
    std::function<int (ij_tuple ij)>& inTileRank,
    std::function<int (ij_tuple ij)>& inTileDevice,
    MPI_Comm mpi_comm)
    : row0_offset_(0),
      col0_offset_(0),
      ioffset_(0),
      joffset_(0),
      uplo_(Uplo::General),
      op_(Op::NoTrans),
      storage_(std::make_shared< MatrixStorage< scalar_t > >(
          inTileMb, inTileNb, inTileRank, inTileDevice, mpi_comm)),
      mpi_comm_(mpi_comm),
      layout_(Layout::ColMajor)
{
    // Count number of block rows.
    mt_ = 0;
    int64_t ii = 0;  // row index (not block row)
    while (ii < m) {
        last_mb_ = std::min(inTileMb(mt_), m - ii);
        assert(last_mb_ != 0);
        ii += last_mb_;
        ++mt_;
    }

    // Count number of block cols.
    nt_ = 0;
    int64_t jj = 0;  // col index (not block col)
    while (jj < n) {
        last_nb_ = std::min(inTileNb(nt_), n - jj);
        assert(last_nb_ != 0);
        jj += last_nb_;
        ++nt_;
    }

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
/// Construct matrix with
///     mt = ceil( m / mb ) block rows and
///     nt = ceil( n / nb ) block columns,
/// with fixed mb-by-nb tile size and 2D block cyclic distribution.
/// No tiles are allocated. Creates empty matrix storage.
///
/// @param[in] m
///     Number of rows of the matrix. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix. n >= 0
///
/// @param[in] mb
///     Row block size in 2D block-cyclic distribution. mb > 0.
///
/// @param[in] nb
///     Column block size in 2D block-cyclic distribution. nb > 0.
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
    int64_t m, int64_t n, int64_t mb, int64_t nb, int p, int q, MPI_Comm mpi_comm)
    : row0_offset_(0),
      col0_offset_(0),
      last_mb_(m % mb == 0 ? mb : m % mb),
      last_nb_(n % nb == 0 ? nb : n % nb),
      ioffset_(0),
      joffset_(0),
      mt_(ceildiv(m, mb)),
      nt_(ceildiv(n, nb)),
      uplo_(Uplo::General),
      op_(Op::NoTrans),
      storage_(std::make_shared< MatrixStorage< scalar_t > >(
          m, n, mb, nb, p, q, mpi_comm)),
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
        // only if first tile remains in sub we have to keep row/col offset
        if (i1 > 0) {
            row0_offset_ = 0;
        }
        if (j1 > 0) {
            col0_offset_ = 0;
        }
    }
    else {
        last_nb_ = tileMb(i2);
        last_mb_ = tileNb(j2);
        ioffset_ += j1;
        joffset_ += i1;
        mt_ = j2 - j1 + 1;
        nt_ = i2 - i1 + 1;
        // only if first tile remains in sub we have to keep row/col offset
        if (j1 > 0) {
            row0_offset_ = 0;
        }
        if (i1 > 0) {
            col0_offset_ = 0;
        }
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
///
template <typename scalar_t>
template <typename out_scalar_t>
BaseMatrix<out_scalar_t> BaseMatrix<scalar_t>::baseEmptyLike(
    int64_t mb, int64_t nb, Op deepOp)
{
    // tileMb, tileNb are of A instead of op(A).
    auto newMb = this->storage_->tileMb;
    auto newNb = this->storage_->tileNb;

    // m, n, mt, nt are of op(A).
    int64_t m  = this->m();
    int64_t n  = this->n();
    int64_t mt = this->mt();
    int64_t nt = this->nt();

    // Undo transpose to get m, n, mt, nt, mb, nb of A instead of op(A).
    if (this->op() != Op::NoTrans) {
        std::swap(m, n);
        std::swap(mt, nt);
        std::swap(mb, nb);
    }

    // Override mb, nb if requested.
    if (mb != 0) {
        newMb = [mb](int64_t i) { return mb; };
        m = mb * mt;
    }
    if (nb != 0) {
        newNb = [nb](int64_t j) { return nb; };
        n = nb * nt;
    }

    // Adjust size to include parent matrix outside this sub-matrix.
    int64_t ioffset = this->ioffset();
    int64_t joffset = this->joffset();
    int64_t parent_m = m;
    for (int i = 0; i < ioffset; ++i) {
        parent_m += newMb(i);
    }
    int64_t parent_n = n;
    for (int j = 0; j < joffset; ++j) {
        parent_n += newNb(j);
    }

    // Create new parent matrix B.
    BaseMatrix<out_scalar_t> B;
    if (deepOp == Op::NoTrans) {
        B = BaseMatrix<out_scalar_t>(
            parent_m, parent_n, newMb, newNb,
            this->storage_->tileRank, this->storage_->tileDevice, this->mpiComm());
    }
    else {
        // todo: just swap and redefine newRank? then use above B constructor.
        auto oldRank = this->storage_->tileRank;
        std::function<int (ij_tuple ij)> newRank = [oldRank](ij_tuple ij) {
            int64_t i = std::get<0>(ij);
            int64_t j = std::get<1>(ij);
            return oldRank( ij_tuple({ j, i }) );
        };
        // todo: what about tileDevice?
        B = BaseMatrix<out_scalar_t>(
            parent_n, parent_m, newNb, newMb,  // transposed
            newRank, this->storage_->tileDevice, this->mpiComm());
        std::swap(ioffset, joffset);
        std::swap(mt, nt);
    }

    // Apply operation and return sub-matrix.
    if (this->op() == Op::Trans) {
        B = transpose( B );
        std::swap(ioffset, joffset);
        std::swap(mt, nt);
    }
    else if (this->op() == Op::ConjTrans) {
        B = conj_transpose( B );
        std::swap(ioffset, joffset);
        std::swap(mt, nt);
    }
    B.initSubmatrix(ioffset, ioffset + mt - 1,
                    joffset, joffset + nt - 1);
    return B;
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
    auto tile = *(storage_->at(globalIndex(i, j, device)).tile());

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
    auto& tile_instance = storage_->tileInsert(index, TileKind::SlateOwned, layout_);
    return tile_instance.tile();
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
Tile<scalar_t>* BaseMatrix<scalar_t>::tileInsertWorkspace(
    int64_t i, int64_t j, int device, Layout layout)
{
    auto index = globalIndex(i, j, device);
    auto& tile_instance = storage_->tileInsert(index, TileKind::Workspace, layout);
    return tile_instance.tile();
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
    auto& tile_instance = storage_->tileInsert(index, data, ld, layout_); // TileKind::UserOwned
    return tile_instance.tile();
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
    storage_->release(globalIndex(i, j, device));
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

    return iter->second->at(device).getState();
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

    tileIter->second->at(device).setState(mosi);
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

    return iter->second->at(device).stateOn(MOSI::OnHold);
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

    iter->second->at(device).setState(~MOSI::OnHold);
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
    auto& tile_node = storage_->at(globalIndex(i, j));

    LockGuard guard(tile_node.getLock());

    auto& tile_instance = tile_node[device];

    // if no need to update
    if (tile_instance.stateOn(MOSI::Modified))
        return;

    tile_instance.setState(MOSI::Modified);

    for (int d = hostNum(); d < num_devices(); ++d) {
        if (d != device && tile_node.existsOn(d)) {
            if (! permissive)
                slate_assert(tile_node[d].stateOn(MOSI::Modified) == false);
            tile_node[d].setState(MOSI::Invalid);
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
        // todo: need to acquire read access lock to TileNode(i, j)
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
            LockGuard guard(storage_->getTilesMapLock());
            auto iter = storage_->find(globalIndex(i, j, host_num_));

            int64_t life = 1;
            if (iter == storage_->end())
                tileInsertWorkspace(i, j, host_num_, layout);
            else
                life += tileLife(i, j);
            tileLife(i, j, life);
        }
        else {
            tileAcquire(i, j, layout);
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

    std::vector< std::set<ij_tuple> > tile_set(num_devices());
    int mpi_size;
    MPI_Comm_size(mpiComm(), &mpi_size);

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
                LockGuard guard(storage_->getTilesMapLock());
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
            std::set<int> dev_set;
            for (auto submatrix : submatrices_list)
                submatrix.getLocalDevices(&dev_set);

            if (mpi_size == 1) {
                for (auto device : dev_set)
                    tile_set[device].insert({i, j});
            }
            else {
                // todo: should each read be an omp task instead?
                #pragma omp task
                {
                    for (auto device : dev_set)
                        tileGetForReading(i, j, device, LayoutConvert::None);
                }
            }
        }
    }

    if (target == Target::Devices) {
        if (mpi_size == 1) {
            for (int d = 0; d < num_devices(); ++d) {
                if (! tile_set[d].empty()) {
                    #pragma omp task default(shared)
                    {
                        tileGetForReading(tile_set[d], d, LayoutConvert::None);
                    }
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
/// @deprecated
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
        tileAcquire(i, j, layout);

        at(i, j).recv(new_vec[recv_from.front()], mpi_comm_, layout, tag);
        tileLayout(i, j, layout);
        tileModified(i, j);
    }

    if (! send_to.empty()) {
        // read tile on host memory
        tileGetForReading(i, j, LayoutConvert(layout));
        // Forward using mpi_send()
        // for (int dst : send_to)
        //     at(i, j).send(new_vec[dst], mpi_comm_, tag);

        // Forward using multiple mpi_isend() calls, followed by a waitall
        std::vector<MPI_Request> isend_req_array(send_to.size(), MPI_REQUEST_NULL);
        int idx=0;
        for (int dst : send_to) {
            at(i, j).isend(new_vec[dst], mpi_comm_, tag, &isend_req_array[idx]);
            idx++;
        }
        slate_mpi_call(
            MPI_Waitall(isend_req_array.size(), &isend_req_array[0], MPI_STATUSES_IGNORE));
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

    if (! (send_to.empty() && recv_from.empty())) {
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
/// [internal]
/// Copies data from src_tile into dst_tile and converts into target_layout
/// Handles all cases of:
///     - Square tiles.
///     - Rectangular contiguous tiles.
///     - Rectangular extended tiles.
/// assumes at least one of src_tile and dst_tile is device resident
/// assumes at most one of src_tile and dst_tile is TileKind::UserOwned
/// attempts to make layout conversion on the device whenever possible
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileCopyDataLayout(Tile<scalar_t>* src_tile,
                                              Tile<scalar_t>* dst_tile,
                                              Layout target_layout,
                                              bool async)
{
    bool src_userOwned = ! src_tile->allocated();
    bool dst_userOwned = ! dst_tile->allocated();
    assert(! (src_userOwned && dst_userOwned));

    // todo: handle square
    bool is_square = src_tile->mb() == src_tile->nb();
    // if (! is_square) return;

    // make sure dst_tile can fit the data in its target_layout
    if (   (! is_square)                         // rectangular tile
        && (  dst_userOwned)                     // TileKind::UserOwned
        && (! dst_tile->extended())              // not extended
        && (  dst_tile->layout() != target_layout) ) // but need to store a non-compatible layout
    {
        // extend its data storage
        storage_->tileMakeTransposable(dst_tile);
    }

    scalar_t* src_data = src_tile->data();
    scalar_t* dst_data = dst_tile->data();
    scalar_t* work_data = nullptr;

    bool need_convert = false;
    bool need_workspace = false;
    bool copy_first = false;

    bool src_extended  = src_tile->extended();
    bool dst_extended  = dst_tile->extended();

    int work_device = host_num_;

    // do we need to convert layout? then, setup workspace
    if (src_tile->layout() != target_layout) {
        need_convert = true;

        if (! is_square) {
            if (  (dst_userOwned && ! dst_extended)
               || (src_userOwned && ! src_extended)
               || (! dst_userOwned && ! src_userOwned) ) {

                need_workspace = true;
                if (dst_tile->device() == host_num_) {
                    work_device = src_tile->device();
                }
                else {
                    work_device = dst_tile->device();
                    copy_first = true;
                }
            }
            else if (dst_userOwned && dst_extended) {
                if (target_layout == dst_tile->userLayout()) {
                    dst_tile->layoutSetFrontDataExt(false);
                    dst_data = dst_tile->userData();

                    if (dst_tile->device() == host_num_) {
                        work_device = src_tile->device();
                        need_workspace = true;
                    }
                    else {
                        work_data = dst_tile->extData();
                        work_device = dst_tile->device();
                        copy_first = true;
                    }
                }
                else {
                    dst_tile->layoutSetFrontDataExt(true);
                    dst_data = dst_tile->extData();

                    if (dst_tile->device() == host_num_) {
                        work_device = src_tile->device();
                        need_workspace = true;
                    }
                    else {
                        work_data = dst_tile->userData();
                        work_device = dst_tile->device();
                        copy_first = true;
                    }
                }
            }
            else if (src_userOwned && src_extended) {
                if (src_tile->device() == host_num_) {
                    work_device = dst_tile->device();
                    copy_first = true;
                    need_workspace = true;
                }
                else {
                    work_device = src_tile->device();
                    if (src_tile->layout() == src_tile->userLayout()) {
                        work_data = src_tile->extData();
                    }
                    else {
                        work_data = src_tile->userData();
                    }
                }
            }
        }
    }
    else if (dst_userOwned && dst_extended) {
        if (target_layout == dst_tile->userLayout()) {
            dst_tile->layoutSetFrontDataExt(false);
            dst_data = dst_tile->userData();
        }
        else {
            dst_tile->layoutSetFrontDataExt(true);
            dst_data = dst_tile->extData();
        }
    }

    if (need_convert && (! is_square)) {
        assert(work_device != host_num_);
        slate_cuda_call(
            cudaSetDevice(work_device));
    }

    if (need_workspace) {
        work_data = storage_->allocWorkspaceBuffer(work_device);
        // printf("%p\n", work_data);
    }

    cudaStream_t stream = comm_stream(dst_tile->device() == host_num_ ?
                                      src_tile->device() :
                                      dst_tile->device());
    if (is_square || (! need_convert)) {
        src_tile->copyData(dst_tile, stream, async);
    }

    if (need_convert) {
        if (is_square) {
            dst_tile->layoutConvert(stream, async);
        }
        else if (copy_first) {
            int64_t work_stride = src_tile->stride();

            Tile<scalar_t> work_tile(src_tile->mb(), src_tile->nb(), work_data,
                                     work_stride, work_device,
                                     TileKind::Workspace, src_tile->layout());
            src_tile->copyData(&work_tile, comm_stream(work_device), async);

            if (dst_tile->isContiguous())
                dst_tile->stride( src_tile->layout() == Layout::ColMajor ?
                                  src_tile->nb() :
                                  src_tile->mb());
            int64_t phys_mb = src_tile->layout() == Layout::ColMajor ?
                              src_tile->mb() :
                              src_tile->nb();
            int64_t phys_nb = src_tile->layout() == Layout::ColMajor ?
                              src_tile->nb() :
                              src_tile->mb();
            device::transpose(phys_mb, phys_nb,
                              work_data, work_stride,
                              dst_data, dst_tile->stride(),
                              comm_stream(work_device));
            if (! async)
                slate_cuda_call(
                    cudaStreamSynchronize(comm_stream(work_device)));
        }
        else {
            int64_t work_stride = src_tile->layout() == Layout::ColMajor ?
                                  src_tile->nb() :
                                  src_tile->mb();
            int64_t phys_mb = src_tile->layout() == Layout::ColMajor ?
                              src_tile->mb() :
                              src_tile->nb();
            int64_t phys_nb = src_tile->layout() == Layout::ColMajor ?
                              src_tile->nb() :
                              src_tile->mb();

            device::transpose(phys_mb, phys_nb,
                              src_data, src_tile->stride(),
                              work_data, work_stride,
                              comm_stream(work_device));
            Tile<scalar_t> work_tile(src_tile->mb(), src_tile->nb(), work_data,
                                     work_stride, work_device,
                                     TileKind::Workspace, target_layout);
            if (dst_tile->isContiguous())
                dst_tile->stride(work_stride);

            work_tile.copyData(dst_tile, comm_stream(work_device), async);

            if (! async)
                slate_cuda_call(
                    cudaStreamSynchronize(comm_stream(work_device)));
        }
    }

    if (need_workspace) {
        storage_->releaseWorkspaceBuffer(work_data, work_device);
    }
}

//------------------------------------------------------------------------------
/// Acquire tile(i, j) on device without copying data if not already exists.
/// This is used when the destination tile's data will be overriden.
/// Converts destination Layout to 'layout' param.
/// Assumes the TileNode(i, j) already exists.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] device
///     Tile's destination: host or device ID.
///
/// @param[in] layout
///     Indicates the required Layout of the received tile:
///     - ColMajor: column major.
///     - RowMajor: row major.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileAcquire(int64_t i, int64_t j, int device,
                                       Layout layout)
{
    auto& tile_instance = storage_->tileAcquire(globalIndex(i, j, device), layout);
    auto tile = tile_instance.tile();

    // Change ColMajor <=> RowMajor if needed.
    if (tile->layout() != layout) {
        if (! tile->isTransposable()) {
            storage_->tileMakeTransposable(tile);
        }
        if (tile->extended())
            tile->layoutSetFrontDataExt(tile->layout() == tile->userLayout());
        tile->layout(layout);
        // tileLayoutConvert(i, j, device, Layout(layout), false);
    }
}

//------------------------------------------------------------------------------
/// Gets tile(i, j) on device.
/// Will copy-in the tile if it does not exist or its state is MOSI::Invalid.
/// Finds a source tile whose state is valid (Modified|Shared) by
/// looping on existing tile instances.
/// Updates source tile's state to shared if copied-in.
/// If 'modify' param is true, marks the destination tile as MOSI::Modified,
/// and invalidates other instances. Otherwise, sets destination tile state
/// to MOSI::Shared if copied-in.
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
    // todo: need to acquire read access to the TilesMap
    // LockGuard guard(storage_->tiles_.get_lock());

    TileInstance<scalar_t>* src_tile_instance = nullptr;
    Layout target_layout = Layout::ColMajor; // default value to silence compiler warning
                                             // it will be ovverriden below

    const int invalid_dev = host_num_-1; // invalid device number
    int src_device = invalid_dev;

    // find tile on destination
    auto& tile_node = storage_->at(globalIndex(i, j));
    auto dst_tile_instance = &(tile_node[dst_device]);

    // acquire write access to the (i, j) TileNode
    LockGuard guard(tile_node.getLock());

    if ((! tile_node.existsOn(dst_device)) ||
        (  tile_node[dst_device].getState() == MOSI::Invalid)) {

        // find a valid source (Modified/Shared) tile

        for (int d = hostNum(); d < num_devices(); ++d) {
            if (d != dst_device && tile_node.existsOn(d)) {
                if (tile_node[d].getState() != MOSI::Invalid) {
                    src_device = d;
                    src_tile_instance = &(tile_node[d]);
                    break;
                }
            }
        }

        // todo: find the shortest path / closest source
        // including possibility of device peer-to-peer copy
        if (src_device == invalid_dev) {
            slate_error(std::string("Error copying tile(")
                         + std::to_string(i) + ", " + std::to_string(j)
                         + "), rank(" + std::to_string(this->mpiRank())
                         + "), invalid source " + std::to_string(src_device)
                         + " -> " + std::to_string(dst_device));
        }

        target_layout = layout == LayoutConvert::None ?
                        src_tile_instance->tile()->layout() :
                        Layout(layout);
    }

    if (! tile_node.existsOn(dst_device)) {
        // Create a copy on the destination.
        storage_->tileAcquire(globalIndex(i, j, dst_device), target_layout);
    }

    if (dst_tile_instance->getState() == MOSI::Invalid) {
        // Update the destination tile's data.
        if (dst_device != host_num_ && src_device != host_num_) {
            // todo: device to device copy
            auto host_tile_instance = &(tile_node[host_num_]);
            {
                // LockGuard host_guard(host_tile_instance->getLock());

                if (! tile_node.existsOn(host_num_)) {
                    // Create a copy on the host.
                    storage_->tileAcquire(globalIndex(i, j, host_num_), target_layout);
                }

                if (tile_node[host_num_].getState() == MOSI::Invalid) {
                    tileCopyDataLayout( src_tile_instance->tile(),
                                        host_tile_instance->tile(),
                                        target_layout,
                                        async);
                    host_tile_instance->setState(MOSI::Shared);
                }

                tileCopyDataLayout( host_tile_instance->tile(),
                                    dst_tile_instance->tile(),
                                    target_layout,
                                    async);
            }
        }
        else {
            // LockGuard guard(src_tile_instance->get_lock());
            tileCopyDataLayout( src_tile_instance->tile(),
                                dst_tile_instance->tile(),
                                target_layout,
                                async);
        }

        dst_tile_instance->setState(MOSI::Shared);
        if (src_tile_instance->stateOn(MOSI::Modified))
            src_tile_instance->setState(MOSI::Shared);
    }
    if (modify) {
        tileModified(i, j, dst_device);
    }
    if (hold) {
        dst_tile_instance->setState(MOSI::OnHold);
    }

    // Change ColMajor <=> RowMajor if needed.
    if (layout != LayoutConvert::None &&
        dst_tile_instance->tile()->layout() != Layout(layout)) {
        tileLayoutConvert(i, j, dst_device, Layout(layout), false, async);
    }
}

//------------------------------------------------------------------------------
/// Gets a set of tiles on device.
/// If destination device is host, forwards LayoutConvert param to tileGet()
/// otherwise, calls tileLayoutConvert() to process layout conversion in batch
/// mode.
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
        {
            tileGet(i, j, device, layoutConvert, modify, hold, async);
        }
    }

    // todo: if modify and target is host, batch convert on device first
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
void BaseMatrix<scalar_t>::tileGetForReading(std::set<ij_tuple>& tile_set,
                                             int device,
                                             LayoutConvert layout)
{
    if (device != hostNum()) {
        LockGuard guard(storage_->getDeviceLock(device));

        // find number of already existing tiles on the device
        int64_t existing_tiles = 0;
        for (auto iter = tile_set.begin(); iter != tile_set.end(); iter++) {
            int64_t i = std::get<0>(*iter);
            int64_t j = std::get<1>(*iter);
            existing_tiles += tileExists(i, j, device);
        }

        // ensure workspace exists for the rest
        if (tile_set.size() > size_t(existing_tiles))
            storage_->ensureDeviceWorkspace(device, tile_set.size() - existing_tiles);
    }

    tileGet(tile_set, device, layout, false, false, device != hostNum());

    if (device != hostNum())
        slate_cuda_call(
            cudaStreamSynchronize(comm_stream(device)));
}

//------------------------------------------------------------------------------
/// Gets a set of tiles for reading on host from a specific device.
/// @see tileGetForReading
///
/// @param[in] tile_set
///     Set of (i, j) tuples indicating indices of Tiles' to be acquired.
///     Tiles should exist on the specified device.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
/// @param[in] from_device
///     Tiles' source device ID.
///
// todo: async version
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileGetForReading(std::set<ij_tuple>& tile_set,
                                             LayoutConvert layout, int from_device)
{
    tileGet(tile_set, hostNum(), layout, false, false, true);

    slate_cuda_call(
        cudaStreamSynchronize(comm_stream(from_device)));
}

//------------------------------------------------------------------------------
/// Gets tile(i, j) for writing on device.
/// Sets destination tile's state to MOSI::Modified.
/// Will copy-in the tile if it does not exist or its state is MOSI::Invalid.
/// Other instances will be invalidated.
/// Finds a source tile whose state is valid (Modified|Shared) by
/// scanning existing tile instances.
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
    if (device != hostNum()) {
        LockGuard guard(storage_->getDeviceLock(device));

        // find number of aready existing tiles on the device
        int64_t existing_tiles = 0;
        for (auto iter = tile_set.begin(); iter != tile_set.end(); iter++) {
            int64_t i = std::get<0>(*iter);
            int64_t j = std::get<1>(*iter);
            existing_tiles += tileExists(i, j, device);
        }

        // ensure workspace exists for the rest
        if (tile_set.size() > size_t(existing_tiles))
            storage_->ensureDeviceWorkspace(device, tile_set.size() - existing_tiles);
    }

    tileGet(tile_set, device, layout, true, false, device != hostNum());

    if (device != hostNum())
        slate_cuda_call(
            cudaStreamSynchronize(comm_stream(device)));
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
    if (device != hostNum()) {
        LockGuard guard(storage_->getDeviceLock(device));

        // find number of aready existing tiles on the device
        int64_t existing_tiles = 0;
        for (auto iter = tile_set.begin(); iter != tile_set.end(); iter++) {
            int64_t i = std::get<0>(*iter);
            int64_t j = std::get<1>(*iter);
            existing_tiles += tileExists(i, j, device);
        }

        // ensure workspace exists for the rest
        if (tile_set.size() > size_t(existing_tiles))
            storage_->ensureDeviceWorkspace(device, tile_set.size() - existing_tiles);
    }

    tileGet(tile_set, device, layout, false, true, device != hostNum());

    if (device != hostNum())
        slate_cuda_call(
            cudaStreamSynchronize(comm_stream(device)));
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
    for (int64_t j = 0; j < nt(); ++j) {
        for (int64_t i = 0; i < mt(); ++i) {
            if (tileIsLocal(i, j)) {
                tiles_set[tileDevice(i, j)].insert({i, j});
            }
        }
    }

    #pragma omp taskgroup
    {
        for (int d = 0; d < num_devices(); ++d) {
            if (! tiles_set[d].empty()) {
                #pragma omp task default(shared)
                {
                    tileGetForReading(tiles_set[d], d, layout);
                }
            }
        }
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
    for (int64_t j = 0; j < nt(); ++j) {
        for (int64_t i = 0; i < mt(); ++i) {
            if (tileIsLocal(i, j)) {
                tiles_set[tileDevice(i, j)].insert({i, j});
            }
        }
    }

    #pragma omp taskgroup
    {
        for (int d = 0; d < num_devices(); ++d) {
            if (! tiles_set[d].empty()) {
                #pragma omp task default(shared)
                {
                    tileGetForWriting(tiles_set[d], d, layout);
                }
            }
        }
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
    for (int64_t j = 0; j < nt(); ++j) {
        for (int64_t i = 0; i < mt(); ++i) {
            if (tileIsLocal(i, j)) {
                tiles_set[tileDevice(i, j)].insert({i, j});
            }
        }
    }

    #pragma omp taskgroup
    {
        for (int d = 0; d < num_devices(); ++d) {
            if (! tiles_set[d].empty()) {
                #pragma omp task default(shared)
                {
                    tileGetAndHold(tiles_set[d], d, layout);
                }
            }
        }
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
    auto& tile_node = storage_->at(globalIndex(i, j));

    LockGuard guard(tile_node.getLock());

    // find on host
    if (tile_node.existsOn(hostNum()) &&
        tile_node[hostNum()].tile()->origin()) {
        if (tile_node[hostNum()].stateOn(MOSI::Invalid)) {
            // todo: should this request Layout conversion to this->layout() ?
            tileGetForReading(i, j, LayoutConvert::None);
        }
        return tile_node[hostNum()].tile();
    }
    else {
        auto device = tileDevice(i, j);
        if (tile_node.existsOn(device) &&
            tile_node[device].tile()->origin()) {
            if (tile_node[device].stateOn(MOSI::Invalid)) {
                // todo: should this request Layout conversion to this->layout() ?
                tileGetForReading(i, j, device, LayoutConvert::None);
            }
        }
        else
            slate_error( std::string("Origin tile not found! tile(")
                        +std::to_string(i)+","+std::to_string(j)+")");
        return tile_node[device].tile();
    }
}

//------------------------------------------------------------------------------
/// Updates all origin instances of local tiles if MOSI::Invalid.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileUpdateAllOrigin()
{
    std::vector< std::set<ij_tuple> > tiles_set_host(num_devices());
    std::vector< std::set<ij_tuple> > tiles_set_dev(num_devices());
    for (int64_t j = 0; j < this->nt(); ++j) {
        for (int64_t i = 0; i < this->mt(); ++i) {
            if (this->tileIsLocal(i, j)) {
                // this->tileUpdateOrigin(i, j);
                auto& tile_node = storage_->at(globalIndex(i, j));

                // find on host
                if (tile_node.existsOn(hostNum()) &&
                    tile_node[hostNum()].tile()->origin()) {
                    if (tile_node[hostNum()].stateOn(MOSI::Invalid)) {
                        // tileGetForReading(i, j, LayoutConvert::None);
                        for (int d = 0; d < num_devices(); ++d) {
                            if (tile_node.existsOn(d)
                                && tile_node[d].getState() != MOSI::Invalid)
                            {
                                tiles_set_host[d].insert({i, j});
                                break;
                            }
                        }
                    }
                }
                else {
                    auto device = tileDevice(i, j);
                    if (tile_node.existsOn(device) &&
                        tile_node[device].tile()->origin()) {
                        if (tile_node[device].stateOn(MOSI::Invalid)) {
                            // tileGetForReading(i, j, device, LayoutConvert::None);
                            tiles_set_dev[device].insert({i, j});
                        }
                    }
                    else
                        slate_error( std::string("Origin tile not found! tile(")
                                    +std::to_string(i)+","+std::to_string(j)+")");
                }
            }
        }
    }

    #pragma omp taskgroup
    {
        for (int d = 0; d < num_devices(); ++d) {
            if (! tiles_set_host[d].empty()) {
                #pragma omp task default(shared)
                {
                    tileGetForReading(tiles_set_host[d], LayoutConvert::None, d);
                }
            }
            if (! tiles_set_dev[d].empty()) {
                #pragma omp task default(shared)
                {
                    tileGetForReading(tiles_set_dev[d], d, LayoutConvert::None);
                }
            }
        }
    }
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
    return storage_->at(globalIndex(i, j, device)).tile()->isTransposable();
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
/// @param[in] reset
///     Optinally resets the tile extended buffers.
///
/// todo: handle op(A), sub-matrix, and sliced-matrix
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileLayoutConvert(int64_t i, int64_t j, int device,
                                             Layout layout, bool reset,
                                             bool async)
{
    LockGuard guard(storage_->at(globalIndex(i, j, device)).getLock());
    auto tile = storage_->at(globalIndex(i, j, device)).tile();
    if (tile->layout() != layout) {
        if (! tile->isTransposable()) {
            assert(! reset); // cannot reset if not transposable
            storage_->tileMakeTransposable(tile);
        }

        scalar_t* work_data = nullptr;
        // if rectangular and not extended, need a workspace buffer
        bool need_workspace = tile->mb() != tile->nb() && (! tile->extended());

        if (need_workspace)
            work_data = storage_->allocWorkspaceBuffer(tile->device());

        tile->layoutConvert(work_data,
                            tile->device() == host_num_ ?
                                              nullptr :
                                              comm_stream(tile->device()),
                            async && (! need_workspace));

        // release the workspace buffer if allocated
        if (need_workspace)
            storage_->releaseWorkspaceBuffer(work_data, tile->device());
    }
    if (reset) {
        assert(tile->layout() == this->layout());
        storage_->tileLayoutReset(tile);
    }
}

//------------------------------------------------------------------------------
/// Converts tiles indicated in 'tile_set' that exist on 'device' into 'layout'
/// if not already in 'layout' major.
/// Tiles should exist on 'device', will throw exception otherwise.
/// Operates in batch mode when tiles are on devices.
/// If device is not Host, will bucket tiles into uniform size and stride
/// batches, then launches each batch transpose.
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
/// @param[in] reset
///     Optinally resets the tiles extended buffers.
///
// todo: async API
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileLayoutConvert(std::set<ij_tuple>& tile_set,
                                             int device, Layout layout,
                                             bool reset)
{
    if (device == host_num_) {
        for (auto iter = tile_set.begin(); iter != tile_set.end(); iter++) {
            int64_t i = std::get<0>(*iter);
            int64_t j = std::get<1>(*iter);
            #pragma omp task
            {
                tileLayoutConvert(i, j, device, layout, reset);
            }
        }
        #pragma omp taskwait
    }
    else {
        // todo: this is not an optimal lock,
        // two calls on different devices will be serialized,
        // ideally, two concurrent read accesses to the TilesMap should be allowed
        // but not a concurrent read and write.
        LockGuard guard(storage_->getTilesMapLock());

        // map key tuple: m, n, extended, stride, work_stride
        using mnss_tuple = std::tuple<int64_t, int64_t, bool, int64_t, int64_t>;
        // map value tuple: data and extended data buffers
        using data_tuple = std::pair<std::vector<scalar_t*>, std::vector<scalar_t*>>;

        using BatchedTilesBuckets = std::map< mnss_tuple, data_tuple >;

        BatchedTilesBuckets tilesBuckets;

        for (auto iter = tile_set.begin(); iter != tile_set.end(); ++iter) {
            int64_t i = std::get<0>(*iter);
            int64_t j = std::get<1>(*iter);

            auto tile = storage_->at(globalIndex(i, j, device)).tile();

            // if we need to convert layout
            if (tile->layout() != layout) {
                // make sure tile is transposable
                if (! tile->isTransposable()) {
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

        // Allocate batch arrays, if not done already.
        int64_t batch_count = 0;
        for (auto bucket = tilesBuckets.begin(); bucket != tilesBuckets.end(); ++bucket) {
            batch_count = std::max(batch_count, int64_t(bucket->second.first.size()));
        }

        int64_t num_arrays =
            (storage_->array_host_.size() <= 0) ? 1 : storage_->array_host_.size();

        // todo: shouldn't we allocate for the current device only?
        allocateBatchArrays(batch_count, num_arrays);

        cudaStream_t stream = comm_stream(device);
        slate_cuda_call(
            cudaSetDevice(device));

        // for each bucket
        for (auto bucket = tilesBuckets.begin(); bucket != tilesBuckets.end(); ++bucket) {
            batch_count = bucket->second.first.size();

            scalar_t** array_dev = this->array_device(device);
            scalar_t** work_array_dev = this->array_device(device) + batch_count;

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

        if (reset) {
            for (auto iter = tile_set.begin(); iter != tile_set.end(); iter++) {
                // #pragma omp task
                {
                    int64_t i = std::get<0>(*iter);
                    int64_t j = std::get<1>(*iter);
                    auto tile = storage_->at(globalIndex(i, j, device)).tile();
                    storage_->tileLayoutReset(tile);
                }
            }
        }
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
/// @param[in] reset
///     Optinally resets the tiles extended buffers.
///
// todo: override on BaseTrapezoidMatrix and BandMatrix
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileLayoutConvert(int device, Layout layout, bool reset)
{
    std::set<ij_tuple> tiles_set;
    for (int64_t j = 0; j < nt(); ++j) {
        for (int64_t i = 0; i < mt(); ++i) {
            // if (tileIsLocal(i, j) && device == tileDevice(i, j) && tileExists(i, j, device))
            if (tileExists(i, j, device)) {
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
/// @param[in] reset
///     Optinally resets the tile extended buffers.
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
    #pragma omp taskgroup
    {
        for (int d = 0; d < num_devices(); ++d) {
            if (! tiles_set[d].empty()) {
                #pragma omp task default(shared)
                {
                    tileLayoutConvert(tiles_set[d], d, layout, reset);
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Converts tile(i, j) into current layout and resets its extended buffer.
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
/// @param[in] layout
///     Intended Layout of tiles:
///     - Layout::ColMajor or
///     - Layout::RowMajor.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileLayoutReset(int64_t i, int64_t j, int device,
                                           Layout layout)
{
    tileLayoutConvert(i, j, device, layout, true);
}

//------------------------------------------------------------------------------
/// Converts set of tiles into current layout and resets their extended buffers.
/// Operates in batch mode.
///
/// @param[in] tile_set
///     Set of (i, j) indices of tiles to be converted and reset.
///
/// @param[in] device
///     Tile's device ID; default is host_num.
///
/// @param[in] layout
///     Intended Layout of tiles:
///     - Layout::ColMajor or
///     - Layout::RowMajor.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileLayoutReset(std::set<ij_tuple>& tile_set,
                                           int device, Layout layout)
{
    tileLayoutConvert(tile_set, device, layout, true);
}

//------------------------------------------------------------------------------
/// Converts all origin tiles into current matrix-layout.
/// Operates in batch mode.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileLayoutReset()
{
    std::set<ij_tuple> tiles_set_host;
    std::vector< std::set<ij_tuple> > tiles_set_dev(num_devices());

    for (int64_t i = 0; i < mt(); ++i) {
        for (int64_t j = 0; j < nt(); ++j) {
            if (tileIsLocal(i, j)) {
                auto tile = tileUpdateOrigin(i, j);
                if (tile->layout() != this->layout()) {
                    assert(tile->isTransposable());
                }

                if (tile->device() == host_num_) {
                    tiles_set_host.insert({i, j});
                }
                else {
                    tiles_set_dev[tile->device()].insert({i, j});
                }
            }
        }
    }

    #pragma omp taskgroup
    {
        if (! tiles_set_host.empty()) {
            #pragma omp task default(shared)
            {
                tileLayoutReset(tiles_set_host, host_num_, this->layout());
            }
        }
        for (int d = 0; d < num_devices(); ++d) {
            if (! tiles_set_dev[d].empty()) {
                #pragma omp task default(shared)
                {
                    tileLayoutReset(tiles_set_dev[d], d, this->layout());
                }
            }
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
    auto& tile_node = storage_->at(globalIndex(i, j));

    // find on host
    if (tile_node.existsOn(hostNum()) &&
        tile_node[hostNum()].tile()->origin()) {
        return tile_node[hostNum()].tile();
    }
    else {
        auto device = tileDevice(i, j);
        if (tile_node.existsOn(hostNum()) &&
            tile_node[device].tile()->origin()) {
            return tile_node[device].tile();
        }
        else
            slate_error(std::string("Origin tile not found! tile(")
                        + std::to_string(i) + "," + std::to_string(j) + ")");
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
int BaseMatrix<scalar_t>::host_num_ = HostNum;

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
