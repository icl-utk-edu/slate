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

#ifndef SLATE_BASE_MATRIX_HH
#define SLATE_BASE_MATRIX_HH

#include "slate_internal_comm.hh"
#include "slate_Map.hh"
#include "slate_Memory.hh"
#include "slate_Storage.hh"
#include "slate_Tile.hh"
#include "slate_types.hh"

#include "lapack.hh"

#include <algorithm>
#include <memory>
#include <set>
#include <list>
#include <tuple>
#include <utility>
#include <vector>

#include "slate_cuda.hh"
#include "slate_cublas.hh"
#include "slate_mpi.hh"
#include "slate_openmp.hh"

namespace slate {

//==============================================================================
/// Base class for all SLATE distributed, tiled matrices.
/// In general, the documentation refers to the current matrix object as op(A),
/// to emphasize that it may be transposed with respect to its parent matrix.
///
template <typename scalar_t>
class BaseMatrix {
public:
    using BcastList = std::list<std::tuple<int64_t, int64_t,
                                           std::list<BaseMatrix<scalar_t> > > >;

    friend class Debug;

    static constexpr bool is_complex = is_complex<scalar_t>::value;
    static constexpr bool is_real    = ! is_complex;

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

    Tile<scalar_t> operator()(int64_t i, int64_t j);
    Tile<scalar_t> operator()(int64_t i, int64_t j, int device);

    /// Alias of operator().
    Tile<scalar_t> at(int64_t i, int64_t j)
    {
        return (*this)(i, j);
    }

    /// Alias of operator().
    Tile<scalar_t> at(int64_t i, int64_t j, int device)
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

    Tile<scalar_t>* tileInsert(int64_t i, int64_t j, int device);
    Tile<scalar_t>* tileInsert(int64_t i, int64_t j, int device,
                               scalar_t* A, int64_t ld);

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

    void tileErase(int64_t i, int64_t j, int device);

    template <Target target = Target::Host>
    void tileBcast(int64_t i, int64_t j, BaseMatrix const& B);

    template <Target target = Target::Host>
    void listBcast(BcastList& bcast_list);

protected:
    void tileBcastToSet(int64_t i, int64_t j, std::set<int> const& bcast_set);
    void tileBcastToSet(int64_t i, int64_t j, std::set<int> const& bcast_set,
                        int radix);

public:
    void tileCopyToDevice(int64_t i, int64_t j, int dst_device);
    void tileCopyToHost(  int64_t i, int64_t j, int src_device);
    void tileMoveToDevice(int64_t i, int64_t j, int dst_device);
    void tileMoveToHost(  int64_t i, int64_t j, int src_device);

    void getRanks(std::set<int>* bcast_set) const;
    void getLocalDevices(std::set<int>* dev_set) const;
    int64_t numLocalTiles() const;
    MPI_Comm mpiComm() const { return mpi_comm_; }


    /// Removes all tiles from matrix.
    void clear()
    {
        storage_->clear();
    }

    /// Removes all temporary host and device workspace tiles from matrix.
    /// Leaves origin tiles.
    void clearWorkspace()
    {
        storage_->clearWorkspace();
    }

    /// Removes batch arrays from matrix for all devices.
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

private:
    ///-------------------------------------------------------------------------
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
    : ioffset_(0),
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
    : ioffset_(0),
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
    assert((0 <= i1 && i1 < mt()) || (i1 > i2));  // todo: remove || ... clause?
    assert((0 <= i2 && i2 < mt()) || (i1 > i2));
    assert((0 <= j1 && j1 < nt()) || (j1 > j2));  // todo: remove || ... clause?
    assert((0 <= j2 && j2 < nt()) || (j1 > j2));

    // adjust i2, j2 for empty matrix
    i2 = std::max(i2, i1 - 1);
    j2 = std::max(j2, j1 - 1);

    if (op_ == Op::NoTrans) {
        ioffset_ += i1;
        joffset_ += j1;
        mt_ = i2 - i1 + 1;
        nt_ = j2 - j1 + 1;
    }
    else {
        ioffset_ += j1;
        joffset_ += i1;
        mt_ = j2 - j1 + 1;
        nt_ = i2 - i1 + 1;
    }
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
/// Get shallow copy of tile {i, j} of op(A) on host,
/// with the tile's op flag set to match the matrix's.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @return Tile {i, j, host}.
///
template <typename scalar_t>
Tile<scalar_t> BaseMatrix<scalar_t>::operator()(
    int64_t i, int64_t j)
{
    return (*this)(i, j, host_num_);
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
///     Tile's device ID.
///
/// @return Tile {i, j, device}.
///
template <typename scalar_t>
Tile<scalar_t> BaseMatrix<scalar_t>::operator()(
    int64_t i, int64_t j, int device)
{
    if (op_ == Op::NoTrans)
        return *(storage_->at(globalIndex(i, j, device)));
    else if (op_ == Op::Trans)
        return transpose(*(storage_->at(globalIndex(i, j, device))));
    else    // if (op_ == Op::ConjTrans)
        return conj_transpose(*(storage_->at(globalIndex(i, j, device))));
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
        return storage_->tileMb(ioffset_ + i);
    else
        return storage_->tileNb(joffset_ + i);
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
        return storage_->tileNb(joffset_ + j);
    else
        return storage_->tileMb(ioffset_ + j);
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
///     Tile's device ID.
///
/// @return Pointer to new tile.
///
template <typename scalar_t>
Tile<scalar_t>* BaseMatrix<scalar_t>::tileInsert(
    int64_t i, int64_t j, int device)
{
    auto index = globalIndex(i, j, device);
    auto tile = storage_->tileInsert(index);
    // set uplo on diagonal tiles (global i == j)
    if (std::get<0>(index) == std::get<1>(index))
        tile->uplo(uplo_);
    return tile;
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
///     Tile's device ID.
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
    auto tile = storage_->tileInsert(index, data, ld);
    // set uplo on diagonal tiles (global i == j)
    if (std::get<0>(index) == std::get<1>(index))
        tile->uplo(uplo_);
    return tile;
}

//------------------------------------------------------------------------------
/// Erase tile {i, j} of op(A).
/// If tile is not origin, then the memory is released to the allocator pool.
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
template <typename scalar_t>
template <Target target>
void BaseMatrix<scalar_t>::tileBcast(
    int64_t i, int64_t j, BaseMatrix<scalar_t> const& B)
{
    BcastList bcast_list_B;
    bcast_list_B.push_back({i, j, {B}});
    listBcast<target>(bcast_list_B);
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
template <typename scalar_t>
template <Target target>
void BaseMatrix<scalar_t>::listBcast(BcastList& bcast_list)
{
    // It is possible that the same tile, with the same data, is sent twice.
    // This happens, e.g., in the hemm and symm routines, where the same tile
    // is sent once as part of A and once as part of A^T.
    // This cannot be avoided without violating the upper bound on the size of
    // communication buffers.
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
            if (!tileIsLocal(i, j)) {

                // If receiving the tile, create tile to receive data,
                // with life span.
                // If tile already exists, add to its life span.
                LockGuard(storage_->tiles_.get_lock());
                auto iter = storage_->find(globalIndex(i, j, host_num_));

                int64_t life = 0;
                for (auto submatrix : submatrices_list)
                    life += submatrix.numLocalTiles();

                if (iter == storage_->end())
                    tileInsert(i, j, host_num_);
                else
                    life += tileLife(i, j); // todo: use temp tile to receive
                tileLife(i, j, life);
            }

            // Send across MPI ranks.  
            // Previous used MPI bcast: tileBcastToSet(i, j, bcast_set);
            // Currently uses 2D hypercube p2p send. 
            tileBcastToSet(i, j, bcast_set, 2);
        }

        // Copy to devices.
        if (target == Target::Devices) {

            std::set<int> dev_set;
            for (auto submatrix : submatrices_list)
                submatrix.getLocalDevices(&dev_set);

            #pragma omp task
            {
                for (auto device : dev_set)
                    tileCopyToDevice(i, j, device);
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
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileBcastToSet(
    int64_t i, int64_t j, std::set<int> const& bcast_set)
{
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
    int64_t i, int64_t j, std::set<int> const& bcast_set, int radix)
{
    // Quit if only root in the broadcast set.
    if (bcast_set.size() == 1)
        return;

    // Convert the set to a vector.
    std::vector<int> bcast_vec(bcast_set.begin(), bcast_set.end());

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
    if (!recv_from.empty())
        at(i, j).recv(new_vec[recv_from.front()], mpi_comm_);

    // Forward.
    for (int dst : send_to)
        at(i, j).send(new_vec[dst], mpi_comm_);
}

//------------------------------------------------------------------------------
/// Copy a tile to a device.
/// If the tile is not on the device, copy the tile to the device.
/// If the tile is on the device, but is not valid,
/// update the device tile's data to make it valid.
/// Do not invalidate the host source tile.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileCopyToDevice(
    int64_t i, int64_t j, int dst_device)
{
    // todo: race condition if multiple threads try to copy tile to device
    // If the tile is not on the device.
    auto iter = storage_->find(globalIndex(i, j, dst_device));
    if (iter == storage_->end()) {
        // Create a copy on the device.
        Tile<scalar_t>* src_tile = storage_->at(globalIndex(i, j, host_num_));
        Tile<scalar_t>* dst_tile = tileInsert(i, j, dst_device);
        src_tile->copyDataToDevice(dst_tile, comm_stream(dst_device));
    }
    else {
        // If the tile on the device is not valid.
        Tile<scalar_t>* dst_tile = iter->second;
        if (! dst_tile->valid()) {
            // Update the device tile's data.
            Tile<scalar_t>* src_tile =
                storage_->at(globalIndex(i, j, host_num_));
            src_tile->copyDataToDevice(dst_tile, comm_stream(dst_device));
            dst_tile->valid(true);
        }
    }
}

//------------------------------------------------------------------------------
/// Copy a tile to the host.
/// If the tile is not on the host, copy the tile to the host.
/// If the tile is on the host, but is not valid,
/// update the host tile's data to make it valid.
/// Do not invalidate the device source tile.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] src_device
///     Tile's source device ID.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileCopyToHost(
    int64_t i, int64_t j, int src_device)
{
    // todo: race condition if multiple threads try to copy tile to device
    // If the tile is not on the host.
    auto iter = storage_->find(globalIndex(i, j, host_num_));
    if (iter == storage_->end()) {
        // Create a copy on the host.
        Tile<scalar_t>* src_tile = storage_->at(globalIndex(i, j, src_device));
        Tile<scalar_t>* dst_tile = tileInsert(i, j, host_num_);
        src_tile->copyDataToHost(dst_tile, comm_stream(src_device));
    }
    else {
        // If the tile on the host is not valid.
        Tile<scalar_t>* dst_tile = iter->second;
        if (! dst_tile->valid()) {
            // Update the host tile's data.
            Tile<scalar_t>* src_tile =
                storage_->at(globalIndex(i, j, src_device));
            src_tile->copyDataToHost(dst_tile, comm_stream(src_device));
            dst_tile->valid(true);
        }
    }
}

//------------------------------------------------------------------------------
/// Move a tile to a device.
/// If the tile is not on the device, copy the tile to the device.
/// If the tile is on the device, but is not valid,
/// update the device tile's data to make it valid.
/// Invalidate the host source tile.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] src_device
///     Tile's destination device ID.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileMoveToDevice(
    int64_t i, int64_t j, int dst_device)
{
    // Copy the tile to the device.
    tileCopyToDevice(i, j, dst_device);

    // If the host tile exists, invalidate it.
    // todo: how could host tile not exist?
    // todo: any races here?
    auto iter = storage_->find(globalIndex(i, j, host_num_));
    if (iter != storage_->end())
        iter->second->valid(false);
}

//------------------------------------------------------------------------------
/// Move a tile to the host.
/// If the tile is not on the host, copy the tile to the host.
/// If the tile is on the host, but is not valid,
/// update the host tile's data to make it valid.
/// Invalidate the device source tile.
///
/// @param[in] i
///     Tile's block row index. 0 <= i < mt.
///
/// @param[in] j
///     Tile's block column index. 0 <= j < nt.
///
/// @param[in] src_device
///     Tile's source device ID.
///
template <typename scalar_t>
void BaseMatrix<scalar_t>::tileMoveToHost(
    int64_t i, int64_t j, int src_device)
{
    // If source is not the host.
    // todo: why this if statement here, but not in other copy/move routines?
    if (src_device != host_num_) {
        // Copy the tile to the host.
        tileCopyToHost(i, j, src_device);

        // If the device tile exists, invalidate it.
        // todo: how could device tile not exist?
        // todo: any races here?
        auto iter = storage_->find(globalIndex(i, j, src_device));
        if (iter != storage_->end())
            iter->second->valid(false);
    }
}

//------------------------------------------------------------------------------
/// Puts all MPI ranks that have tiles in the matrix into the set.
///
/// @param[out] bcast_set
///     On output, set of MPI ranks that this sub-matrix has tiles on.
///
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
