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

#include "slate_Map.hh"
#include "slate_Memory.hh"
#include "slate_Storage.hh"
#include "slate_Tile.hh"
#include "slate_types.hh"

#include "lapack.hh"

#include <algorithm>
#include <functional>
#include <memory>
#include <set>
#include <utility>
#include <vector>
#include <iostream>
#include <sstream>

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

///=============================================================================
/// Base class for all SLATE distributed, tiled matrices.
template <typename scalar_t>
class BaseMatrix {
public:
    friend class Debug;

    ///-------------------------------------------------------------------------
    /// Default constructor; does not allocate any memory.
    BaseMatrix():
        ioffset_(0),
        joffset_(0),
        mt_(0),
        nt_(0),
        op_(Op::NoTrans),
        storage_(nullptr)
    {}

    // Defaults okay:
    // 1. destructor
    // 2. copy constructor
    // 3. move constructor
    // 4. copy assignment
    // 5. move assignment

    ///-------------------------------------------------------------------------
    /// Construct with mt block rows and nt block columns.
    /// Creates empty matrix storage.
    BaseMatrix(int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm):
        ioffset_(0),
        joffset_(0),
        mt_(ceildiv(m, nb)),
        nt_(ceildiv(n, nb)),
        op_(Op::NoTrans),
        storage_(std::make_shared< MatrixStorage< scalar_t > >(m, n, nb, p, q, mpi_comm)),
        mpi_comm_(mpi_comm)
    {
        int err;
        err = MPI_Comm_rank(mpi_comm_, &mpi_rank_);
        assert(err == MPI_SUCCESS);
        err = MPI_Comm_group(mpi_comm_, &mpi_group_);
        assert(err == MPI_SUCCESS);

        // todo: these are static, but we (re-)initialize with each matrix.
        // todo: similar code in BaseMatrix(...) and MatrixStorage(...)
        host_num_    = storage_->host_num_;
        num_devices_ = storage_->num_devices_;
    }

    ///-------------------------------------------------------------------------
    /// Sub-matrix constructor
    /// creates shallow copy view of parent matrix, A[ i1:i2, j1:j2 ].
    /// A[ 0:mt-1, 0:nt-1 ] is the entire A matrix.
    /// If i2 < i1 or j2 < j1, produces empty matrix.
    BaseMatrix(BaseMatrix& orig,
               int64_t i1, int64_t i2,
               int64_t j1, int64_t j2):
        BaseMatrix(orig)
    {
        assert(0 <= i1 && i1 < mt());
        assert(0 <= i2 && i2 < mt());
        assert(0 <= j1 && j1 < nt());
        assert(0 <= j2 && j2 < nt());

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

    ///-------------------------------------------------------------------------
    /// Swap contents of matrices A and B.
    friend void swap(BaseMatrix& A, BaseMatrix& B)
    {
        using std::swap;
        swap(A.ioffset_, B.ioffset_);
        swap(A.joffset_, B.joffset_);
        swap(A.mt_,      B.mt_);
        swap(A.nt_,      B.nt_);
        swap(A.op_,      B.op_);
        swap(A.storage_, B.storage_);
    }

    ///-------------------------------------------------------------------------
    /// @return shallow copy of tile {i, j} of op(A) on host,
    /// with the tile's op flag set to match the matrix's.
    Tile<scalar_t> operator() (int64_t i, int64_t j)
    {
        return (*this)(i, j, host_num_);
    }

    ///-------------------------------------------------------------------------
    /// @return shallow copy of tile {i, j} of op(A) on given device,
    /// with the tile's op flag set to match the matrix's.
    Tile<scalar_t> operator() (int64_t i, int64_t j, int device)
    {
        if (op_ == Op::NoTrans) {
            return *(storage_->at(globalIndex(i, j, device)));
        }
        else if (op_ == Op::Trans) {
            return transpose(*(storage_->at(globalIndex(i, j, device))));
        }
        else {  // if (op_ == Op::ConjTrans)
            return conj_transpose(*(storage_->at(globalIndex(i, j, device))));
        }
    }

    ///-------------------------------------------------------------------------
    /// alias of operator()
    Tile<scalar_t> at(int64_t i, int64_t j)
    {
        return (*this)(i, j);
    }

    ///-------------------------------------------------------------------------
    /// alias of operator()
    Tile<scalar_t> at(int64_t i, int64_t j, int device)
    {
        return (*this)(i, j, device);
    }

    // todo: it's a bit hard to keep track of m and n for sub-matrices,
    // esp. if all tiles are not same size.
    // basically need to sum up tileMb/tileNb for tiles in sub-matrix.
    // do that in the sub-matrix constructor?
    // int64_t m() const { ... }
    // int64_t n() const { ... }

    ///-------------------------------------------------------------------------
    /// @return number of devices (per MPI process) to distribute matrix to.
    int64_t num_devices() const { return num_devices_; }

    ///-------------------------------------------------------------------------
    /// @return number of block rows in op(A).
    int64_t mt() const { return (op_ == Op::NoTrans ? mt_ : nt_); }

    ///-------------------------------------------------------------------------
    /// @return number of block cols in op(A).
    int64_t nt() const { return (op_ == Op::NoTrans ? nt_ : mt_); }

    ///-------------------------------------------------------------------------
    /// @return transpose operation op(A).
    Op op() const { return op_; }

    ///-------------------------------------------------------------------------
    /// @return shallow copy transpose of A.
    /// The template ensures that the type matches the type of A,
    /// so for SymmetricMatrix A, transpose(A) is SymmetricMatrix.
    template< typename MatrixType >
    friend MatrixType transpose(MatrixType& A);

    ///-------------------------------------------------------------------------
    /// @return shallow copy conjugate-transpose of A.
    /// @see transpose
    template< typename MatrixType >
    friend MatrixType conj_transpose(MatrixType& A);

    ///-------------------------------------------------------------------------
    /// @return number of allocated tiles in global matrix.
    // todo: is it worthwhile to figure out the number in the local matrix?
    // for general matrices, it's just mt() * nt(), but for triangular or band
    // matrices, only some tiles are allocated.
    // maybe deprecated this routine.
    size_t size() const
    {
        if (storage_)
            return storage_->size();
        else
            return 0;
    }

    ///-------------------------------------------------------------------------
    /// @return MPI rank of tile {i, j} of op(A).
    int tileRank(int64_t i, int64_t j) const
    {
        return storage_->tileRank(globalIndex(i, j));
    }

    ///-------------------------------------------------------------------------
    /// @return device of tile {i, j} of op(A).
    int tileDevice(int64_t i, int64_t j) const
    {
        return storage_->tileDevice(globalIndex(i, j));
    }

    ///-------------------------------------------------------------------------
    /// @return whether tile {i, j} of op(A) is local.
    bool tileIsLocal(int64_t i, int64_t j) const
    {
        return storage_->tileIsLocal(globalIndex(i, j));
    }

    ///-------------------------------------------------------------------------
    /// @return number of rows (mb) in block row i of op(A).
    int64_t tileMb(int64_t i)
    {
        if (op_ == Op::NoTrans) {
            return storage_->tileMb(ioffset_ + i);
        }
        else {
            return storage_->tileNb(joffset_ + i);
        }
    }

    ///-------------------------------------------------------------------------
    /// @return number of cols (nb) in block col j of op(A).
    int64_t tileNb(int64_t j)
    {
        if (op_ == Op::NoTrans) {
            return storage_->tileNb(joffset_ + j);
        }
        else {
            return storage_->tileMb(ioffset_ + j);
        }
    }

    ///-------------------------------------------------------------------------
    /// Insert tile and allocate its data.
    Tile<scalar_t>* tileInsert(int64_t i, int64_t j, int device)
    {
        return storage_->tileInsert(globalIndex(i, j, device));
    }

    ///-------------------------------------------------------------------------
    /// Insert tile with existing data.
    Tile<scalar_t>* tileInsert(int64_t i, int64_t j, int device,
                               scalar_t* A, int64_t ld)
    {
        return storage_->tileInsert(globalIndex(i, j, device), A, ld);
    }

    ///-------------------------------------------------------------------------
    /// @return life counter of tile {i, j}.
    int64_t tileLife(int64_t i, int64_t j)
    {
        return storage_->tileLife(globalIndex(i, j));
    }

    ///-------------------------------------------------------------------------
    /// Set life counter of tile {i, j}.
    void tileLife(int64_t i, int64_t j, int64_t life)
    {
        storage_->tileLife(globalIndex(i, j), life);
    }

    ///-------------------------------------------------------------------------
    /// For workspace tiles (i.e., not local), decrements tile's life by 1.
    /// Then, if life reaches 0, deletes tile on all devices.
    void tileTick(int64_t i, int64_t j)
    {
        storage_->tileTick(globalIndex(i, j));
    }

    ///-------------------------------------------------------------------------
    /// \brief Erase a tile.
    /// If the memory is not origin, then the memory is released to the allocator pool.
    void tileErase(int64_t i, int64_t j, int device)
    {
        storage_->erase(globalIndex(i, j, device));
    }

    ///-------------------------------------------------------------------------
    /// Send tile {i, j} to all MPI ranks in matrix A.
    /// If target is Devices, also copies tile to all devices on each MPI rank.
    /// This should be called by at least all ranks with local tiles in A;
    /// ones that do not have any local tiles are excluded from the broadcast.
    // todo: should these be "tileBcast"? there is no tileRecv.
    template <Target target = Target::Host>
    void tileBcast(int64_t i, int64_t j, BaseMatrix const& A)
    {
        // Find the set of participating ranks.
        std::set<int> bcast_set;
        bcast_set.insert(tileRank(i, j));
        A.getRanks(&bcast_set);

        // If this rank is in the set.
        if (bcast_set.find(mpi_rank_) != bcast_set.end()) {
            // If receiving the tile, create tile to receive data, with life span.
            if (! tileIsLocal(i, j)) {
                tileInsert(i, j, host_num_);
                tileLife(i, j, A.numLocalTiles());
            }

            // Send across MPI ranks.
            tileBcastToSet(i, j, bcast_set);

            // Copy to devices.
            if (target == Target::Devices)
                for (int device = 0; device < num_devices_; ++device)
                    tileCopyToDevice(i, j, device);
        }
    }

    ///-------------------------------------------------------------------------
    /// Send tile {i, j} to all MPI ranks in matrix A1 or A2.
    /// If target is Devices, also copies tile to all devices on each MPI rank.
    /// This should be called by at least all ranks with local tiles in A1 or A2;
    /// ones that do not have any local tiles are excluded from the broadcast.
    // todo: this code is nearly identical to above tileBcast. Generalize?
    // todo: should these be "tileBcast"? there is no tileRecv.
    template <Target target = Target::Host>
    void tileBcast(int64_t i, int64_t j, BaseMatrix const& A1, BaseMatrix const& A2)
    {
        // Find the set of participating ranks.
        std::set<int> bcast_set;
        bcast_set.insert(tileRank(i, j));
        A1.getRanks(&bcast_set);
        A2.getRanks(&bcast_set);

        // If this rank is in the set.
        if (bcast_set.find(mpi_rank_) != bcast_set.end()) {
            // If receiving the tile, create tile to receive data, with life span.
            if (! tileIsLocal(i, j)) {
                tileInsert(i, j, host_num_);
                tileLife(i, j, A1.numLocalTiles() + A2.numLocalTiles());
            }

            // Send across MPI ranks.
            tileBcastToSet(i, j, bcast_set);

            // Copy to devices.
            if (target == Target::Devices)
                for (int device = 0; device < num_devices_; ++device)
                    tileCopyToDevice(i, j, device);
        }
    }

    ///-------------------------------------------------------------------------
    /// [internal]
    /// Broadcast tile {i, j} to all MPI ranks in the bcast_set.
    /// This should be called by all (and only) ranks that are in bcast_set,
    /// as either the root sender or a receiver.
    // todo: should these be "tileBcast"? there is no tileRecv.
    void tileBcastToSet(int64_t i, int64_t j, std::set<int> &bcast_set)
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
        #pragma omp critical(slate_mpi)
        retval = MPI_Comm_create_group(mpi_comm_, bcast_group, tag, &bcast_comm);
        assert(retval == MPI_SUCCESS);
        assert(bcast_comm != MPI_COMM_NULL);

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
        at(i, j).bcast(bcast_root, bcast_comm);

        // Free the group.
        #pragma omp critical(slate_mpi)
        retval = MPI_Group_free(&bcast_group);
        assert(retval == MPI_SUCCESS);

        // Free the communicator.
        #pragma omp critical(slate_mpi)
        retval = MPI_Comm_free(&bcast_comm);
        assert(retval == MPI_SUCCESS);
    }

    ///-------------------------------------------------------------------------
    /// \brief Copy a tile to a device.
    ///
    /// If the tile is not on the device, copy the tile to the device.
    /// If the tile is on the device, but is not valid,
    /// update the device tile's data to make it valid.
    /// Do not invalidate the host source tile.
    void tileCopyToDevice(int64_t i, int64_t j, int dst_device)
    {
        // If the tile is not on the device.
        auto iter = storage_->find(globalIndex(i, j, dst_device));
        if (iter == storage_->end()) {
            // Create a copy on the device.
            Tile<scalar_t> *src_tile = storage_->at(globalIndex(i, j, host_num_));
            Tile<scalar_t> *dst_tile = tileInsert(i, j, dst_device);
            src_tile->copyDataToDevice(dst_tile, comm_stream(dst_device));
        }
        else {
            // If the tile on the device is not valid.
            Tile<scalar_t> *dst_tile = iter->second;
            if (! dst_tile->valid()) {
                // Update the device tile's data.
                Tile<scalar_t> *src_tile = storage_->at(globalIndex(i, j, host_num_));
                src_tile->copyDataToDevice(dst_tile, comm_stream(dst_device));
                dst_tile->valid(true);
            }
        }
    }

    ///-------------------------------------------------------------------------
    /// \brief Copy a tile to the host.
    ///
    /// If the tile not on the host, copy the tile to the host.
    /// If the tile is on the host, but is not valid,
    /// update the host tile's data to make it valid.
    /// Do not invalidate the device source tile.
    void tileCopyToHost(int64_t i, int64_t j, int src_device)
    {
        // If the tile is not on the host.
        auto iter = storage_->find(globalIndex(i, j, host_num_));
        if (iter == storage_->end()) {
            // Create a copy on the host.
            Tile<scalar_t> *src_tile = storage_->at(globalIndex(i, j, src_device));
            Tile<scalar_t> *dst_tile = tileInsert(i, j, host_num_);
            src_tile->copyDataToHost(dst_tile, comm_stream(src_device));
        }
        else {
            // If the tile on the host is not valid.
            Tile<scalar_t> *dst_tile = iter->second;
            if (! dst_tile->valid()) {
                // Update the host tile's data.
                Tile<scalar_t> *src_tile = storage_->at(globalIndex(i, j, src_device));
                src_tile->copyDataToHost(dst_tile, comm_stream(src_device));
                dst_tile->valid(true);
            }
        }
    }

    ///-----------------------------------------------------------------------------
    /// \brief Move a tile to a device.
    ///
    /// If the tile is not on the device, copy the tile to the device.
    /// If the tile is on the device, but is not valid,
    /// update the device tile's data to make it valid.
    /// Invalidate the host source tile.
    void tileMoveToDevice(int64_t i, int64_t j, int dst_device)
    {
        // Copy the tile to the device.
        tileCopyToDevice(i, j, dst_device);

        // If the host tile exists, invalidate it.
        // todo: how could host tile not exist?
        auto iter = storage_->find(globalIndex(i, j, host_num_));
        if (iter != storage_->end())
            iter->second->valid(false);
    }

    ///-----------------------------------------------------------------------------
    /// \brief Move a tile to the host.
    ///
    /// If the tile is not on the host, copy the tile to the host.
    /// If the tile is on the host, but is not valid,
    /// update the host tile's data to make it valid.
    /// Invalidate the device source tile.
    void tileMoveToHost(int64_t i, int64_t j, int src_device)
    {
        // If source is not the host.
        // todo: why this if statement here, but not in other copy/move routines?
        if (src_device != host_num_) {
            // Copy the tile to the host.
            tileCopyToHost(i, j, src_device);

            // If the device tile exists, invalidate it.
            // todo: how could device tile not exist?
            auto iter = storage_->find(globalIndex(i, j, src_device));
            if (iter != storage_->end())
                iter->second->valid(false);
        }
    }

    ///-------------------------------------------------------------------------
    /// Puts all MPI ranks that have tiles in the matrix into the set.
    void getRanks(std::set<int> *bcast_set) const
    {
        for (int64_t i = 0; i < mt_; ++i)
            for (int64_t j = 0; j < nt_; ++j)
                bcast_set->insert(tileRank(i, j));
    }

    ///-------------------------------------------------------------------------
    /// Determines the lifespan of a temporary tile that updates every tile in
    /// the matrix, that is, the number of local tiles in the matrix.
    int64_t numLocalTiles() const
    {
        // Find the tile's lifespan.
        int64_t life = 0;
        for (int64_t i = 0; i < mt_; ++i)
            for (int64_t j = 0; j < nt_; ++j)
                if (tileIsLocal(i, j))
                    ++life;

        return life;
    }

    ///-------------------------------------------------------------------------
    /// Removes all tiles from matrix.
    void clear()
    {
        storage_->clear();
    }

    ///-------------------------------------------------------------------------
    /// Removes all temporary host and device workspace tiles from matrix.
    /// Leaves origin tiles.
    // todo: make sure all origin tiles are valid?
    void clearWorkspace()
    {
        storage_->clearWorkspace();
    }

    ///-------------------------------------------------------------------------
    /// Removes batch arrays from matrix for all devices.
    void clearBatchArrays()
    {
        storage_->clearBatchArrays();
    }

    ///-------------------------------------------------------------------------
    /// @return batch arrays for the A, B, or C matrices, on host, to send to device
    /// Throws error if arrays were not allocated with allocateBatchArrays.
    scalar_t** a_array_host(int device) { return storage_->a_array_host_.at(device); }
    scalar_t** b_array_host(int device) { return storage_->b_array_host_.at(device); }
    scalar_t** c_array_host(int device) { return storage_->c_array_host_.at(device); }

    ///-------------------------------------------------------------------------
    /// @return batch arrays for the A, B, or C matrices, on device
    /// Throws error if arrays were not allocated with allocateBatchArrays.
    scalar_t** a_array_device(int device) { return storage_->a_array_dev_.at(device); }
    scalar_t** b_array_device(int device) { return storage_->b_array_dev_.at(device); }
    scalar_t** c_array_device(int device) { return storage_->c_array_dev_.at(device); }

    ///-------------------------------------------------------------------------
    /// @return CUDA streams and cuBLAS handles
    cublasHandle_t cublas_handle (int device) { return storage_->cublas_handles_ .at(device); }
    cudaStream_t   compute_stream(int device) { return storage_->compute_streams_.at(device); }
    cudaStream_t   comm_stream   (int device) { return storage_->comm_streams_   .at(device); }

protected:
    ///-------------------------------------------------------------------------
    /// [internal]
    /// @return index {i, j} in global matrix, taking into account the
    /// local offset and transpose.
    std::tuple< int64_t, int64_t >
        globalIndex(int64_t i, int64_t j) const
    {
        assert(0 <= i && i < mt());
        assert(0 <= j && j < nt());
        if (op_ == Op::NoTrans)
            return { ioffset_ + i, joffset_ + j };
        else
            return { ioffset_ + j, joffset_ + i };
    }

    ///-------------------------------------------------------------------------
    /// [internal]
    /// @return index {i, j, dev} in global matrix, taking into account the
    /// local offset and transpose.
    std::tuple< int64_t, int64_t, int >
        globalIndex(int64_t i, int64_t j, int device) const
    {
        assert(0 <= i && i < mt());
        assert(0 <= j && j < nt());
        assert(device == host_num_ || (0 <= device && device < num_devices_));
        if (op_ == Op::NoTrans)
            return { ioffset_ + i, joffset_ + j, device };
        else
            return { ioffset_ + j, joffset_ + i, device };
    }

private:
    ///-------------------------------------------------------------------------
    int64_t ioffset_;   ///< block row offset with respect to original matrix
    int64_t joffset_;   ///< block col offset with respect to original matrix
    int64_t mt_;        ///< number of local block rows in this view
    int64_t nt_;        ///< number of local block cols in this view

protected:
    Op op_;             ///< transpose operation with respect to original matrix
    std::shared_ptr< MatrixStorage< scalar_t > > storage_;   ///< shared storage of tiles and buffers

    // ----- consider where to put these, here or in MatrixStorage
    static int host_num_;
    static int num_devices_;

    MPI_Comm  mpi_comm_;
    MPI_Group mpi_group_;
    int mpi_rank_;
};

template <typename scalar_t>
int BaseMatrix< scalar_t >::host_num_ = -1;

template <typename scalar_t>
int BaseMatrix< scalar_t >::num_devices_ = 0;

} // namespace slate

#endif // SLATE_BASE_MATRIX_HH
