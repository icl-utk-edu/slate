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

#ifndef SLATE_TILE_GETRF_HH
#define SLATE_TILE_GETRF_HH

#include "slate_internal.hh"
#include "slate_Tile.hh"
#include "slate_Tile_blas.hh"
#include "slate_Tile_lapack.hh"
#include "slate_types.hh"
#include "slate_util.hh"

#include <list>

#include <blas.hh>
#include <lapack.hh>

namespace slate {
namespace internal {
// todo: Perhaps we should put all Tile routines in "internal".

///-----------------------------------------------------------------------------
/// \brief
/// Swap a single row in the panel factorization.
///
/// \param[in] i
///     pivot index
///
/// \param[in] j
///     first column of the swap
///
/// \param[in] n
///     length of the swap
///
/// \param[inout] tiles
///     vector of local panel tiles
///
/// \param[in] pivot
///     vector of pivot indices for the diagonal tile
///
/// \param[in] mpi_rank
///     MPI rank in the panel factorization
///
/// \param[in] mpi_root
///     MPI rank of the root for the panel factorization
///
/// \param[in] mpi_comm
///     MPI subcommunicator for the panel factorization
///
template <typename scalar_t>
void getrf_swap(
    int64_t i, int64_t j, int64_t n,
    std::vector< Tile<scalar_t> >& tiles,
    std::vector< AuxPivot<scalar_t> >& pivot,
    int mpi_rank, int mpi_root, MPI_Comm mpi_comm)
{
    bool root = mpi_rank == mpi_root;

    // If I own the pivot.
    if (pivot[i].rank() == mpi_rank) {
        // If I am the root.
        if (root) {
            // if pivot not on the diagonal
            if (pivot[i].localTileIndex() > 0 ||
                pivot[i].elementOffset() > i)
            {
                // local swap
                swap(j, n,
                     tiles.at(0), i,
                     tiles.at(pivot[i].localTileIndex()),
                              pivot[i].elementOffset());
            }
        }
        // I am not the root.
        else {
            // MPI swap with the root
            swap(j, n,
                 tiles.at(pivot[i].localTileIndex()),
                 pivot[i].elementOffset(),
                 mpi_root, mpi_comm);
        }
    }
    // I don't own the pivot.
    else {
        // I am the root.
        if (root) {
            // MPI swap with the pivot owner
            swap(j, n,
                 tiles.at(0), i,
                 pivot[i].rank(), mpi_comm);
        }
    }
}

///-----------------------------------------------------------------------------
/// \brief
/// Compute the LU factorization of a panel.
///
/// \param[in] diag_len
///     length of the panel diagonal
///
/// \param[in] ib
///     internal blocking in the panel
///
/// \param[inout] tiles
///     local tiles in the panel
///
/// \param[in] tile_indices
///     i indices of the tiles in the panel
///
/// \param[in] tile_offsets
///     i element offsets of the tiles in the panel
///
/// \param[inout] pivot
///     pivots produced by the panel factorization
///
/// \param[in] mpi_rank
///     MPI rank in the panel factorization
///
/// \param[in] mpi_root
///     MPI rank of the root for the panel factorization
///
/// \param[in] mpi_comm
///     MPI subcommunicator for the panel factorization
///
/// \param[in] thread_rank
///     rank of this thread
///
/// \param[in] thread_size
///     number of local threads
///
/// \param[in] thread_barrier
///     barrier for synchronizing local threads
///
/// \param[out] max_value
///     workspace for per-thread pivot value
///
/// \param[in] max_index
///     workspace for per-thread pivot index
//      (local index of the tile containing the pivot)
///
/// \param[in] max_offset
///     workspace for per-thread pivot offset
///     (pivot offset in the tile)
///
/// \param[in] tob_block
///     workspace for broadcasting the top row for the geru operation
///     and the top block for the gemm operation.
///
template <typename scalar_t>
int64_t getrf(int64_t diag_len, int64_t ib,
              std::vector< Tile<scalar_t> >& tiles,
              std::vector<int64_t>& tile_indices,
              std::vector<int64_t>& tile_offsets,
              std::vector< AuxPivot<scalar_t> >& pivot,
              int mpi_rank, int mpi_root, MPI_Comm mpi_comm,
              int thread_rank, int thread_size,
              ThreadBarrier& thread_barrier,
              std::vector<scalar_t>& max_value,
              std::vector<int64_t>& max_index,
              std::vector<int64_t>& max_offset,
              std::vector<scalar_t>& top_block)
{
    trace::Block trace_block("lapack::getrf");

    using namespace blas;
    using namespace lapack;
    using real_t = real_type<scalar_t>;

    const bool root = mpi_rank == mpi_root;
    const int64_t nb = tiles.at(0).nb();

    // Loop over ib-wide stripes.
    for (int64_t k = 0; k < diag_len; k += ib) {

        //=======================
        // ib panel factorization
        int64_t kb = std::min(diag_len-k, ib);

        // Loop over ib columns of a stripe.
        for (int64_t j = k; j < k+kb; ++j) {

            if (root) {
                max_value[thread_rank] = tiles.at(0)(j, j);
                max_index[thread_rank] = 0;
                max_offset[thread_rank] = j;
            }
            else {
                max_value[thread_rank] = tiles.at(0)(0, j);
                max_index[thread_rank] = 0;
                max_offset[thread_rank] = 0;
            }

            //------------------
            // thread max search
            for (int64_t idx = thread_rank;
                 idx < tiles.size();
                 idx += thread_size)
            {
                auto tile = tiles.at(idx);
                auto i_index = tile_indices.at(idx);

                // if diagonal tile
                if (i_index == 0) {
                    for (int64_t i = j+1; i < tile.mb(); ++i) {
                        if (cabs1(tile(i, j)) >
                            cabs1(max_value[thread_rank]))
                        {
                            max_value[thread_rank] = tile(i, j);
                            max_index[thread_rank] = idx;
                            max_offset[thread_rank] = i;
                        }
                    }

                }
                // off diagonal tiles
                else {
                    for (int64_t i = 0; i < tile.mb(); ++i) {
                        if (cabs1(tile(i, j)) >
                            cabs1(max_value[thread_rank]))
                        {
                            max_value[thread_rank] = tile(i, j);
                            max_index[thread_rank] = idx;
                            max_offset[thread_rank] = i;
                        }
                    }
                }
            }
            thread_barrier.wait(thread_size);

            //------------------------------------
            // global max reduction and pivot swap
            if (thread_rank == 0) {
                // threads max reduction
                for (int rank = 1; rank < thread_size; ++rank) {
                    if (cabs1(max_value[rank]) > cabs1(max_value[0])) {
                        max_value[0] = max_value[rank];
                        max_index[0] = max_index[rank];
                        max_offset[0] = max_offset[rank];
                    }
                }

                // MPI max abs reduction
                struct { real_t max; int loc; } max_loc_in, max_loc;
                max_loc_in.max = cabs1(max_value[0]);
                max_loc_in.loc = mpi_rank;
                #pragma omp critical(slate_mpi)
                {
                    slate_mpi_call(
                        MPI_Allreduce(&max_loc_in, &max_loc, 1,
                                      mpi_type< max_loc_type<real_t> >::value,
                                      MPI_MAXLOC, mpi_comm));
                }

                // Broadcast the pivot information.
                pivot[j] = AuxPivot<scalar_t>(tile_indices[max_index[0]],
                                              max_offset[0],
                                              max_index[0],
                                              max_value[0],
                                              max_loc.loc);
                #pragma omp critical(slate_mpi)
                {
                    slate_mpi_call(
                        MPI_Bcast(&pivot[j], sizeof(AuxPivot<scalar_t>),
                                  MPI_BYTE, max_loc.loc, mpi_comm));
                }

                // pivot swap
                getrf_swap(j, k, kb,
                           tiles, pivot,
                           mpi_rank, mpi_root, mpi_comm);

                // Broadcast the top row for the geru operation.
                if (k+kb > j+1) {
                    if (root) {
                        auto top_tile = tiles.at(0);
                        // todo: make it a tile operation
                        copy(k+kb-j-1,
                             &top_tile.at(j, j+1), top_tile.stride(),
                             top_block.data(), 1);
                    }
                    #pragma omp critical(slate_mpi)
                    {
                        slate_mpi_call(
                            MPI_Bcast(top_block.data(),
                                      k+kb-j-1, mpi_type<scalar_t>::value,
                                      mpi_root, mpi_comm));
                    }
                }
            }
            thread_barrier.wait(thread_size);

            // column scaling and trailing update
            for (int64_t idx = thread_rank;
                 idx < tiles.size();
                 idx += thread_size)
            {
                auto tile = tiles.at(idx);
                auto i_index = tile_indices.at(idx);

                // column scaling
                real_t sfmin = std::numeric_limits<real_t>::min();
                if (cabs1(pivot[j].value()) >= sfmin) {
                    if (i_index == 0) {
                        // diagonal tile
                        for (int64_t i = j+1; i < tile.mb(); ++i)
                            tile.at(i, j) /= tile(j, j);
                    }
                    else {
                        // off diagonal tile
                        for (int64_t i = 0; i < tile.mb(); ++i)
                            tile.at(i, j) /= pivot[j].value();
                    }
                }
                else {
                    // todo: make it a tile operation
                    if (i_index == 0) {
                        // diagonal tile
                        scalar_t one = 1.0;
                        scalar_t alpha = one / tile(j, j);
                        scal(tile.mb()-j-1, alpha, &tile.at(j+1, j), 1);
                    }
                    else {
                        // off diagonal tile
                        scalar_t one = 1.0;
                        scalar_t alpha = one / pivot[j].value();
                        scal(tile.mb(), alpha, &tile.at(0, j), 1);
                    }
                }

                // trailing update
                // todo: make it a tile operation
                if (k+kb > j+1) {
                    if (i_index == 0) {
                        geru(Layout::ColMajor,
                             tile.mb()-j-1, k+kb-j-1,
                             -1.0, &tile.at(j+1, j), 1,
                                   top_block.data(), 1,
                                   &tile.at(j+1, j+1), tile.stride());
                    }
                    else {
                        geru(Layout::ColMajor,
                             tile.mb(), k+kb-j-1,
                             -1.0, &tile.at(0, j), 1,
                                   top_block.data(), 1,
                                   &tile.at(0, j+1), tile.stride());
                    }
                }
            }
            thread_barrier.wait(thread_size);
        }

        // If there is a trailing submatrix.
        if (k+kb < nb) {

            //======================
            // pivoting to the right
            if (thread_rank == 0) {
                for (int64_t i = k; i < k+kb; ++i) {
                    getrf_swap(i, k+kb, nb-k-kb,
                               tiles, pivot,
                               mpi_rank, mpi_root, mpi_comm);
                }
            }
            thread_barrier.wait(thread_size);

            //=================
            // triangular solve
            if (root && thread_rank == 0) {

                auto top_tile = tiles.at(0);
                trsm(Layout::ColMajor,
                     Side::Left, Uplo::Lower,
                     Op::NoTrans, Diag::Unit,
                     kb, nb-k-kb,
                     1.0, &top_tile.at(k, k), top_tile.stride(),
                          &top_tile.at(k, k+kb), top_tile.stride());
            }
            thread_barrier.wait(thread_size);

            // Broadcast the top block for gemm.
            if (thread_rank == 0) {
                if (root) {
                    auto top_tile = tiles.at(0);
                    lacpy(MatrixType::General,
                          kb, nb-k-kb,
                          &top_tile.at(k, k+kb), top_tile.stride(),
                          top_block.data(), kb);
                }
                #pragma omp critical(slate_mpi)
                {
                    slate_mpi_call(
                        MPI_Bcast(top_block.data(),
                                  kb*(nb-k-kb), mpi_type<scalar_t>::value,
                                  mpi_root, mpi_comm));
                }
            }
            thread_barrier.wait(thread_size);

            //============================
            // rank-ib update to the right
            for (int64_t idx = thread_rank;
                 idx < tiles.size();
                 idx += thread_size)
            {
                auto tile = tiles.at(idx);
                auto i_index = tile_indices.at(idx);

                if (i_index == 0) {
                    if (k+kb < tile.mb()) {
                        gemm(Layout::ColMajor,
                             Op::NoTrans, Op::NoTrans,
                             tile.mb()-k-kb, nb-k-kb, kb,
                             -1.0, &tile.at(k+kb,k   ), tile.stride(),
                                   &tile.at(k,   k+kb), tile.stride(),
                              1.0, &tile.at(k+kb,k+kb), tile.stride());
                    }
                }
                else {
                    gemm(Layout::ColMajor,
                         Op::NoTrans, Op::NoTrans,
                         tile.mb(), nb-k-kb, kb,
                         -1.0, &tile.at(0, k), tile.stride(),
                               top_block.data(), kb,
                          1.0, &tile.at(0, k+kb), tile.stride());
                }
            }
            thread_barrier.wait(thread_size);
        }
    }

    //=====================
    // pivoting to the left
    for (int64_t k = ib; k < diag_len; k += ib) {
        if (thread_rank == 0) {
            for (int64_t i = k; i < k+ib && i < diag_len; ++i) {
                getrf_swap(i, 0, k,
                           tiles, pivot,
                           mpi_rank, mpi_root, mpi_comm);
            }
        }
    }
}

} // namespace internal
} // namespace slate

#endif // SLATE_TILE_GETRF_HH
