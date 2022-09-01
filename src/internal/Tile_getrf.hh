// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_TILE_GETRF_HH
#define SLATE_TILE_GETRF_HH

#include "internal/internal.hh"
#include "internal/internal_swap.hh"
#include "slate/Tile.hh"
#include "slate/Tile_blas.hh"
#include "internal/Tile_lapack.hh"
#include "slate/types.hh"
#include "slate/internal/util.hh"

#include <list>

#include <blas.hh>
#include <lapack.hh>

namespace slate {
namespace internal {
// todo: Perhaps we should put all Tile routines in "internal".

//------------------------------------------------------------------------------
/// Swap a single row in the panel factorization.
///
/// @param[in] i
///     pivot index
///
/// @param[in] j
///     first column of the swap
///
/// @param[in] n
///     length of the swap
///
/// @param[in,out] tiles
///     vector of local panel tiles
///
/// @param[in] pivot
///     vector of pivot indices for the diagonal tile
///
/// @param[in] mpi_rank
///     MPI rank in the panel factorization
///
/// @param[in] mpi_root
///     MPI rank of the root for the panel factorization
///
/// @param[in] mpi_comm
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
                swapLocalRow(j, n,
                             tiles[0], i,
                             tiles[pivot[i].localTileIndex()],
                             pivot[i].elementOffset());
            }
        }
        // I am not the root.
        else {
            // MPI swap with the root
            swapRemoteRow(j, n,
                          tiles[pivot[i].localTileIndex()],
                          pivot[i].elementOffset(),
                          mpi_root, mpi_comm);
        }
    }
    // I don't own the pivot.
    else {
        // I am the root.
        if (root) {
            // MPI swap with the pivot owner
            swapRemoteRow(j, n,
                          tiles[0], i,
                          pivot[i].rank(), mpi_comm);
        }
    }
}

//------------------------------------------------------------------------------
/// Compute the LU factorization of a panel.
///
/// @param[in] diag_len
///     length of the panel diagonal
///
/// @param[in] ib
///     internal blocking in the panel
///
/// @param[in,out] tiles
///     local tiles in the panel
///
/// @param[in] tile_indices
///     i indices of the tiles in the panel
///
/// @param[in,out] pivot
///     pivots produced by the panel factorization
///
/// @param[in] mpi_rank
///     MPI rank in the panel factorization
///
/// @param[in] mpi_root
///     MPI rank of the root for the panel factorization
///
/// @param[in] mpi_comm
///     MPI subcommunicator for the panel factorization
///
/// @param[in] thread_rank
///     rank of this thread
///
/// @param[in] thread_size
///     number of local threads
///
/// @param[in] thread_barrier
///     barrier for synchronizing local threads
///
/// @param[out] max_value
///     workspace for per-thread pivot value
///
/// @param[in] max_index
///     workspace for per-thread pivot index
//      (local index of the tile containing the pivot)
///
/// @param[in] max_offset
///     workspace for per-thread pivot offset
///     (pivot offset in the tile)
///
/// @param[in] tob_block
///     workspace for broadcasting the top row for the geru operation
///     and the top block for the gemm operation.
///
/// @param[in] pivot_threshold
///     threshold for pivoting.  1 is partial pivoting, 0 is no pivoting
///
/// @ingroup gesv_tile
///
template <typename scalar_t>
void getrf(
    int64_t diag_len, int64_t ib,
    std::vector< Tile<scalar_t> >& tiles,
    std::vector<int64_t>& tile_indices,
    std::vector< AuxPivot<scalar_t> >& pivot,
    int mpi_rank, int mpi_root, MPI_Comm mpi_comm,
    int thread_rank, int thread_size,
    ThreadBarrier& thread_barrier,
    std::vector<scalar_t>& max_value,
    std::vector<int64_t>& max_index,
    std::vector<int64_t>& max_offset,
    std::vector<scalar_t>& top_block,
    blas::real_type<scalar_t> pivot_threshold)
{
    trace::Block trace_block("lapack::getrf");

    using real_t = blas::real_type<scalar_t>;

    const scalar_t zero = 0.0;
    const scalar_t one  = 1.0;

    bool root = mpi_rank == mpi_root;
    int64_t nb = tiles[0].nb();

    // Loop over ib-wide stripes.
    for (int64_t k = 0; k < diag_len; k += ib) {

        //=======================
        // ib panel factorization
        int64_t kb = std::min(diag_len-k, ib);

        // Loop over ib columns of a stripe.
        for (int64_t j = k; j < k+kb; ++j) {

            if (root && thread_rank == 0) {
                max_value[thread_rank] = tiles[0](j, j);
                max_index[thread_rank] = 0;
                max_offset[thread_rank] = j;
            }
            else {
                max_value[thread_rank] = tiles[thread_rank](0, j);
                max_index[thread_rank] = thread_rank;
                max_offset[thread_rank] = 0;
            }

            //------------------
            // thread max search
            for (int64_t idx = thread_rank;
                 idx < int64_t(tiles.size());
                 idx += thread_size)
            {
                auto tile = tiles[idx];
                auto i_index = tile_indices[idx];

                // if diagonal tile
                if (i_index == 0) {
                    for (int64_t i = j+1; i < tile.mb(); ++i) {
                        if (cabs1(tile(i, j)) > cabs1(max_value[thread_rank])) {
                            max_value[thread_rank] = tile(i, j);
                            max_index[thread_rank] = idx;
                            max_offset[thread_rank] = i;
                        }
                    }
                }
                // off diagonal tiles
                else {
                    for (int64_t i = 0; i < tile.mb(); ++i) {
                        if (cabs1(tile(i, j)) > cabs1(max_value[thread_rank])) {
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
                // Do two reductions that differ in the root's value
                // * the diagonal entry
                // * the largest entry
                struct { real_t max; int loc; } max_loc_in[2], max_loc[2];
                if (mpi_rank == mpi_root) {
                    max_loc_in[0].max = cabs1(tiles[0](j, j));
                    max_loc_in[1].max = cabs1(max_value[0]);
                }
                else {
                    max_loc_in[0].max = cabs1(max_value[0])*pivot_threshold;
                    max_loc_in[1].max = max_loc_in[0].max;
                }
                max_loc_in[0].loc = mpi_rank;
                max_loc_in[1].loc = mpi_rank;
                slate_mpi_call(
                    MPI_Allreduce(max_loc_in, max_loc, 2,
                                  mpi_type< max_loc_type<real_t> >::value,
                                  MPI_MAXLOC, mpi_comm));

                int bcast_rank;
                if (max_loc[0].loc != mpi_root) {
                    // if diagonal isn't good enough for the remote entries,
                    // use the result of the second reduction
                    bcast_rank = max_loc[1].loc;
                }
                else {
                    bcast_rank = mpi_root;

                    // if the diagonal is good enough for the local entries,
                    // update that max_* variables on the root
                    if (mpi_rank == mpi_root
                        && max_loc[0].max >= cabs1(max_value[0])*pivot_threshold) {
                        max_offset[0] = j;
                        max_index[0] = 0;
                        max_value[0] = tiles[0](j, j);
                    }
                }


                // todo: can this Bcast info be merged into the Allreduce?
                // Broadcast the pivot information.
                pivot[j] = AuxPivot<scalar_t>(tile_indices[max_index[0]],
                                              max_offset[0],
                                              max_index[0],
                                              max_value[0],
                                              bcast_rank);
                slate_mpi_call(
                    MPI_Bcast(&pivot[j], sizeof(AuxPivot<scalar_t>),
                              MPI_BYTE, bcast_rank, mpi_comm));

                // pivot swap
                getrf_swap(j, 0, nb,
                           tiles, pivot,
                           mpi_rank, mpi_root, mpi_comm);

                // Broadcast the top row for the geru operation.
                if (k+kb > j+1) {
                    if (root) {
                        auto top_tile = tiles[0];
                        // todo: make it a tile operation
                        blas::copy(k+kb-j-1,
                                   &top_tile.at(j, j+1), top_tile.stride(),
                                   top_block.data(), 1);
                    }
                    slate_mpi_call(
                        MPI_Bcast(top_block.data(),
                                  k+kb-j-1, mpi_type<scalar_t>::value,
                                  mpi_root, mpi_comm));
                }
            }
            thread_barrier.wait(thread_size);

            // column scaling and trailing update
            for (int64_t idx = thread_rank;
                 idx < int64_t(tiles.size());
                 idx += thread_size)
            {
                auto tile = tiles[idx];
                auto i_index = tile_indices[idx];

                // column scaling
                real_t sfmin = std::numeric_limits<real_t>::min();
                if (cabs1(pivot[j].value()) >= sfmin) {
                    // todo: make it a tile operation
                    if (i_index == 0) {
                        // diagonal tile
                        scalar_t alpha = one / tile(j, j);
                        int64_t m = tile.mb()-j-1;
                        if (m > 0)
                            blas::scal(tile.mb()-j-1, alpha, &tile.at(j+1, j), 1);
                    }
                    else {
                        // off diagonal tile
                        scalar_t alpha = one / pivot[j].value();
                        blas::scal(tile.mb(), alpha, &tile.at(0, j), 1);
                    }
                }
                else if (pivot[j].value() != zero) {
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
                    // pivot[j].value() = 0, The factorization has been completed
                    // but the factor U is exactly singular
                    // todo: how to handle a zero pivot
                }

                // trailing update
                // todo: make it a tile operation
                if (k+kb > j+1) {
                    if (i_index == 0) {
                        blas::geru(Layout::ColMajor,
                                   tile.mb()-j-1, k+kb-j-1,
                                   -one, &tile.at(j+1, j), 1,
                                         top_block.data(), 1,
                                         &tile.at(j+1, j+1), tile.stride());
                    }
                    else {
                        blas::geru(Layout::ColMajor,
                                   tile.mb(), k+kb-j-1,
                                   -one, &tile.at(0, j), 1,
                                         top_block.data(), 1,
                                         &tile.at(0, j+1), tile.stride());
                    }
                }
            }
            // Next instructions only use thread's assigned tiles
            // So no thread barrier is needed here
        }

        // If there is a trailing submatrix.
        if (k+kb < nb) {
            if (thread_rank == 0) {
                if (root) {
                    // triangular solve
                    auto top_tile = tiles[0];
                    blas::trsm(Layout::ColMajor,
                               Side::Left, Uplo::Lower,
                               Op::NoTrans, Diag::Unit,
                               kb, nb-k-kb,
                               one, &top_tile.at(k, k), top_tile.stride(),
                                    &top_tile.at(k, k+kb), top_tile.stride());

                    // Broadcast the top block for gemm.
                    lapack::lacpy(lapack::MatrixType::General,
                                  kb, nb-k-kb,
                                  &top_tile.at(k, k+kb), top_tile.stride(),
                                  top_block.data(), kb);
                }
                slate_mpi_call(
                    MPI_Bcast(top_block.data(),
                              kb*(nb-k-kb), mpi_type<scalar_t>::value,
                              mpi_root, mpi_comm));
            }
            thread_barrier.wait(thread_size);

            //============================
            // rank-ib update to the right
            for (int64_t idx = thread_rank;
                 idx < int64_t(tiles.size());
                 idx += thread_size)
            {
                auto tile = tiles[idx];
                auto i_index = tile_indices[idx];

                if (i_index == 0) {
                    if (k+kb < tile.mb()) {
                        blas::gemm(blas::Layout::ColMajor,
                                   Op::NoTrans, Op::NoTrans,
                                   tile.mb()-k-kb, nb-k-kb, kb,
                                   -one, &tile.at( k+kb, k    ), tile.stride(),
                                         &tile.at( k,    k+kb ), tile.stride(),
                                   one,  &tile.at( k+kb, k+kb ), tile.stride());
                    }
                }
                else {
                    blas::gemm(blas::Layout::ColMajor,
                               Op::NoTrans, Op::NoTrans,
                               tile.mb(), nb-k-kb, kb,
                               -one, &tile.at(0, k), tile.stride(),
                                     top_block.data(), kb,
                               one,  &tile.at(0, k+kb), tile.stride());
                }
            }
        }
    }
}

} // namespace internal
} // namespace slate

#endif // SLATE_TILE_GETRF_HH
