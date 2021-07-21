// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_TILE_GETRF_TNTPIV_HH
#define SLATE_TILE_GETRF_TNTPIV_HH

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
void getrf_tntpiv_swap(
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
/// @ingroup gesv_tile
///
template <typename scalar_t>
void getrf_tntpiv(
    int64_t diag_len, int64_t ib, //int stage, //TODO
    std::vector< Tile<scalar_t> >& tiles,
    std::vector<int64_t>& tile_indices,
    std::vector< AuxPivot<scalar_t> >& pivot,
    int mpi_rank, //int mpi_root, MPI_Comm mpi_comm, //TODO
    int thread_rank, int thread_size,
    ThreadBarrier& thread_barrier,
    std::vector<scalar_t>& max_value,
    std::vector<int64_t>& max_index,
    std::vector<int64_t>& max_offset,
    std::vector<scalar_t>& top_block)
{
    trace::Block trace_block("lapack::getrf_tntpiv");

    using real_t = blas::real_type<scalar_t>;

    const scalar_t one = 1.0;

    //bool root = mpi_rank == mpi_root; //TODO
    int64_t nb = tiles[0].nb();

    // Loop over ib-wide stripes.
    for (int64_t k = 0; k < diag_len; k += ib) {

        //=======================
        // ib panel factorization
        int64_t kb = std::min(diag_len-k, ib);

        // Loop over ib columns of a stripe.
        for (int64_t j = k; j < k+kb; ++j) {

         //   if (root) { //TODO
                max_value[thread_rank] = tiles[0](j, j);
                max_index[thread_rank] = 0;
                max_offset[thread_rank] = j;
           /* } //TODO
            else {
                max_value[thread_rank] = tiles[0](0, j);
                max_index[thread_rank] = 0;
                max_offset[thread_rank] = 0;
            }*/

            //------------------
            // thread max search
            for (int64_t idx = thread_rank;
                 idx < int64_t(tiles.size());
                 idx += thread_size)
            {
                auto tile = tiles[idx];
                //auto i_index = tile_indices[idx]; TODO

                // if diagonal tile
                if (idx == 0) { //TODO
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
                /*struct { real_t max; int loc; } max_loc_in, max_loc; //TODO
                max_loc_in.max = cabs1(max_value[0]);
                max_loc_in.loc = mpi_rank;
                #pragma omp critical(slate_mpi)
                {
                    slate_mpi_call(
                        MPI_Allreduce(&max_loc_in, &max_loc, 1,
                                      mpi_type< max_loc_type<real_t> >::value,
                                      MPI_MAXLOC, mpi_comm));
                }*/

                // todo: can this Bcast info be merged into the Allreduce?
                // Broadcast the pivot information.
                pivot[j] = AuxPivot<scalar_t>(tile_indices[max_index[0]],
                                              max_offset[0],
                                              max_index[0],
                                              max_value[0],
                                              mpi_rank); //TODO
                /*#pragma omp critical(slate_mpi) //TODO
                {
                    slate_mpi_call(
                        MPI_Bcast(&pivot[j], sizeof(AuxPivot<scalar_t>),
                                  MPI_BYTE, max_loc.loc, mpi_comm));
                }*/ 


                 //TODO::RABAB only local copy
                // pivot swap
                /*getrf_tntpiv_swap(j, k, kb,
                           tiles, pivot,
                           mpi_rank, mpi_root, mpi_comm);*/

                // if pivot not on the diagonal 
                if (pivot[j].localTileIndex() > 0 ||
                    pivot[j].elementOffset() > j)
                {
                   // local swap
                    swapLocalRow(k, kb,
                                 tiles[0], j,
                                 tiles[pivot[j].localTileIndex()],
                                 pivot[j].elementOffset());
                }  
                // Broadcast the top row for the geru operation.
                if (k+kb > j+1) {
                    //if (root) { //TODO
                        auto top_tile = tiles[0];
                        // todo: make it a tile operation
                        blas::copy(k+kb-j-1,
                                   &top_tile.at(j, j+1), top_tile.stride(),
                                   top_block.data(), 1);
                    /*} //TODO
                    #pragma omp critical(slate_mpi)
                    {
                        slate_mpi_call(
                            MPI_Bcast(top_block.data(),
                                      k+kb-j-1, mpi_type<scalar_t>::value,
                                      mpi_root, mpi_comm));
                    }*/
                }
            }
            thread_barrier.wait(thread_size);

            scalar_t one  = 1.0;
            scalar_t zero = 0.0;
            // column scaling and trailing update
            for (int64_t idx = thread_rank;
                 idx < int64_t(tiles.size());
                 idx += thread_size)
            {
                auto tile = tiles[idx];
                //auto i_index = tile_indices[idx]; TODO

                // column scaling
                real_t sfmin = std::numeric_limits<real_t>::min();
                if (cabs1(pivot[j].value()) >= sfmin) {
                    // todo: make it a tile operation
                    if (idx == 0) { //TODO
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
                    if (idx == 0) { //TODO
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
                    if (idx == 0) { //TODO
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
            thread_barrier.wait(thread_size);
        }

        // If there is a trailing submatrix.
        if (k+kb < nb) {
            //======================
            // pivoting to the right
            if (thread_rank == 0) {
                for (int64_t i = k; i < k+kb; ++i) {
                /*    getrf_tntpiv_swap(i, k+kb, nb-k-kb, //TODO local copy
                               tiles, pivot,
                               mpi_rank, mpi_root, mpi_comm);
                 */
                  // if pivot not on the diagonal 
                  if (pivot[i].localTileIndex() > 0 ||
                      pivot[i].elementOffset() > i)
                  {
                      // local swap
                      swapLocalRow(k+kb, nb-k-kb,
                                   tiles[0], i,
                                   tiles[pivot[i].localTileIndex()],
                                   pivot[i].elementOffset());
                  }
                         
                }
            }
            thread_barrier.wait(thread_size);

            //=================
            // triangular solve
            //if (root && thread_rank == 0) { TODO
            if (thread_rank == 0) {
                auto top_tile = tiles[0];
                blas::trsm(Layout::ColMajor,
                           Side::Left, Uplo::Lower,
                           Op::NoTrans, Diag::Unit,
                           kb, nb-k-kb,
                           one, &top_tile.at(k, k), top_tile.stride(),
                                &top_tile.at(k, k+kb), top_tile.stride());
            }
            thread_barrier.wait(thread_size);

            // Broadcast the top block for gemm.
            if (thread_rank == 0) {
                //if (root) { //TODO
                    auto top_tile = tiles[0];
                    lapack::lacpy(lapack::MatrixType::General,
                                  kb, nb-k-kb,
                                  &top_tile.at(k, k+kb), top_tile.stride(),
                                  top_block.data(), kb);
                /*} //TODO
                #pragma omp critical(slate_mpi)
                {
                    slate_mpi_call(
                        MPI_Bcast(top_block.data(),
                                  kb*(nb-k-kb), mpi_type<scalar_t>::value,
                                  mpi_root, mpi_comm));
                }*/
            }
            thread_barrier.wait(thread_size);

            //============================
            // rank-ib update to the right
            for (int64_t idx = thread_rank;
                 idx < int64_t(tiles.size());
                 idx += thread_size)
            {
                auto tile = tiles[idx];
                //auto i_index = tile_indices[idx]; TODO

                if (idx == 0) { //TODO
                    if (k+kb < tile.mb()) {
                        blas::gemm(blas::Layout::ColMajor,
                                   Op::NoTrans, Op::NoTrans,
                                   tile.mb()-k-kb, nb-k-kb, kb,
                                   -one, &tile.at(k+kb,k   ), tile.stride(),
                                         &tile.at(k,   k+kb), tile.stride(),
                                   one,  &tile.at(k+kb,k+kb), tile.stride());
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
            thread_barrier.wait(thread_size);
        }
    }

    //=====================
    // pivoting to the left
    for (int64_t k = ib; k < diag_len; k += ib) {
        if (thread_rank == 0) {
            for (int64_t i = k; i < k+ib && i < diag_len; ++i) {
                /*getrf_tntpiv_swap(i, 0, k,
                           tiles, pivot,
                           mpi_rank, mpi_root, mpi_comm);*/ //TODO
                //TODO::RABAB I might not need it
                // if pivot not on the diagonal 
                 if (pivot[i].localTileIndex() > 0 ||
                     pivot[i].elementOffset() > i)
                 {
                    // local swap
                    swapLocalRow(0, k,
                                 tiles[0], i,
                                 tiles[pivot[i].localTileIndex()],
                                 pivot[i].elementOffset());
                 }
            }
        }
    }
}

} // namespace internal
} // namespace slate

#endif // SLATE_TILE_GETRF_TNTPIV_HH
