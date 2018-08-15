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

#include "slate_Tile.hh"
#include "slate_Tile_blas.hh"
#include "slate_Tile_lapack.hh"
#include "slate_types.hh"
#include "slate_util.hh"

#include <list>

#include <blas.hh>
#include <lapack.hh>

namespace slate {

///-----------------------------------------------------------------------------
/// \brief
/// Compute the LU factorization of a panel.
template <typename scalar_t>
int64_t getrf(int64_t diagonal_length, int64_t ib,
              std::vector< Tile<scalar_t> >& tiles,
              std::vector<int64_t>& tile_indices,
              std::vector<int64_t>& tile_offsets,
              int thread_rank, int thread_size,
              ThreadBarrier& thread_barrier,
              std::vector<scalar_t>& max_value, std::vector<int64_t>& max_index,
              std::vector<int64_t>& max_offset,
              std::vector<scalar_t>& top_block,
              int mpi_rank, int mpi_root, MPI_Comm mpi_comm,
              std::vector< Pivot<scalar_t> >& pivot_vector)
{
    trace::Block trace_block("lapack::getrf");

    using namespace blas;
    using namespace lapack;
    using real_t = real_type<scalar_t>;

    bool root = mpi_rank == mpi_root;
    int64_t nb = tiles.at(0).nb();

    // Loop over ib-wide stripes.
    for (int64_t k = 0; k < diagonal_length; k += ib) {

        //=======================
        // ib panel factorization
        int64_t kb = std::min(diagonal_length-k, ib);

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
                pivot_vector[j] = {max_loc.loc,
                                   max_value[0],
                                   max_index[0],
                                   max_offset[0]};
                #pragma omp critical(slate_mpi)
                {
                    slate_mpi_call(
                        MPI_Bcast(&pivot_vector[j], sizeof(Pivot<scalar_t>),
                                  MPI_BYTE, max_loc.loc, mpi_comm));
                }

                //-----------
                // pivot swap

                // If I own the pivot.
                if (max_loc.loc == mpi_rank) {
                    // if I am the root
                    if (root) {
                        // if pivot not on the diagonal
                        if (max_index[0] > 0 || max_offset[0] > j) {
                            // local swap
                            swap(k, kb,
                                 tiles.at(0), j,
                                 tiles.at(max_index[0]), max_offset[0]);
                        }
                    }
                    // I am not the root.
                    else {
                        // MPI swap with the root
                        swap(k, kb,
                             tiles.at(max_index[0]), max_offset[0],
                             mpi_root, mpi_comm);
                    }
                }
                // I don't own the pivot.
                else {
                    // I am the root.
                    if (root) {
                        // MPI swap with the pivot owner
                        swap(k, kb,
                             tiles.at(0), j, max_loc.loc, mpi_comm);
                    }
                }

                // Broadcast the top row for the geru operation.
                if (root) {
                    auto top_tile = tiles.at(0);
                    // todo: make it a tile operation
                    copy(k+kb-j-1,
                         &top_tile.data()[j+(j+1)*top_tile.stride()],
                         top_tile.stride(), top_block.data(), 1);
                }
                #pragma omp critical(slate_mpi)
                {
                    slate_mpi_call(
                        MPI_Bcast(top_block.data(),
                                  k+kb-j-1, mpi_type<scalar_t>::value,
                                  mpi_root, mpi_comm));
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

                real_t sfmin = std::numeric_limits<real_t>::min();
                if (cabs1(tile(j, j)) >= sfmin) {
                    if (i_index == 0) {
                        // diagonal tile
                        for (int64_t i = j+1; i < tile.mb(); ++i)
                            tile.at(i, j) /= tile(j, j);
                    }
                    else {
                        // off diagonal tile
                        for (int64_t i = 0; i < tile.mb(); ++i)
                            tile.at(i, j) /= pivot_vector[j].value;
                    }
                }
                else {
                    // todo: make it a tile operation
                    if (i_index == 0) {
                        // diagonal tile
                        scalar_t one = 1.0;
                        scalar_t alpha = one / tile(j, j);
                        scal(tile.mb()-j-1, alpha,
                             &tile.data()[j+1+j*tile.stride()], 1);
                    }
                    else {
                        // off diagonal tile
                        scalar_t one = 1.0;
                        scalar_t alpha = one / pivot_vector[j].value;
                        scal(tile.mb(), alpha,
                             &tile.data()[j*tile.stride()], 1);
                    }
                }

                // todo: make it a tile operation
                if (i_index == 0) {
                    geru(Layout::ColMajor,
                         tile.mb()-j-1, k+kb-j-1,
                         -1.0, &tile.data()[j+1+j*tile.stride()], 1,
                               top_block.data(), 1,
                               &tile.data()[j+1+(j+1)*tile.stride()],
                               tile.stride());
                }
                else {
                    geru(Layout::ColMajor,
                         tile.mb(), k+kb-j-1,
                         -1.0, &tile.data()[j*tile.stride()], 1,
                               top_block.data(), 1,
                               &tile.data()[(j+1)*tile.stride()],
                               tile.stride());
                }

            }
            thread_barrier.wait(thread_size);
        }

        //======================
        // pivoting to the right
        if (thread_rank == 0) {

            for (int64_t i = k; i < k+kb; ++i) {

                // If I own the pivot.
                if (pivot_vector[i].rank == mpi_rank) {
                    // if I am the root
                    if (root) {
                        // if pivot not on the diagonal
                        if (pivot_vector[i].tile_index > 0 ||
                            pivot_vector[i].element_offset > i)
                        {
                            // local swap
                            swap(k+kb, nb-k-kb,
                                 tiles.at(0), i,
                                 tiles.at(pivot_vector[i].tile_index),
                                 pivot_vector[i].element_offset);
                        }
                    }
                    // I am not the root.
                    else {
                        // MPI swap with the root
                        swap(k+kb, nb-k-kb,
                             tiles.at(pivot_vector[i].tile_index),
                             pivot_vector[i].element_offset,
                             mpi_root, mpi_comm);
                    }
                }
                // I don't own the pivot.
                else {
                    // I am the root.
                    if (root) {
                        // MPI swap with the pivot owner
                        swap(k+kb, nb-k-kb,
                             tiles.at(0), i, pivot_vector[i].rank, mpi_comm);
                    }
                }
            }
        }
        thread_barrier.wait(thread_size);

        //=================
        // triangular solve
        if (root && thread_rank == 0) {

            auto top_tile = tiles.at(0);
            int64_t stride = top_tile.stride();
            trsm(Layout::ColMajor,
                 Side::Left, Uplo::Lower,
                 Op::NoTrans, Diag::Unit,
                 kb, nb-k-kb,
                 1.0, &top_tile.data()[k+k*stride], stride,
                      &top_tile.data()[k+(k+kb)*stride], stride);
        }
        thread_barrier.wait(thread_size);

        // Broadcast the top block for gemm.
        if (thread_rank == 0) {
            if (root) {
                auto top_tile = tiles.at(0);
                lacpy(MatrixType::General,
                      kb, nb-k-kb,
                      &top_tile.data()[k+(k+kb)*top_tile.stride()],
                      top_tile.stride(),
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

            const int64_t& stride = tile.stride();

            if (i_index == 0) {
                gemm(Layout::ColMajor,
                     Op::NoTrans, Op::NoTrans,
                     tile.mb()-k-kb, nb-k-kb, kb,
                     -1.0, &tile.data()[k+kb+k*stride], stride,
                           &tile.data()[k+(k+kb)*stride], stride,
                      1.0, &tile.data()[(k+kb)+(k+kb)*stride], stride);
            }
            else {
                gemm(Layout::ColMajor,
                     Op::NoTrans, Op::NoTrans,
                     tile.mb(), nb-k-kb, kb,
                     -1.0, &tile.data()[k*stride], stride,
                           top_block.data(), kb,
                      1.0, &tile.data()[(k+kb)*stride], stride);
            }


        }
        thread_barrier.wait(thread_size);
    }

    //=====================
    // pivoting to the left

    for (int64_t k = ib; k < diagonal_length; k += ib) {

        if (thread_rank == 0) {

            for (int64_t i = k; i < k+ib; ++i) {

                // If I own the pivot.
                if (pivot_vector[i].rank == mpi_rank) {
                    // if I am the root
                    if (root) {
                        // if pivot not on the diagonal
                        if (pivot_vector[i].tile_index > 0 ||
                            pivot_vector[i].element_offset > i)
                        {
                            // local swap
                            swap(k-ib, ib,
                                 tiles.at(0), i,
                                 tiles.at(pivot_vector[i].tile_index),
                                 pivot_vector[i].element_offset);
                        }
                    }
                    // I am not the root.
                    else {
                        // MPI swap with the root
                        swap(k-ib, ib,
                             tiles.at(pivot_vector[i].tile_index),
                             pivot_vector[i].element_offset,
                             mpi_root, mpi_comm);
                    }
                }
                // I don't own the pivot.
                else {
                    // I am the root.
                    if (root) {
                        // MPI swap with the pivot owner
                        swap(k-ib, ib,
                             tiles.at(0), i, pivot_vector[i].rank, mpi_comm);
                    }
                }
            }
        }
    }
}

} // namespace slate

#endif // SLATE_TILE_GETRF_HH
