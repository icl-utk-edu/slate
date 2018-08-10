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

#include <blas.hh>

#include "slate_Tile.hh"
#include "slate_Tile_blas.hh"
#include "slate_types.hh"
#include "slate_util.hh"

#include <list>

namespace slate {

///-----------------------------------------------------------------------------
/// \brief
/// Compute the LU factorization of a panel.
template <typename scalar_t>
int64_t getrf(std::vector< Tile<scalar_t> >& tiles,
              std::vector<int64_t>& i_indices, std::vector<int64_t>& i_offsets,
              int thread_rank, int thread_size,
              ThreadBarrier& thread_barrier,
              std::vector<scalar_t>& max_val, std::vector<int64_t>& max_idx,
              std::vector<int64_t>& max_offs,
              scalar_t& piv_val, std::vector<scalar_t>& top_row,
              int mpi_rank, int mpi_root, MPI_Comm mpi_comm)
{
    trace::Block trace_block("lapack::getrf");

    using namespace blas;
    using real_t = blas::real_type<scalar_t>;

    int64_t ib = 4;
    bool root = mpi_rank == mpi_root;
    if (root)
        assert(i_indices[0] == 0);

    auto mb = tiles.at(0).mb();
    auto nb = tiles.at(0).nb();
    int64_t diag_len = std::min(nb, mb);

    // Loop over ib-wide stripes.
    for (int64_t k = 0; k < diag_len; k += ib) {

        // Loop over ib columns of a stripe.
        for (int64_t j = k; j < k+ib && j < diag_len; ++j) {

            if (root) {
                max_val[thread_rank] = tiles.at(0)(j, j);
                max_idx[thread_rank] = 0;
                max_offs[thread_rank] = j;
            }
            else {
                max_val[thread_rank] = tiles.at(0)(0, j);
                max_idx[thread_rank] = 0;
                max_offs[thread_rank] = 0;
            }

            //------------------
            // thread max search
            for (int64_t idx = thread_rank;
                 idx < tiles.size();
                 idx += thread_size)
            {
                auto tile = tiles.at(idx);
                auto i_index = i_indices.at(idx);

                // if diagonal tile
                if (i_index == 0) {
                    for (int64_t i = j+1; i < tile.mb(); ++i) {
                        if (std::abs(tile(i, j)) >
                            std::abs(max_val[thread_rank]))
                        {
                            max_val[thread_rank] = tile(i, j);
                            max_idx[thread_rank] = idx;
                            max_offs[thread_rank] = i;
                        }
                    }

                }
                // off diagonal tiles
                else {
                    for (int64_t i = 0; i < tile.mb(); ++i) {
                        if (std::abs(tile(i, j)) >
                            std::abs(max_val[thread_rank]))
                        {
                            max_val[thread_rank] = tile(i, j);
                            max_idx[thread_rank] = idx;
                            max_offs[thread_rank] = i;
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
                    if (std::abs(max_val[rank]) > std::abs(max_val[0])) {
                        max_val[0] = max_val[rank];
                        max_idx[0] = max_idx[rank];
                        max_offs[0] = max_offs[rank];
                    }
                }

                // MPI max abs reduction
                // todo: if root & IN_PLACE
                struct { real_t max; int loc; } max_loc_in, max_loc;
                max_loc_in.max = std::abs(max_val[0]);
                max_loc_in.loc = mpi_rank;
                MPI_Allreduce(&max_loc_in, &max_loc, 1, MPI_DOUBLE_INT,
                              MPI_MAXLOC, mpi_comm);

                // Broadcast the pivot actual value (not abs).
                piv_val = max_val[0];
                MPI_Bcast(&piv_val, 1, mpi_type<scalar_t>::value,
                          max_loc.loc, mpi_comm);

                //-----------
                // pivot swap

                // if I own the pivot
                if (max_loc.loc == mpi_rank) {
                    // if I am the root
                    if (root) {
                        // if pivot not on the diagonal
                        if (max_idx[0] > 0 && max_offs[0] > j) {
                            // local swap
                            swap(k, std::min(diag_len-k, ib),
                                 tiles.at(0), j,
                                 tiles.at(max_idx[0]), max_offs[0]);
                        }
                    }
                    // I am not the root
                    else {
                        // MPI swap with the root
                        swap(k, std::min(diag_len-k, ib),
                             tiles.at(max_idx[0]), max_offs[0], mpi_root,
                             mpi_comm);
                    }
                }
                // I don't own the pivot
                else {
                    // I am the root
                    if (root) {
                        // MPI swap with the pivot owner
                        swap(k, std::min(diag_len-k, ib),
                             tiles.at(0), j, max_loc.loc, mpi_comm);
                    }
                }

                // Broadcast the top row for the geru operation.
                if (root) {
                    auto top_tile = tiles.at(0);
                    blas::copy(std::min(k+ib-j-1, diag_len-j-1),
                               &top_tile.data()[j+(j+1)*top_tile.stride()],
                               top_tile.stride(), top_row.data(), 1);
                }
                MPI_Bcast(top_row.data(), std::min(k+ib-j-1, diag_len-j-1),
                          mpi_type<scalar_t>::value, mpi_root, mpi_comm);
            }
            thread_barrier.wait(thread_size);

            // column scaling and trailing update
            for (int64_t idx = thread_rank;
                 idx < tiles.size();
                 idx += thread_size)
            {
                auto tile = tiles.at(idx);
                auto i_index = i_indices.at(idx);

                if (i_index == 0) {
                    // diagonal tile
                    for (int64_t i = j+1; i < tile.mb(); ++i)
                        tile.at(i, j) /= tile(j, j);
                }
                else {
                    // off diagonal tile
                    for (int64_t i = 0; i < tile.mb(); ++i)
                        tile.at(i, j) /= piv_val;
                }

                // todo: make it a tile operation
                if (i_index == 0) {
                    blas::geru(blas::Layout::ColMajor,
                               tile.mb()-j-1, std::min(k+ib-j-1, diag_len-j-1),
                               -1.0, &tile.data()[j+1+j*tile.stride()], 1,
                                     top_row.data(), 1,
                                     &tile.data()[j+1+(j+1)*tile.stride()],
                                     tile.stride());
                }
                else {
                    blas::geru(blas::Layout::ColMajor,
                               tile.mb(), std::min(k+ib-j-1, diag_len-j-1),
                               -1.0, &tile.data()[j*tile.stride()], 1,
                                     top_row.data(), 1,
                                     &tile.data()[(j+1)*tile.stride()],
                                     tile.stride());
                }

            }
            thread_barrier.wait(thread_size);

        }

        return 0;

    }


}

} // namespace slate

#endif // SLATE_TILE_GETRF_HH
