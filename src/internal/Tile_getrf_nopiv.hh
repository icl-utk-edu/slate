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

#ifndef SLATE_TILE_GETRF_NOPIV_HH
#define SLATE_TILE_GETRF_NOPIV_HH

#include "internal/internal.hh"
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
/// @param[out] diag_value
///     workspace for communicating diagonal values
///
/// @param[in] tob_block
///     workspace for broadcasting the top row for the geru operation
///     and the top block for the gemm operation.
///
/// @ingroup getrf_nopiv_tile
///
template <typename scalar_t>
void getrf_nopiv(
    int64_t diag_len, int64_t ib,
    std::vector< Tile<scalar_t> >& tiles,
    std::vector<int64_t>& tile_indices,
    int mpi_rank, int mpi_root, MPI_Comm mpi_comm,
    int thread_rank, int thread_size,
    ThreadBarrier& thread_barrier,
    scalar_t& diag_value,
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

            //------------------------------------
            // the pivot is the value on the diagonal
            if (thread_rank == 0) {
                if (root) {
                    diag_value = tiles.at(0)(j, j);
                }

                // Broadcast the diag_value information.
                #pragma omp critical(slate_mpi)
                {
                    slate_mpi_call(
                        MPI_Bcast(&diag_value, 1,
                                  mpi_type<scalar_t>::value, mpi_root, mpi_comm));
                }

                // Broadcast the top row for the geru operation.
                if (k+kb > j+1) {
                    if (root) {
                        auto top_tile = tiles.at(0);
                        // todo: make it a tile operation
                        blas::copy(k+kb-j-1,
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

            scalar_t one  = 1.0;
            scalar_t zero = 0.0;
            // column scaling and trailing update
            for (int64_t idx = thread_rank;
                 idx < int64_t(tiles.size());
                 idx += thread_size)
            {
                auto tile = tiles.at(idx);
                auto i_index = tile_indices.at(idx);

                // column scaling
                real_t sfmin = std::numeric_limits<real_t>::min();
                if (cabs1(diag_value) >= sfmin) {
                    // todo: make it a tile operation
                    if (i_index == 0) {
                        // diagonal tile
                        scalar_t alpha = one / diag_value;
                        int64_t m = tile.mb()-j-1;
                        if (m > 0)
                            scal(tile.mb()-j-1, alpha, &tile.at(j+1, j), 1);
                    }
                    else {
                        // off diagonal tile
                        scalar_t alpha = one / diag_value;
                        scal(tile.mb(), alpha, &tile.at(0, j), 1);
                    }
                }
                else if (tile(j, j) != zero) {
                    if (i_index == 0) {
                        // diagonal tile
                        for (int64_t i = j+1; i < tile.mb(); ++i)
                            tile.at(i, j) /= tile(j, j);
                    }
                    else {
                        // off diagonal tile
                        for (int64_t i = 0; i < tile.mb(); ++i)
                            tile.at(i, j) /= diag_value;
                    }
                }
                else {
                    // diag_value = 0, The factorization has been completed
                    // but the factor U is exactly singular
                    // todo: how to handle a zero diag_value (pivot)
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
                 idx < int64_t(tiles.size());
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
}

} // namespace internal
} // namespace slate

#endif // SLATE_TILE_GETRF_NOPIV_HH
