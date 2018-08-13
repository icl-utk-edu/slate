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

#include "slate_Matrix.hh"
#include "slate_HermitianMatrix.hh"
#include "slate_types.hh"
#include "slate_Tile_getrf.hh"
#include "slate_internal.hh"

namespace slate {
namespace internal {

///-----------------------------------------------------------------------------
/// \brief
/// LU factorization of a column of tiles.
/// Dispatches to target implementations.
template <Target target, typename scalar_t>
void getrf(Matrix<scalar_t>&& A, int64_t ib, int max_panel_threads,
           int priority)
{
    getrf(internal::TargetType<target>(), A, ib, max_panel_threads, priority);
}

///-----------------------------------------------------------------------------
/// \brief
/// LU factorization of a column of tiles, host implementation.
template <typename scalar_t>
void getrf(internal::TargetType<Target::HostTask>,
           Matrix<scalar_t>& A, int64_t ib, int max_panel_threads,
           int priority)
{
    assert(A.nt() == 1);

    // Move the panel to the host.
    for (int64_t i = 0; i < A.mt(); ++i) {
        if (A.tileIsLocal(i, 0)) {
            #pragma omp task shared(A) priority(priority)
            {
                A.tileMoveToHost(i, 0, A.tileDevice(i, 0));
            }
        }
    }
    #pragma omp taskwait

    // lists of local tiles, indices, and offsets
    std::vector< Tile<scalar_t> > tiles;
    std::vector<int64_t> tile_indices;
    std::vector<int64_t> tile_offsets;

    // Build the broadcast set.
    // Build lists of local tiles, indices, and offsets.
    int64_t tile_offset = 0;
    std::set<int> bcast_set;
    for (int64_t i = 0; i < A.mt(); ++i) {
        bcast_set.insert(A.tileRank(i, 0));
        if (A.tileIsLocal(i, 0)) {
            tiles.push_back(A(i, 0));
            tile_indices.push_back(i);
            tile_offsets.push_back(tile_offset);
        }
        tile_offset += A.tileMb(i);
    }

    // Create the broadcast communicator.
    // Translate the root rank.
    int bcast_rank;
    int bcast_root;
    MPI_Comm bcast_comm;
    bcast_comm = commFromSet(bcast_set,
                             A.mpiComm(), A.mpiGroup(),
                             A.tileRank(0, 0), bcast_root);
    // Find the local rank.
    MPI_Comm_rank(bcast_comm, &bcast_rank);

    // Launch the panel tasks.
    int thread_size = max_panel_threads;
    if (tiles.size() < max_panel_threads)
        thread_size = tiles.size();

    std::vector<scalar_t> max_value(thread_size);
    std::vector<int64_t> max_index(thread_size);
    std::vector<int64_t> max_offset(thread_size);
    std::vector<scalar_t> top_row(A.tileNb(0));
    ThreadBarrier thread_barrier;
    std::vector< pivot_t<scalar_t> > pivot_vector(A.tileMb(0));

    // #pragma omp parallel for \
    //     num_threads(thread_size) \
    //     shared(thread_barrier, max_value, max_index, max_offset, top_row, \
                  pivot_vector)
    #pragma omp taskloop \
        num_tasks(thread_size) \
        shared(thread_barrier, max_value, max_index, max_offset, top_row, \
               pivot_vector)
    for (int thread_rank = 0; thread_rank < thread_size; ++thread_rank)
    {
        // Factor the panel in parallel.
        getrf(ib,
              tiles, tile_indices, tile_offsets,
              thread_rank, thread_size,
              thread_barrier,
              max_value, max_index, max_offset, top_row,
              bcast_rank, bcast_root, bcast_comm,
              pivot_vector);
    }

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void getrf<Target::HostTask, float>(
    Matrix<float>&& A, int64_t ib, int max_panel_threads,
    int priority);

// ----------------------------------------
template
void getrf<Target::HostTask, double>(
    Matrix<double>&& A, int64_t ib, int max_panel_threads,
    int priority);

// ----------------------------------------
template
void getrf< Target::HostTask, std::complex<float> >(
    Matrix< std::complex<float> >&& A, int64_t ib, int max_panel_threads,
    int priority);

// ----------------------------------------
template
void getrf< Target::HostTask, std::complex<double> >(
    Matrix< std::complex<double> >&& A, int64_t ib, int max_panel_threads,
    int priority);

} // namespace internal
} // namespace slate
