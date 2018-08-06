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
#include "slate_Tile_blas.hh"
#include "slate_internal.hh"

namespace slate {
namespace internal {

///-----------------------------------------------------------------------------
/// \brief
/// LU factorization of a column of tiles.
/// Dispatches to target implementations.
template <Target target, typename scalar_t>
void getrf(Matrix<scalar_t>&& A, int priority)
{
    getrf(internal::TargetType<target>(), A, priority);
}

///-----------------------------------------------------------------------------
/// \brief
/// LU factorization of a column of tiles, host implementation.
template <typename scalar_t>
void getrf(internal::TargetType<Target::HostTask>,
           Matrix<scalar_t>& A, int priority)
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

    // lists of tiles, indices, and offsets for each thread
    std::vector< std::vector< Tile<scalar_t> > > tiles;
    std::vector< std::vector<int64_t> > i_indices;
    std::vector< std::vector<int64_t> > i_offsets;

    const int thread_size = 2;
    tiles.resize(thread_size);
    i_indices.resize(thread_size);
    i_offsets.resize(thread_size);

    // Build lists of tiles, indices, and offsets for each thread.
    // Assing local tiles to threads in a round-robin fashion.
    int thread_rank = 0;
    int64_t i_offset = 0;
    for (int64_t i = 0; i < A.mt(); ++i) {

        if (A.tileIsLocal(i, 0)) {

            tiles[thread_rank].push_back(A(i, 0));
            i_indices[thread_rank].push_back(i);
            i_offsets[thread_rank].push_back(i_offset);

            ++thread_rank;
            if (thread_rank == thread_size)
                thread_rank = 0;
        }
        i_offset += A.tileMb(i);
    }

    // Launch the panel tasks.
    std::vector<scalar_t> max_val(thread_size);
    std::vector<int64_t> max_idx(thread_size);
    ThreadBarrier thread_barrier;\

    #pragma omp taskloop num_tasks(thread_size) \
                         shared(thread_barrier, max_val, max_idx)
    for (int thread_rank = 0; thread_rank < thread_size; ++thread_rank)
    {
        // Factor the panel in parallel.
        getrf(tiles[thread_rank],
              i_indices[thread_rank], i_offsets[thread_rank],
              thread_rank, thread_size,
              thread_barrier,
              max_val, max_idx);
    }

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void getrf<Target::HostTask, float>(
    Matrix<float>&& A,
    int priority);

// ----------------------------------------
template
void getrf<Target::HostTask, double>(
    Matrix<double>&& A,
    int priority);

// ----------------------------------------
template
void getrf< Target::HostTask, std::complex<float> >(
    Matrix< std::complex<float> >&& A,
    int priority);

// ----------------------------------------
template
void getrf< Target::HostTask, std::complex<double> >(
    Matrix< std::complex<double> >&& A,
    int priority);

} // namespace internal
} // namespace slate
