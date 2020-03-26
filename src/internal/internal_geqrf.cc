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

#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/types.hh"
#include "internal/Tile_geqrf.hh"
#include "internal/internal.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// QR factorization of a column of tiles.
/// Dispatches to target implementations.
/// @ingroup geqrf_internal
///
template <Target target, typename scalar_t>
void geqrf(Matrix<scalar_t>&& A, Matrix<scalar_t>&& T,
           int64_t ib, int max_panel_threads, int priority)
{
    geqrf(internal::TargetType<target>(),
          A, T, ib, max_panel_threads, priority);
}

//------------------------------------------------------------------------------
/// QR factorization of a column of tiles, host implementation.
/// @ingroup geqrf_internal
///
template <typename scalar_t>
void geqrf(internal::TargetType<Target::HostTask>,
           Matrix<scalar_t>& A, Matrix<scalar_t>& T,
           int64_t ib, int max_panel_threads, int priority)
{
    using real_t = blas::real_type<scalar_t>;

    assert(A.nt() == 1);

    // Move the panel to the host.
    for (int64_t i = 0; i < A.mt(); ++i) {
        if (A.tileIsLocal(i, 0)) {
            #pragma omp task shared(A) priority(priority)
            {
                A.tileGetForWriting(i, 0, LayoutConvert::ColMajor);
            }
        }
    }
    #pragma omp taskwait

    // todo: What about coherency protocol for T?
    // Should be invalidated in device memory if exists.

    // Build lists of local tiles and indices.
    std::vector< Tile<scalar_t> > tiles;
    std::vector<int64_t> tile_indices;

    for (int64_t i = 0; i < A.mt(); ++i) {
        if (A.tileIsLocal(i, 0)) {
            tiles.push_back(A(i, 0));
            tile_indices.push_back(i);
        }
    }

    // If participating in the panel factorization.
    if (tiles.size() > 0) {

        // Launch the panel tasks.
        int thread_size = max_panel_threads;
        if (int(tiles.size()) < max_panel_threads)
            thread_size = tiles.size();

        T.tileInsert(tile_indices[0], 0);
        T.tileModified(tile_indices[0], 0);// todo: is this necessary?
        auto T00 = T(tile_indices[0], 0);
        T00.set(0);

        ThreadBarrier thread_barrier;
        std::vector<real_t> scale(thread_size);
        std::vector<real_t> sumsq(thread_size);
        real_t xnorm;
        std::vector< std::vector<scalar_t> > W(thread_size);

        #if 1
        omp_set_nested(1);
        #pragma omp parallel for \
            num_threads(thread_size) \
            shared(thread_barrier, scale, sumsq, xnorm, W)
        #else
        #pragma omp taskloop \
            num_tasks(thread_size) \
            shared(thread_barrier, scale, sumsq, xnorm, W)
        #endif
        for (int thread_rank = 0; thread_rank < thread_size; ++thread_rank) {
            // Factor the panel in parallel.
            // todo: double check the size of W.
            W.at(thread_rank).resize(ib*A.tileNb(0));
            geqrf(ib,
                  tiles, tile_indices, T00,
                  thread_rank, thread_size,
                  thread_barrier,
                  scale, sumsq, xnorm, W);
        }
        #pragma omp taskwait
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void geqrf<Target::HostTask, float>(
    Matrix<float>&& A, Matrix<float>&& T,
    int64_t ib, int max_panel_threads, int priority);

// ----------------------------------------
template
void geqrf<Target::HostTask, double>(
    Matrix<double>&& A, Matrix<double>&& T,
    int64_t ib, int max_panel_threads, int priority);

// ----------------------------------------
template
void geqrf< Target::HostTask, std::complex<float> >(
    Matrix< std::complex<float> >&& A, Matrix< std::complex<float> >&& T,
    int64_t ib, int max_panel_threads, int priority);

// ----------------------------------------
template
void geqrf< Target::HostTask, std::complex<double> >(
    Matrix< std::complex<double> >&& A, Matrix< std::complex<double> >&& T,
    int64_t ib, int max_panel_threads, int priority);

} // namespace internal
} // namespace slate
