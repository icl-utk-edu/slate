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

#include "slate/slate.hh"
#include "aux/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

#include <atomic>

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::getrs from internal::specialization::getrs
namespace internal {
namespace specialization {

template <typename scalar_t>
using Reflectors = std::map< std::pair<int64_t, int64_t>,
                             std::vector<scalar_t> >;

using Progress = std::vector< std::atomic<int64_t> >;

//------------------------------------------------------------------------------
template <typename scalar_t>
void tb2bd_step(Matrix<scalar_t>& A, int64_t band,
                int64_t sweep, int64_t step,
                Reflectors<scalar_t>& reflectors, omp_lock_t& lock)
{
    int64_t task = step == 0 ? 0 : (step+1)%2 + 1;
    int64_t block = (step+1)/2;
    int64_t i;
    int64_t j;

    switch (task) {
        case 0:
            i =   sweep;
            j = 1+sweep;
            if (i < A.m() && j < A.n()) {
                omp_set_lock(&lock);
                auto& v1 = reflectors[{i, j}];
                auto& v2 = reflectors[{i+1, j}];
                omp_unset_lock(&lock);
                internal::gebr1<Target::HostTask>(
                    A.slice(i, std::min(i+band-1, A.m()-1),
                            j, std::min(j+band-2, A.n()-1)),
                    v1, v2);
            }
            break;
        case 1:
            i = (block-1)*(band-1)+1+sweep;
            j =  block   *(band-1)+1+sweep;
            if (i < A.m() && j < A.n()) {
                omp_set_lock(&lock);
                auto& v1 = reflectors[{i, j-(band-1)}];
                auto& v2 = reflectors[{i, j}];
                omp_unset_lock(&lock);
                internal::gebr2<Target::HostTask>(
                    v1,
                    A.slice(i, std::min(i+band-2, A.m()-1),
                            j, std::min(j+band-2, A.n()-1)),
                    v2);
            }
            break;
        case 2:
            i = block*(band-1)+1+sweep;
            j = block*(band-1)+1+sweep;
            if (i < A.m() && j < A.n()) {
                omp_set_lock(&lock);
                auto& v1 = reflectors[{i-(band-1), j}];
                auto& v2 = reflectors[{i, j}];
                omp_unset_lock(&lock);
                internal::gebr3<Target::HostTask>(
                    v1,
                    A.slice(i, std::min(i+band-2, A.m()-1),
                            j, std::min(j+band-2, A.n()-1)),
                    v2);
            }
            break;
    }
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void tb2bd_run(Matrix<scalar_t>& A,
               int64_t band, int64_t diag_len,
               int64_t num_passes, int64_t chunk_size,
               int thread_rank, int thread_size,
               Reflectors<scalar_t>& reflectors, omp_lock_t& lock,
               Progress& progress)
{
    for (int64_t pass = 0; pass < num_passes; ++pass) {

        int64_t width = diag_len-1-(pass*chunk_size);
        int64_t num_blocks = width / (band-1);
        if (width % (band-1) > 0)
            ++num_blocks;

        for (int64_t i = 0; i < num_blocks+chunk_size-1; ++i) {
            for (int64_t j = 0; j <= i && j < chunk_size; ++j) {

                int64_t sweep = (pass*chunk_size)+j;
                int64_t block = i-j;
                if (block < num_blocks) {
                    if (block == 0) {
                        int64_t step = 0;
                        if (step % thread_size == thread_rank) {

                            // Wait until sweep-1 is two tasks ahead.
                            if (sweep > 0 && block < num_blocks-1)
                                while (progress.at(sweep-1).load() < step+2);

                            tb2bd_step(A, band, sweep, step,
                                       reflectors, lock);

                            // Mark step as done.
                            progress.at(sweep).store(step);
                        }
                    }
                    else {
                        int64_t step = 2*block-1;
                        if (step % thread_size == thread_rank) {

                            // Wait until step-1 is done in this sweep.
                            while (progress.at(sweep).load() < step-1);

                            // Wait until sweep-1 is two tasks ahead.
                            if (sweep > 0 && block < num_blocks-1)
                                while (progress.at(sweep-1).load() < step+2);

                            tb2bd_step(A, band, sweep, step,
                                       reflectors, lock);

                            // Mark step as done.
                            progress.at(sweep).store(step);
                        }

                        step = 2*block;
                        if (step % thread_size == thread_rank) {

                            // Wait until step-1 is done in this sweep.
                            while (progress.at(sweep).load() < step-1);

                            // Wait until sweep-1 is two tasks ahead.
                            if (sweep > 0 && block < num_blocks-1)
                                while (progress.at(sweep-1).load() < step+2);

                            tb2bd_step(A, band, sweep, step,
                                       reflectors, lock);

                            // Mark step as done.
                            progress.at(sweep).store(step);
                        }
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Reduced a block-bidiagonal (triangular-band) matrix to a bidiagonal form.
/// Generic implementation for any target.
/// @ingroup tb2bd_specialization
///
template <Target target, typename scalar_t>
void tb2bd(slate::internal::TargetType<target>,
           Matrix<scalar_t>& A, int64_t band, int64_t chunk_size)
{
    int64_t diag_len = std::min(A.m(), A.n());
    int64_t num_passes = (diag_len-2) / chunk_size;
    if ((diag_len-2) % chunk_size > 0)
        ++num_passes;

    omp_lock_t lock;
    omp_init_lock(&lock);
    Reflectors<scalar_t> reflectors;

    Progress progress(num_passes*chunk_size);
    for (int64_t i = 0; i < num_passes*chunk_size; ++i)
        progress.at(i).store(-1);

    #pragma omp parallel
    #pragma omp master
    {
        int thread_size = omp_get_max_threads();
        #if 1
        // Launching new threads for the band reduction guarantees progression.
        // This should never deadlock, but may be detrimental to performance.
        omp_set_nested(1);
        #pragma omp parallel for \
            num_threads(thread_size) \
            shared(reflectors, lock, progress)
        #else
        // Issuing panel operation as tasks may cause a deadlock.
        #pragma omp taskloop \
            num_tasks(thread_size) \
            shared(reflectors, lock, progress)
        #endif
        for (int thread_rank = 0; thread_rank < thread_size; ++thread_rank)
        {
            tb2bd_run(A,
                      band, diag_len,
                      num_passes, chunk_size,
                      thread_rank, thread_size,
                      reflectors, lock, progress);
        }
        #pragma omp taskwait
    }

    omp_destroy_lock(&lock);
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup tb2bd_specialization
///
template <Target target, typename scalar_t>
void tb2bd(Matrix<scalar_t>& A, int64_t band,
           const std::map<Option, Value>& opts)
{
    int64_t chunk_size;
    try {
        chunk_size = opts.at(Option::ChunkSize).i_;
        assert(chunk_size >= 1);
    }
    catch (std::out_of_range) {
        chunk_size = 1;
    }

    internal::specialization::tb2bd(internal::TargetType<target>(),
                                    A, band, chunk_size);
}

//------------------------------------------------------------------------------
///
template <typename scalar_t>
void tb2bd(Matrix<scalar_t>& A, int64_t band,
           const std::map<Option, Value>& opts)
{
    Target target;
    try {
        target = Target(opts.at(Option::Target).i_);
    }
    catch (std::out_of_range) {
        target = Target::HostTask;
    }

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            tb2bd<Target::HostTask>(A, band, opts);
            break;
        case Target::HostNest:
            tb2bd<Target::HostNest>(A, band, opts);
            break;
        case Target::HostBatch:
            tb2bd<Target::HostBatch>(A, band, opts);
            break;
        case Target::Devices:
            tb2bd<Target::Devices>(A, band, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void tb2bd<float>(
    Matrix<float>& A, int64_t band,
    const std::map<Option, Value>& opts);

template
void tb2bd<double>(
    Matrix<double>& A, int64_t band,
    const std::map<Option, Value>& opts);

template
void tb2bd< std::complex<float> >(
    Matrix< std::complex<float> >& A, int64_t band,
    const std::map<Option, Value>& opts);

template
void tb2bd< std::complex<double> >(
    Matrix< std::complex<double> >& A, int64_t band,
    const std::map<Option, Value>& opts);

} // namespace slate
