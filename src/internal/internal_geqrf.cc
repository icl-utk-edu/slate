// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

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
    #pragma omp taskgroup
    for (int64_t i = 0; i < A.mt(); ++i) {
        if (A.tileIsLocal(i, 0)) {
            #pragma omp task default(none) shared(A) firstprivate(i) priority(priority)
            {
                A.tileGetForWriting(i, 0, LayoutConvert::ColMajor);
            }
        }
    }

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
        T.tileModified(tile_indices[0], 0);
        auto T00 = T(tile_indices[0], 0);
        T00.set(0);

        ThreadBarrier thread_barrier;
        std::vector<real_t> scale(thread_size);
        std::vector<real_t> sumsq(thread_size);
        real_t xnorm;
        std::vector< std::vector<scalar_t> > W(thread_size);

        #if 1
            #pragma omp parallel default(none) \
                num_threads(thread_size) \
                shared(thread_barrier, scale, sumsq, xnorm, W, A, T00) \
                shared(tile_indices, tiles) \
                firstprivate(ib, thread_size)
        #else
            #pragma omp taskloop default(none) \
                num_tasks(thread_size) \
                shared(thread_barrier, scale, sumsq, xnorm, W, A, T00) \
                shared(tile_indices, tiles) \
                firstprivate(ib, thread_size)
        #endif
        {
            // Factor the panel in parallel.
            // todo: double check the size of W.
            int thread_rank = omp_get_thread_num();
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
