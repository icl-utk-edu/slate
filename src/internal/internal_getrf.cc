// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/types.hh"
#include "internal/Tile_getrf.hh"
#include "internal/internal.hh"

namespace slate {

namespace internal {

//------------------------------------------------------------------------------
/// LU factorization of a column of tiles, host implementation.
/// @ingroup gesv_internal
///
template <typename scalar_t>
void getrf_panel(
    internal::TargetType<Target::HostTask>,
    Matrix<scalar_t>& A, int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    blas::real_type<scalar_t> pivot_threshold,
    int max_panel_threads, int priority, int tag)
{
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;
    assert(A.nt() == 1);

    // Move the panel to the host.
    std::set<ij_tuple> A_tiles_set;
    for (int64_t i = 0; i < A.mt(); ++i) {
        if (A.tileIsLocal(i, 0)) {
            A_tiles_set.insert({i, 0});
        }
    }
    A.tileGetForWriting(A_tiles_set, LayoutConvert::ColMajor);

    // lists of local tiles, indices, and offsets
    std::vector< Tile<scalar_t> > tiles;
    std::vector<int64_t> tile_indices;

    // Build the set of ranks in the panel.
    // Build lists of local tiles and their indices.
    std::set<int> ranks_set;
    for (int64_t i = 0; i < A.mt(); ++i) {
        ranks_set.insert( A.tileRank( i, 0 ) );
        if (A.tileIsLocal(i, 0)) {
            tiles.push_back(A(i, 0));
            tile_indices.push_back(i);
        }
    }

    // If participating in the panel factorization.
    if (ranks_set.find( A.mpiRank() ) != ranks_set.end()) {

        // Create the broadcast communicator.
        // Translate the root rank.
        int bcast_rank;
        int bcast_root;
        MPI_Comm bcast_comm;
        bcast_comm = commFromSet(ranks_set,
                                 A.mpiComm(), A.mpiGroup(),
                                 A.tileRank(0, 0), bcast_root,
                                 tag);
        // Find the local rank.
        MPI_Comm_rank(bcast_comm, &bcast_rank);

        // Launch the panel tasks.
        int thread_size = max_panel_threads;
        if (int(tiles.size()) < max_panel_threads)
            thread_size = tiles.size();

        ThreadBarrier thread_barrier;
        std::vector<scalar_t> max_value(thread_size);
        std::vector<int64_t> max_index(thread_size);
        std::vector<int64_t> max_offset(thread_size);
        std::vector<scalar_t> top_block(ib*A.tileNb(0));
        std::vector< AuxPivot<scalar_t> > aux_pivot(diag_len);

        #if 1
            // Launching new threads for the panel guarantees progression.
            // This should never deadlock, but may be detrimental to performance.
            #pragma omp parallel for num_threads(thread_size) slate_omp_default_none \
                shared(thread_barrier, max_value, max_index, max_offset) \
                shared(top_block, aux_pivot, tiles, bcast_comm) \
                firstprivate( tile_indices, bcast_root, bcast_rank, ib, \
                              diag_len, thread_size, pivot_threshold )
        #else
            // Issuing panel operation as tasks may cause a deadlock.
            #pragma omp taskloop num_tasks(thread_size) slate_omp_default_none \
                shared(thread_barrier, max_value, max_index, max_offset) \
                shared(top_block, aux_pivot, tiles, bcast_comm) \
                firstprivate( tile_indices, bcast_root, bcast_rank, ib, \
                              diag_len, thread_size, pivot_threshold )
        #endif
        for (int thread_rank = 0; thread_rank < thread_size; ++thread_rank) {
            // Factor the panel in parallel.
            getrf(diag_len, ib,
                  tiles, tile_indices,
                  aux_pivot,
                  bcast_rank, bcast_root, bcast_comm,
                  thread_rank, thread_size,
                  thread_barrier,
                  max_value, max_index, max_offset, top_block,
                  pivot_threshold);
        }

        // Copy pivot information from aux_pivot to pivot.
        for (int64_t i = 0; i < diag_len; ++i) {
            pivot[i] = Pivot(aux_pivot[i].tileIndex(),
                             aux_pivot[i].elementOffset());
        }

        // Free the broadcast communicator.
        slate_mpi_call(MPI_Comm_free(&bcast_comm));
    }
}

//------------------------------------------------------------------------------
/// LU factorization of a column of tiles.
/// Dispatches to target implementations.
/// @ingroup gesv_internal
///
template <Target target, typename scalar_t>
void getrf_panel(
    Matrix<scalar_t>&& A, int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    blas::real_type<scalar_t> pivot_threshold,
    int max_panel_threads, int priority, int tag)
{
    getrf_panel(
        internal::TargetType<target>(),
        A, diag_len, ib, pivot,
        pivot_threshold, max_panel_threads, priority, tag );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void getrf_panel<Target::HostTask, float>(
    Matrix<float>&& A, int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    blas::real_type<float> pivot_threshold,
    int max_panel_threads, int priority, int tag);

// ----------------------------------------
template
void getrf_panel<Target::HostTask, double>(
    Matrix<double>&& A, int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    blas::real_type<double> pivot_threshold,
    int max_panel_threads, int priority, int tag);

// ----------------------------------------
template
void getrf_panel< Target::HostTask, std::complex<float> >(
    Matrix< std::complex<float> >&& A, int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    blas::real_type<std::complex<float>> pivot_threshold,
    int max_panel_threads, int priority, int tag);

// ----------------------------------------
template
void getrf_panel< Target::HostTask, std::complex<double> >(
    Matrix< std::complex<double> >&& A, int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    blas::real_type<std::complex<double>> pivot_threshold,
    int max_panel_threads, int priority, int tag);

} // namespace internal
} // namespace slate
