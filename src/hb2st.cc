// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/HermitianBandMatrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

#include <atomic>

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::getrs from internal::specialization::getrs
namespace internal {
namespace specialization {

using ProgressVector = std::vector< std::atomic<int64_t> >;

//------------------------------------------------------------------------------
/// @internal
/// Implements the tasks of tridiagonal bulge chasing.
///
/// @param[in,out] A
///     The band Hermitian matrix A.
///
/// @param[out] V
///     Matrix of Householder reflectors produced in the process.
///     Dimension 2*band-by-... todo.
///
/// @param[in] sweep
///     The sweep number.
///     One sweep eliminates one row and sweeps the entire matrix.
///
/// @param[in] step
///     The step number.
///     Steps in each sweep have consecutive numbers.
///
template <typename scalar_t>
void hb2st_step(HermitianBandMatrix<scalar_t>& A,
                Matrix<scalar_t>& V,
                int64_t sweep, int64_t step)
{
    int64_t n = A.n();
    int64_t band = A.bandwidth();

    // Steps 0, 1, ... map to task types 0, 1, 2, 1, 2, ...
    int64_t task = step == 0 ? 0 : (step + 1) % 2 + 1;

    // Block-column that step updates. Blocks are not aligned with Matrix tiles.
    int64_t block = step/2;

    // First row/col of the block:
    // (i, i) for diagonal blocks, (i, j) for off-diagonal blocks.
    int64_t i, j;

    // (vi, vj) is offset within tile of V for Householder vector.
    // vindex + step is tile of V for the step.
    int64_t vj = sweep % band;
    int64_t vi = vj + 1;
    int64_t k  = sweep / band;
    int64_t vindex = k*A.nt() - k*(k - 1)/2;

    switch (task) {
        // task 0 - the first task of the sweep
        // Brings col i to tridiagonal and updates the diagonal block.
        case 0:
            i = sweep;
            j = sweep;
            if (i < n && j < n) {
                int64_t m1 = std::min(i+band, n-1) - i;
                auto V1 = V(0, vindex);
                internal::hebr1<Target::HostTask>(
                    m1, &V1.at(vi, vj),
                    A.slice(i, m1 + i));
            }
            break;

        // task 1 - an off-diagonal block in the sweep
        // Applies update from task max(0, step-2), of type 0 or 1, to an
        // off-diagonal block, creating a bulge, then brings col i back
        // to the original bandwidth and updates the off-diagonal block.
        case 1:
            i = (block+1)*band + 1 + sweep;
            j =  block   *band + 1 + sweep;
            if (i < n && j < n) {
                int64_t m1 = std::min(j+band-1, n-1) - i + 1;
                int64_t m2 = std::min(i+band-1, n-1) - i + 1;
                auto V1 = V(0, vindex + (step-1)/2);
                auto V2 = V(0, vindex + (step+1)/2);
                internal::hebr2<Target::HostTask>(
                    m1, &V1.at(vi, vj),
                    m2, &V2.at(vi, vj),
                    A.slice(i, m2 + i - 1,
                            j, m1 + i - 1));
            }
            break;

        // task 2 - a diagonal block in the sweep
        // Applies update from task (step-1), of type 1, to a diagonal block.
        case 2:
            i = block*band + 1 + sweep;
            j = block*band + 1 + sweep;
            if (i < n && j < n) {
                int64_t m1 = std::min(i+band-1, n-1) - i + 1;
                auto V1 = V(0, vindex + step/2);
                internal::hebr3<Target::HostTask>(
                    m1, &V1.at(vi, vj),
                    A.slice(i, m1 + i - 1));
            }
            break;
    }
}

//------------------------------------------------------------------------------
/// @internal
/// Implements multithreaded tridiagonal bulge chasing.
/// This is the main routine that each thread runs.
///
/// @param[in,out] A
///     The band Hermitian matrix A.
///
/// @param[out] V
///     Matrix of Householder reflectors produced in the process.
///     Dimension 2*band-by-XYZ todo
///
/// @param[in] thread_rank
///     rank of this thread
///
/// @param[in] thread_size
///     number of threads
///
/// @param[in] progress
///     progress table for synchronizing threads
///
template <typename scalar_t>
void hb2st_run(HermitianBandMatrix<scalar_t>& A,
               Matrix<scalar_t>& V,
               int thread_rank, int thread_size,
               ProgressVector& progress)
{
    int64_t n = A.n();
    int64_t band = A.bandwidth();
    int64_t pass_size = ceildiv(thread_size, 3);

    // Thread that starts each pass.
    int64_t start_thread = 0;

    // Pass is indexed by the sweep that starts each pass.
    // Loop bound `pass < n-2` would be sufficient to get complex bidiagonal,
    // but `pass < n-1` makes last 2 entries real for steqr2.
    for (int64_t pass = 0; pass < n-1; pass += pass_size) {
        int64_t sweep_end = std::min(pass + pass_size, n-1);
        // Steps in first sweep of this pass; later sweeps may have fewer steps.
        int64_t nsteps_pass = 2*ceildiv(n - 1 - pass, band) - 1;
        // Step that this thread starts on, in this pass.
        int64_t step_begin = (thread_rank - start_thread + thread_size) % thread_size;
        for (int64_t step = step_begin; step < nsteps_pass; step += thread_size) {
            for (int64_t sweep = pass; sweep < sweep_end; ++sweep) {
                int64_t nsteps_sweep = 2*ceildiv(n - 1 - sweep, band) - 1;
                int64_t nsteps_last  = 2*ceildiv(n - 1 - (sweep-1), band) - 1;

                if (step < nsteps_sweep) {
                    if (sweep > 0) {
                        // Wait until sweep-1 is two tasks ahead,
                        // or sweep-1 is finished.
                        int64_t depend = std::min(step+2, nsteps_last-1);
                        while (progress.at(sweep-1).load() < depend) {}
                    }
                    if (step > 0) {
                        // Wait until step-1 is done in this sweep.
                        while (progress.at(sweep).load() < step-1) {}
                    }
                    ///printf( "tid %d pass %lld, task %lld, %lld\n",
                    //         thread_rank, pass, sweep, step );
                    hb2st_step(A, V, sweep, step);

                    // Mark step as done.
                    progress.at(sweep).store(step);
                }
            }
        }
        // Update start thread for next pass.
        start_thread = (start_thread + nsteps_pass) % thread_size;
    }
}

//------------------------------------------------------------------------------
/// @internal
/// Reduces a band Hermitian matrix to a tridiagonal matrix using bulge chasing.
/// @ingroup heev_specialization
///
template <Target target, typename scalar_t>
void hb2st(slate::internal::TargetType<target>,
           HermitianBandMatrix<scalar_t>& A,
           Matrix<scalar_t>& V)
{
    const scalar_t zero = 0.0;

    int64_t n = A.n();
    int64_t band = A.bandwidth();

    ProgressVector progress(n-1);
    for (int64_t i = 0; i < n-1; ++i)
        progress.at(i).store(-1);

    set(zero, V);

    // Insert workspace tiles needed for fill-in in bulge chasing
    // and set tile entries outside the band to 0.
    // todo: should release these tiles when done
    // WARNING: assumes lower matrix, todo:
    int jj = 0; // col index
    for (int j = 0; j < A.nt(); ++j) {
        int ii = 0; // row index
        for (int i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal(i, j)
                && (ii == jj
                    || (ii > jj && ii - (jj + A.tileNb(j) - 1) <= band + 1) ) )
            {
                if (i == j && j < A.nt()-1) {
                    auto T_ptr = A.tileInsertWorkspace( i, j+1 );
                    lapack::laset(
                        lapack::MatrixType::General, T_ptr->mb(), T_ptr->nb(),
                        zero, zero, T_ptr->data(), T_ptr->stride());
                }

                if (j > 0 && i == j + 1) {
                    auto T_ptr = A.tileInsertWorkspace( i, j-1 );
                    lapack::laset(
                        lapack::MatrixType::General, T_ptr->mb(), T_ptr->nb(),
                        zero, zero, T_ptr->data(), T_ptr->stride());
                }

                if (i == j) {
                    auto Aij = A(i, j);
                    Aij.uplo(Uplo::Upper);
                    tile::tzset( zero, Aij );
                }

                if (i == j + 1) {
                    auto Aij = A(i, j);
                    Aij.uplo(Uplo::Lower);
                    tile::tzset( zero, Aij );
                }
            }
            ii += A.tileMb(i);
        }
        jj += A.tileNb(j);
    }

    // set min number for omp nested active parallel regions
    slate::OmpSetMaxActiveLevels set_active_levels( MinOmpActiveLevels );

    #pragma omp parallel
    #pragma omp master
    {
        int thread_size = omp_get_max_threads();

        #if 1
            // Launching new threads for the band reduction guarantees progress.
            // This should never deadlock, but may be detrimental to performance.
            #pragma omp parallel for \
                        num_threads(thread_size) \
                        shared(V, progress)
        #else
            // Issuing panel operation as tasks may cause a deadlock.
            #pragma omp taskloop \
                        num_tasks(thread_size) \
                        shared(V, progress)
        #endif
        for (int thread_rank = 0; thread_rank < thread_size; ++thread_rank) {
            hb2st_run(A, V, thread_rank, thread_size, progress);
        }
        #pragma omp taskwait
    }

    // Now that chasing is over, matrix is reduced to symmetric tridiagonal.
    A.bandwidth(1);
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup heev_specialization
///
template <Target target, typename scalar_t>
void hb2st(HermitianBandMatrix<scalar_t>& A,
           Matrix<scalar_t>& V,
           Options const& opts)
{
    internal::specialization::hb2st(internal::TargetType<target>(),
                                    A, V);
}

//------------------------------------------------------------------------------
/// Reduces a band Hermitian matrix to a bidiagonal matrix using bulge chasing.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     The band Hermitian matrix A.
///
/// @param[out] V
///     Matrix of Householder reflectors produced in the process.
///     Dimension 2*band-by-XYZ todo
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  nested OpenMP parallel for loop on CPU host.
///       - HostBatch: batched BLAS on CPU host.
///       - Devices:   batched BLAS on GPU device.
///
/// @ingroup heev_computational
///
template <typename scalar_t>
void hb2st(HermitianBandMatrix<scalar_t>& A,
           Matrix<scalar_t>& V,
           Options const& opts)
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            hb2st<Target::HostTask>(A, V, opts);
            break;
        case Target::HostNest:
            hb2st<Target::HostNest>(A, V, opts);
            break;
        case Target::HostBatch:
            hb2st<Target::HostBatch>(A, V, opts);
            break;
        case Target::Devices:
            hb2st<Target::Devices>(A, V, opts);
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void hb2st<float>(
    HermitianBandMatrix<float>& A,
    Matrix<float>& V,
    Options const& opts);

template
void hb2st<double>(
    HermitianBandMatrix<double>& A,
    Matrix<double>& V,
    Options const& opts);

template
void hb2st< std::complex<float> >(
    HermitianBandMatrix< std::complex<float> >& A,
    Matrix< std::complex<float> >& V,
    Options const& opts);

template
void hb2st< std::complex<double> >(
    HermitianBandMatrix< std::complex<double> >& A,
    Matrix< std::complex<double> >& V,
    Options const& opts);

} // namespace slate
