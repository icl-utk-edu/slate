// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
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

template <typename scalar_t>
using Reflectors = std::map< std::pair<int64_t, int64_t>,
                             std::vector<scalar_t> >;

using Progress = std::vector< std::atomic<int64_t> >;

//------------------------------------------------------------------------------
/// @internal
/// Implements the tasks of tridiagonal bulge chasing.
///
/// @param[in,out] A
///     The band Hermitian matrix A.
///
/// @param[in] band
///     The bandwidth of matrix A.
///
/// @param[in] sweep
///     The sweep number.
///     One sweep eliminates one row and sweeps the entire matrix.
///
/// @param[in] step
///     The step number.
///     Steps in each sweep have consecutive numbers.
///
/// @param[out] reflectors
///     Householder reflectors produced by the step.
///
/// @param[in] lock
///     Lock for protecting access to reflectors.
///
template <typename scalar_t>
void hb2st_step(HermitianBandMatrix<scalar_t>& A, int64_t band,
                int64_t sweep, int64_t step,
                Reflectors<scalar_t>& reflectors, omp_lock_t& lock)
{
    int64_t task = step == 0 ? 0 : (step+1)%2 + 1;
    int64_t block = step/2;
    int64_t i;
    int64_t j;

    switch (task) {
        // task 0 - the first task of the sweep
        case 0:
            i = sweep;
            j = sweep;
            if (i < A.n() && j < A.n()) {
                omp_set_lock(&lock);
                auto& v = reflectors[{i+1, j}];
                omp_unset_lock(&lock);
                int64_t m = std::min(i+band, A.n()-1) - i;
                v.resize(m);
                internal::hebr1<Target::HostTask>(
                    A.slice(i, std::min(i+band, A.n()-1)),
                    v.size(), v.data());
            }
            break;
        // task 1 - an off-diagonal block in the sweep
        case 1:
            i = (block+1)*band+1+sweep;
            j =  block   *band+1+sweep;
            if (i < A.n() && j < A.n()) {
                omp_set_lock(&lock);
                auto& v1 = reflectors[{i-band,
                                      step == 1 ? j-1 : j-band}];
                auto& v2 = reflectors[{i, j}];
                omp_unset_lock(&lock);
                int64_t m = std::min(i+band-1, A.n()-1) - i + 1;
                v2.resize(m);
                internal::hebr2<Target::HostTask>(
                    v1.size(), v1.data(),
                    A.slice(i, std::min(i+band-1, A.n()-1),
                            j, std::min(j+band-1, A.n()-1)),
                    v2.size(), v2.data());
            }
            break;
        // task 2 - a diagonal block in the sweep
        case 2:
            i = block*band+1+sweep;
            j = block*band+1+sweep;
            if (i < A.n() && j < A.n()) {
                omp_set_lock(&lock);
                auto& v = reflectors[{i, j-band}];
                omp_unset_lock(&lock);
                internal::hebr3<Target::HostTask>(
                    v.size(), v.data(),
                    A.slice(i, std::min(i+band-1, A.n()-1)));
            }
            break;
    }
}

//------------------------------------------------------------------------------
/// @internal
/// Implements multithreaded tridiagonal bulge chasing.
///
/// @param[in,out] A
///     The band Hermitian matrix A.
///
/// @param[in] band
///     The bandwidth of matrix A.
///
/// @param[in] diag_len
///     The length of the diagonal.
///
/// @param[in] pass_size
///     The number of rows eliminated at a time.
///
/// @param[in] thread_rank
///     rank of this thread
///
/// @param[in] thread_size
///     number of threads
///
/// @param[out] reflectors
///     Householder reflectors produced in the process.
///
/// @param[in] lock
///     lock for protecting access to reflectors
///
/// @param[in] progress
///     progress table for synchronizing threads
///
template <typename scalar_t>
void hb2st_run(HermitianBandMatrix<scalar_t>& A,
               int64_t band, int64_t diag_len,
               int64_t pass_size,
               int thread_rank, int thread_size,
               Reflectors<scalar_t>& reflectors, omp_lock_t& lock,
               Progress& progress)
{
    // Thread that starts each pass.
    int64_t start_thread = 0;

    // Pass is indexed by the sweep that starts each pass.
    // pass < diag_len-2 would be sufficient to get complex bidiagonal,
    // but pass < diag_len-1 makes last 2 entries real for steqr2.
    for (int64_t pass = 0; pass < diag_len-1; pass += pass_size) {
        int64_t sweep_end = std::min(pass + pass_size, diag_len-1);
        // Steps in first sweep of this pass; later sweeps may have fewer steps.
        int64_t nsteps_pass = 2*ceildiv(diag_len - 1 - pass, band) - 1;
        // Step that this thread starts on, in this pass.
        int64_t step_begin = (thread_rank - start_thread + thread_size) % thread_size;
        for (int64_t step = step_begin; step < nsteps_pass; step += thread_size) {
            for (int64_t sweep = pass; sweep < sweep_end; ++sweep) {
                int64_t nsteps_sweep = 2*ceildiv(diag_len - 1 - sweep, band) - 1;
                int64_t nsteps_last  = 2*ceildiv(diag_len - 1 - (sweep-1), band) - 1;

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
                    ///printf( "tid %d pass %lld, task %lld, %lld\n", thread_rank, pass, sweep, step );
                    hb2st_step(A, band, sweep, step,
                               reflectors, lock);

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
           HermitianBandMatrix<scalar_t>& A)
{
    int64_t diag_len = A.n();
    int64_t band = A.bandwidth();

    omp_lock_t lock;
    omp_init_lock(&lock);
    Reflectors<scalar_t> reflectors;

    Progress progress(diag_len-1);
    for (int64_t i = 0; i < diag_len-1; ++i)
        progress.at(i).store(-1);

    // insert workspace tiles needed for fill-in in bulge chasing
    // and set tile entries outside the band to 0
    // todo: should release these tiles when done
    // WARNING: assumes lower matrix, todo:
    int jj = 0; // col index
    for (int j = 0; j < A.nt(); ++j) {
        int ii = 0; // row index
        for (int i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal(i, j) &&
                ((ii == jj) ||
                 ( ii > jj && (ii - (jj + A.tileNb(j) - 1)) <= (band+1) ) ) )
            {
                if (i == j && j < A.nt()-1) {
                    auto T_ptr = A.tileInsertWorkspace( i, j+1 );
                    lapack::laset(lapack::MatrixType::General, T_ptr->mb(), T_ptr->nb(),
                          0, 0, T_ptr->data(), T_ptr->stride());
                }

                if ((j > 0) && (i == (j + 1))) {
                    auto T_ptr = A.tileInsertWorkspace( i, j-1 );
                    lapack::laset(lapack::MatrixType::General, T_ptr->mb(), T_ptr->nb(),
                          0, 0, T_ptr->data(), T_ptr->stride());
                }

                if (i == j) {
                    auto Aij = A(i, j);
                    Aij.uplo(Uplo::Upper);
                    tzset(scalar_t(0), Aij);
                }

                if (i == (j + 1)) {
                    auto Aij = A(i, j);
                    Aij.uplo(Uplo::Lower);
                    tzset(scalar_t(0), Aij);
                }
            }
            ii += A.tileMb(i);
        }
        jj += A.tileNb(j);
    }

    #pragma omp parallel
    #pragma omp master
    {
        int thread_size = omp_get_max_threads();
        int64_t pass_size = ceildiv(thread_size, 3);

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
        for (int thread_rank = 0; thread_rank < thread_size; ++thread_rank) {
            hb2st_run(A,
                      band, diag_len,
                      pass_size,
                      thread_rank, thread_size,
                      reflectors, lock, progress);
        }
        #pragma omp taskwait
    }

    omp_destroy_lock(&lock);

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
           Options const& opts)
{
    internal::specialization::hb2st(internal::TargetType<target>(),
                                    A);
}

//------------------------------------------------------------------------------
/// Reduces a band Hermitian matrix to a bidiagonal matrix using bulge chasing.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///         One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///         The band Hermitian matrix A.
///
/// @param[in] band
///         The bandwidth of matrix A.
///
/// @param[in] opts
///         Additional options, as map of name = value pairs. Possible options:
///         - Option::Target:
///           Implementation to target. Possible values:
///           - HostTask:  OpenMP tasks on CPU host [default].
///           - HostNest:  nested OpenMP parallel for loop on CPU host.
///           - HostBatch: batched BLAS on CPU host.
///           - Devices:   batched BLAS on GPU device.
///
/// @ingroup heev_computational
///
// todo: Change Matrix to BandMatrix and remove the band parameter.
template <typename scalar_t>
void hb2st(HermitianBandMatrix<scalar_t>& A,
           Options const& opts)
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            hb2st<Target::HostTask>(A, opts);
            break;
        case Target::HostNest:
            hb2st<Target::HostNest>(A, opts);
            break;
        case Target::HostBatch:
            hb2st<Target::HostBatch>(A, opts);
            break;
        case Target::Devices:
            hb2st<Target::Devices>(A, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void hb2st<float>(
    HermitianBandMatrix<float>& A,
    Options const& opts);

template
void hb2st<double>(
    HermitianBandMatrix<double>& A,
    Options const& opts);

template
void hb2st< std::complex<float> >(
    HermitianBandMatrix< std::complex<float> >& A,
    Options const& opts);

template
void hb2st< std::complex<double> >(
    HermitianBandMatrix< std::complex<double> >& A,
    Options const& opts);

} // namespace slate
