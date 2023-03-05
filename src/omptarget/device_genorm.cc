// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Exception.hh"
#include "slate/internal/device.hh"

#include "device_util.hh"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <complex>

namespace slate {
namespace device {

//------------------------------------------------------------------------------
/// Batched routine that returns the largest absolute value of elements for
/// each tile in Aarray. Sets
///     tiles_maxima[k] = max_{i, j}( abs( A^(k)_(i, j) )),
/// for each tile A^(k), where
/// A^(k) = Aarray[k],
/// k = 0, ..., blockDim.x-1,
/// i = 0, ..., m-1,
/// j = 0, ..., n-1.
///
/// @param[in] m
///     Number of rows of each tile. m >= 0.
///
/// @param[in] n
///     Number of columns of each tile. n >= 0.
///
/// @param[in] Aarray
///     Array in GPU memory of dimension batch_count, containing pointers to tiles,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= m.
///
/// @param[out] values
///     Array in GPU memory, dimension batch_count * ldv.
///     - Norm::Max: ldv = 1.
///         On exit, values[k] = max_{i, j} abs( A^(k)_(i, j) )
///         for 0 <= k < batch_count.
///
///     - Norm::One: ldv >= n.
///         On exit, values[k*ldv + j] = sum_{i} abs( A^(k)_(i, j) )
///         for 0 <= k < batch_count, 0 <= j < n.
///
///     - Norm::Inf: ldv >= m.
///         On exit, values[k*ldv + i] = sum_{j} abs( A^(k)_(i, j) )
///         for 0 <= k < batch_count, 0 <= i < m.
///
///     - Norm::Fro: ldv = 2.
///         On exit,
///             values[k*2 + 0] = scale_k
///             values[k*2 + 1] = sumsq_k
///         where scale_k^2 sumsq_k = sum_{i,j} abs( A^(k)_(i, j) )^2
///         for 0 <= k < batch_count.
///
/// @param[in] ldv
///     Leading dimension of tiles_sums (values) array.
///
/// @param[in] batch_count
///     Size of Aarray. batch_count >= 0.
///
/// @param[in] stream
///     device to execute in.
///
template <typename scalar_t>
void genorm(
    lapack::Norm norm, NormScope scope,
    int64_t m, int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue)
{
    using real_t = blas::real_type<scalar_t>;

    // quick return
    if (batch_count == 0)
        return;

    if (scope == NormScope::Matrix) {

        //---------
        // max norm
        if (norm == lapack::Norm::Max) {
            if (m == 0 || n == 0) {
                blas::device_memset(values, 0, batch_count, queue);
            }
            else {
                assert(ldv == 1);
                blas::device_memset(values, 0, batch_count, queue);
                queue.sync(); // sync queue before switching to openmp device execution
                // Use omp target offload
                // note: the max_nan_reduction preserves nans
                #pragma omp target is_device_ptr(Aarray, values) device(queue.device())
                #pragma omp teams distribute
                for (int64_t k = 0; k < batch_count; ++k) {
                    const scalar_t* tileA = Aarray[k];
                    #pragma omp parallel for reduction(max_nan_reduction:values[k]) schedule(static, 1)
                    for (int64_t i = 0; i < m; ++i) {
                        const scalar_t* rowA = &tileA[i];
                        real_t max = 0;
                        for (int64_t j = 0; j < n; ++j) {
                            max = max_nan(max, abs_val(rowA[j*lda]));
                        }
                        values[k] = max_nan(values[k], max);
                    }
                }
            }
        }
        //---------
        // one norm
        else if (norm == lapack::Norm::One) {
            if (m == 0 || n == 0) {
                blas::device_memset(values, 0, batch_count * n, queue);
            }
            else {
                assert(ldv >= n);
                queue.sync(); // sync queue before switching to openmp device execution
                // use omp target offload
                #pragma omp target is_device_ptr(Aarray, values) device(queue.device())
                #pragma omp teams distribute
                for (int64_t k = 0; k < batch_count; ++k) {
                    // distribute cols to threads (j)
                    #pragma omp parallel for schedule(static, 1)
                    for (int64_t j = 0; j < n; ++j) {
                        values[k*ldv + j] = 0;
                        for (int64_t i = 0; i < m; ++i) {
                            values[k*ldv + j] += abs_val( Aarray[k][i + j*lda] );
                        }
                    }
                }
            }
        }
        //---------
        // inf norm
        else if (norm == lapack::Norm::Inf) {
            if (m == 0 || n == 0) {
                blas::device_memset(values, 0, batch_count * m, queue);
            }
            else {
                assert(ldv >= m);
                queue.sync(); // sync queue before switching to openmp device execution
                // use omp target offload
                #pragma omp target is_device_ptr(Aarray, values) device(queue.device())
                #pragma omp teams distribute
                for (int64_t k = 0; k < batch_count; ++k) {
                    // distribute rows to threads (i)
                    #pragma omp parallel for schedule(static, 1)
                    for (int64_t i = 0; i < m; ++i) {
                        values[k*ldv + i] = 0;
                        for (int64_t j = 0; j < n; ++j) {
                            values[k*ldv + i] += abs_val( Aarray[k][i + j*lda] );
                        }
                    }
                }
            }
        }
        //---------
        // Frobenius norm
        else if (norm == lapack::Norm::Fro) {
            if (m == 0 || n == 0) {
                blas::device_memset(values, 0, batch_count * 2, queue);
            }
            else {
                assert(ldv == 2);
                queue.sync(); // sync queue before switching to openmp device execution
                // use omp target offload
                #pragma omp target is_device_ptr(Aarray, values) device(queue.device())
                #pragma omp teams distribute
                for (int64_t k = 0; k < batch_count; ++k) {
                    const scalar_t* tileA = Aarray[k];
                    values[k*2 + 0] = 0; // scale_k
                    values[k*2 + 1] = 1; // sumsq_k
                    // distribute rows to threads (i)
                    #pragma omp parallel for schedule(static, 1)
                    for (int64_t i = 0; i < m; ++i) {
                        real_t scale_ki = 0;
                        real_t sumsq_ki = 1;
                        for (int64_t j = 0; j < n; ++j)
                            add_sumsq(scale_ki, sumsq_ki, abs_val( tileA[i + j*lda] ));
                        // accumulate the scale and sumsq for each k
                        // todo: critical section is slow, is there a better way?
                        #pragma omp critical
                        {
                            real_t scale_k = values[k*2 + 0];
                            real_t sumsq_k = values[k*2 + 1];
                            if (scale_k > scale_ki) {
                                sumsq_k = sumsq_k + sumsq_ki*(scale_ki / scale_k)*(scale_ki / scale_k);
                                // scale_k stays same
                            }
                            else if (scale_ki != 0) {
                                sumsq_k = sumsq_k*(scale_k / scale_ki)*(scale_k / scale_ki) + sumsq_ki;
                                scale_k = scale_ki;
                            }
                            values[k*2 + 0] = scale_k;
                            values[k*2 + 1] = sumsq_k;
                        }
                    }
                }
            }
        }
    }
    else if (scope == NormScope::Columns) {

        if (norm == Norm::Max) {

            if (m == 0 || n == 0) {
                blas::device_memset(values, 0, batch_count * n, queue);
            }
            else {
                // todo:  NOTE THIS NORM IS NOT CHECKED YET
                assert(ldv >= n);
                blas::device_memset(values, 0, batch_count * n, queue);
                queue.sync(); // sync queue before switching to openmp device execution
                // Use omp target offload
                #pragma omp target is_device_ptr(Aarray, values) device(queue.device())
                #pragma omp teams distribute
                for (int64_t k = 0; k < batch_count; ++k) {
                    const scalar_t* tileA = Aarray[k];
                    // distribute cols to threads (j)
                    #pragma omp parallel for collapse(1) schedule(static, 1)
                    for (int64_t j = 0; j < n; ++j) {
                        const scalar_t* colA = &tileA[j*lda];
                        for (int64_t i = 0; i < m; ++i) {
                            values[j*ldv] = max_nan(values[j*ldv], abs_val(colA[i]));
                        }
                    }
                }
            }
        }
        else {
            slate_not_implemented("The norm isn't yet supported");
        }
    }
    else {
        slate_not_implemented("The norm scope isn't yet supported.");
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void genorm(
    lapack::Norm norm, NormScope scope,
    int64_t m, int64_t n,
    float const* const* Aarray, int64_t lda,
    float* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue);

template
void genorm(
    lapack::Norm norm, NormScope scope,
    int64_t m, int64_t n,
    double const* const* Aarray, int64_t lda,
    double* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue);

template
void genorm(
    lapack::Norm norm, NormScope scope,
    int64_t m, int64_t n,
    std::complex<float> const* const* Aarray, int64_t lda,
    float* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue);

template
void genorm(
    lapack::Norm norm, NormScope scope,
    int64_t m, int64_t n,
    std::complex<double> const* const* Aarray, int64_t lda,
    double* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue);

} // namespace device
} // namespace slate
