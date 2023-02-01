// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Exception.hh"
#include "slate/internal/device.hh"

#include "device_util.hh"

#include <cstdio>

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
///     - Norm::Max: ldv = 2.
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
void trnorm(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag,
    int64_t m, int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue)
{
    using real_t = blas::real_type<scalar_t>;

    // quick return
    if (batch_count == 0)
        return;

    //---------
    // max norm
    if (norm == lapack::Norm::Max) {
        if (m == 0 || n == 0) {
            blas::device_memset(values, 0, batch_count, queue);
        }
        else {
            assert(ldv == 1);
            // use omp offload
            blas::device_memset(values, 0, batch_count, queue);
            queue.sync(); // sync queue before switching to openmp device execution
            #pragma omp target is_device_ptr(Aarray, values) device(queue.device())
            #pragma omp teams distribute
            for (int64_t k = 0; k < batch_count; ++k) {
                const scalar_t* tileA = Aarray[k];
                // distribute rows (i) to threads, each thread computes 1 column max
                // nan-preserving max reduction operation
                #pragma omp parallel for reduction(max_nan_reduction:values[k]) schedule(static, 1)
                for (int64_t i = 0; i < m; ++i) {
                    const scalar_t* row = &tileA[i];
                    real_t max = 0;
                    if (uplo == lapack::Uplo::Lower) {
                        if (diag == lapack::Diag::Unit) {
                            if (i < n) // unit diag
                                max = 1;
                            for (int64_t j = 0; j < i && j < n; ++j) // strictly lower
                                max = max_nan(max, abs_val(row[j*lda]));
                        }
                        else { // non-unit
                            for (int64_t j = 0; j <= i && j < n; ++j) // lower
                                max = max_nan(max, abs_val(row[j*lda]));
                        }
                    }
                    else { // Upper
                        // Loop backwards (n-1 down to i)
                        if (diag == lapack::Diag::Unit) {
                            if (i < n) // diag
                                max = 1;
                            for (int64_t j = n-1; j > i; --j) // strictly upper
                                max = max_nan(max, abs_val(row[j*lda]));
                        }
                        else { // non-unit
                            for (int64_t j = n-1; j >= i; --j) // upper
                                max = max_nan(max, abs_val(row[j*lda]));
                        }
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
            blas::device_memset(values, 0, batch_count * n, queue);
            queue.sync(); // sync queue before switching to openmp device execution
            #pragma omp target is_device_ptr(Aarray, values) device(queue.device())
            #pragma omp teams distribute
            for (int64_t k = 0; k < batch_count; ++k) {
                const scalar_t* tileA = Aarray[k];
                // distribute cols (j) to threads
                // each thread computes one column sum
                #pragma omp parallel for schedule(static, 1)
                for (int64_t j = 0; j < n; ++j) {
                    const scalar_t* column = &tileA[lda*j];
                    real_t sum = 0;
                    if (uplo == lapack::Uplo::Lower) {
                        if (diag == lapack::Diag::Unit) {
                            if (j < m) // diag
                                sum += 1;
                            for (int64_t i = j+1; i < m; ++i) // strictly lower
                                sum += abs_val(column[i]);
                        }
                        else { // diag == non-unit
                            for (int64_t i = j; i < m; ++i) // lower
                                sum += abs_val(column[i]);
                        }
                    }
                    else { // uplo = upper
                        if (diag == lapack::Diag::Unit) {
                            if (j < m) // diag
                                sum += 1;
                            for (int64_t i = 0; i < j && i < m; ++i) // strictly upper
                                sum += abs_val(column[i]);
                        }
                        else {
                            for (int64_t i = 0; i <= j && i < m; ++i) // upper
                                sum += abs_val(column[i]);
                        }
                    }
                    values[k*ldv + j] = sum;
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
            #pragma omp target is_device_ptr(Aarray, values) device(queue.device())
            #pragma omp teams distribute
            for (int64_t k = 0; k < batch_count; ++k) {
                const scalar_t* tileA = Aarray[k];
                // distribute i to threads, each thread sums one row
                #pragma omp parallel for schedule(static, 1)
                for (int64_t i = 0; i < m; ++i) {
                    scalar_t const* row = &tileA[i];
                    real_t sum = 0;
                    if (uplo == lapack::Uplo::Lower) {
                        if (diag == lapack::Diag::Unit) {
                            if (i < n) // diag
                                sum += 1;
                            for (int64_t j = 0; j < i && j < n; ++j) // strictly lower
                                sum += abs_val(row[j*lda]);
                        }
                        else {
                            for (int64_t j = 0; j <= i && j < n; ++j) // lower
                                sum += abs_val(row[j*lda]);
                        }
                    }
                    else {
                        // Loop backwards (n-1 down to i)
                        if (diag == lapack::Diag::Unit) {
                            if (i < n) // diag
                                sum += 1;
                            for (int64_t j = n-1; j > i; --j) // strictly upper
                                sum += abs_val(row[j*lda]);
                        }
                        else {
                            for (int64_t j = n-1; j >= i; --j) // upper
                                sum += abs_val(row[j*lda]);
                        }
                    }
                    values[k*ldv + i] = sum;
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
            blas::device_memset(values, 0, batch_count * 2, queue);
            queue.sync(); // sync queue before switching to openmp device execution
            #pragma omp target is_device_ptr(Aarray, values) device(queue.device())
            #pragma omp teams distribute
            // distribute each batch array to a team
            for (int64_t k = 0; k < batch_count; ++k) {
                const scalar_t* tileA = Aarray[k];
                // distribute rows (i) to threads
                #pragma omp parallel for schedule(static, 1)
                for (int64_t i = 0; i < m; ++i) {
                    const scalar_t* row = &tileA[i];
                    real_t scale = 0;
                    real_t sumsq = 1;
                    if (uplo == lapack::Uplo::Lower) {
                        if (diag == lapack::Diag::Unit) {
                            if (i < n) // diag
                                add_sumsq(scale, sumsq, real_t(1));
                            for (int64_t j = 0; j < i && j < n; ++j) // strictly lower
                                add_sumsq(scale, sumsq, abs_val(row[j*lda]));
                        }
                        else { // diag == non-unit
                            for (int64_t j = 0; j <= i && j < n; ++j) // lower
                                add_sumsq(scale, sumsq, abs_val(row[j*lda]));
                        }
                    }
                    else { // uplo == upper
                        // Loop backwards (n-1 down to i)
                        if (diag == lapack::Diag::Unit) {
                            if (i < n) // diag
                                add_sumsq(scale, sumsq, real_t(1));
                            for (int64_t j = n-1; j > i; --j) // strictly upper
                                add_sumsq(scale, sumsq, abs_val(row[j*lda]));
                        }
                        else { // diag == non-unit
                            for (int64_t j = n-1; j >= i; --j) // upper
                                add_sumsq(scale, sumsq, abs_val(row[j*lda]));
                        }
                    }
                    // accumulate the scale and sumsq for each k
                    // todo: this is slow; (reduction with 2 values? two-reductions?)
                    #pragma omp critical
                    {
                        real_t scale_k = values[k*2 + 0];
                        real_t sumsq_k = values[k*2 + 1];
                        if (scale_k > scale) {
                            sumsq_k = sumsq_k + sumsq*(scale / scale_k)*(scale / scale_k);
                            // scale_k stays same
                        }
                        else if (scale != 0) {
                            sumsq_k = sumsq_k*(scale_k / scale)*(scale_k / scale) + sumsq;
                            scale_k = scale;
                        }
                        values[k*2 + 0] = scale_k;
                        values[k*2 + 1] = sumsq_k;
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void trnorm(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag,
    int64_t m, int64_t n,
    float const* const* Aarray, int64_t lda,
    float* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue);

template
void trnorm(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag,
    int64_t m, int64_t n,
    double const* const* Aarray, int64_t lda,
    double* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue);

template
void trnorm(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag,
    int64_t m, int64_t n,
    std::complex<float> const* const* Aarray, int64_t lda,
    float* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue);

template
void trnorm(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag,
    int64_t m, int64_t n,
    std::complex<double> const* const* Aarray, int64_t lda,
    double* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue);

} // namespace device
} // namespace slate
