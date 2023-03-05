// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
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
/// Batched routine for element-wise trapezoidal tile addition.
/// Sets upper or lower part of
/// \[
///     Barray[k] = \alpha Aarray[k] + \beta Barray[k].
/// \]
///
/// @param[in] uplo
///     Whether each Aarray[k] is upper or lower trapezoidal.
///
/// @param[in] m
///     Number of rows of each tile. m >= 0.
///
/// @param[in] n
///     Number of columns of each tile. n >= 0.
///
/// @param[in] alpha
///     The scalar alpha.
///
/// @param[in] Aarray
///     Array in GPU memory of dimension batch_count, containing pointers to tiles,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] lda
///     Leading dimension of each tile in A. lda >= m.
///
/// @param[in] beta
///     The scalar beta.
///
/// @param[in,out] Barray
///     Array in GPU memory of dimension batch_count, containing pointers to tiles,
///     where each Barray[k] is an m-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] ldb
///     Leading dimension of each tile in B. ldb >= m.
///
/// @param[in] batch_count
///     Size of Aarray and Barray. batch_count >= 0.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t>
void tzadd(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    scalar_t const& alpha, scalar_t** Aarray, int64_t lda,
    scalar_t const& beta, scalar_t** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
    // quick return
    if (batch_count == 0)
        return;

    queue.sync(); // sync queue before switching to openmp device execution
    #pragma omp target is_device_ptr(Aarray, Barray) device(queue.device())
    #pragma omp teams distribute
    for (int64_t k = 0; k < batch_count; ++k) {
        scalar_t* tileA = Aarray[ k ];
        scalar_t* tileB = Barray[ k ];
        // distribute rows (i) to thread
        #pragma omp parallel for collapse(1) schedule(static, 1)
        for (int64_t i = 0; i < m; ++i) {
            scalar_t* rowA = &tileA[ i ];
            scalar_t* rowB = &tileB[ i ];
            // add lower/upper
            if (uplo == lapack::Uplo::Lower) {
                for (int64_t j = 0; j <= i && j < n; ++j) { // lower
                    rowB[j*ldb] = alpha * rowA[j*lda] + beta * rowB[j*ldb];
                }
            }
            else {
                for (int64_t j = n-1; j >= i; --j) { // upper
                    rowB[j*ldb] = alpha * rowA[j*lda] + beta * rowB[j*ldb];
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void tzadd(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    float const& alpha, float** Aarray, int64_t lda,
    float const& beta, float** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

template
void tzadd(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    double const& alpha, double** Aarray, int64_t lda,
    double const& beta, double** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

template
void tzadd(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<float> const& alpha, std::complex<float>** Aarray, int64_t lda,
    std::complex<float> const& beta, std::complex<float>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

template
void tzadd(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<double> const& alpha, std::complex<double>** Aarray, int64_t lda,
    std::complex<double> const& beta, std::complex<double>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

} // namespace device
} // namespace slate
