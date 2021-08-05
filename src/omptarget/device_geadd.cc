// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Exception.hh"
#include "slate/internal/device.hh"

#include <cstdio>

namespace slate {
namespace device {

//------------------------------------------------------------------------------
/// Batched routine for element-wise tile addition.
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
///     Leading dimension of each tile in A. lda >= m.
///
/// @param[out] Barray
///     Array in GPU memory of dimension batch_count, containing pointers to tiles,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] ldb
///     Leading dimension of each tile in B. ldb >= m.
///
/// @param[in] batch_count
///     Size of Aarray and Barray. batch_count >= 0.
///
/// @param[in] queue
///     blas device queue to execute in.
///
template <typename scalar_t>
void geadd(
    int64_t m, int64_t n,
    scalar_t alpha, scalar_t** Aarray, int64_t lda,
    scalar_t beta, scalar_t** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
    // quick return
    if (batch_count == 0)
        return;

    // Use omp target offload
    #pragma omp target is_device_ptr(Aarray, Barray) device(queue.device())
    #pragma omp teams distribute
    for (int64_t k = 0; k < batch_count; ++k) {
        scalar_t* tileA = Aarray[k];
        scalar_t* tileB = Barray[k];
        // distribute rows (i) to threads
        #pragma omp parallel for simd schedule(static, 1)
        for (int64_t i = 0; i < m; ++i) {
            scalar_t* rowA = &tileA[i];
            scalar_t* rowB = &tileB[i];
            for (int64_t j = 0; j < n; ++j) {
                rowB[j*ldb] = alpha * rowA[j*lda] + beta * rowB[j*ldb];
            }
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void geadd(
    int64_t m, int64_t n,
    float alpha, float** Aarray, int64_t lda,
    float beta, float** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

template
void geadd(
    int64_t m, int64_t n,
    double alpha, double** Aarray, int64_t lda,
    double beta, double** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

template
void geadd(
    int64_t m, int64_t n,
    std::complex<float> alpha, std::complex<float>** Aarray, int64_t lda,
    std::complex<float> beta, std::complex<float>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

template
void geadd(
    int64_t m, int64_t n,
    std::complex<double> alpha, std::complex<double>** Aarray, int64_t lda,
    std::complex<double> beta, std::complex<double>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

} // namespace device
} // namespace slate
