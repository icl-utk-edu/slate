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
/// Batched routine for element-wise copy and precision conversion.
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
///     where each Barray[k] is an m-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] ldb
///     Leading dimension of each tile in B. ldb >= m.
///
/// @param[in] batch_count
///     Size of Aarray and Barray. batch_count >= 0.
///
/// @param[in] stream
///     Device to execute in.
///
template <typename src_scalar_t, typename dst_scalar_t>
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    src_scalar_t const* const* Aarray, int64_t lda,
    dst_scalar_t** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
    // quick return
    if (batch_count == 0)
        return;

    queue.sync(); // sync queue before switching to openmp device execution
    #pragma omp target is_device_ptr(Aarray, Barray) device(queue.device())
    #pragma omp teams distribute
    for (int64_t k = 0; k < batch_count; ++k) {
        src_scalar_t const* tileA = Aarray[k];
        dst_scalar_t* tileB = Barray[k];
        // distribute rows (i) to threads
        #pragma omp parallel for schedule(static, 1)
        for (int64_t i = 0; i < m; ++i) {
            src_scalar_t const* rowA = &tileA[i];
            dst_scalar_t* rowB = &tileB[i];
            if (uplo == lapack::Uplo::Lower) {
                for (int64_t j = 0; j <= i && j < n; ++j) // lower
                    rowB[j*ldb] = rowA[j*lda];
            }
            else {
                for (int64_t j = n-1; j >= i; --j) // upper
                    rowB[j*ldb] = rowA[j*lda];
            }
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    float const* const* Aarray, int64_t lda,
    float** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

template
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    float const* const* Aarray, int64_t lda,
    double** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

template
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    double const* const* Aarray, int64_t lda,
    double** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

template
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    double const* const* Aarray, int64_t lda,
    float** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

template
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<float> const* const* Aarray, int64_t lda,
    std::complex<float>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

template
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<float> const* const* Aarray, int64_t lda,
    std::complex<double>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

template
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<double> const* const* Aarray, int64_t lda,
    std::complex<double>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

template
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<double> const* const* Aarray, int64_t lda,
    std::complex<float>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

} // namespace device
} // namespace slate
