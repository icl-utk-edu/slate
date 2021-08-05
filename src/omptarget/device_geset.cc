// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/internal/device.hh"

#include <cstdio>

namespace slate {
namespace device {

//------------------------------------------------------------------------------
/// Batched routine for element-wise tile set.
///
/// @param[in] m
///     Number of rows of each tile. m >= 0.
///
/// @param[in] n
///     Number of columns of each tile. n >= 0.
///
/// @param[in] diag_value
///     The value to set on the diagonal.
///
/// @param[in] offdiag_value
///     The value to set outside of the diagonal.
///
/// @param[in] Aarray
///     Array in GPU memory of dimension batch_count, containing pointers to tiles,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] lda
///     Leading dimension of each tile in A. lda >= m.
///
/// @param[in] batch_count
///     Size of Aarray. batch_count >= 0.
///
/// @param[in] stream
///     device to execute in.
///
template <typename scalar_t>
void geset(
    int64_t m, int64_t n,
    scalar_t diag_value, scalar_t offdiag_value, scalar_t** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue)
{
    // quick return
    if (batch_count == 0)
        return;

    // Use omp target offload
    #pragma omp target is_device_ptr(Aarray) device(queue.device())
    #pragma omp teams distribute
    for (int64_t k = 0; k < batch_count; ++k) {
        #pragma omp parallel for simd collapse(2)
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                scalar_t* tileA = Aarray[k];
                scalar_t* rowA = &tileA[i];
                rowA[j*lda] = (j != i) ? offdiag_value : diag_value;
            }
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void geset(
    int64_t m, int64_t n,
    float diag_value, float offdiag_value, float** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue);

template
void geset(
    int64_t m, int64_t n,
    double diag_value, double offdiag_value, double** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue);

template
void geset(
    int64_t m, int64_t n,
    std::complex<float> diag_value, std::complex<float> offdiag_value,
    std::complex<float>** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue);

template
void geset(
    int64_t m, int64_t n,
    std::complex<double> diag_value, std::complex<double> offdiag_value,
    std::complex<double>** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue);

} // namespace device
} // namespace slate
