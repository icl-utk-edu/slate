// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/internal/device.hh"

#include <cstdio>

namespace slate {
namespace device {

//------------------------------------------------------------------------------
/// Element-wise m-by-n matrix A
/// to diag_value on the diagonal and offdiag_value on the off-diagonals.
///
/// @param[in] m
///     Number of rows of A. m >= 0.
///
/// @param[in] n
///     Number of columns of A. n >= 0.
///
/// @param[in] offdiag_value
///     The value to set outside of the diagonal.
///
/// @param[in] diag_value
///     The value to set on the diagonal.
///
/// @param[out] A
///     An m-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] lda
///     Leading dimension of A. lda >= m.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t>
void geset(
    int64_t m, int64_t n,
    scalar_t const& offdiag_value, scalar_t const& diag_value,
    scalar_t* A, int64_t lda,
    blas::Queue &queue)
{
    // quick return
    if (m == 0 || n == 0)
        return;

    queue.sync(); // sync queue before switching to openmp device execution
    // Use omp target offload
    #pragma omp target is_device_ptr(A) device(queue.device())
    #pragma omp teams distribute parallel for schedule(static, 1)
    for (int64_t i = 0; i < m; ++i) {
        scalar_t* rowA = &A[i];
        for (int64_t j = 0; j < n; ++j) {
            rowA[j*lda] = (j != i) ? offdiag_value : diag_value;
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void geset(
    int64_t m, int64_t n,
    float const& offdiag_value, float const& diag_value,
    float* A, int64_t lda,
    blas::Queue &queue);

template
void geset(
    int64_t m, int64_t n,
    double const& offdiag_value, double const& diag_value,
    double* A, int64_t lda,
    blas::Queue &queue);

template
void geset(
    int64_t m, int64_t n,
    std::complex<float> const& offdiag_value, std::complex<float> const& diag_value,
    std::complex<float>* A, int64_t lda,
    blas::Queue &queue);

template
void geset(
    int64_t m, int64_t n,
    std::complex<double> const& offdiag_value, std::complex<double> const& diag_value,
    std::complex<double>* A, int64_t lda,
    blas::Queue &queue);

//==============================================================================
namespace batch {

//------------------------------------------------------------------------------
/// Initializes a batch of m-by-n matrices Aarray[k]
/// to diag_value on the diagonal and offdiag_value on the off-diagonals.
///
/// @param[in] m
///     Number of rows of each tile. m >= 0.
///
/// @param[in] n
///     Number of columns of each tile. n >= 0.
///
/// @param[in] offdiag_value
///     The value to set outside of the diagonal.
///
/// @param[in] diag_value
///     The value to set on the diagonal.
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
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t>
void geset(
    int64_t m, int64_t n,
    scalar_t const& offdiag_value, scalar_t const& diag_value,
    scalar_t** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue)
{
    // quick return
    if (batch_count == 0)
        return;
    // quick return
    if (m == 0 || n == 0)
        return;

    queue.sync(); // sync queue before switching to openmp device execution
    // Use omp target offload
    #pragma omp target is_device_ptr(Aarray) device(queue.device())
    #pragma omp teams distribute
    for (int64_t k = 0; k < batch_count; ++k) {
        scalar_t* tileA = Aarray[k];
        // distribute i to threads
        #pragma omp parallel for schedule(static, 1)
        for (int64_t i = 0; i < m; ++i) {
            scalar_t* rowA = &tileA[i];
            for (int64_t j = 0; j < n; ++j) {
                rowA[j*lda] = (j == i ? diag_value : offdiag_value);
            }
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void geset(
    int64_t m, int64_t n,
    float const& offdiag_value, float const& diag_value,
    float** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue);

template
void geset(
    int64_t m, int64_t n,
    double const& offdiag_value, double const& diag_value,
    double** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue);

template
void geset(
    int64_t m, int64_t n,
    std::complex<float> const& offdiag_value, std::complex<float> const& diag_value,
    std::complex<float>** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue);

template
void geset(
    int64_t m, int64_t n,
    std::complex<double> const& offdiag_value, std::complex<double> const& diag_value,
    std::complex<double>** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue);

} // namespace batch
} // namespace device
} // namespace slate
