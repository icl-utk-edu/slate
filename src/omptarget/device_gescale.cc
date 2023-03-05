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
/// Kernel implementing element-wise tile scale.
/// Each thread block deals with one tile.
/// Each thread deals with one row.
/// Launched by gescale().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///
/// @param[in] numer
///     Scale value numerator.
///
/// @param[in] denom
///     Scale value denominator.
///
/// @param[in,out] A
///     An m-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] lda
///     Leading dimension of each tile in Aarray. lda >= m.
///
template <typename scalar_t, typename scalar_t2>
void gescale(
    int64_t m, int64_t n,
    scalar_t2 numer, scalar_t2 denom,
    scalar_t* A, int64_t lda,
    blas::Queue& queue)
{
    // quick return
    if (m == 0 || n == 0)
        return;

    scalar_t2 mul = numer / denom;

    queue.sync(); // sync queue before switching to openmp device execution
    // Use omp target offload
    #pragma omp target is_device_ptr(A) device(queue.device())
    #pragma omp teams distribute parallel for schedule(static, 1)
    for (int64_t i = 0; i < m; ++i) {
        scalar_t* rowA = &A[ i ];
        for (int64_t j = 0; j < n; ++j)
            rowA[ j*lda ] = rowA[ j*lda ] * mul;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gescale(
    int64_t m, int64_t n,
    float numer, float denom,
    float* A, int64_t lda,
    blas::Queue& queue);

template
void gescale(
    int64_t m, int64_t n,
    double numer, double denom,
    double* A, int64_t lda,
    blas::Queue& queue);

template
void gescale(
    int64_t m, int64_t n,
    float numer, float denom,
    std::complex<float>* A, int64_t lda,
    blas::Queue& queue);

template
void gescale(
    int64_t m, int64_t n,
    std::complex<float> numer, std::complex<float> denom,
    std::complex<float>* A, int64_t lda,
    blas::Queue& queue);

template
void gescale(
    int64_t m, int64_t n,
    double numer,  double denom,
    std::complex<double>* A, int64_t lda,
    blas::Queue& queue);

template
void gescale(
    int64_t m, int64_t n,
    std::complex<double> numer, std::complex<double> denom,
    std::complex<double>* A, int64_t lda,
    blas::Queue& queue);


//==============================================================================
namespace batch {

//------------------------------------------------------------------------------
/// Batched routine for element-wise tile scale. Sets
/// \[
///     Aarray[k] *= (numer / denom).
/// \]
/// This does NOT currently take extra care to avoid over/underflow.
///
/// @param[in] m
///     Number of rows of each tile. m >= 0.
///
/// @param[in] n
///     Number of columns of each tile. n >= 0.
///
/// @param[in] numer
///     Scale value numerator.
///
/// @param[in] denom
///     Scale value denominator.
///
/// @param[in,out] Aarray
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
template <typename scalar_t, typename scalar_t2>
void gescale(
    int64_t m, int64_t n,
    scalar_t2 numer, scalar_t2 denom,
    scalar_t** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue)
{
    // quick return
    if (m == 0 || n == 0)
        return;
    // quick return
    if (batch_count == 0)
        return;

    scalar_t2 mul = numer / denom;

    queue.sync(); // sync queue before switching to openmp device execution
    // Use omp target offload
    #pragma omp target is_device_ptr(Aarray) device(queue.device())
    #pragma omp teams distribute
    for (int64_t k = 0; k < batch_count; ++k) {
        scalar_t* A = Aarray[k];
        // distribute rows (i) to threads
        #pragma omp parallel for schedule(static, 1)
        for (int64_t i = 0; i < m; ++i) {
            scalar_t* rowA = &A[ i ];
            for (int64_t j = 0; j < n; ++j)
                rowA[ j*lda ] = rowA[ j*lda ] * mul;
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gescale(
    int64_t m, int64_t n,
    float numer, float denom,
    float** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

template
void gescale(
    int64_t m, int64_t n,
    double numer, double denom,
    double** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

template
void gescale(
    int64_t m, int64_t n,
    float numer, float denom,
    std::complex<float>** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

template
void gescale(
    int64_t m, int64_t n,
    std::complex<float> numer, std::complex<float> denom,
    std::complex<float>** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

template
void gescale(
    int64_t m, int64_t n,
    double numer,  double denom,
    std::complex<double>** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

template
void gescale(
    int64_t m, int64_t n,
    std::complex<double> numer, std::complex<double> denom,
    std::complex<double>** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

} // namespace batch
} // namespace device
} // namespace slate
