#include "hip/hip_runtime.h"
// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Exception.hh"
#include "slate/internal/device.hh"

#include "device_util.hip.hh"

#include <cstdio>

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
/// @param[in,out] Aarray
///     Array of tiles of dimension gridDim.x,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile in Aarray. lda >= m.
///
template <typename scalar_t>
__global__ void tzscale_kernel(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    blas::real_type<scalar_t> numer, blas::real_type<scalar_t> denom,
    scalar_t** Aarray, int64_t lda)
{
    scalar_t* tileA = Aarray[ blockIdx.x ];
    blas::real_type<scalar_t> mul = numer / denom;

    // thread per row, if more rows than threads, loop by blockDim.x
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        scalar_t* rowA = &tileA[ i ];

        if (uplo == lapack::Uplo::Lower) {
            for (int64_t j = 0; j <= i && j < n; ++j) { // lower
                rowA[j*lda] = rowA[j*lda] * mul;
            }
        }
        else {
            for (int64_t j = n-1; j >= i; --j) // upper
                rowA[j*lda] = rowA[j*lda] * mul;
        }
    }
}

//------------------------------------------------------------------------------
/// Batched routine for element-wise trapezoidal tile scale.
/// Sets upper or lower part of
/// \[
///     Aarray[k] *= (numer / denom).
/// \]
/// This does NOT currently take extra care to avoid over/underflow.
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
template <typename scalar_t>
void tzscale(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    blas::real_type<scalar_t> numer, blas::real_type<scalar_t> denom,
    scalar_t** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue)
{
    // quick return
    if (batch_count == 0)
        return;

    hipSetDevice( queue.device() );

    // Max threads/block=1024 for current CUDA compute capability (<= 7.5)
    int64_t nthreads = std::min( int64_t( 1024 ), m );

    hipLaunchKernelGGL(tzscale_kernel, dim3(batch_count), dim3(nthreads), 0, queue.stream(),
        uplo, m, n,
        numer, denom, Aarray, lda);

    hipError_t error = hipGetLastError();
    slate_assert(error == hipSuccess);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void tzscale(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    float numer, float denom, float** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

template
void tzscale(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    double numer, double denom, double** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

template
void tzscale(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    float numer, float denom,
    hipFloatComplex** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

template
void tzscale(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    double numer, double denom,
    hipDoubleComplex** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

} // namespace device
} // namespace slate
