// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Exception.hh"
#include "slate/internal/device.hh"

#include "device_util.cuh"

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
///     Scale value on the numerator.
///
/// @param[in] denom
///     Scale value on the denominator.
///
/// @param[in] Atiles
///     Array of tiles of dimension gridDim.x,
///     where each Atiles[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile in Atiles. lda >= m.
///
template <typename scalar_t>
__global__ void gescaleKernel(
    int64_t m, int64_t n,
    scalar_t numer, scalar_t denom, scalar_t** tilesA, int64_t lda)
{
    scalar_t* tileA = tilesA[blockIdx.x];

    // thread per row, if more rows than threads, loop by blockDim.x
    for (int64_t ridx = threadIdx.x; ridx < m; ridx += blockDim.x) {
        // todo: should the increment be ridx += 1024?
        scalar_t* rowA = &tileA[ridx];

        for (int64_t j = 0; j < n; ++j)
            rowA[j*lda] = rowA[j*lda] * numer / denom;
    }
}

//------------------------------------------------------------------------------
/// Batched routine for element-wise tile scale.
///
/// @param[in] m
///     Number of rows of each tile. m >= 0.
///
/// @param[in] n
///     Number of columns of each tile. n >= 0.
///
/// @param[in] numer
///     Scale value on the numerator.
///
/// @param[in] denom
///     Scale value on the denominator.
///
/// @param[in] Aarray
///     Array in GPU memory of dimension batch_count, containing pointers to tiles,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] lda
///     Leading dimension of each tile in A. lda >= m.
///
/// @param[in] batch_count
///     Size of Aarray and Barray. batch_count >= 0.
///
/// @param[in] stream
///     CUDA stream to execute in.
///
template <typename scalar_t>
void gescale(
    int64_t m, int64_t n,
    scalar_t numer, scalar_t denom, scalar_t** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue)
{
    // quick return
    if (batch_count == 0)
        return;

    // Max threads/block=1024 for current CUDA compute capability (<=7.5)
    int64_t nthreads = std::min((int64_t)1024 , m);

    gesscaleKernel<<<batch_count, nthreads, 0, queue.stream()>>>(
        m, n,
        numer, denom, Aarray, lda);

    cudaError_t error = cudaGetLastError();
    slate_assert(error == cudaSuccess);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gescale(
    int64_t m, int64_t n,
    float numer, float denom, float** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue);

template
void gescale(
    int64_t m, int64_t n,
    double numer, double denom, double** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue);

template
void gescale(
    int64_t m, int64_t n,
    cuFloatComplex numer, cuFloatComplex denom,
    cuFloatComplex** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue);

template
void gescale(
    int64_t m, int64_t n,
    cuDoubleComplex numer, cuDoubleComplex denom,
    cuDoubleComplex** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue);

} // namespace device
} // namespace slate
