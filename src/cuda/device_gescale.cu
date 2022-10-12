// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
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
/// Device function implementing element-wise tile scale.
/// Each thread block deals with one tile. gridDim.x == batch_count.
/// Each thread deals with one row.
/// Called by gescale_kernel and gescale_batch_kernel.
///
/// @copydoc gescale
///
template <typename scalar_t, typename scalar_t2>
__device__ void gescale_func(
    int64_t m, int64_t n,
    scalar_t2 mul,
    scalar_t* A, int64_t lda)
{
    // thread per row, if more rows than threads, loop by blockDim.x
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        scalar_t* rowA = &A[ i ];
        for (int64_t j = 0; j < n; ++j)
            rowA[ j*lda ] = rowA[ j*lda ] * mul;
    }
}

//------------------------------------------------------------------------------
/// Kernel implementing element-wise tile scale.
/// @copydoc gescale
template <typename scalar_t, typename scalar_t2>
__global__ void gescale_kernel(
    int64_t m, int64_t n,
    scalar_t2 mul,
    scalar_t* A, int64_t lda)
{
    gescale_func( m, n, mul, A, lda );
}

//------------------------------------------------------------------------------
/// Kernel implementing element-wise tile scale.
/// @copydoc gescale_batch
template <typename scalar_t, typename scalar_t2>
__global__ void gescale_batch_kernel(
    int64_t m, int64_t n,
    scalar_t2 mul,
    scalar_t** Aarray, int64_t lda)
{
    gescale_func( m, n, mul, Aarray[ blockIdx.x ], lda );
}

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

    cudaSetDevice( queue.device() );

    // Max threads/block=1024 for current CUDA compute capability (<= 7.5)
    int64_t nthreads = std::min( int64_t( 1024 ), m );

    scalar_t2 mul = numer / denom;

    gescale_kernel<<<1, nthreads, 0, queue.stream()>>>(
        m, n, mul, A, lda );

    cudaError_t error = cudaGetLastError();
    slate_assert(error == cudaSuccess);
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
    cuFloatComplex* A, int64_t lda,
    blas::Queue& queue);

template
void gescale(
    int64_t m, int64_t n,
    cuFloatComplex numer, cuFloatComplex denom,
    cuFloatComplex* A, int64_t lda,
    blas::Queue& queue);

template
void gescale(
    int64_t m, int64_t n,
    double numer,  double denom,
    cuDoubleComplex* A, int64_t lda,
    blas::Queue& queue);

template
void gescale(
    int64_t m, int64_t n,
    cuDoubleComplex numer, cuDoubleComplex denom,
    cuDoubleComplex* A, int64_t lda,
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

    cudaSetDevice( queue.device() );

    // Max threads/block=1024 for current CUDA compute capability (<= 7.5)
    int64_t nthreads = std::min( int64_t( 1024 ), m );

    scalar_t2 mul = numer / denom;

    gescale_batch_kernel<<<batch_count, nthreads, 0, queue.stream()>>>(
        m, n,
        mul, Aarray, lda);

    cudaError_t error = cudaGetLastError();
    slate_assert(error == cudaSuccess);
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
    cuFloatComplex** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

template
void gescale(
    int64_t m, int64_t n,
    cuFloatComplex numer, cuFloatComplex denom,
    cuFloatComplex** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

template
void gescale(
    int64_t m, int64_t n,
    double numer,  double denom,
    cuDoubleComplex** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

template
void gescale(
    int64_t m, int64_t n,
    cuDoubleComplex numer, cuDoubleComplex denom,
    cuDoubleComplex** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

} // namespace batch
} // namespace device
} // namespace slate
