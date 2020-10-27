// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/internal/device.hh"
#include "device_util.cuh"

#include <cstdio>
#include <cuComplex.h>

namespace slate {
namespace device {

//------------------------------------------------------------------------------
__host__ __device__
inline
float real(float a)
{
    return a;
}

__host__ __device__
float imag(float a)
{
    return 0;
}

//------------------------------------------------------------------------------
__host__ __device__
inline
double real(double a)
{
    return a;
}

__host__ __device__
inline
double imag(double a)
{
    return 0;
}

//------------------------------------------------------------------------------
__host__ __device__
inline
float real(cuFloatComplex a)
{
    return a.x;
}

__host__ __device__
inline
float imag(cuFloatComplex a)
{
    return a.y;
}

//------------------------------------------------------------------------------
__host__ __device__
inline
double real(cuDoubleComplex a)
{
    return a.x;
}

__host__ __device__
inline
double imag(cuDoubleComplex a)
{
    return a.y;
}

//------------------------------------------------------------------------------
/// Finds the largest absolute value of elements, for each tile in tiles.
/// Each thread block deals with one tile.
/// Each thread deals with one row, followed by a reduction.
/// Uses dynamic shared memory array of length sizeof(real_t) * n.
/// Kernel assumes non-trivial tiles (n >= 1).
/// Launched by henorm().
///
/// @param[in] n
///     Number of rows and columns of each tile. n >= 1.
///     Also the number of threads per block (blockDim.x), hence,
///     n <= 1024 for current CUDA architectures (2.x to 6.x).
///
/// @param[in] tiles
///     Array of tiles of dimension gridDim.x,
///     where each tiles[k] is an n-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= n.
///
/// @param[out] tiles_maxima
///     Array of dimension gridDim.x.
///     On exit, tiles_maxima[k] = max_{i, j} abs( A^(k)_(i, j) )
///     for tile A^(k).
///
template <typename scalar_t>
__global__ void henormMaxKernel(
    lapack::Uplo uplo,
    int64_t n,
    scalar_t const* const* tiles, int64_t lda,
    blas::real_type<scalar_t>* tiles_maxima)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const* tile = tiles[blockIdx.x];
    scalar_t const* row = &tile[threadIdx.x];

    // Each thread finds max of one row.
    // This does coalesced reads of one column at a time in parallel.
    real_t max = 0;
    if (uplo == lapack::Uplo::Lower) {
        for (int64_t j = 0; j < threadIdx.x && j < n; ++j) // strictly lower
            max = max_nan(max, abs(row[j*lda]));
        int64_t j = threadIdx.x;
        max = max_nan(max, abs( real( row[j*lda] )));  // diag (real)
    }
    else {
        // Loop backwards (n-1 down to i) to maintain coalesced reads.
        for (int64_t j = n-1; j > threadIdx.x; --j) // strictly upper
            max = max_nan(max, abs(row[j*lda]));
        int64_t j = threadIdx.x;
        max = max_nan(max, abs( real( row[j*lda] )));  // diag (real)
    }

    // Save partial results in shared memory.
    extern __shared__ char dynamic_data[];
    real_t* row_max = (real_t*) dynamic_data;
    row_max[threadIdx.x] = max;
    __syncthreads();

    // Reduction to find max of tile.
    max_nan_reduce(blockDim.x, threadIdx.x, row_max);
    if (threadIdx.x == 0) {
        tiles_maxima[blockIdx.x] = row_max[0];
    }
}

//------------------------------------------------------------------------------
/// Sum of absolute values of each column of elements, for each tile in tiles.
/// Each thread block deals with one tile.
/// Each thread deals with one column.
/// Kernel assumes non-trivial tiles (n >= 1).
/// Launched by henorm().
///
/// @param[in] n
///     Number of rows and columns of each tile. n >= 1.
///     Also the number of threads per block (blockDim.x), hence,
///     n <= 1024 for current CUDA architectures (2.x to 6.x).
///
/// @param[in] tiles
///     Array of tiles of dimension gridDim.x,
///     where each tiles[k] is an n-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= n.
///
/// @param[out] tiles_sums
///     Array of dimension gridDim.x * ldv.
///     On exit, tiles_sums[k*ldv + j] = max_{i} abs( A^(k)_(i, j) )
///     for row j of tile A^(k).
///
/// @param[in] ldv
///     Leading dimension of tiles_sums (values) array.
///
template <typename scalar_t>
__global__ void henormOneKernel(
    lapack::Uplo uplo,
    int64_t n,
    scalar_t const* const* tiles, int64_t lda,
    blas::real_type<scalar_t>* tiles_sums, int64_t ldv)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const* tile = tiles[blockIdx.x];
    scalar_t const* row    = &tile[threadIdx.x];
    scalar_t const* column = &tile[lda*threadIdx.x];

    // Each thread sums one row/column.
    // todo: the row reads are coalesced, but the col reads are not coalesced
    real_t sum = 0;
    if (uplo == lapack::Uplo::Lower) {
        for (int64_t j = 0; j < threadIdx.x; ++j) // strictly lower
            sum += abs(row[j*lda]);
        int64_t j = threadIdx.x;
        sum += abs( real( row[j*lda] )); // diag (real)
        for (int64_t i = threadIdx.x + 1; i < n; ++i) // strictly lower
            sum += abs(column[i]);
    }
    else {
        // Loop backwards (n-1 down to i) to maintain coalesced reads.
        for (int64_t j = n-1; j > threadIdx.x; --j) // strictly upper
            sum += abs(row[j*lda]);
        int64_t j = threadIdx.x;
        sum += abs( real( row[j*lda] )); // diag (real)
        for (int64_t i = 0; i < threadIdx.x && i < n; ++i) // strictly upper
            sum += abs(column[i]);
    }

    tiles_sums[blockIdx.x*ldv + threadIdx.x] = sum;
}

//------------------------------------------------------------------------------
/// Sum of squares, in scaled representation, for each tile in tiles.
/// Each thread block deals with one tile.
/// Each thread deals with one row, followed by a reduction.
/// Kernel assumes non-trivial tiles (n >= 1).
/// Launched by henorm().
///
/// @param[in] n
///     Number of rows and columns of each tile. n >= 1.
///     Also the number of threads per block, hence,
///     n <= 1024 for current CUDA architectures (2.x to 6.x).
///
/// @param[in] tiles
///     Array of tiles of dimension blockDim.x,
///     where each tiles[k] is an n-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= n.
///
/// @param[out] tiles_values
///     Array of dimension 2 * blockDim.x.
///     On exit,
///         tiles_values[2*k + 0] = scale
///         tiles_values[2*k + 1] = sumsq
///     such that scale^2 * sumsq = sum_{i,j} abs( A^(k)_{i,j} )^2
///     for tile A^(k).
///
template <typename scalar_t>
__global__ void henormFroKernel(
    lapack::Uplo uplo,
    int64_t n,
    scalar_t const* const* tiles, int64_t lda,
    blas::real_type<scalar_t>* tiles_values)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const* tile = tiles[blockIdx.x];
    scalar_t const* row = &tile[threadIdx.x];

    // Each thread finds sum-of-squares of one row.
    // This does coalesced reads of one column at a time in parallel.
    real_t scale = 0;
    real_t sumsq = 1;
    if (uplo == lapack::Uplo::Lower) {
        for (int64_t j = 0; j < threadIdx.x && j < n; ++j) // strictly lower
            add_sumsq(scale, sumsq, abs(row[j*lda]));
        // double for symmetric entries
        sumsq *= 2;
        // diagonal (real)
        add_sumsq(scale, sumsq, abs( real( row[threadIdx.x*lda] )));
    }
    else {
        // Loop backwards (n-1 down to i) to maintain coalesced reads.
        for (int64_t j = n-1; j > threadIdx.x; --j) // strictly upper
            add_sumsq(scale, sumsq, abs(row[j*lda]));
        // double for symmetric entries
        sumsq *= 2;
        // diagonal (real)
        add_sumsq(scale, sumsq, abs( real( row[threadIdx.x*lda] )));
    }

    // Save partial results in shared memory.
    extern __shared__ char dynamic_data[];
    real_t* row_scale = (real_t*) &dynamic_data[0];
    real_t* row_sumsq = &row_scale[n];
    row_scale[threadIdx.x] = scale;
    row_sumsq[threadIdx.x] = sumsq;
    __syncthreads();

    // Reduction to find sum-of-squares of tile.
    // todo: parallel reduction.
    if (threadIdx.x == 0) {
        real_t tile_scale = row_scale[0];
        real_t tile_sumsq = row_sumsq[0];
        for (int64_t i = 1; i < n; ++i)
            add_sumsq(tile_scale, tile_sumsq, row_scale[i], row_sumsq[i]);

        tiles_values[blockIdx.x*2 + 0] = tile_scale;
        tiles_values[blockIdx.x*2 + 1] = tile_sumsq;
    }
}

//------------------------------------------------------------------------------
/// Batched routine that returns the largest absolute value of elements for
/// each tile in Aarray. Sets
///     tiles_maxima[k] = max_{i, j}( abs( A^(k)_(i, j) )),
/// for each tile A^(k), where
/// A^(k) = Aarray[k],
/// k = 0, ..., blockDim.x-1,
/// i = 0, ..., n-1,
/// j = 0, ..., n-1.
///
/// @param[in] n
///     Number of rows and columns of each tile. n >= 0.
///     Currently, n <= 1024 due to CUDA implementation.
///
/// @param[in] Aarray
///     Array in GPU memory of dimension batch_count, containing pointers to tiles,
///     where each Aarray[k] is an n-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= n.
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
///     - Norm::Inf: for symmetric, same as Norm::One
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
///     CUDA stream to execute in.
///
template <typename scalar_t>
void henorm(
    lapack::Norm norm, lapack::Uplo uplo,
    int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* values, int64_t ldv, int64_t batch_count,
    cudaStream_t stream)
{
    using real_t = blas::real_type<scalar_t>;

    // quick return
    if (batch_count == 0)
        return;

    //---------
    // max norm
    if (norm == lapack::Norm::Max) {
        if (n == 0) {
            cudaMemsetAsync(values, 0, sizeof(real_t) * batch_count, stream);
        }
        else {
            assert(n <= 1024);
            assert(ldv == 1);
            // Max 1024 threads * 8 bytes = 8 KiB shared memory in double [complex].
            henormMaxKernel<<<batch_count, n, sizeof(real_t) * n, stream>>>
                (uplo, n, Aarray, lda, values);
        }
    }
    //---------
    // one norm
    else if (norm == lapack::Norm::One || norm == lapack::Norm::Inf) {
        if (n == 0) {
            cudaMemsetAsync(values, 0, sizeof(real_t) * batch_count * n, stream);
        }
        else {
            assert(n <= 1024);
            assert(ldv >= n);
            henormOneKernel<<<batch_count, n, 0, stream>>>
                (uplo, n, Aarray, lda, values, ldv);
        }
    }
    //---------
    // Frobenius norm
    else if (norm == lapack::Norm::Fro) {
        if (n == 0) {
            cudaMemsetAsync(values, 0, sizeof(real_t) * batch_count * 2, stream);
        }
        else {
            assert(n <= 1024);
            assert(ldv == 2);
            // Max 1024 threads * 16 bytes = 16 KiB shared memory in double [complex].
            henormFroKernel<<<batch_count, n, sizeof(real_t) * n * 2, stream>>>
                (uplo, n, Aarray, lda, values);
        }
    }

    slate_cuda_call(cudaGetLastError());
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void henorm(
    lapack::Norm norm, lapack::Uplo uplo,
    int64_t n,
    float const* const* Aarray, int64_t lda,
    float* values, int64_t ldv, int64_t batch_count,
    cudaStream_t stream);

template
void henorm(
    lapack::Norm norm, lapack::Uplo uplo,
    int64_t n,
    double const* const* Aarray, int64_t lda,
    double* values, int64_t ldv, int64_t batch_count,
    cudaStream_t stream);

template
void henorm(
    lapack::Norm norm, lapack::Uplo uplo,
    int64_t n,
    cuFloatComplex const* const* Aarray, int64_t lda,
    float* values, int64_t ldv, int64_t batch_count,
    cudaStream_t stream);

template
void henorm(
    lapack::Norm norm, lapack::Uplo uplo,
    int64_t n,
    cuDoubleComplex const* const* Aarray, int64_t lda,
    double* values, int64_t ldv, int64_t batch_count,
    cudaStream_t stream);

} // namespace device
} // namespace slate
