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
/// Finds the largest absolute value of elements, for each tile in Aarray.
/// Each thread block deals with one tile.
/// Each thread deals with one row, followed by a reduction.
/// Uses dynamic shared memory array of length sizeof(real_t) * m.
/// Kernel assumes non-trivial tiles (m, n >= 1).
/// Launched by genorm().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///     Also the number of threads per block (blockDim.x), hence,
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///
/// @param[in] Aarray
///     Array of tiles of dimension gridDim.x,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= m.
///
/// @param[out] tiles_maxima
///     Array of dimension gridDim.x.
///     On exit, tiles_maxima[k] = max_{i, j} abs( A^(k)_(i, j) )
///     for tile A^(k).
///
template <typename scalar_t>
__global__ void genorm_max_kernel(
    int64_t m, int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* tiles_maxima)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const* tile = Aarray[ blockIdx.x ];

    // Save partial results in shared memory.
    HIP_DYNAMIC_SHARED( char, dynamic_data)
    real_t* row_max = (real_t*) dynamic_data;
    int chunk;
    if (threadIdx.x < blockDim.x) {
        row_max[threadIdx.x] = 0;
    }

    // This does coalesced reads of one column at a time in parallel.
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        chunk = i % blockDim.x;
        scalar_t const* row = &tile[ i ];
        real_t max = 0;

        // Each thread finds max of one row.
        for (int64_t j = 0; j < n; ++j)
            max = max_nan(max, abs(row[j*lda]));

        // Save partial results in shared memory.
        row_max[chunk] = max_nan(max, row_max[chunk]);
    }

    // Reduction to find max of tile.
    __syncthreads();
    max_nan_reduce(blockDim.x, threadIdx.x, row_max);
    if (threadIdx.x == 0) {
        tiles_maxima[blockIdx.x] = row_max[0];
    }
}

const int ib  = 32;  ///< block size for genorm_one_kernel
const int ib1 = 33;  ///< ib + 1 for stride to avoid GPU bank conflicts

//------------------------------------------------------------------------------
/// Sum of absolute values of each column of elements, for each tile in Aarray.
/// Each thread block deals with one tile.
/// Each thread deals with one column.
/// Kernel assumes non-trivial tiles (m, n >= 1).
/// Launched by genorm().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///     Also the number of threads per block (blockDim.x), hence,
///
/// @param[in] Aarray
///     Array of tiles of dimension gridDim.x,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= m.
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
__global__ void genorm_one_kernel(
    int64_t m, int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* tiles_sums, int64_t ldv)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const* tile = Aarray[ blockIdx.x ];
    HIP_DYNAMIC_SHARED( char, dynamic_data)
    real_t* shmem_tile = (real_t*)dynamic_data;
    const int k = threadIdx.x;

    for (int64_t jj = 0; jj < n; jj += ib) {
        real_t sum = 0.0;
        for (int64_t ii = 0; ii < m; ii += ib) {
            // Read 32x32 sub-tile into shared memory.
            // This does coalesced reads of one column at a time in parallel.
            for (int64_t j = 0; j < ib; ++j)
                if (jj+j < n && ii+k < m)
                    shmem_tile[ j*ib1 + k ] = abs( tile[ (jj+j)*lda + ii+k ] );
            __syncthreads();  // shmem_tile loaded

            // Each thread sums one column.
            for (int64_t i = 0; i < ib; ++i)
                if (jj+k < n && ii+i < m)
                    sum += shmem_tile[ k*ib1 + i ];
            __syncthreads();  // done with shmem_tile
        }

        if (jj+k < n)
            tiles_sums[ blockIdx.x*ldv + jj+k ] = sum;
    }
}

//------------------------------------------------------------------------------
/// Sum of absolute values of each row of elements, for each tile in Aarray.
/// Each thread block deals with one tile.
/// Each thread deals with one row.
/// Kernel assumes non-trivial tiles (m, n >= 1).
/// Launched by genorm().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///     Also the number of threads per block, hence,
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///
/// @param[in] Aarray
///     Array of tiles of dimension gridDim.x,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= m.
///
/// @param[out] tiles_sums
///     Array of dimension gridDim.x * ldv.
///     On exit, tiles_sums[k*ldv + i] = sum_{j} abs( A^(k)_(i, j) )
///     for row i of tile A^(k).
///
/// @param[in] ldv
///     Leading dimension of tiles_sums (values) array.
///
template <typename scalar_t>
__global__ void genorm_inf_kernel(
    int64_t m, int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* tiles_sums, int64_t ldv)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const* tile = Aarray[ blockIdx.x ];

    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        scalar_t const* row = &tile[ i ];

        // Each thread sums one row.
        // This does coalesced reads of one column at a time in parallel.
        real_t sum = abs(row[0]);
        for (int64_t j = 1; j < n; ++j)
            sum += abs(row[j*lda]);

        tiles_sums[ blockIdx.x*ldv + i ] = sum;
    }
}

//------------------------------------------------------------------------------
/// Sum of squares, in scaled representation, for each tile in Aarray.
/// Each thread block deals with one tile.
/// Each thread deals with one row, followed by a reduction.
/// Kernel assumes non-trivial tiles (m, n >= 1).
/// Launched by genorm().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///     Also the number of threads per block, hence,
///
/// @param[in] Aarray
///     Array of tiles of dimension blockDim.x,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= m.
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
__global__ void genorm_fro_kernel(
    int64_t m, int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* tiles_values)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const* tile = Aarray[ blockIdx.x ];
    int chunk;

    // Save partial results in shared memory.
    HIP_DYNAMIC_SHARED( char, dynamic_data)
    real_t* row_scale = (real_t*) &dynamic_data[0];
    real_t* row_sumsq = &row_scale[blockDim.x];

    real_t tile_scale = row_scale[0];
    real_t tile_sumsq = row_sumsq[0];

    // Each thread finds sum-of-squares of one row.
    // This does coalesced reads of one column at a time in parallel.
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        scalar_t const* row = &tile[ i ];
        real_t scale = 0;
        real_t sumsq = 1;
        chunk = i % blockDim.x;

        for (int64_t j = 0; j < n; ++j) {
            add_sumsq(scale, sumsq, abs(row[j*lda]));
        }

        if (i < blockDim.x) {
            row_scale[chunk] = 0;
            row_sumsq[chunk] = 1;
        }

        // Save partial results in shared memory.
        combine_sumsq(row_scale[chunk], row_sumsq[chunk], scale, sumsq);
        __syncthreads();
    }

    // Reduction to find sum-of-squares of tile.
    // todo: parallel reduction.
    if (threadIdx.x == 0) {
        tile_scale = row_scale[0];
        tile_sumsq = row_sumsq[0];
        for (int64_t chunk = 1; chunk < blockDim.x && chunk < m; ++chunk) {
            combine_sumsq(tile_scale, tile_sumsq, row_scale[chunk], row_sumsq[chunk]);
        }

        tiles_values[blockIdx.x*2 + 0] = tile_scale;
        tiles_values[blockIdx.x*2 + 1] = tile_sumsq;
    }
}

//------------------------------------------------------------------------------
// todo docs
template <typename scalar_t>
__global__ void ge_col_norms_max_kernel(
    int64_t m, int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* col_max, int64_t ldv)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const* tile = Aarray[ blockIdx.x ];
    HIP_DYNAMIC_SHARED( char, dynamic_data)
    real_t* shmem_tile = (real_t*)dynamic_data;
    const int k = threadIdx.x;

    for (int64_t jj = 0; jj < n; jj += ib) {
        real_t max = 0.0;
        for (int64_t ii = 0; ii < m; ii += ib) {
            // Read 32x32 sub-tile into shared memory.
            // This does coalesced reads of one column at a time in parallel.
            for (int64_t j = 0; j < ib; ++j)
                if (jj+j < n && ii+k < m)
                    shmem_tile[ j*ib1 + k ] = abs( tile[ (jj+j)*lda + ii+k ] );
            __syncthreads();  // shmem_tile loaded

            // Each thread compute max of one column.
            for (int64_t i = 0; i < ib; ++i)
                if (jj+k < n && ii+i < m)
                    max = max_nan( shmem_tile[ k*ib1 + i ], max );
            __syncthreads();  // done with shmem_tile
        }

        if (jj+k < n)
            col_max[ blockIdx.x*ldv + jj+k ] = max;
    }
}

//------------------------------------------------------------------------------
/// Batched routine that computes a partial norm for each tile.
///
/// @param[in] norm
///     Norm to compute. See values for description.
///
/// @param[in] scope
///     Scope of the norm.
///     - NormScope::Matrix  computes partial norm of each tile.
///     - NormScope::Columns computes norm of each column.
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
///     Leading dimension of each tile. lda >= m.
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
///     - Norm::Inf: ldv >= m.
///         On exit, values[k*ldv + i] = sum_{j} abs( A^(k)_(i, j) )
///         for 0 <= k < batch_count, 0 <= i < m.
///
///     - Norm::Max: ldv = 2.
///         On exit,
///             values[k*2 + 0] = scale_k
///             values[k*2 + 1] = sumsq_k
///         where scale_k^2 sumsq_k = sum_{i,j} abs( A^(k)_(i, j) )^2
///         for 0 <= k < batch_count.
///
/// @param[in] ldv
///     Leading dimension of values array.
///
/// @param[in] batch_count
///     Size of Aarray. batch_count >= 0.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t>
void genorm(
    lapack::Norm norm, NormScope scope,
    int64_t m, int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue)
{
    using real_t = blas::real_type<scalar_t>;
    int64_t nb = 512;

    // quick return
    if (batch_count == 0)
        return;

    hipSetDevice( queue.device() );

    if (scope == NormScope::Matrix) {

        //---------
        // max norm
        if (norm == lapack::Norm::Max) {
            if (m == 0 || n == 0) {
                blas::device_memset(values, 0, batch_count, queue);
            }
            else {
                assert(ldv == 1);
                size_t shared_mem = sizeof(real_t) * nb;
                hipLaunchKernelGGL(genorm_max_kernel, dim3(batch_count), dim3(nb), shared_mem, queue.stream(), m, n, Aarray, lda, values);
            }
        }
        //---------
        // one norm
        else if (norm == lapack::Norm::One) {
            if (m == 0 || n == 0) {
                blas::device_memset(values, 0, batch_count * n, queue);
            }
            else {
                assert(ldv >= n);
                size_t shared_mem = sizeof(real_t) * ib * ib1;
                hipLaunchKernelGGL(genorm_one_kernel, dim3(batch_count), dim3(ib), shared_mem, queue.stream(), m, n, Aarray, lda, values, ldv);
            }
        }
        //---------
        // inf norm
        else if (norm == lapack::Norm::Inf) {
            if (m == 0 || n == 0) {
                blas::device_memset(values, 0, batch_count * m, queue);
            }
            else {
                assert(ldv >= m);
                hipLaunchKernelGGL(genorm_inf_kernel, dim3(batch_count), dim3(nb), 0, queue.stream(), m, n, Aarray, lda, values, ldv);
            }
        }
        //---------
        // Frobenius norm
        else if (norm == lapack::Norm::Fro) {
            if (m == 0 || n == 0) {
                blas::device_memset(values, 0, batch_count * 2, queue);
            }
            else {
                assert(ldv == 2);
                size_t shared_mem = sizeof(real_t) * nb * 2;
                hipLaunchKernelGGL(genorm_fro_kernel, dim3(batch_count), dim3(nb), shared_mem, queue.stream(), m, n, Aarray, lda, values);
            }
        }
    }
    else if (scope == NormScope::Columns) {

        if (norm == Norm::Max) {

            if (m == 0 || n == 0) {
                blas::device_memset(values, 0, batch_count * n, queue);
            }
            else {
                assert(ldv >= n);
                size_t shared_mem = sizeof(real_t) * ib * ib1;
                hipLaunchKernelGGL(ge_col_norms_max_kernel, dim3(batch_count), dim3(ib), shared_mem, queue.stream(), m, n, Aarray, lda, values, ldv);
            }
        }
        else {
            slate_not_implemented("The norm isn't yet supported");
        }
    }
    else {
        slate_not_implemented("The norm scope isn't yet supported.");
    }

    hipError_t error = hipGetLastError();
    slate_assert(error == hipSuccess);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void genorm(
    lapack::Norm norm, NormScope scope,
    int64_t m, int64_t n,
    float const* const* Aarray, int64_t lda,
    float* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue);

template
void genorm(
    lapack::Norm norm, NormScope scope,
    int64_t m, int64_t n,
    double const* const* Aarray, int64_t lda,
    double* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue);

template
void genorm(
    lapack::Norm norm, NormScope scope,
    int64_t m, int64_t n,
    hipFloatComplex const* const* Aarray, int64_t lda,
    float* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue);

template
void genorm(
    lapack::Norm norm, NormScope scope,
    int64_t m, int64_t n,
    hipDoubleComplex const* const* Aarray, int64_t lda,
    double* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue);

} // namespace device
} // namespace slate
