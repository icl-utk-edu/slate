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
/// Finds the largest absolute value of elements, for each tile in tiles.
/// Each thread block deals with one tile.
/// Each thread deals with one row, followed by a reduction.
/// Uses dynamic shared memory array of length sizeof(real_t) * m.
/// Kernel assumes non-trivial tiles (m, n >= 1).
/// Launched by trnorm().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///     Also the number of threads per block (blockDim.x), hence,
///     m <= 1024 for current CUDA architectures (2.x to 6.x).
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///
/// @param[in] tiles
///     Array of tiles of dimension gridDim.x,
///     where each tiles[k] is an m-by-n matrix stored in an lda-by-n array.
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
__global__ void trnormMaxKernel(
    lapack::Uplo uplo, lapack::Diag diag,
    int64_t m, int64_t n,
    scalar_t const* const* tiles, int64_t lda,
    blas::real_type<scalar_t>* tiles_maxima)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const* tile = tiles[blockIdx.x];
    int chunk;

    // Save partial results in shared memory.
    extern __shared__ char dynamic_data[];
    real_t* row_max = (real_t*) dynamic_data;

    if (threadIdx.x < blockDim.x) {
        row_max[threadIdx.x] = 0;
    }
    // Each thread finds max of one row.
    // This does coalesced reads of one column at a time in parallel.
    for (int idx = threadIdx.x; idx < m; idx += blockDim.x) {
        chunk = idx % blockDim.x;

        scalar_t const* row = &tile[idx];

        real_t max = 0;
        if (uplo == lapack::Uplo::Lower) {
            if (diag == lapack::Diag::Unit) {
                if (idx < n) // diag
                    max = 1;
                for (int64_t j = 0; j < idx && j < n; ++j) // strictly lower
                    max = max_nan(max, abs(row[j*lda]));
            }
            else {
                for (int64_t j = 0; j <= idx && j < n; ++j) // lower
                    max = max_nan(max, abs(row[j*lda]));
            }
        }
        else {
            // Loop backwards (n-1 down to i) to maintain coalesced reads.
            if (diag == lapack::Diag::Unit) {
                if (idx < n) // diag
                    max = 1;
                for (int64_t j = n-1; j > idx; --j) // strictly upper
                    max = max_nan(max, abs(row[j*lda]));
            }
            else {
                for (int64_t j = n-1; j >= idx; --j) // upper
                    max = max_nan(max, abs(row[j*lda]));
            }
        }

        row_max[chunk] = max_nan(max, row_max[chunk]);
    }

    // Reduction to find max of tile.
    __syncthreads();
    max_nan_reduce(blockDim.x, threadIdx.x, row_max);
    if (threadIdx.x == 0) {
        tiles_maxima[blockIdx.x] = row_max[0];
    }
}

//------------------------------------------------------------------------------
/// Sum of absolute values of each column of elements, for each tile in tiles.
/// Each thread block deals with one tile.
/// Each thread deals with one column.
/// Kernel assumes non-trivial tiles (m, n >= 1).
/// Launched by trnorm().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///     Also the number of threads per block (blockDim.x), hence,
///     n <= 1024 for current CUDA architectures (2.x to 6.x).
///
/// @param[in] tiles
///     Array of tiles of dimension gridDim.x,
///     where each tiles[k] is an m-by-n matrix stored in an lda-by-n array.
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
__global__ void trnormOneKernel(
    lapack::Uplo uplo, lapack::Diag diag,
    int64_t m, int64_t n,
    scalar_t const* const* tiles, int64_t lda,
    blas::real_type<scalar_t>* tiles_sums, int64_t ldv)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const* tile = tiles[blockIdx.x];

    // Each thread sums one column.
    // todo: this doesn't do coalesced reads
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {

        scalar_t const* column = &tile[lda*idx];
        real_t sum = 0;

        if (uplo == lapack::Uplo::Lower) {
            if (diag == lapack::Diag::Unit) {
                if (idx < m) // diag
                    sum += 1;
                for (int64_t i = idx+1; i < m; ++i) // strictly lower
                    sum += abs(column[i]);
            }
            else {
                for (int64_t i = idx; i < m; ++i) // lower
                    sum += abs(column[i]);
            }
        }
        else {
            if (diag == lapack::Diag::Unit) {
                if (idx < m) // diag
                    sum += 1;
                for (int64_t i = 0; i < idx && i < m; ++i) // strictly upper
                    sum += abs(column[i]);
            }
            else {
                for (int64_t i = 0; i <= idx && i < m; ++i) // upper
                    sum += abs(column[i]);
            }
        }
        tiles_sums[blockIdx.x*ldv + idx] = sum;
    }
}

//------------------------------------------------------------------------------
/// Sum of absolute values of each row of elements, for each tile in tiles.
/// Each thread block deals with one tile.
/// Each thread deals with one row.
/// Kernel assumes non-trivial tiles (m, n >= 1).
/// Launched by trnorm().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///     Also the number of threads per block, hence,
///     m <= 1024 for current CUDA architectures (2.x to 6.x).
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///
/// @param[in] tiles
///     Array of tiles of dimension gridDim.x,
///     where each tiles[k] is an m-by-n matrix stored in an lda-by-n array.
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
__global__ void trnormInfKernel(
    lapack::Uplo uplo, lapack::Diag diag,
    int64_t m, int64_t n,
    scalar_t const* const* tiles, int64_t lda,
    blas::real_type<scalar_t>* tiles_sums, int64_t ldv)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const* tile = tiles[blockIdx.x];
    int chunk;

    // Each thread sums one row.
    // This does coalesced reads of one column at a time in parallel.
    for (int idx = threadIdx.x; idx < m; idx += blockDim.x) {
        chunk = idx % blockDim.x;
        scalar_t const* row = &tile[idx];
        real_t sum = 0;
        if (uplo == lapack::Uplo::Lower) {
            if (diag == lapack::Diag::Unit) {
                if (idx < n) // diag
                    sum += 1;
                for (int64_t j = 0; j < idx && j < n; ++j) // strictly lower
                    sum += abs(row[j*lda]);
            }
            else {
                for (int64_t j = 0; j <= idx && j < n; ++j) // lower
                    sum += abs(row[j*lda]);
            }
        }
        else {
            // Loop backwards (n-1 down to i) to maintain coalesced reads.
            if (diag == lapack::Diag::Unit) {
                if (idx < n) // diag
                    sum += 1;
                for (int64_t j = n-1; j > idx; --j) // strictly upper
                    sum += abs(row[j*lda]);
            }
            else {
                for (int64_t j = n-1; j >= idx; --j) // upper
                    sum += abs(row[j*lda]);
            }
        }
        tiles_sums[blockIdx.x*ldv + idx] = sum;
    }
}

//------------------------------------------------------------------------------
/// Sum of squares, in scaled representation, for each tile in tiles.
/// Each thread block deals with one tile.
/// Each thread deals with one row, followed by a reduction.
/// Kernel assumes non-trivial tiles (m, n >= 1).
/// Launched by trnorm().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///     Also the number of threads per block, hence,
///     n <= 1024 for current CUDA architectures (2.x to 6.x).
///
/// @param[in] tiles
///     Array of tiles of dimension blockDim.x,
///     where each tiles[k] is an m-by-n matrix stored in an lda-by-n array.
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
__global__ void trnormFroKernel(
    lapack::Uplo uplo, lapack::Diag diag,
    int64_t m, int64_t n,
    scalar_t const* const* tiles, int64_t lda,
    blas::real_type<scalar_t>* tiles_values)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const* tile = tiles[blockIdx.x];
    int chunk;

    // Save partial results in shared memory.
    extern __shared__ char dynamic_data[];
    real_t* row_scale = (real_t*) &dynamic_data[0];
    real_t* row_sumsq = &row_scale[blockDim.x];

    // Each thread finds sum-of-squares of one row.
    // This does coalesced reads of one column at a time in parallel.
    for (int idx = threadIdx.x; idx < m; idx += blockDim.x) {
        real_t scale = 0;
        real_t sumsq = 1;
        chunk = idx % blockDim.x;
        scalar_t const* row = &tile[idx];

        if (uplo == lapack::Uplo::Lower) {
            if (diag == lapack::Diag::Unit) {
                if (idx < n) // diag
                    add_sumsq(scale, sumsq, real_t(1));
                for (int64_t j = 0; j < idx && j < n; ++j) // strictly lower
                    add_sumsq(scale, sumsq, abs(row[j*lda]));
            }
            else {
                for (int64_t j = 0; j <= idx && j < n; ++j) // lower
                    add_sumsq(scale, sumsq, abs(row[j*lda]));
            }
        }
        else {
            // Loop backwards (n-1 down to i) to maintain coalesced reads.
            if (diag == lapack::Diag::Unit) {
                if (idx < n) // diag
                    add_sumsq(scale, sumsq, real_t(1));
                for (int64_t j = n-1; j > idx; --j) // strictly upper
                    add_sumsq(scale, sumsq, abs(row[j*lda]));
            }
            else {
                for (int64_t j = n-1; j >= idx; --j) // upper
                    add_sumsq(scale, sumsq, abs(row[j*lda]));
            }
        }

        if (idx < blockDim.x) {
            row_scale[chunk] = 0;
            row_sumsq[chunk] = 1;
        }

        add_sumsq(row_scale[chunk], row_sumsq[chunk], scale, sumsq);
        __syncthreads();
    }

    // Reduction to find sum-of-squares of tile.
    // todo: parallel reduction.
    if (threadIdx.x == 0) {
        real_t tile_scale = row_scale[0];
        real_t tile_sumsq = row_sumsq[0];
        for (int64_t chunk = 1; chunk < blockDim.x && chunk < m; ++chunk) {
            add_sumsq(tile_scale, tile_sumsq, row_scale[chunk], row_sumsq[chunk]);
        }

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
/// i = 0, ..., m-1,
/// j = 0, ..., n-1.
///
/// @param[in] m
///     Number of rows of each tile. m >= 0.
///
/// @param[in] n
///     Number of columns of each tile. n >= 0.
///     Currently, n <= 1024 due to CUDA implementation.
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
///     Leading dimension of tiles_sums (values) array.
///
/// @param[in] batch_count
///     Size of Aarray. batch_count >= 0.
///
/// @param[in] stream
///     CUDA stream to execute in.
///
template <typename scalar_t>
void trnorm(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag,
    int64_t m, int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* values, int64_t ldv, int64_t batch_count,
    cudaStream_t stream)
{
    using real_t = blas::real_type<scalar_t>;
    int64_t nb = 512;

    // quick return
    if (batch_count == 0)
        return;

    //---------
    // max norm
    if (norm == lapack::Norm::Max) {
        if (m == 0 || n == 0) {
            cudaMemsetAsync(values, 0, sizeof(real_t) * batch_count, stream);
        }
        else {
            assert(ldv == 1);
            trnormMaxKernel<<<batch_count, nb, sizeof(real_t) * nb, stream>>>
                (uplo, diag, m, n, Aarray, lda, values);
        }
    }
    //---------
    // one norm
    else if (norm == lapack::Norm::One) {
        if (m == 0 || n == 0) {
            cudaMemsetAsync(values, 0, sizeof(real_t) * batch_count * n, stream);
        }
        else {
            assert(ldv >= n);
            trnormOneKernel<<<batch_count, nb, 0, stream>>>
                (uplo, diag, m, n, Aarray, lda, values, ldv);
        }
    }
    //---------
    // inf norm
    else if (norm == lapack::Norm::Inf) {
        if (m == 0 || n == 0) {
            cudaMemsetAsync(values, 0, sizeof(real_t) * batch_count * m, stream);
        }
        else {
            assert(ldv >= m);
            trnormInfKernel<<<batch_count, nb, 0, stream>>>
                (uplo, diag, m, n, Aarray, lda, values, ldv);
        }
    }
    //---------
    // Frobenius norm
    else if (norm == lapack::Norm::Fro) {
        if (m == 0 || n == 0) {
            cudaMemsetAsync(values, 0, sizeof(real_t) * batch_count * 2, stream);
        }
        else {
            assert(ldv == 2);
            // Max 1024 threads * 16 bytes = 16 KiB shared memory in double [complex].
            trnormFroKernel<<<batch_count, nb, sizeof(real_t) * nb * 2, stream>>>
                (uplo, diag, m, n, Aarray, lda, values);
        }
    }

    slate_cuda_call(cudaGetLastError());
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void trnorm(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag,
    int64_t m, int64_t n,
    float const* const* Aarray, int64_t lda,
    float* values, int64_t ldv, int64_t batch_count,
    cudaStream_t stream);

template
void trnorm(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag,
    int64_t m, int64_t n,
    double const* const* Aarray, int64_t lda,
    double* values, int64_t ldv, int64_t batch_count,
    cudaStream_t stream);

template
void trnorm(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag,
    int64_t m, int64_t n,
    cuFloatComplex const* const* Aarray, int64_t lda,
    float* values, int64_t ldv, int64_t batch_count,
    cudaStream_t stream);

template
void trnorm(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag,
    int64_t m, int64_t n,
    cuDoubleComplex const* const* Aarray, int64_t lda,
    double* values, int64_t ldv, int64_t batch_count,
    cudaStream_t stream);

} // namespace device
} // namespace slate
