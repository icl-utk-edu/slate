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
/// Finds the largest absolute value of elements, for each tile in tiles.
/// Each thread block deals with one tile.
/// Each thread deals with one row, followed by a reduction.
/// Uses dynamic shared memory array of length sizeof(real_t) * n.
/// Kernel assumes non-trivial tiles (n >= 1).
/// Launched by synorm().
///
/// @param[in] n
///     Number of rows and columns of each tile. n >= 1.
///     Also the number of threads per block (blockDim.x), hence,
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
__global__ void synormMaxKernel(
    lapack::Uplo uplo,
    int64_t n,
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
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        chunk = idx % blockDim.x;

        scalar_t const* row = &tile[idx];
        if (idx < blockDim.x) {
            row_max[chunk] = 0;
        }

        real_t max = 0;
        if (uplo == lapack::Uplo::Lower) {
            for (int64_t j = 0; j <= idx && j < n; ++j) // lower
                max = max_nan(max, abs(row[j*lda]));
        }
        else {
            // Loop backwards (n-1 down to i) to maintain coalesced reads.
            for (int64_t j = n-1; j >= idx; --j) // upper
                max = max_nan(max, abs(row[j*lda]));
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
/// Kernel assumes non-trivial tiles (n >= 1).
/// Launched by synorm().
///
/// @param[in] n
///     Number of rows and columns of each tile. n >= 1.
///     Also the number of threads per block (blockDim.x), hence,
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
__global__ void synormOneKernel(
    lapack::Uplo uplo,
    int64_t n,
    scalar_t const* const* tiles, int64_t lda,
    blas::real_type<scalar_t>* tiles_sums, int64_t ldv)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const* tile = tiles[blockIdx.x];

    // Each thread sums one row/column.
    // todo: the row reads are coalesced, but the col reads are not coalesced
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        scalar_t const* row    = &tile[idx];
        scalar_t const* column = &tile[lda*idx];
        real_t sum = 0;

        if (uplo == lapack::Uplo::Lower) {
            for (int64_t j = 0; j <= idx; ++j) // lower
                sum += abs(row[j*lda]);
            for (int64_t i = idx + 1; i < n; ++i) // strictly lower
                sum += abs(column[i]);
        }
        else {
            // Loop backwards (n-1 down to i) to maintain coalesced reads.
            for (int64_t j = n-1; j >= idx; --j) // upper
                sum += abs(row[j*lda]);
            for (int64_t i = 0; i < idx && i < n; ++i) // strictly upper
                sum += abs(column[i]);
        }
        tiles_sums[blockIdx.x*ldv + idx] = sum;
    }
}

//------------------------------------------------------------------------------
/// Sum of squares, in scaled representation, for each tile in tiles.
/// Each thread block deals with one tile.
/// Each thread deals with one row, followed by a reduction.
/// Kernel assumes non-trivial tiles (n >= 1).
/// Launched by synorm().
///
/// @param[in] n
///     Number of rows and columns of each tile. n >= 1.
///     Also the number of threads per block, hence,
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
__global__ void synormFroKernel(
    lapack::Uplo uplo,
    int64_t n,
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
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        real_t scale = 0;
        real_t sumsq = 1;
        chunk = idx % blockDim.x;
        scalar_t const* row = &tile[idx];

        if (uplo == lapack::Uplo::Lower) {
            for (int64_t j = 0; j < idx && j < n; ++j) // strictly lower
                add_sumsq(scale, sumsq, abs(row[j*lda]));
            // double for symmetric entries
            sumsq *= 2;
            // diagonal
            add_sumsq(scale, sumsq, abs(row[idx*lda]));
        }
        else {
            // Loop backwards (n-1 down to i) to maintain coalesced reads.
            for (int64_t j = n-1; j > idx; --j) // strictly upper
                add_sumsq(scale, sumsq, abs(row[j*lda]));
            // double for symmetric entries
            sumsq *= 2;
            // diagonal
            add_sumsq(scale, sumsq, abs(row[idx*lda]));
        }

        if (idx < blockDim.x) {
            row_scale[chunk] = 0;
            row_sumsq[chunk] = 1;
        }
        combine_sumsq(row_scale[chunk], row_sumsq[chunk], scale, sumsq);
        __syncthreads();
    }

    // Reduction to find sum-of-squares of tile.
    // todo: parallel reduction.
    if (threadIdx.x == 0) {
        real_t tile_scale = row_scale[0];
        real_t tile_sumsq = row_sumsq[0];
        for (int64_t chunk = 1; chunk < blockDim.x && chunk < n; ++chunk)
            combine_sumsq(tile_scale, tile_sumsq, row_scale[chunk], row_sumsq[chunk]);

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
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t>
void synorm(
    lapack::Norm norm, lapack::Uplo uplo,
    int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue)
{
    using real_t = blas::real_type<scalar_t>;
    int64_t nb = 512;

    // quick return
    if (batch_count == 0)
        return;

    //---------
    // max norm
    if (norm == lapack::Norm::Max) {
        if (n == 0) {
            blas::device_memset(values, 0, batch_count, queue);
        }
        else {
            assert(ldv == 1);
            synormMaxKernel<<<batch_count, nb, sizeof(real_t) * nb, queue.stream()>>>
                (uplo, n, Aarray, lda, values);
        }
    }
    //---------
    // one norm
    else if (norm == lapack::Norm::One || norm == lapack::Norm::Inf) {
        if (n == 0) {
            blas::device_memset(values, 0, batch_count * n, queue);
        }
        else {
            assert(ldv >= n);
            synormOneKernel<<<batch_count, nb, 0, queue.stream()>>>
                (uplo, n, Aarray, lda, values, ldv);
        }
    }
    //---------
    // Frobenius norm
    else if (norm == lapack::Norm::Fro) {
        if (n == 0) {
            blas::device_memset(values, 0, batch_count * 2, queue);
        }
        else {
            assert(ldv == 2);
            synormFroKernel<<<batch_count, nb, sizeof(real_t) * nb * 2, queue.stream()>>>
                (uplo, n, Aarray, lda, values);
        }
    }

    cudaError_t error = cudaGetLastError();
    slate_assert(error == cudaSuccess);
}

const int one_ib = 32;
const int one_ib1 = 33;

//------------------------------------------------------------------------------
/// Sum of absolute values of each row and each column of elements,
/// for each tile in tiles.
/// Each thread block deals with one tile.
/// Kernel assumes non-trivial tiles (m, n >= 1).
/// Launched by synormOffdiag().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
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
///     On exit,
///         tiles_sums[k*ldv + j]     = sum_{i} abs( A^(k)_(i, j) )
///     for column j of tile A^(k), and
///         tiles_sums[k*ldv + i + n] = sum_{j} abs( A^(k)_(i, j) )
///     for row i of tile A^(k).
///
/// @param[in] ldv
///     Leading dimension of tiles_sums (values) array.
///
template <typename scalar_t>
__global__ void synormOffdiagOneKernel(
    int64_t m, int64_t n,
    scalar_t const* const* tiles, int64_t lda,
    blas::real_type<scalar_t>* tiles_sums, int64_t ldv)
{
    // row_sums doesn't need to be shared, it could be in registers,
    // but we don't know how large it is beforehand -- each thread uses
    // ceil(m/ib) entries; in total it is ceil(m/ib)*ib entries.
    using real_t = blas::real_type<scalar_t>;
    scalar_t const* tile = tiles[blockIdx.x];
    extern __shared__ char dynamic_data[];
    real_t* shmem_tile = (real_t*)dynamic_data;
    real_t* row_sums = &shmem_tile[one_ib1*one_ib];
    const int idx = threadIdx.x;

    // Initialize row sums.
    for (int64_t ii = 0; ii < m; ii += one_ib) {
        row_sums[ii+idx] = 0;
    }

    for (int64_t jj = 0; jj < n; jj += one_ib) {
        real_t sum = 0.0;
        for (int64_t ii = 0; ii < m; ii += one_ib) {
            // Read 32 x 32 (ib x ib) sub-tile into shared memory.
            // This does coalesced reads of one column at a time in parallel.
            for (int64_t j = 0; j < one_ib; ++j)
                if (jj+j < n && ii+idx < m)
                    shmem_tile[j*one_ib1 + idx] = abs(tile[(jj+j)*lda + ii+idx]);
            __syncthreads();  // shmem_tile loaded

            // Each thread sums one column.
            for (int64_t i = 0; i < one_ib; ++i)
                if (ii+i < m)
                    sum += shmem_tile[idx*one_ib1 + i];

            // Each thread sums one row.
            for (int64_t j = 0; j < one_ib; ++j)
                if (jj+j < n)
                    row_sums[ii+idx] += shmem_tile[j*one_ib1 + idx];
            __syncthreads();  // done with shmem_tile
        }

        if (jj+idx < n)
            tiles_sums[blockIdx.x*ldv + jj+idx] = sum;
    }

    // Save row sums.
    for (int64_t ii = 0; ii < m; ii += one_ib) {
        if (ii+idx < m)
            tiles_sums[blockIdx.x*ldv + ii+idx + n] = row_sums[ii+idx];
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
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t>
void synormOffdiag(
    lapack::Norm norm,
    int64_t m, int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* values, int64_t ldv,
    int64_t batch_count,
    blas::Queue &queue)
{
    using real_t = blas::real_type<scalar_t>;

    // quick return
    if (batch_count == 0)
        return;

    //---------
    // one norm
    if (norm == lapack::Norm::One || norm == lapack::Norm::Inf) {
        assert(ldv >= n);
        size_t lwork = sizeof(real_t) * (one_ib*one_ib1 + roundup(m, int64_t(one_ib)));
        assert(lwork <= 48*1024); // max 48 KiB
        synormOffdiagOneKernel<<<batch_count, 32, lwork, queue.stream()>>>
            (m, n, Aarray, lda, values, ldv);
    }
    else {
        slate_not_implemented("Only Norm::One and Norm::Inf is supported.");
    }

    cudaError_t error = cudaGetLastError();
    slate_assert(error == cudaSuccess);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void synorm(
    lapack::Norm norm, lapack::Uplo uplo,
    int64_t n,
    float const* const* Aarray, int64_t lda,
    float* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue);

template
void synorm(
    lapack::Norm norm, lapack::Uplo uplo,
    int64_t n,
    double const* const* Aarray, int64_t lda,
    double* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue);

template
void synorm(
    lapack::Norm norm, lapack::Uplo uplo,
    int64_t n,
    cuFloatComplex const* const* Aarray, int64_t lda,
    float* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue);

template
void synorm(
    lapack::Norm norm, lapack::Uplo uplo,
    int64_t n,
    cuDoubleComplex const* const* Aarray, int64_t lda,
    double* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue);

//----------------------------------------
template
void synormOffdiag(
    lapack::Norm norm,
    int64_t m, int64_t n,
    float const* const* Aarray, int64_t lda,
    float* values, int64_t ldv,
    int64_t batch_count,
    blas::Queue &queue);

template
void synormOffdiag(
    lapack::Norm norm,
    int64_t m, int64_t n,
    double const* const* Aarray, int64_t lda,
    double* values, int64_t ldv,
    int64_t batch_count,
    blas::Queue &queue);

template
void synormOffdiag(
    lapack::Norm norm,
    int64_t m, int64_t n,
    cuFloatComplex const* const* Aarray, int64_t lda,
    float* values, int64_t ldv,
    int64_t batch_count,
    blas::Queue &queue);

template
void synormOffdiag(
    lapack::Norm norm,
    int64_t m, int64_t n,
    cuDoubleComplex const* const* Aarray, int64_t lda,
    double* values, int64_t ldv,
    int64_t batch_count,
    blas::Queue &queue);

} // namespace device
} // namespace slate
