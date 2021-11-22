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
/// Kernel implementing element-wise tile addition.
/// Each thread block deals with one tile.
/// Each thread deals with one row.
/// Launched by geadd().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///
/// @param[in] Atiles
///     Array of tiles of dimension gridDim.x,
///     where each Atiles[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile in Atiles. lda >= m.
///
/// @param[in] Btiles
///     Array of tiles of dimension gridDim.x,
///     where each Btiles[k] is an m-by-n matrix stored in an ldb-by-n array.
///
/// @param[in] ldb
///     Leading dimension of each tile in Btiles. ldb >= m.
///
template <typename scalar_t>
__global__ void geaddKernel(
    int64_t m, int64_t n,
    scalar_t alpha, scalar_t** tilesA, int64_t lda,
    scalar_t beta, scalar_t** tilesB, int64_t ldb)
{
    scalar_t* tileA = tilesA[blockIdx.x];
    scalar_t* tileB = tilesB[blockIdx.x];

    // thread per row, if more rows than threads, loop by blockDim.x
    for (int64_t ridx = threadIdx.x; ridx < m; ridx += blockDim.x) {
        // todo: should the increment be ridx += 1024?
        scalar_t* rowA = &tileA[ridx];
        scalar_t* rowB = &tileB[ridx];

        for (int64_t j = 0; j < n; ++j)
            rowB[j*ldb] = axpby(alpha, rowA[j*lda], beta, rowB[j*ldb]);
    }
}

//------------------------------------------------------------------------------
/// Batched routine for element-wise tile addition.
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
///     Leading dimension of each tile in A. lda >= m.
///
/// @param[out] Barray
///     Array in GPU memory of dimension batch_count, containing pointers to tiles,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] ldb
///     Leading dimension of each tile in B. ldb >= m.
///
/// @param[in] batch_count
///     Size of Aarray and Barray. batch_count >= 0.
///
/// @param[in] stream
///     CUDA stream to execute in.
///
template <typename scalar_t>
void geadd(
    int64_t m, int64_t n,
    scalar_t alpha, scalar_t** Aarray, int64_t lda,
    scalar_t beta, scalar_t** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
    // quick return
    if (batch_count == 0)
        return;

    // Max threads/block=1024 for current CUDA compute capability (<=7.5)
    int64_t nthreads = std::min((int64_t)1024, m);

    cudaSetDevice( queue.device() );

    geaddKernel<<<batch_count, nthreads, 0, queue.stream()>>>(
        m, n,
        alpha, Aarray, lda,
        beta, Barray, ldb);

    cudaError_t error = cudaGetLastError();
    slate_assert(error == cudaSuccess);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void geadd(
    int64_t m, int64_t n,
    float alpha, float** Aarray, int64_t lda,
    float beta, float** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

template
void geadd(
    int64_t m, int64_t n,
    double alpha, double** Aarray, int64_t lda,
    double beta, double** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

template
void geadd(
    int64_t m, int64_t n,
    cuFloatComplex alpha, cuFloatComplex** Aarray, int64_t lda,
    cuFloatComplex beta, cuFloatComplex** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

template
void geadd(
    int64_t m, int64_t n,
    cuDoubleComplex alpha, cuDoubleComplex** Aarray, int64_t lda,
    cuDoubleComplex beta, cuDoubleComplex** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

} // namespace device
} // namespace slate
