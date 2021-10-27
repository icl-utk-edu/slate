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
/// Kernel implementing copy and precision conversions.
/// Each thread block deals with one tile.
/// Each thread deals with one row.
/// Launched by tzcopy().
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
/// @param[in,out] Btiles
///     Array of tiles of dimension gridDim.x,
///     where each Btiles[k] is an m-by-n matrix stored in an ldb-by-n array.
///
/// @param[in] ldb
///     Leading dimension of each tile in Btiles. ldb >= m.
///
template <typename src_scalar_t, typename dst_scalar_t>
__global__ void tzcopyKernel(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    src_scalar_t** tilesA, int64_t lda,
    dst_scalar_t** tilesB, int64_t ldb)
{
    src_scalar_t* tileA = tilesA[blockIdx.x];
    dst_scalar_t* tileB = tilesB[blockIdx.x];

    // thread per row, if more rows than threads, loop by blockDim.x
    for (int ridx = threadIdx.x; ridx < m; ridx += blockDim.x) {
        src_scalar_t* rowA = &tileA[ridx];
        dst_scalar_t* rowB = &tileB[ridx];

        if (uplo == lapack::Uplo::Lower) {
            for (int64_t j = 0; j <= ridx && j < n; ++j) { // lower
                copy(rowA[j*lda], rowB[j*ldb]);
            }
        }
        else {
            for (int64_t j = n-1; j >= ridx; --j) { // upper
                copy(rowA[j*lda], rowB[j*ldb]);
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Batched routine for element-wise copy and precision conversion.
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
///     where each Barray[k] is an m-by-n matrix stored in an lda-by-n array in GPU memory.
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
template <typename src_scalar_t, typename dst_scalar_t>
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    src_scalar_t** Aarray, int64_t lda,
    dst_scalar_t** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
    // quick return
    if (batch_count == 0)
        return;

    // Max threads/block=1024 for current CUDA compute capability (<=7.5)
    int64_t nthreads = std::min((int64_t)1024 , m);

    cudaSetDevice( queue.device() );

    tzcopyKernel<<<batch_count, nthreads, 0, queue.stream()>>>(
          uplo,
          m, n,
          Aarray, lda,
          Barray, ldb);

    cudaError_t error = cudaGetLastError();

    assert(error == cudaSuccess);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    float** Aarray, int64_t lda,
    float** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

template
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    float** Aarray, int64_t lda,
    double** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

template
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    double** Aarray, int64_t lda,
    double** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

template
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    double** Aarray, int64_t lda,
    float** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

template
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    cuFloatComplex** Aarray, int64_t lda,
    cuFloatComplex** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

template
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    cuFloatComplex** Aarray, int64_t lda,
    cuDoubleComplex** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

template
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    cuDoubleComplex** Aarray, int64_t lda,
    cuDoubleComplex** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

template
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    cuDoubleComplex** Aarray, int64_t lda,
    cuFloatComplex** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

} // namespace device
} // namespace slate
