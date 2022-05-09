#include "hip/hip_runtime.h"
// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
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
/// Kernel implementing element-wise tile set.
/// Each thread block deals with one tile.
/// Each thread deals with one row.
/// Launched by geset().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///
/// @param[in] offdiag_value
///     The value to set outside of the diagonal.
///
/// @param[in] diag_value
///     The value to set on the diagonal.
///
/// @param[in] Atiles
///     Array of tiles of dimension gridDim.x,
///     where each Atiles[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile in Atiles. lda >= m.
///
template <typename scalar_t>
__global__ void gesetKernel(
    int64_t m, int64_t n,
    scalar_t offdiag_value, scalar_t diag_value, scalar_t** tilesA, int64_t lda)
{
    scalar_t* tileA = tilesA[blockIdx.x];

    // thread per row, if more rows than threads, loop by blockDim.x
    for (int64_t ridx = threadIdx.x; ridx < m; ridx += blockDim.x) {
        // todo: should the increment be ridx += 1024?
        scalar_t* rowA = &tileA[ridx];

        for (int64_t j = 0; j < n; ++j)
            rowA[j*lda] = (j != ridx) ? offdiag_value : diag_value;
    }
}

//------------------------------------------------------------------------------
/// Batched routine for element-wise tile set.
///
/// @param[in] m
///     Number of rows of each tile. m >= 0.
///
/// @param[in] n
///     Number of columns of each tile. n >= 0.
///
/// @param[in] diag_value
///     The value to set on the diagonal.
///
/// @param[in] offdiag_value
///     The value to set outside of the diagonal.
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
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t>
void geset(
    int64_t m, int64_t n,
    scalar_t diag_value, scalar_t offdiag_value, scalar_t** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue)
{
    // quick return
    if (batch_count == 0)
        return;

    // Max threads/block=1024 for current CUDA compute capability (<= 7.5)
    int64_t nthreads = std::min( int64_t( 1024 ), m );

    hipLaunchKernelGGL(gesetKernel, dim3(batch_count), dim3(nthreads), 0, queue.stream(), 
        m, n,
        diag_value, offdiag_value, Aarray, lda);

    hipError_t error = hipGetLastError();
    slate_assert(error == hipSuccess);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void geset(
    int64_t m, int64_t n,
    float diag_value, float offdiag_value, float** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue);

template
void geset(
    int64_t m, int64_t n,
    double diag_value, double offdiag_value, double** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue);

template
void geset(
    int64_t m, int64_t n,
    hipFloatComplex diag_value, hipFloatComplex offdiag_value,
    hipFloatComplex** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue);

template
void geset(
    int64_t m, int64_t n,
    hipDoubleComplex diag_value, hipDoubleComplex offdiag_value,
    hipDoubleComplex** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue);

} // namespace device
} // namespace slate
