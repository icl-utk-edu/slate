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
/// Kernel implementing element-wise tile set.
/// Each thread block deals with one tile.
/// Each thread deals with one row.
/// Launched by geset_kernel() and geset_batch_kernel().
///
/// @copydoc geset
///
template <typename scalar_t>
__device__ void geset_func(
    int64_t m, int64_t n,
    scalar_t offdiag_value, scalar_t diag_value,
    scalar_t* A, int64_t lda)
{
    // thread per row, if more rows than threads, loop by blockDim.x
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        scalar_t* rowA = &A[ i ];

        for (int64_t j = 0; j < n; ++j)
            rowA[ j*lda ] = (j != i) ? offdiag_value : diag_value;
    }
}

//------------------------------------------------------------------------------
/// Kernel implementing element-wise tile.
/// @copydoc geset
template <typename scalar_t>
__global__ void geset_kernel(
    int64_t m, int64_t n,
    scalar_t offdiag_value, scalar_t diag_value,
    scalar_t* A, int64_t lda)
{
    geset_func( m, n, offdiag_value, diag_value, A, lda );
}

//------------------------------------------------------------------------------
/// Kernel implementing element-wise tile set.
/// @copydoc geset_batch
template <typename scalar_t>
__global__ void geset_batch_kernel(
    int64_t m, int64_t n,
    scalar_t offdiag_value, scalar_t diag_value,
    scalar_t** Aarray, int64_t lda)
{
    geset_func( m, n, offdiag_value, diag_value,
                Aarray[ blockIdx.x ], lda );
}

//------------------------------------------------------------------------------
/// Element-wise m-by-n matrix A
/// to diag_value on the diagonal and offdiag_value on the off-diagonals.
///
/// @param[in] m
///     Number of rows of A. m >= 0.
///
/// @param[in] n
///     Number of columns of A. n >= 0.
///
/// @param[in] offdiag_value
///     The value to set outside of the diagonal.
///
/// @param[in] diag_value
///     The value to set on the diagonal.
///
/// @param[out] A
///     An m-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] lda
///     Leading dimension of A. lda >= m.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t>
void geset(
    int64_t m, int64_t n,
    scalar_t const& offdiag_value, scalar_t const& diag_value,
    scalar_t* A, int64_t lda,
    blas::Queue &queue)
{
    // quick return
    if (m == 0 || n == 0)
        return;

    hipSetDevice( queue.device() );

    // Max threads/block=1024 for current CUDA compute capability (<= 7.5)
    int64_t nthreads = std::min( int64_t( 1024 ), m );

    geset_kernel<<<1, nthreads, 0, queue.stream()>>>(
        m, n,
        offdiag_value, diag_value, A, lda);

    hipError_t error = hipGetLastError();
    slate_assert(error == hipSuccess);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void geset(
    int64_t m, int64_t n,
    float const& offdiag_value, float const& diag_value,
    float* A, int64_t lda,
    blas::Queue &queue);

template
void geset(
    int64_t m, int64_t n,
    double const& offdiag_value, double const& diag_value,
    double* A, int64_t lda,
    blas::Queue &queue);

template
void geset(
    int64_t m, int64_t n,
    hipFloatComplex const& offdiag_value, hipFloatComplex const& diag_value,
    hipFloatComplex* A, int64_t lda,
    blas::Queue &queue);

template
void geset(
    int64_t m, int64_t n,
    hipDoubleComplex const& offdiag_value, hipDoubleComplex const& diag_value,
    hipDoubleComplex* A, int64_t lda,
    blas::Queue &queue);

//==============================================================================
namespace batch {

//------------------------------------------------------------------------------
/// Initializes a batch of m-by-n matrices Aarray[k]
/// to diag_value on the diagonal and offdiag_value on the off-diagonals.
///
/// @param[in] m
///     Number of rows of each tile. m >= 0.
///
/// @param[in] n
///     Number of columns of each tile. n >= 0.
///
/// @param[in] offdiag_value
///     The value to set outside of the diagonal.
///
/// @param[in] diag_value
///     The value to set on the diagonal.
///
/// @param[out] Aarray
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
void geset(
    int64_t m, int64_t n,
    scalar_t const& offdiag_value, scalar_t const& diag_value,
    scalar_t** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue)
{
    // quick return
    if (batch_count == 0)
        return;
    // quick return
    if (m == 0 || n == 0)
        return;

    hipSetDevice( queue.device() );

    // Max threads/block=1024 for current CUDA compute capability (<= 7.5)
    int64_t nthreads = std::min( int64_t( 1024 ), m );

    geset_batch_kernel<<<batch_count, nthreads, 0, queue.stream()>>>(
        m, n,
        offdiag_value, diag_value, Aarray, lda);

    hipError_t error = hipGetLastError();
    slate_assert(error == hipSuccess);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void geset(
    int64_t m, int64_t n,
    float const& offdiag_value, float const& diag_value,
    float** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue);

template
void geset(
    int64_t m, int64_t n,
    double const& offdiag_value, double const& diag_value,
    double** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue);

template
void geset(
    int64_t m, int64_t n,
    hipFloatComplex const& offdiag_value, hipFloatComplex const& diag_value,
    hipFloatComplex** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue);

template
void geset(
    int64_t m, int64_t n,
    hipDoubleComplex const& offdiag_value, hipDoubleComplex const& diag_value,
    hipDoubleComplex** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue);

} // namespace batch
} // namespace device
} // namespace slate
