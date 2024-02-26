// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
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
/// Launched by tzadd().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///
/// @param[in] Aarray
///     Array of tiles of dimension gridDim.x,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile in Aarray. lda >= m.
///
/// @param[in,out] Barray
///     Array of tiles of dimension gridDim.x,
///     where each Barray[k] is an m-by-n matrix stored in an ldb-by-n array.
///
/// @param[in] ldb
///     Leading dimension of each tile in Barray. ldb >= m.
///
template <typename scalar_t>
__global__ void tzadd_kernel(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    scalar_t alpha, scalar_t** Aarray, int64_t lda,
    scalar_t beta,  scalar_t** Barray, int64_t ldb)
{
    scalar_t* tileA = Aarray[ blockIdx.x ];
    scalar_t* tileB = Barray[ blockIdx.x ];

    // thread per row, if more rows than threads, loop by blockDim.x
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        scalar_t* rowA = &tileA[ i ];
        scalar_t* rowB = &tileB[ i ];

        if (uplo == lapack::Uplo::Lower) {
            for (int64_t j = 0; j <= i && j < n; ++j) { // lower
                rowB[j*ldb] = alpha * rowA[j*lda] + beta * rowB[ j*ldb ];
            }
        }
        else {
            for (int64_t j = n-1; j >= i; --j) { // upper
                 rowB[j*ldb] = alpha * rowA[ j*lda ] + beta * rowB[ j*ldb ];
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Batched routine for element-wise trapezoidal tile addition.
/// Sets upper or lower part of
/// \[
///     Barray[k] = \alpha Aarray[k] + \beta Barray[k].
/// \]
///
/// @param[in] uplo
///     Whether each Aarray[k] is upper or lower trapezoidal.
///
/// @param[in] m
///     Number of rows of each tile. m >= 0.
///
/// @param[in] n
///     Number of columns of each tile. n >= 0.
///
/// @param[in] alpha
///     The scalar alpha.
///
/// @param[in] Aarray
///     Array in GPU memory of dimension batch_count, containing pointers to tiles,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] lda
///     Leading dimension of each tile in A. lda >= m.
///
/// @param[in] beta
///     The scalar beta.
///
/// @param[in,out] Barray
///     Array in GPU memory of dimension batch_count, containing pointers to tiles,
///     where each Barray[k] is an m-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] ldb
///     Leading dimension of each tile in B. ldb >= m.
///
/// @param[in] batch_count
///     Size of Aarray and Barray. batch_count >= 0.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t>
void tzadd(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    scalar_t const& alpha, scalar_t** Aarray, int64_t lda,
    scalar_t const& beta, scalar_t** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
    // quick return
    if (batch_count == 0)
        return;

    // Max threads/block=1024 for current CUDA compute capability (<= 7.5)
    int64_t nthreads = std::min( int64_t( 1024 ), m );

    cudaSetDevice( queue.device() );

    tzadd_kernel<<<batch_count, nthreads, 0, queue.stream()>>>(
        uplo,
        m, n,
        alpha, Aarray, lda,
        beta, Barray, ldb);

    cudaError_t error = cudaGetLastError();
    slate_assert(error == cudaSuccess);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void tzadd(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    float const& alpha, float** Aarray, int64_t lda,
    float const& beta, float** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

template
void tzadd(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    double const& alpha, double** Aarray, int64_t lda,
    double const& beta, double** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

//------------------------------------------------------------------------------
// Specializations to cast std::complex => cuComplex.
template <>
void tzadd(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<float> const& alpha, std::complex<float>** Aarray, int64_t lda,
    std::complex<float> const& beta,  std::complex<float>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
    tzadd( uplo, m, n,
           make_cuFloatComplex( real( alpha ), imag( alpha ) ),
           (cuFloatComplex**) Aarray, lda,
           make_cuFloatComplex( real( beta ), imag( beta ) ),
           (cuFloatComplex**) Barray, ldb,
           batch_count, queue );
}

template <>
void tzadd(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<double> const& alpha, std::complex<double>** Aarray, int64_t lda,
    std::complex<double> const& beta,  std::complex<double>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
    tzadd( uplo, m, n,
           make_cuDoubleComplex( real( alpha ), imag( alpha ) ),
           (cuDoubleComplex**) Aarray, lda,
           make_cuDoubleComplex( real( beta ), imag( beta ) ),
           (cuDoubleComplex**) Barray, ldb,
           batch_count, queue );
}

} // namespace device
} // namespace slate
