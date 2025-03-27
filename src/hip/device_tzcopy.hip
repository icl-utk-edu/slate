#include "hip/hip_runtime.h"
// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Exception.hh"
#include "slate/internal/device.hh"

#include "device_util.hip.hh"

namespace slate {
namespace device {

//------------------------------------------------------------------------------
/// Kernel implementing copy and precision conversions, copying A to B.
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
/// @param[in] Aarray
///     Array of tiles of dimension gridDim.x,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile in Aarray. lda >= m.
///
/// @param[out] Barray
///     Array of tiles of dimension gridDim.x,
///     where each Barray[k] is an m-by-n matrix stored in an ldb-by-n array.
///
/// @param[in] ldb
///     Leading dimension of each tile in Barray. ldb >= m.
///
template <typename src_scalar_t, typename dst_scalar_t>
__global__ void tzcopy_kernel(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    src_scalar_t const* const* Aarray, int64_t lda,
    dst_scalar_t** Barray, int64_t ldb)
{
    src_scalar_t const* tileA = Aarray[ blockIdx.x ];
    dst_scalar_t*       tileB = Barray[ blockIdx.x ];

    // thread per row, if more rows than threads, loop by blockDim.x
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        src_scalar_t const* rowA = &tileA[ i ];
        dst_scalar_t*       rowB = &tileB[ i ];

        if (uplo == lapack::Uplo::Lower) {
            for (int64_t j = 0; j <= i && j < n; ++j) { // lower
                copy(rowA[j*lda], rowB[j*ldb]);
            }
        }
        else {
            for (int64_t j = n-1; j >= i; --j) { // upper
                copy(rowA[j*lda], rowB[j*ldb]);
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Batched routine for element-wise trapezoidal copy and precision conversion,
/// copying A to B. Sets upper or lower part of
/// \[
///     Barray[k] = Aarray[k].
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
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename src_scalar_t, typename dst_scalar_t>
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    src_scalar_t const* const* Aarray, int64_t lda,
    dst_scalar_t** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
    // quick return
    if (batch_count == 0)
        return;

    // Max threads/block=1024 for current CUDA compute capability (<= 7.5)
    int64_t nthreads = std::min( int64_t( 1024 ), m );

    hipSetDevice( queue.device() );

    tzcopy_kernel<<<batch_count, nthreads, 0, queue.stream()>>>(
          uplo,
          m, n,
          Aarray, lda,
          Barray, ldb);

    hipError_t error = hipGetLastError();
    slate_assert(error == hipSuccess);
}

//------------------------------------------------------------------------------
// Explicit instantiations.

// float => float
template
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    float const* const* Aarray, int64_t lda,
    float** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

// float => double
template
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    float const* const* Aarray, int64_t lda,
    double** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

// double => double
template
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    double const* const* Aarray, int64_t lda,
    double** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

// double => float
template
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    double const* const* Aarray, int64_t lda,
    float** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

//------------------------------------------------------------------------------
// Specializations to cast std::complex => hipComplex.

// complex-float => complex-float
template <>
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<float> const* const* Aarray, int64_t lda,
    std::complex<float>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
    tzcopy( uplo, m, n,
            (rocblas_float_complex**) Aarray, lda,
            (rocblas_float_complex**) Barray, ldb,
            batch_count, queue );
}

// complex-float => complex-double
template <>
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<float> const* const* Aarray, int64_t lda,
    std::complex<double>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
    tzcopy( uplo, m, n,
            (rocblas_float_complex**)  Aarray, lda,
            (rocblas_double_complex**) Barray, ldb,
            batch_count, queue );
}

// complex-double => complex-double
template <>
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<double> const* const* Aarray, int64_t lda,
    std::complex<double>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
    tzcopy( uplo, m, n,
            (rocblas_double_complex**) Aarray, lda,
            (rocblas_double_complex**) Barray, ldb,
            batch_count, queue );
}

// complex-double => complex-float
template <>
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<double> const* const* Aarray, int64_t lda,
    std::complex<float>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
    tzcopy( uplo, m, n,
            (rocblas_double_complex**) Aarray, lda,
            (rocblas_float_complex**)  Barray, ldb,
            batch_count, queue );
}

} // namespace device
} // namespace slate
