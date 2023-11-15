// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "slate/Exception.hh"
#include "slate/internal/device.hh"

#include "device_util.dp.hpp"

#include <cstdio>
#include <complex>

namespace slate {
namespace device {

//------------------------------------------------------------------------------
/// Kernel implementing element-wise tile scale.
/// Each thread block deals with one tile.
/// Each thread deals with one row.
/// Launched by gescale().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///
/// @param[in] numer
///     Scale value numerator.
///
/// @param[in] denom
///     Scale value denominator.
///
/// @param[in,out] Aarray
///     Array of tiles of dimension gridDim.x,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile in Aarray. lda >= m.
///
template <typename scalar_t>
void tzscale_kernel(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    blas::real_type<scalar_t> numer, blas::real_type<scalar_t> denom,
    scalar_t** Aarray, int64_t lda, const sycl::nd_item<3> &item_ct1)
{
    scalar_t *tileA = Aarray[item_ct1.get_group(2)];
    blas::real_type<scalar_t> mul = numer / denom;

    // thread per row, if more rows than threads, loop by blockDim.x
    for (int64_t i = item_ct1.get_local_id(2); i < m;
         i += item_ct1.get_local_range(2)) {
        scalar_t* rowA = &tileA[ i ];

        if (uplo == lapack::Uplo::Lower) {
            for (int64_t j = 0; j <= i && j < n; ++j) { // lower
                rowA[j*lda] = rowA[j*lda] * mul;
                // rowA[j * lda] =
                //     dpct_operator_overloading::operator*(rowA[j * lda], mul);
            }
        }
        else {
            for (int64_t j = n-1; j >= i; --j) // upper
                rowA[j*lda] = rowA[j*lda] * mul;
                // rowA[j * lda] =
                //     dpct_operator_overloading::operator*(rowA[j * lda], mul);
        }
    }
}

//------------------------------------------------------------------------------
/// Batched routine for element-wise trapezoidal tile scale.
/// Sets upper or lower part of
/// \[
///     Aarray[k] *= (numer / denom).
/// \]
/// This does NOT currently take extra care to avoid over/underflow.
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
/// @param[in] numer
///     Scale value numerator.
///
/// @param[in] denom
///     Scale value denominator.
///
/// @param[in,out] Aarray
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
void tzscale(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    blas::real_type<scalar_t> numer, blas::real_type<scalar_t> denom,
    scalar_t** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue)
{
    // quick return
    if (batch_count == 0)
        return;

    /*
    DPCT1093:132: The "queue.device()" device may be not the one intended for
    use. Adjust the selected device if needed.
    */
    dpct::select_device(queue.device());

    // Max threads/block=1024 for current CUDA compute capability (<= 7.5)
    int64_t nthreads = std::min( int64_t( 1024 ), m );

    /*
    DPCT1049:22: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(&queue.stream()))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, batch_count) *
                                             sycl::range<3>(1, 1, nthreads),
                                         sycl::range<3>(1, 1, nthreads)),
                       [=](sycl::nd_item<3> item_ct1) {
                           tzscale_kernel(uplo, m, n, numer, denom, Aarray, lda,
                                          item_ct1);
                       });

    /*
    DPCT1010:133: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    dpct::err0 error = 0;
    slate_assert(error == 0);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void tzscale(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    float numer, float denom, float** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

template
void tzscale(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    double numer, double denom, double** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

//------------------------------------------------------------------------------
// Specializations to cast std::complex => cuComplex.
template <>
void tzscale(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    float numer, float denom,
    std::complex<float>** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue)
{
    tzscale(uplo, m, n, numer, denom, (sycl::float2 **)Aarray, lda, batch_count,
            queue);
}

template <>
void tzscale(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    double numer, double denom,
    std::complex<double>** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue)
{
    tzscale(uplo, m, n, numer, denom, (sycl::double2 **)Aarray, lda,
            batch_count, queue);
}

} // namespace device
} // namespace slate
