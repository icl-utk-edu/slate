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
/// Kernel implementing element-wise tile set.
/// Each thread block deals with one tile.
/// Each thread deals with one row.
/// Launched by geset_kernel() and geset_batch_kernel().
///
/// @copydoc geset
///
template <typename scalar_t>
void geset_func(
    int64_t m, int64_t n,
    scalar_t offdiag_value,
    scalar_t diag_value,
    scalar_t* A, int64_t lda, const sycl::nd_item<3> &item_ct1)
{
    // thread per row, if more rows than threads, loop by blockDim.x
    for (int64_t i = item_ct1.get_local_id(2); i < m;
         i += item_ct1.get_local_range(2)) {
        scalar_t* rowA = &A[ i ];

        for (int64_t j = 0; j < n; ++j)
            rowA[ j*lda ] = (j != i) ? offdiag_value : diag_value;
    }
}

//------------------------------------------------------------------------------
/// Kernel implementing element-wise tile.
/// @copydoc geset
template <typename scalar_t>
void geset_kernel(
    int64_t m, int64_t n,
    scalar_t offdiag_value,
    scalar_t diag_value,
    scalar_t* A, int64_t lda, const sycl::nd_item<3> &item_ct1)
{
    geset_func(m, n, offdiag_value, diag_value, A, lda, item_ct1);
}

//------------------------------------------------------------------------------
/// Kernel implementing element-wise tile set.
/// @copydoc geset_batch
template <typename scalar_t>
void geset_batch_kernel(
    int64_t m, int64_t n,
    scalar_t offdiag_value,
    scalar_t diag_value,
    scalar_t** Aarray, int64_t lda, const sycl::nd_item<3> &item_ct1)
{
    geset_func(m, n, offdiag_value, diag_value, Aarray[item_ct1.get_group(2)],
               lda, item_ct1);
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
    scalar_t const& offdiag_value,
    scalar_t const& diag_value,
    scalar_t* A, int64_t lda,
    blas::Queue &queue)
{
    // quick return
    if (m == 0 || n == 0)
        return;

    /*
    DPCT1093:134: The "queue.device()" device may be not the one intended for
    use. Adjust the selected device if needed.
    */
    dpct::select_device(queue.device());

    // Max threads/block=1024 for current CUDA compute capability (<= 7.5)
    int64_t nthreads = std::min( int64_t( 1024 ), m );

    /*
    DPCT1049:23: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(&queue.stream()))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nthreads),
                                         sycl::range<3>(1, 1, nthreads)),
                       [=](sycl::nd_item<3> item_ct1) {
                           geset_kernel(m, n, offdiag_value, diag_value, A, lda,
                                        item_ct1);
                       });

    /*
    DPCT1010:135: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    dpct::err0 error = 0;
    slate_assert(error == 0);
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

//------------------------------------------------------------------------------
// Specializations to cast std::complex => cuComplex.
template <>
void geset(
    int64_t m, int64_t n,
    std::complex<float> const& offdiag_value,
    std::complex<float> const& diag_value,
    std::complex<float>* A, int64_t lda,
    blas::Queue &queue)
{
    geset(m, n, sycl::float2(real(offdiag_value), imag(offdiag_value)),
          sycl::float2(real(diag_value), imag(diag_value)), (sycl::float2 *)A,
          lda, queue);
}

template <>
void geset(
    int64_t m, int64_t n,
    std::complex<double> const& offdiag_value,
    std::complex<double> const& diag_value,
    std::complex<double>* A, int64_t lda,
    blas::Queue &queue)
{
    geset(m, n, sycl::double2(real(offdiag_value), imag(offdiag_value)),
          sycl::double2(real(diag_value), imag(diag_value)), (sycl::double2 *)A,
          lda, queue);
}

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
    scalar_t const& offdiag_value,
    scalar_t const& diag_value,
    scalar_t** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue)
{
    // quick return
    if (batch_count == 0)
        return;
    // quick return
    if (m == 0 || n == 0)
        return;

    /*
    DPCT1093:136: The "queue.device()" device may be not the one intended for
    use. Adjust the selected device if needed.
    */
    dpct::select_device(queue.device());

    // Max threads/block=1024 for current CUDA compute capability (<= 7.5)
    int64_t nthreads = std::min( int64_t( 1024 ), m );

    /*
    DPCT1049:24: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(&queue.stream()))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, batch_count) *
                                             sycl::range<3>(1, 1, nthreads),
                                         sycl::range<3>(1, 1, nthreads)),
                       [=](sycl::nd_item<3> item_ct1) {
                           geset_batch_kernel(m, n, offdiag_value, diag_value,
                                              Aarray, lda, item_ct1);
                       });

    /*
    DPCT1010:137: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    dpct::err0 error = 0;
    slate_assert(error == 0);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void geset(
    int64_t m, int64_t n,
    float const& offdiag_value,
    float const& diag_value,
    float** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue);

template
void geset(
    int64_t m, int64_t n,
    double const& offdiag_value,
    double const& diag_value,
    double** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue);

//------------------------------------------------------------------------------
// Specializations to cast std::complex => cuComplex.
template <>
void geset(
    int64_t m, int64_t n,
    std::complex<float> const& offdiag_value,
    std::complex<float> const& diag_value,
    std::complex<float>** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue)
{
    geset(m, n, sycl::float2(real(offdiag_value), imag(offdiag_value)),
          sycl::float2(real(diag_value), imag(diag_value)),
          (sycl::float2 **)Aarray, lda, batch_count, queue);
}

template <>
void geset(
    int64_t m, int64_t n,
    std::complex<double> const& offdiag_value,
    std::complex<double> const& diag_value,
    std::complex<double>** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue)
{
    geset(m, n, sycl::double2(real(offdiag_value), imag(offdiag_value)),
          sycl::double2(real(diag_value), imag(diag_value)),
          (sycl::double2 **)Aarray, lda, batch_count, queue);
}

} // namespace batch
} // namespace device
} // namespace slate
