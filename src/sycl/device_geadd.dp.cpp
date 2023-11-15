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
/// Kernel implementing element-wise tile addition.
/// Each thread deals with one row.
/// Launched by geadd_kernel() and geadd_batch_kernel().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///
/// @param[in] Aarray
///     is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile in Aarray. lda >= m.
///
/// @param[in,out] B
///     is an m-by-n matrix stored in an ldb-by-n array.
///
/// @param[in] ldb
///     Leading dimension of each tile in Barray. ldb >= m.
///
/// @copydoc geadd
///
template <typename scalar_t>
void geadd_func(
    int64_t m, int64_t n,
    scalar_t alpha, scalar_t* A, int64_t lda,
    scalar_t beta,  scalar_t* B, int64_t ldb, const sycl::nd_item<3> &item_ct1)
{
    // thread per row, if more rows than threads, loop by blockDim.x
    for (int64_t i = item_ct1.get_local_id(2); i < m;
         i += item_ct1.get_local_range(2)) {
        scalar_t* rowA = &A[ i ];
        scalar_t* rowB = &B[ i ];

        for (int64_t j = 0; j < n; ++j)
            rowB[ j*ldb ] = (alpha * rowA[ j*lda ]) + (beta * rowB[ j*ldb ]);
            // rowB[j * ldb] = dpct_operator_overloading::operator+(
            //                     dpct_operator_overloading::operator*(alpha, rowA[j * lda]),
            //                     dpct_operator_overloading::operator*(beta, rowB[j * ldb]));
    }
}

//------------------------------------------------------------------------------
/// Kernel implementing element-wise tile.
/// @copydoc geadd
template <typename scalar_t>
void geadd_kernel(
    int64_t m, int64_t n,
    scalar_t alpha, scalar_t* A, int64_t lda,
    scalar_t beta,  scalar_t* B, int64_t ldb, const sycl::nd_item<3> &item_ct1)
{
    geadd_func(m, n, alpha, A, lda, beta, B, ldb, item_ct1);
}

//------------------------------------------------------------------------------
/// Kernel implementing element-wise tile set.
/// @copydoc geadd_batch
template <typename scalar_t>
void geadd_batch_kernel(
    int64_t m, int64_t n,
    scalar_t alpha, scalar_t** Aarray, int64_t lda,
    scalar_t beta,  scalar_t** Barray, int64_t ldb,
    const sycl::nd_item<3> &item_ct1)
{
    geadd_func(m, n, alpha, Aarray[item_ct1.get_group(2)], lda, beta,
               Barray[item_ct1.get_group(2)], ldb, item_ct1);
}

//------------------------------------------------------------------------------
/// Routine for element-wise tile addition.
/// Sets
/// \[
///     B = \alpha A + \beta B.
/// \]
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
/// @param[in] A
///     is an m-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] lda
///     Leading dimension of each tile in A. lda >= m.
///
/// @param[in] beta
///     The scalar beta.
///
/// @param[in,out] B
///     is an m-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] ldb
///     Leading dimension of each tile in B. ldb >= m.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t>
void geadd(
    int64_t m, int64_t n,
    scalar_t const& alpha, scalar_t* A, int64_t lda,
    scalar_t const& beta, scalar_t* B, int64_t ldb,
    blas::Queue &queue)
{
    // quick return
    if (m == 0 || n == 0)
        return;

    /*
    DPCT1093:146: The "queue.device()" device may be not the one intended for
    use. Adjust the selected device if needed.
    */
    dpct::select_device(queue.device());

    // Max threads/block=1024 for current CUDA compute capability (<= 7.5)
    int64_t nthreads = std::min( int64_t( 1024 ), m );

    /*
    DPCT1049:49: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(&queue.stream()))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nthreads),
                                         sycl::range<3>(1, 1, nthreads)),
                       [=](sycl::nd_item<3> item_ct1) {
                           geadd_kernel(m, n, alpha, A, lda, beta, B, ldb,
                                        item_ct1);
                       });

    /*
    DPCT1010:147: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    dpct::err0 error = 0;
    slate_assert(error == 0);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void geadd(
    int64_t m, int64_t n,
    float const& alpha, float* Aarray, int64_t lda,
    float const& beta,  float* Barray, int64_t ldb,
    blas::Queue &queue);

template
void geadd(
    int64_t m, int64_t n,
    double const& alpha, double* Aarray, int64_t lda,
    double const& beta,  double* Barray, int64_t ldb,
    blas::Queue &queue);

//------------------------------------------------------------------------------
// Specializations to cast std::complex => cuComplex.
template <>
void geadd(
    int64_t m, int64_t n,
    std::complex<float> const& alpha, std::complex<float>* Aarray, int64_t lda,
    std::complex<float> const& beta,  std::complex<float>* Barray, int64_t ldb,
    blas::Queue &queue)
{
    geadd(m, n, sycl::float2(real(alpha), imag(alpha)), (sycl::float2 *)Aarray,
          lda, sycl::float2(real(beta), imag(beta)), (sycl::float2 *)Barray,
          ldb, queue);
}

template <>
void geadd(
    int64_t m, int64_t n,
    std::complex<double> const& alpha, std::complex<double>* Aarray, int64_t lda,
    std::complex<double> const& beta,  std::complex<double>* Barray, int64_t ldb,
    blas::Queue &queue)
{
    geadd(m, n, sycl::double2(real(alpha), imag(alpha)),
          (sycl::double2 *)Aarray, lda, sycl::double2(real(beta), imag(beta)),
          (sycl::double2 *)Barray, ldb, queue);
}

//==============================================================================
namespace batch {

//------------------------------------------------------------------------------
/// Batched routine for element-wise tile addition.
/// Sets
/// \[
///     Barray[k] = \alpha Aarray[k] + \beta Barray[k].
/// \]
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
void geadd(
    int64_t m, int64_t n,
    scalar_t const& alpha, scalar_t** Aarray, int64_t lda,
    scalar_t const& beta, scalar_t** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
    // quick return
    if (m == 0 || n == 0)
        return;
    // quick return
    if (batch_count == 0)
        return;

    /*
    DPCT1093:148: The "queue.device()" device may be not the one intended for
    use. Adjust the selected device if needed.
    */
    dpct::select_device(queue.device());

    // Max threads/block=1024 for current CUDA compute capability (<= 7.5)
    int64_t nthreads = std::min( int64_t( 1024 ), m );

    /*
    DPCT1049:50: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(&queue.stream()))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, batch_count) *
                                             sycl::range<3>(1, 1, nthreads),
                                         sycl::range<3>(1, 1, nthreads)),
                       [=](sycl::nd_item<3> item_ct1) {
                           geadd_batch_kernel(m, n, alpha, Aarray, lda, beta,
                                              Barray, ldb, item_ct1);
                       });

    /*
    DPCT1010:149: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    dpct::err0 error = 0;
    slate_assert(error == 0);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void geadd(
    int64_t m, int64_t n,
    float const& alpha, float** Aarray, int64_t lda,
    float const& beta, float** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

template
void geadd(
    int64_t m, int64_t n,
    double const& alpha, double** Aarray, int64_t lda,
    double const& beta, double** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue);

//------------------------------------------------------------------------------
// Specializations to cast std::complex => cuComplex.
template <>
void geadd(
    int64_t m, int64_t n,
    std::complex<float> const& alpha, std::complex<float>** Aarray, int64_t lda,
    std::complex<float> const& beta, std::complex<float>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
    geadd(m, n, sycl::float2(real(alpha), imag(alpha)), (sycl::float2 **)Aarray,
          lda, sycl::float2(real(beta), imag(beta)), (sycl::float2 **)Barray,
          ldb, batch_count, queue);
}

template <>
void geadd(
    int64_t m, int64_t n,
    std::complex<double> const& alpha, std::complex<double>** Aarray, int64_t lda,
    std::complex<double> const& beta, std::complex<double>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
    geadd(m, n, sycl::double2(real(alpha), imag(alpha)),
          (sycl::double2 **)Aarray, lda, sycl::double2(real(beta), imag(beta)),
          (sycl::double2 **)Barray, ldb, batch_count, queue);
}

} // namespace batch
} // namespace device
} // namespace slate
