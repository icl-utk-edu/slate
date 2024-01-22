// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <sycl/sycl.hpp>
#include "slate/Exception.hh"
#include "slate/internal/device.hh"

#include "device_util.dp.hpp"

#include <cstdio>
#include <complex>

namespace slate {
namespace device {

//------------------------------------------------------------------------------
/// Finds the largest absolute value of elements, for each tile in Aarray.
/// Each thread block deals with one tile.
/// Each thread deals with one row, followed by a reduction.
/// Uses dynamic shared memory array of length sizeof(real_t) * m.
/// Kernel assumes non-trivial tiles (m, n >= 1).
/// Launched by trnorm().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///     Also the number of threads per block (blockDim.x), hence,
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///
/// @param[in] Aarray
///     Array of tiles of dimension gridDim.x,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= m.
///
/// @param[out] tiles_maxima
///     Array of dimension gridDim.x.
///     On exit, tiles_maxima[k] = max_{i, j} abs( A^(k)_(i, j) )
///     for tile A^(k).
///
template <typename scalar_t>
void trnorm_max_kernel(
    lapack::Uplo uplo, lapack::Diag diag,
    int64_t m, int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* tiles_maxima, const sycl::nd_item<3> &item_ct1,
    uint8_t *dpct_local)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const *tile = Aarray[item_ct1.get_group(2)];
    int chunk;

    // Save partial results in shared memory.
    auto dynamic_data = (char *)dpct_local;
    real_t* row_max = (real_t*) dynamic_data;

    if (item_ct1.get_local_id(2) < item_ct1.get_local_range(2)) {
        row_max[item_ct1.get_local_id(2)] = 0;
    }
    // Each thread finds max of one row.
    // This does coalesced reads of one column at a time in parallel.
    for (int i = item_ct1.get_local_id(2); i < m;
         i += item_ct1.get_local_range(2)) {
        chunk = i % item_ct1.get_local_range(2);

        scalar_t const* row = &tile[ i ];

        real_t max = 0;
        if (uplo == lapack::Uplo::Lower) {
            if (diag == lapack::Diag::Unit) {
                if (i < n) // diag
                    max = 1;
                for (int64_t j = 0; j < i && j < n; ++j) // strictly lower
                    max = max_nan(max, abs(row[j*lda]));
            }
            else {
                for (int64_t j = 0; j <= i && j < n; ++j) // lower
                    max = max_nan(max, abs(row[j*lda]));
            }
        }
        else {
            // Loop backwards (n-1 down to i) to maintain coalesced reads.
            if (diag == lapack::Diag::Unit) {
                if (i < n) // diag
                    max = 1;
                for (int64_t j = n-1; j > i; --j) // strictly upper
                    max = max_nan(max, abs(row[j*lda]));
            }
            else {
                for (int64_t j = n-1; j >= i; --j) // upper
                    max = max_nan(max, abs(row[j*lda]));
            }
        }

        row_max[chunk] = max_nan(max, row_max[chunk]);
    }

    // Reduction to find max of tile.
    /*
    DPCT1065:51: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    max_nan_reduce(item_ct1.get_local_range(2), item_ct1.get_local_id(2),
                   row_max, item_ct1);
    if (item_ct1.get_local_id(2) == 0) {
        tiles_maxima[item_ct1.get_group(2)] = row_max[0];
    }
}

//------------------------------------------------------------------------------
/// Sum of absolute values of each column of elements, for each tile in Aarray.
/// Each thread block deals with one tile.
/// Each thread deals with one column.
/// Kernel assumes non-trivial tiles (m, n >= 1).
/// Launched by trnorm().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///     Also the number of threads per block (blockDim.x), hence,
///
/// @param[in] Aarray
///     Array of tiles of dimension gridDim.x,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= m.
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
void trnorm_one_kernel(
    lapack::Uplo uplo, lapack::Diag diag,
    int64_t m, int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* tiles_sums, int64_t ldv,
    const sycl::nd_item<3> &item_ct1)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const *tile = Aarray[item_ct1.get_group(2)];

    // Each thread sums one column.
    // todo: this doesn't do coalesced reads
    for (int j = item_ct1.get_local_id(2); j < n;
         j += item_ct1.get_local_range(2)) {

        scalar_t const* column = &tile[ lda*j ];
        real_t sum = 0;

        if (uplo == lapack::Uplo::Lower) {
            if (diag == lapack::Diag::Unit) {
                if (j < m) // diag
                    sum += 1;
                for (int64_t i = j+1; i < m; ++i) // strictly lower
                    sum += abs(column[i]);
            }
            else {
                for (int64_t i = j; i < m; ++i) // lower
                    sum += abs(column[i]);
            }
        }
        else {
            if (diag == lapack::Diag::Unit) {
                if (j < m) // diag
                    sum += 1;
                for (int64_t i = 0; i < j && i < m; ++i) // strictly upper
                    sum += abs(column[i]);
            }
            else {
                for (int64_t i = 0; i <= j && i < m; ++i) // upper
                    sum += abs(column[i]);
            }
        }
        tiles_sums[item_ct1.get_group(2) * ldv + j] = sum;
    }
}

//------------------------------------------------------------------------------
/// Sum of absolute values of each row of elements, for each tile in Aarray.
/// Each thread block deals with one tile.
/// Each thread deals with one row.
/// Kernel assumes non-trivial tiles (m, n >= 1).
/// Launched by trnorm().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///     Also the number of threads per block, hence,
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///
/// @param[in] Aarray
///     Array of tiles of dimension gridDim.x,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= m.
///
/// @param[out] tiles_sums
///     Array of dimension gridDim.x * ldv.
///     On exit, tiles_sums[k*ldv + i] = sum_{j} abs( A^(k)_(i, j) )
///     for row i of tile A^(k).
///
/// @param[in] ldv
///     Leading dimension of tiles_sums (values) array.
///
template <typename scalar_t>
void trnorm_inf_kernel(
    lapack::Uplo uplo, lapack::Diag diag,
    int64_t m, int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* tiles_sums, int64_t ldv,
    const sycl::nd_item<3> &item_ct1)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const *tile = Aarray[item_ct1.get_group(2)];

    // Each thread sums one row.
    // This does coalesced reads of one column at a time in parallel.
    for (int i = item_ct1.get_local_id(2); i < m;
         i += item_ct1.get_local_range(2)) {
        scalar_t const* row = &tile[ i ];
        real_t sum = 0;
        if (uplo == lapack::Uplo::Lower) {
            if (diag == lapack::Diag::Unit) {
                if (i < n) // diag
                    sum += 1;
                for (int64_t j = 0; j < i && j < n; ++j) // strictly lower
                    sum += abs(row[j*lda]);
            }
            else {
                for (int64_t j = 0; j <= i && j < n; ++j) // lower
                    sum += abs(row[j*lda]);
            }
        }
        else {
            // Loop backwards (n-1 down to i) to maintain coalesced reads.
            if (diag == lapack::Diag::Unit) {
                if (i < n) // diag
                    sum += 1;
                for (int64_t j = n-1; j > i; --j) // strictly upper
                    sum += abs(row[j*lda]);
            }
            else {
                for (int64_t j = n-1; j >= i; --j) // upper
                    sum += abs(row[j*lda]);
            }
        }
        tiles_sums[item_ct1.get_group(2) * ldv + i] = sum;
    }
}

//------------------------------------------------------------------------------
/// Sum of squares, in scaled representation, for each tile in Aarray.
/// Each thread block deals with one tile.
/// Each thread deals with one row, followed by a reduction.
/// Kernel assumes non-trivial tiles (m, n >= 1).
/// Launched by trnorm().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///     Also the number of threads per block, hence,
///
/// @param[in] Aarray
///     Array of tiles of dimension blockDim.x,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= m.
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
void trnorm_fro_kernel(
    lapack::Uplo uplo, lapack::Diag diag,
    int64_t m, int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* tiles_values, const sycl::nd_item<3> &item_ct1,
    uint8_t *dpct_local)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const *tile = Aarray[item_ct1.get_group(2)];
    int chunk;

    // Save partial results in shared memory.
    auto dynamic_data = (char *)dpct_local;
    real_t* row_scale = (real_t*) &dynamic_data[0];
    real_t *row_sumsq = &row_scale[item_ct1.get_local_range(2)];

    // Each thread finds sum-of-squares of one row.
    // This does coalesced reads of one column at a time in parallel.
    for (int i = item_ct1.get_local_id(2); i < m;
         i += item_ct1.get_local_range(2)) {
        real_t scale = 0;
        real_t sumsq = 1;
        chunk = i % item_ct1.get_local_range(2);
        scalar_t const* row = &tile[ i ];

        if (uplo == lapack::Uplo::Lower) {
            if (diag == lapack::Diag::Unit) {
                if (i < n) // diag
                    add_sumsq(scale, sumsq, real_t(1));
                for (int64_t j = 0; j < i && j < n; ++j) // strictly lower
                    add_sumsq(scale, sumsq, abs(row[j*lda]));
            }
            else {
                for (int64_t j = 0; j <= i && j < n; ++j) // lower
                    add_sumsq(scale, sumsq, abs(row[j*lda]));
            }
        }
        else {
            // Loop backwards (n-1 down to i) to maintain coalesced reads.
            if (diag == lapack::Diag::Unit) {
                if (i < n) // diag
                    add_sumsq(scale, sumsq, real_t(1));
                for (int64_t j = n-1; j > i; --j) // strictly upper
                    add_sumsq(scale, sumsq, abs(row[j*lda]));
            }
            else {
                for (int64_t j = n-1; j >= i; --j) // upper
                    add_sumsq(scale, sumsq, abs(row[j*lda]));
            }
        }

        if (i < item_ct1.get_local_range(2)) {
            row_scale[chunk] = 0;
            row_sumsq[chunk] = 1;
        }

        combine_sumsq(row_scale[chunk], row_sumsq[chunk], scale, sumsq);
    }

    // Reduction to find sum-of-squares of tile.
    // todo: parallel reduction.
    /*
    DPCT1065:52: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) == 0) {
        real_t tile_scale = row_scale[0];
        real_t tile_sumsq = row_sumsq[0];
        for (int64_t chunk1 = 1;
             chunk1 < item_ct1.get_local_range(2) && chunk1 < m; ++chunk1) {
            combine_sumsq(tile_scale, tile_sumsq, row_scale[chunk1], row_sumsq[chunk1]);
        }

        tiles_values[item_ct1.get_group(2) * 2 + 0] = tile_scale;
        tiles_values[item_ct1.get_group(2) * 2 + 1] = tile_sumsq;
    }
}

//------------------------------------------------------------------------------
/// Batched routine that computes a partial norm for each trapezoidal tile.
///
/// todo: rename to tznorm for consistency with other tz routines.
///
/// @param[in] norm
///     Norm to compute. See values for description.
///
/// @param[in] uplo
///     Whether each Aarray[k] is upper or lower trapezoidal.
///
/// @param[in] diag
///     Whether or not each Aarray[k] has unit diagonal.
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
///     Leading dimension of each tile. lda >= m.
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
///     - Norm::Inf: ldv >= m.
///         On exit, values[k*ldv + i] = sum_{j} abs( A^(k)_(i, j) )
///         for 0 <= k < batch_count, 0 <= i < m.
///
///     - Norm::Max: ldv = 2.
///         On exit,
///             values[k*2 + 0] = scale_k
///             values[k*2 + 1] = sumsq_k
///         where scale_k^2 sumsq_k = sum_{i,j} abs( A^(k)_(i, j) )^2
///         for 0 <= k < batch_count.
///
/// @param[in] ldv
///     Leading dimension of values array.
///
/// @param[in] batch_count
///     Size of Aarray. batch_count >= 0.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t>
void trnorm(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag,
    int64_t m, int64_t n,
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
        if (m == 0 || n == 0) {
            blas::device_memset(values, 0, batch_count, queue);
        }
        else {
            assert(ldv == 1);
            /*
            DPCT1083:54: The size of local memory in the migrated code may be
            different from the original code. Check that the allocated memory
            size in the migrated code is correct.
            */
            size_t shared_mem = sizeof(real_t) * nb;
            /*
            DPCT1049:53: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(&queue.stream()))->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(shared_mem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, batch_count) *
                                          sycl::range<3>(1, 1, nb),
                                      sycl::range<3>(1, 1, nb)),
                    [=](sycl::nd_item<3> item_ct1) {
                        trnorm_max_kernel(uplo, diag, m, n, Aarray, lda, values,
                                          item_ct1,
                                          dpct_local_acc_ct1.get_pointer());
                    });
            });
        }
    }
    //---------
    // one norm
    else if (norm == lapack::Norm::One) {
        if (m == 0 || n == 0) {
            blas::device_memset(values, 0, batch_count * n, queue);
        }
        else {
            assert(ldv >= n);
            /*
            DPCT1049:55: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(&queue.stream()))
                ->parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, batch_count) *
                                          sycl::range<3>(1, 1, nb),
                                      sycl::range<3>(1, 1, nb)),
                    [=](sycl::nd_item<3> item_ct1) {
                        trnorm_one_kernel(uplo, diag, m, n, Aarray, lda, values,
                                          ldv, item_ct1);
                    });
        }
    }
    //---------
    // inf norm
    else if (norm == lapack::Norm::Inf) {
        if (m == 0 || n == 0) {
            blas::device_memset(values, 0, batch_count * m, queue);
        }
        else {
            assert(ldv >= m);
            /*
            DPCT1049:56: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(&queue.stream()))
                ->parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, batch_count) *
                                          sycl::range<3>(1, 1, nb),
                                      sycl::range<3>(1, 1, nb)),
                    [=](sycl::nd_item<3> item_ct1) {
                        trnorm_inf_kernel(uplo, diag, m, n, Aarray, lda, values,
                                          ldv, item_ct1);
                    });
        }
    }
    //---------
    // Frobenius norm
    else if (norm == lapack::Norm::Fro) {
        if (m == 0 || n == 0) {
            blas::device_memset(values, 0, batch_count * 2, queue);
        }
        else {
            assert(ldv == 2);
            /*
            DPCT1083:58: The size of local memory in the migrated code may be
            different from the original code. Check that the allocated memory
            size in the migrated code is correct.
            */
            size_t shared_mem = sizeof(real_t) * nb * 2;
            /*
            DPCT1049:57: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(&queue.stream()))->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(shared_mem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, batch_count) *
                                          sycl::range<3>(1, 1, nb),
                                      sycl::range<3>(1, 1, nb)),
                    [=](sycl::nd_item<3> item_ct1) {
                        trnorm_fro_kernel(uplo, diag, m, n, Aarray, lda, values,
                                          item_ct1,
                                          dpct_local_acc_ct1.get_pointer());
                    });
            });
        }
    }

}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void trnorm(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag,
    int64_t m, int64_t n,
    float const* const* Aarray, int64_t lda,
    float* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue);

template
void trnorm(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag,
    int64_t m, int64_t n,
    double const* const* Aarray, int64_t lda,
    double* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue);

//------------------------------------------------------------------------------
// Specializations to cast std::complex => cuComplex.
template <>
void trnorm(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag,
    int64_t m, int64_t n,
    std::complex<float> const* const* Aarray, int64_t lda,
    float* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue)
{
    trnorm(norm, uplo, diag, m, n, (sycl::float2 **)Aarray, lda, values, ldv,
           batch_count, queue);
}

template <>
void trnorm(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag,
    int64_t m, int64_t n,
    std::complex<double> const* const* Aarray, int64_t lda,
    double* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue)
{
    trnorm(norm, uplo, diag, m, n, (sycl::double2 **)Aarray, lda, values, ldv,
           batch_count, queue);
}

} // namespace device
} // namespace slate
