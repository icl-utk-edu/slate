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
/// Uses dynamic shared memory array of length sizeof(real_t) * n.
/// Kernel assumes non-trivial tiles (n >= 1).
/// Launched by synorm().
///
/// @param[in] n
///     Number of rows and columns of each tile. n >= 1.
///     Also the number of threads per block (blockDim.x), hence,
///
/// @param[in] Aarray
///     Array of tiles of dimension gridDim.x,
///     where each Aarray[k] is an n-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= n.
///
/// @param[out] tiles_maxima
///     Array of dimension gridDim.x.
///     On exit, tiles_maxima[k] = max_{i, j} abs( A^(k)_(i, j) )
///     for tile A^(k).
///
template <typename scalar_t>
void synorm_max_kernel(
    lapack::Uplo uplo,
    int64_t n,
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
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) {
        chunk = i % item_ct1.get_local_range(2);

        scalar_t const* row = &tile[ i ];
        if (i < item_ct1.get_local_range(2)) {
            row_max[chunk] = 0;
        }

        real_t max = 0;
        if (uplo == lapack::Uplo::Lower) {
            for (int64_t j = 0; j <= i && j < n; ++j) // lower
                max = max_nan(max, abs(row[j*lda]));
        }
        else {
            // Loop backwards (n-1 down to i) to maintain coalesced reads.
            for (int64_t j = n-1; j >= i; --j) // upper
                max = max_nan(max, abs(row[j*lda]));
        }
        row_max[chunk] = max_nan(max, row_max[chunk]);
    }

    // Reduction to find max of tile.
    /*
    DPCT1065:73: Consider replacing sycl::nd_item::barrier() with
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
/// Kernel assumes non-trivial tiles (n >= 1).
/// Launched by synorm().
///
/// @param[in] n
///     Number of rows and columns of each tile. n >= 1.
///     Also the number of threads per block (blockDim.x), hence,
///
/// @param[in] Aarray
///     Array of tiles of dimension gridDim.x,
///     where each Aarray[k] is an n-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= n.
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
void synorm_one_kernel(
    lapack::Uplo uplo,
    int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* tiles_sums, int64_t ldv,
    const sycl::nd_item<3> &item_ct1)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const *tile = Aarray[item_ct1.get_group(2)];

    // Each thread sums one row/column.
    // todo: the row reads are coalesced, but the col reads are not coalesced
    for (int k = item_ct1.get_local_id(2); k < n;
         k += item_ct1.get_local_range(2)) {
        scalar_t const* row    = &tile[ k ];
        scalar_t const* column = &tile[ lda*k ];
        real_t sum = 0;

        if (uplo == lapack::Uplo::Lower) {
            for (int64_t j = 0; j <= k; ++j) // lower
                sum += abs(row[j*lda]);
            for (int64_t i = k + 1; i < n; ++i) // strictly lower
                sum += abs(column[i]);
        }
        else {
            // Loop backwards (n-1 down to i) to maintain coalesced reads.
            for (int64_t j = n-1; j >= k; --j) // upper
                sum += abs(row[j*lda]);
            for (int64_t i = 0; i < k && i < n; ++i) // strictly upper
                sum += abs(column[i]);
        }
        tiles_sums[item_ct1.get_group(2) * ldv + k] = sum;
    }
}

//------------------------------------------------------------------------------
/// Sum of squares, in scaled representation, for each tile in Aarray.
/// Each thread block deals with one tile.
/// Each thread deals with one row, followed by a reduction.
/// Kernel assumes non-trivial tiles (n >= 1).
/// Launched by synorm().
///
/// @param[in] n
///     Number of rows and columns of each tile. n >= 1.
///     Also the number of threads per block, hence,
///
/// @param[in] Aarray
///     Array of tiles of dimension blockDim.x,
///     where each Aarray[k] is an n-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= n.
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
void synorm_fro_kernel(
    lapack::Uplo uplo,
    int64_t n,
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
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) {
        real_t scale = 0;
        real_t sumsq = 1;
        chunk = i % item_ct1.get_local_range(2);
        scalar_t const* row = &tile[ i ];

        if (uplo == lapack::Uplo::Lower) {
            for (int64_t j = 0; j < i && j < n; ++j) // strictly lower
                add_sumsq(scale, sumsq, abs(row[j*lda]));
            // double for symmetric entries
            sumsq *= 2;
            // diagonal
            add_sumsq( scale, sumsq, abs( row[ i*lda ] ) );
        }
        else {
            // Loop backwards (n-1 down to i) to maintain coalesced reads.
            for (int64_t j = n-1; j > i; --j) // strictly upper
                add_sumsq(scale, sumsq, abs(row[j*lda]));
            // double for symmetric entries
            sumsq *= 2;
            // diagonal
            add_sumsq( scale, sumsq, abs( row[ i*lda ] ) );
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
    DPCT1065:74: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) == 0) {
        real_t tile_scale = row_scale[0];
        real_t tile_sumsq = row_sumsq[0];
        for (int64_t chunk1 = 1;
             chunk1 < item_ct1.get_local_range(2) && chunk1 < n; ++chunk1) {
            combine_sumsq(tile_scale, tile_sumsq, row_scale[chunk1], row_sumsq[chunk1]);
        }

        tiles_values[item_ct1.get_group(2) * 2 + 0] = tile_scale;
        tiles_values[item_ct1.get_group(2) * 2 + 1] = tile_sumsq;
    }
}

//------------------------------------------------------------------------------
/// Batched routine that computes a partial norm for each tile.
///
/// @param[in] norm
///     Norm to compute. See values for description.
///
/// @param[in] uplo
///     Whether each Aarray[k] is stored in the upper or lower triangle.
///
/// @param[in] n
///     Number of rows and columns of each tile. n >= 0.
///
/// @param[in] Aarray
///     Array in GPU memory of dimension batch_count, containing pointers to tiles,
///     where each Aarray[k] is an n-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= n.
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
///     - Norm::Inf: for symmetric, same as Norm::One
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
void synorm(
    lapack::Norm norm, lapack::Uplo uplo,
    int64_t n,
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
        if (n == 0) {
            blas::device_memset(values, 0, batch_count, queue);
        }
        else {
            assert(ldv == 1);
            /*
            DPCT1083:76: The size of local memory in the migrated code may be
            different from the original code. Check that the allocated memory
            size in the migrated code is correct.
            */
            size_t shared_mem = sizeof(real_t) * nb;
            /*
            DPCT1049:75: The work-group size passed to the SYCL kernel may
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
                        synorm_max_kernel(uplo, n, Aarray, lda, values,
                                          item_ct1,
                                          dpct_local_acc_ct1.get_pointer());
                    });
            });
        }
    }
    //---------
    // one norm
    else if (norm == lapack::Norm::One || norm == lapack::Norm::Inf) {
        if (n == 0) {
            blas::device_memset(values, 0, batch_count * n, queue);
        }
        else {
            assert(ldv >= n);
            /*
            DPCT1049:77: The work-group size passed to the SYCL kernel may
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
                        synorm_one_kernel(uplo, n, Aarray, lda, values, ldv,
                                          item_ct1);
                    });
        }
    }
    //---------
    // Frobenius norm
    else if (norm == lapack::Norm::Fro) {
        if (n == 0) {
            blas::device_memset(values, 0, batch_count * 2, queue);
        }
        else {
            assert(ldv == 2);
            /*
            DPCT1083:79: The size of local memory in the migrated code may be
            different from the original code. Check that the allocated memory
            size in the migrated code is correct.
            */
            size_t shared_mem = sizeof(real_t) * nb * 2;
            /*
            DPCT1049:78: The work-group size passed to the SYCL kernel may
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
                        synorm_fro_kernel(uplo, n, Aarray, lda, values,
                                          item_ct1,
                                          dpct_local_acc_ct1.get_pointer());
                    });
            });
        }
    }

}

const int ib  = 32;
const int ib1 = 33;

//------------------------------------------------------------------------------
/// Sum of absolute values of each row and each column of elements,
/// for each tile in tiles.
/// Each thread block deals with one tile.
/// Kernel assumes non-trivial tiles (m, n >= 1).
/// Launched by synormOffdiag().
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
///     Leading dimension of each tile. lda >= m.
///
/// @param[out] tiles_sums
///     Array of dimension gridDim.x * ldv.
///     On exit,
///         tiles_sums[k*ldv + j]     = sum_{i} abs( A^(k)_(i, j) )
///     for column j of tile A^(k), and
///         tiles_sums[k*ldv + i + n] = sum_{j} abs( A^(k)_(i, j) )
///     for row i of tile A^(k).
///
/// @param[in] ldv
///     Leading dimension of tiles_sums (values) array.
///
template <typename scalar_t>
void synorm_offdiag_one_kernel(
    int64_t m, int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* tiles_sums, int64_t ldv,
    const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local)
{
    // row_sums doesn't need to be shared, it could be in registers,
    // but we don't know how large it is beforehand -- each thread uses
    // ceil(m/ib) entries; in total it is ceil(m/ib)*ib entries.
    using real_t = blas::real_type<scalar_t>;
    scalar_t const *tile = Aarray[item_ct1.get_group(2)];
    auto dynamic_data = (char *)dpct_local;
    real_t* shmem_tile = (real_t*)dynamic_data;
    real_t* row_sums = &shmem_tile[ ib1*ib ];
    const int k = item_ct1.get_local_id(2);

    // Initialize row sums.
    for (int64_t ii = 0; ii < m; ii += ib) {
        row_sums[ ii+k ] = 0;
    }

    for (int64_t jj = 0; jj < n; jj += ib) {
        real_t sum = 0.0;
        for (int64_t ii = 0; ii < m; ii += ib) {
            // Read 32 x 32 (ib x ib) sub-tile into shared memory.
            // This does coalesced reads of one column at a time in parallel.
            for (int64_t j = 0; j < ib; ++j)
                if (jj+j < n && ii+k < m)
                    shmem_tile[ j*ib1 + k ] = abs( tile[ (jj+j)*lda + ii+k ] );
            /*
            DPCT1065:80: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier(); // shmem_tile loaded

            // Each thread sums one column.
            for (int64_t i = 0; i < ib; ++i)
                if (ii+i < m)
                    sum += shmem_tile[ k*ib1 + i ];

            // Each thread sums one row.
            for (int64_t j = 0; j < ib; ++j)
                if (jj+j < n)
                    row_sums[ ii+k ] += shmem_tile[ j*ib1 + k ];
            /*
            DPCT1065:81: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier(); // done with shmem_tile
        }

        if (jj+k < n)
            tiles_sums[item_ct1.get_group(2) * ldv + jj + k] = sum;
    }

    // Save row sums.
    for (int64_t ii = 0; ii < m; ii += ib) {
        if (ii+k < m)
            tiles_sums[item_ct1.get_group(2) * ldv + ii + k + n] = row_sums[ii + k];
    }
}

//------------------------------------------------------------------------------
/// Batched routine that computes a partial norm for each tile.
/// Used for full, off-diagonal tiles within a symmetric matrix,
/// where element Aij contributes to both column i and j.
///
/// @param[in] norm
///     Norm to compute. See values for description.
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
///     - Norm::Inf: for symmetric, same as Norm::One
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
void synormOffdiag(
    lapack::Norm norm,
    int64_t m, int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* values, int64_t ldv,
    int64_t batch_count,
    blas::Queue &queue)
{
    using real_t = blas::real_type<scalar_t>;

    // quick return
    if (batch_count == 0)
        return;

    //---------
    // one norm
    if (norm == lapack::Norm::One || norm == lapack::Norm::Inf) {
        assert(ldv >= n);
        size_t shared_mem
            /*
            DPCT1083:82: The size of local memory in the migrated code may be
            different from the original code. Check that the allocated memory
            size in the migrated code is correct.
            */
            = sizeof(real_t) * (ib * ib1 + roundup(m, int64_t(ib)));
        assert( shared_mem <= 48*1024 ); // max 48 KiB
        ((sycl::queue *)(&queue.stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, batch_count) *
                                      sycl::range<3>(1, 1, 32),
                                  sycl::range<3>(1, 1, 32)),
                [=](sycl::nd_item<3> item_ct1) {
                    synorm_offdiag_one_kernel(m, n, Aarray, lda, values, ldv,
                                              item_ct1,
                                              dpct_local_acc_ct1.get_pointer());
                });
        });
    }
    else {
        slate_not_implemented("Only Norm::One and Norm::Inf is supported.");
    }

}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void synorm(
    lapack::Norm norm, lapack::Uplo uplo,
    int64_t n,
    float const* const* Aarray, int64_t lda,
    float* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue);

template
void synorm(
    lapack::Norm norm, lapack::Uplo uplo,
    int64_t n,
    double const* const* Aarray, int64_t lda,
    double* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue);

//------------------------------------------------------------------------------
// Specializations to cast std::complex => cuComplex.
template <>
void synorm(
    lapack::Norm norm, lapack::Uplo uplo,
    int64_t n,
    std::complex<float> const* const* Aarray, int64_t lda,
    float* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue)
{
    synorm(norm, uplo, n, (sycl::float2 **)Aarray, lda, values, ldv,
           batch_count, queue);
}

template <>
void synorm(
    lapack::Norm norm, lapack::Uplo uplo,
    int64_t n,
    std::complex<double> const* const* Aarray, int64_t lda,
    double* values, int64_t ldv, int64_t batch_count,
    blas::Queue &queue)
{
    synorm(norm, uplo, n, (sycl::double2 **)Aarray, lda, values, ldv,
           batch_count, queue);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void synormOffdiag(
    lapack::Norm norm,
    int64_t m, int64_t n,
    float const* const* Aarray, int64_t lda,
    float* values, int64_t ldv,
    int64_t batch_count,
    blas::Queue &queue);

template
void synormOffdiag(
    lapack::Norm norm,
    int64_t m, int64_t n,
    double const* const* Aarray, int64_t lda,
    double* values, int64_t ldv,
    int64_t batch_count,
    blas::Queue &queue);

//------------------------------------------------------------------------------
// Specializations to cast std::complex => cuComplex.
template <>
void synormOffdiag(
    lapack::Norm norm,
    int64_t m, int64_t n,
    std::complex<float> const* const* Aarray, int64_t lda,
    float* values, int64_t ldv,
    int64_t batch_count,
    blas::Queue &queue)
{
    synormOffdiag(norm, m, n, (sycl::float2 **)Aarray, lda, values, ldv,
                  batch_count, queue);
}

template <>
void synormOffdiag(
    lapack::Norm norm,
    int64_t m, int64_t n,
    std::complex<double> const* const* Aarray, int64_t lda,
    double* values, int64_t ldv,
    int64_t batch_count,
    blas::Queue &queue)

{
    synormOffdiag(norm, m, n, (sycl::double2 **)Aarray, lda, values, ldv,
                  batch_count, queue);
}

} // namespace device
} // namespace slate
