// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/internal/device.hh"
#include "internal/internal_batch.hh"
#include "internal/internal.hh"
#include "slate/internal/util.hh"
#include "slate/Matrix.hh"
#include "internal/Tile_lapack.hh"
#include "slate/types.hh"

#include <vector>

namespace slate {

//------------------------------------------------------------------------------
// On macOS, nvcc using clang++ generates a different C++ name mangling
// (std::__1::complex) than g++ for std::complex. This solution is to use
// cu*Complex in .cu files, and cast from std::complex here.
namespace device {

template <>
void genorm(
    Norm in_norm, NormScope scope,
    int64_t m, int64_t n,
    std::complex<float> const* const* Aarray, int64_t lda,
    float* values, int64_t ldv,
    int64_t batch_count,
    blas::Queue &queue)
{
#if defined( BLAS_HAVE_CUBLAS )
    genorm(in_norm, scope, m, n, (cuFloatComplex**) Aarray, lda,
           values, ldv, batch_count, queue);

#elif defined( BLAS_HAVE_ROCBLAS )
    genorm(in_norm, scope, m, n, (hipFloatComplex**) Aarray, lda,
           values, ldv, batch_count, queue);
#endif
}

template <>
void genorm(
    Norm in_norm, NormScope scope,
    int64_t m, int64_t n,
    std::complex<double> const* const* Aarray, int64_t lda,
    double* values, int64_t ldv,
    int64_t batch_count,
    blas::Queue &queue)
{
#if defined( BLAS_HAVE_CUBLAS )
    genorm(in_norm, scope, m, n, (cuDoubleComplex**) Aarray, lda,
           values, ldv, batch_count, queue);

#elif defined( BLAS_HAVE_ROCBLAS )
    genorm(in_norm, scope, m, n, (hipDoubleComplex**) Aarray, lda,
           values, ldv, batch_count, queue);
#endif
}

#if ! defined( SLATE_HAVE_DEVICE )
// Specializations to allow compilation without CUDA or HIP.
template <>
void genorm(
    Norm in_norm, NormScope scope,
    int64_t m, int64_t n,
    double const* const* Aarray, int64_t lda,
    double* values, int64_t ldv,
    int64_t batch_count,
    blas::Queue &queue)
{
}

template <>
void genorm(
    Norm in_norm, NormScope scope,
    int64_t m, int64_t n,
    float const* const* Aarray, int64_t lda,
    float* values, int64_t ldv,
    int64_t batch_count,
    blas::Queue &queue)
{
}
#endif // not SLATE_HAVE_DEVICE

} // namespace device

namespace internal {

//------------------------------------------------------------------------------
/// General matrix norm.
/// Dispatches to target implementations.
///
/// @param[in] in_norm
/// - Norm::Max: values is dimension 1 and contains the local max.
/// - Norm::One: values is dimension n and contains the local column sum.
/// - Norm::Inf: values is dimension m and contains the local row sum.
/// - Norm::Fro: values is dimension 2 and contains the local scale and
///              sum-of-squares.
///
template <Target target, typename scalar_t>
void norm(
    Norm in_norm, NormScope scope, Matrix<scalar_t>&& A,
    blas::real_type<scalar_t>* values,
    int priority, int queue_index)
{
    norm(internal::TargetType<target>(),
         in_norm, scope, A, values,
         priority, queue_index);
}

//------------------------------------------------------------------------------
/// General matrix norm.
/// Host OpenMP task implementation.
/// @ingroup norm_internal
///
template <typename scalar_t>
void norm(
    internal::TargetType<Target::HostTask>,
    Norm in_norm, NormScope scope, Matrix<scalar_t>& A,
    blas::real_type<scalar_t>* values,
    int priority, int queue_index)
{
    using real_t = blas::real_type<scalar_t>;

    // norms assume column major
    // todo: relax this assumption, a few cases need to be adjusted only
    const Layout layout = Layout::ColMajor;

    // i, j are tile row, tile col indices; ii, jj are row, col indices.
    //---------
    // max norm
    // max_{ii,jj} abs( A_{ii,jj} )
    if (scope == NormScope::Matrix) {

        if (in_norm == Norm::Max) {

            // Find max of each tile, append to tiles_maxima.
            std::vector<real_t> tiles_maxima;
            #pragma omp taskgroup
            for (int64_t i = 0; i < A.mt(); ++i) {
                for (int64_t j = 0; j < A.nt(); ++j) {
                    if (A.tileIsLocal(i, j)) {
                        #pragma omp task slate_omp_default_none \
                            shared( A, tiles_maxima ) \
                            firstprivate(i, j, layout, in_norm, scope) priority(priority)
                        {
                            A.tileGetForReading(i, j, LayoutConvert(layout));
                            real_t tile_max;
                            genorm(in_norm, scope, A(i, j), &tile_max);
                            #pragma omp critical
                            {
                                tiles_maxima.push_back(tile_max);
                            }
                        }
                    }
                }
            }

            // Find max of tiles_maxima.
            *values = lapack::lange(in_norm,
                                    1, tiles_maxima.size(),
                                    tiles_maxima.data(), 1);

        }
        //---------
        // one norm
        // max col sum = max_jj sum_ii abs( A_{ii,jj} )
        else if (in_norm == Norm::One) {

            // Sum each column within a tile.
            std::vector<real_t> tiles_sums(A.n()*A.mt(), 0.0);
            #pragma omp taskgroup
            for (int64_t i = 0; i < A.mt(); ++i) {
                int64_t jj = 0;
                for (int64_t j = 0; j < A.nt(); ++j) {
                    if (A.tileIsLocal(i, j)) {
                        #pragma omp task slate_omp_default_none \
                            shared( A, tiles_sums ) \
                            firstprivate(i, j, layout, in_norm, scope, jj) priority(priority)
                        {
                            A.tileGetForReading(i, j, LayoutConvert(layout));
                            genorm(in_norm, scope, A(i, j), &tiles_sums[A.n()*i+jj]);
                        }
                    }
                    jj += A.tileNb(j);
                }
            }

            // Sum tile results into local results.
            // Summing up local contributions only.
            std::fill_n(values, A.n(), 0.0);
            {
                trace::Block trace_block("slate::Tiles_sum");

                int64_t jj = 0;
                for (int64_t j = 0; j < A.nt(); ++j) {
                    int64_t nb = A.tileNb(j);
                    for (int64_t i = 0; i < A.mt(); ++i) {
                        if (A.tileIsLocal(i, j)) {
                                blas::axpy(
                                    nb, 1.0,
                                    &tiles_sums[A.n()*i + jj ], 1,
                                    &values[jj], 1);
                        }
                    }
                    jj += A.tileNb(j);
                }
            }
        }
        //---------
        // inf norm
        // max row sum = max_ii sum_jj abs( A_{ii,jj} )
        else if (in_norm == Norm::Inf) {

            // Sum each row within a tile.
            std::vector<real_t> tiles_sums(A.m()*A.nt(), 0.0);
            trace::Block trace_block("slate::Rows_sum");
            #pragma omp taskgroup
            for (int64_t j = 0; j < A.nt(); ++j) {
                int64_t ii = 0;
                for (int64_t i = 0; i < A.mt(); ++i) {
                    if (A.tileIsLocal(i, j)) {
                        #pragma omp task slate_omp_default_none \
                            shared( A, tiles_sums ) \
                            firstprivate(i, j, layout, in_norm, scope, ii) priority(priority)
                        {
                            A.tileGetForReading(i, j, LayoutConvert(layout));
                            genorm(in_norm, scope, A(i, j), &tiles_sums[A.m()*j + ii]);
                        }
                    }
                ii += A.tileMb(i);
                }
            }

            // Sum tile results into local results.
            // Summing up local contributions only.
            std::fill_n(values, A.m(), 0.0);
            {
                trace::Block trace_block2("slate::Tiles_sum");

                int64_t ii = 0;
                for (int64_t i = 0; i < A.mt(); ++i) {
                    for (int64_t j = 0; j < A.nt(); ++j) {
                        int64_t mb = A.tileMb(i);
                        if (A.tileIsLocal(i, j)) {
                            blas::axpy(
                                mb, 1.0,
                                &tiles_sums[A.m()*j + ii ], 1,
                                &values[ii], 1);
                        }
                    }
                    ii += A.tileMb(i);
                }
            }
        }
        //---------
        // Frobenius norm
        // sqrt( sum_{ii,jj} abs( A_{ii,jj} )^2 )
        // In scaled form: scale^2 sumsq = sum abs( A_{ii,jj}^2 )
        else if (in_norm == Norm::Fro) {

            values[0] = 0;  // scale
            values[1] = 1;  // sumsq
            #pragma omp taskgroup
            for (int64_t i = 0; i < A.mt(); ++i) {
                for (int64_t j = 0; j < A.nt(); ++j) {
                    if (A.tileIsLocal(i, j)) {
                        #pragma omp task slate_omp_default_none \
                            shared( A, values ) \
                            firstprivate(i, j, layout, in_norm, scope) priority(priority)
                        {
                            A.tileGetForReading(i, j, LayoutConvert(layout));
                            real_t tile_values[2];
                            genorm(in_norm, scope, A(i, j), tile_values);
                            #pragma omp critical
                            {
                                combine_sumsq(values[0], values[1],
                                          tile_values[0], tile_values[1]);
                            }
                        }
                    }
                }
            }
        }
    }
    else if (scope == NormScope::Columns) {

        if (in_norm == Norm::Max) {
            // Find max of each column in each tile.
            std::vector<real_t> cols_maxima(A.n()*A.mt(), 0.0);
            #pragma omp taskgroup
            for (int64_t i = 0; i < A.mt(); ++i) {
                int64_t jj = 0;
                for (int64_t j = 0; j < A.nt(); ++j) {
                    if (A.tileIsLocal(i, j)) {
                        #pragma omp task slate_omp_default_none \
                            shared( A, cols_maxima ) \
                            firstprivate(i, j, layout, in_norm, scope, jj) priority(priority)
                        {
                            A.tileGetForReading(i, j, LayoutConvert(layout));
                            genorm(in_norm, scope, A(i, j), &cols_maxima[A.n()*i+jj]);
                        }
                    }
                    jj += A.tileNb(j);
                }
            }

            // Find max of each column.
            // we are looking for absolute value, thus it is safe to initialize to 0.
            // optimized out for zero blocks
            std::fill_n(values, A.n(), 0.0);
            for (int64_t i = 0; i < A.mt(); ++i) {
                int64_t jj = 0;
                for (int64_t j = 0; j < A.nt(); ++j) {
                    if (A.tileIsLocal(i, j)) {
                        for (int64_t ll = 0; ll < A.tileNb(j); ++ll)
                            values[ll + jj] = max_nan(values[ll + jj], cols_maxima[A.n()*i + ll + jj]);
                    }
                    jj += A.tileNb(j);
                }
            }

        }
        else {
            slate_not_implemented("The NormScope isn't yet supported.");
        }
    }
    else {
        slate_not_implemented("The NormScope isn't yet supported.");
    }
}

//------------------------------------------------------------------------------
/// General matrix norm.
/// Host nested OpenMP implementation.
/// TODO: currently, this does only max norm.
/// @ingroup norm_internal
///
template <typename scalar_t>
void norm(
    internal::TargetType<Target::HostNest>,
    Norm in_norm, NormScope scope, Matrix<scalar_t>& A,
    blas::real_type<scalar_t>* values,
    int priority, int queue_index)
{
    using real_t = blas::real_type<scalar_t>;
    if (in_norm != Norm::Max)
        slate_not_implemented("The NormScope isn't yet supported.");

    // norms assumes column major
    // todo: relax this assumption, a few cases need to be adjusted only
    const Layout layout = Layout::ColMajor;

    int64_t A_mt = A.mt();
    int64_t A_nt = A.nt();

    if (scope == NormScope::Matrix) {

        std::vector<real_t> tiles_maxima;

        #pragma omp parallel for collapse(2) schedule(dynamic, 1) \
            slate_omp_default_none \
            shared( A, tiles_maxima ) \
            firstprivate( A_mt, A_nt, scope, in_norm )
        for (int64_t i = 0; i < A_mt; ++i) {
            for (int64_t j = 0; j < A_nt; ++j) {
                if (A.tileIsLocal(i, j)) {
                    A.tileGetForReading(i, j, LayoutConvert(layout));
                    real_t tile_max;
                    genorm(in_norm, scope, A(i, j), &tile_max);
                    #pragma omp critical
                    {
                        tiles_maxima.push_back(tile_max);
                    }
                }
            }
        }

        *values = lapack::lange(in_norm,
                                1, tiles_maxima.size(),
                                tiles_maxima.data(), 1);

    }
    else if (scope == NormScope::Columns) {

        // Find max of each column in each tile.
        std::vector<real_t> cols_maxima(A.n()*A.mt(), 0.0);

        #pragma omp parallel for collapse(1) schedule(dynamic, 1) \
            slate_omp_default_none \
            shared( A, cols_maxima ) \
            firstprivate( A_mt, A_nt, layout, scope, in_norm )
        for (int64_t i = 0; i < A_mt; ++i) {
            int64_t jj = 0;
            for (int64_t j = 0; j < A_nt; ++j) {
                if (A.tileIsLocal(i, j)) {
                    A.tileGetForReading(i, j, LayoutConvert(layout));
                    genorm(in_norm, scope, A(i, j), &cols_maxima[A.n()*i+jj]);
                }
                jj += A.tileNb(j);
            }
        }

        // Find max of each column.
        // we are looking for absolute value, thus it is safe to initialize to 0.
        // optimized out for zero blocks
        std::fill_n(values, A.n(), 0.0);
        for (int64_t i = 0; i < A.mt(); ++i) {
            int64_t jj = 0;
            for (int64_t j = 0; j < A.nt(); ++j) {
                if (A.tileIsLocal(i, j)) {
                    for (int64_t ll = 0; ll < A.tileNb(j); ++ll)
                        values[ll + jj] = max_nan(values[ll + jj], cols_maxima[A.n()*i + ll + jj]);
                }
                jj += A.tileNb(j);
            }
        }

    }
    else {
        slate_not_implemented("The NormScope isn't yet supported.");
    }

}

//------------------------------------------------------------------------------
/// General matrix norm.
/// GPU device implementation.
/// @ingroup norm_internal
///
template <typename scalar_t>
void norm(
    internal::TargetType<Target::Devices>,
    Norm in_norm, NormScope scope, Matrix<scalar_t>& A,
    blas::real_type<scalar_t>* values,
    int priority, int queue_index)
{
    using real_t = blas::real_type<scalar_t>;

    // norms assume column major
    // todo: relax this assumption, a few cases need to be adjusted only
    const Layout layout = Layout::ColMajor;
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    assert(A.num_devices() > 0);

    std::vector<std::vector<scalar_t*> > a_host_arrays(A.num_devices());
    std::vector<std::vector<real_t> > vals_host_arrays(A.num_devices());

    std::vector<scalar_t**> a_dev_arrays(A.num_devices());
    std::vector<real_t*> vals_dev_arrays(A.num_devices());

    // devices_values used for max and Frobenius norms.
    std::vector<real_t> devices_values;

    int64_t ldv = 0;
    if (scope == NormScope::Matrix) {
        if (in_norm == Norm::Max) {
            ldv = 1;
            devices_values.resize(A.num_devices());
        }
        else if (in_norm == Norm::One) {
            // todo: this assumes all tiles with uniform nb
            ldv = A.tileNb(0);
        }
        else if (in_norm == Norm::Inf) {
            // todo: this assumes all tiles with uniform mb
            ldv = A.tileMb(0);
        }
        else if (in_norm == Norm::Fro) {
            ldv = 2;
            devices_values.resize(A.num_devices() * 2);
        }
    }
    else if (scope == NormScope::Columns) {
        if (in_norm == Norm::Max) {
            // todo: this assumes all tiles with uniform nb
            ldv = A.tileNb(0);
        }
        else {
            slate_not_implemented("The NormScope isn't yet supported.");
        }
    }
    else {
        slate_not_implemented("The NormScope isn't yet supported.");
    }

    // TODO: Why are we doing this?
    // Use the batch arrays in the matrix class.
    for (int device = 0; device < A.num_devices(); ++device) {

        int64_t num_tiles = A.getMaxDeviceTiles(device);

        a_host_arrays[device].resize(num_tiles);
        vals_host_arrays[device].resize(num_tiles*ldv);

        blas::Queue* queue = A.compute_queue(device, queue_index);
        a_dev_arrays[device] = blas::device_malloc<scalar_t*>(num_tiles, *queue);
        vals_dev_arrays[device] = blas::device_malloc<real_t>(num_tiles*ldv, *queue);
    }

    // Define index ranges for regions of matrix.
    // Tiles in each region are all the same size.
    int64_t irange[4][2] = {
        { 0,        A.mt()-1 },
        { A.mt()-1, A.mt()   },
        { 0,        A.mt()-1 },
        { A.mt()-1, A.mt()   }
    };
    int64_t jrange[4][2] = {
        { 0,        A.nt()-1 },
        { 0,        A.nt()-1 },
        { A.nt()-1, A.nt()   },
        { A.nt()-1, A.nt()   }
    };

    #pragma omp taskgroup
    for (int device = 0; device < A.num_devices(); ++device) {
        #pragma omp task slate_omp_default_none \
            priority( priority ) shared( A, devices_values ) \
            shared(a_host_arrays, a_dev_arrays, vals_host_arrays, vals_dev_arrays) \
            firstprivate(device, irange, jrange, queue_index, ldv, scope, in_norm, layout)
        {
            std::set<ij_tuple> A_tiles_set;

            for (int64_t i = 0; i < A.mt(); ++i) {
                for (int64_t j = 0; j < A.nt(); ++j) {
                    if (A.tileIsLocal(i, j) && device == A.tileDevice(i, j)) {
                        A_tiles_set.insert({i, j});
                    }
                }
            }
            A.tileGetForReading(A_tiles_set, device, LayoutConvert(layout));

            // Setup batched arguments.
            scalar_t** a_host_array = a_host_arrays[device].data();
            scalar_t** a_dev_array = a_dev_arrays[device];

            int64_t batch_count = 0;
            int64_t mb[4], nb[4], lda[4], group_count[4];
            for (int q = 0; q < 4; ++q) {
                group_count[q] = 0;
                lda[q] = 0;
                mb[q] = A.tileMb(irange[q][0]);
                nb[q] = A.tileNb(jrange[q][0]);
                for (int64_t i = irange[q][0]; i < irange[q][1]; ++i) {
                    for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                        if (A.tileIsLocal(i, j) &&
                            device == A.tileDevice(i, j))
                        {
                            a_host_array[batch_count] = A(i, j, device).data();
                            lda[q] = A(i, j, device).stride();
                            ++group_count[q];
                            ++batch_count;
                        }
                    }
                }
            }

            real_t* vals_host_array = vals_host_arrays[device].data();
            real_t* vals_dev_array = vals_dev_arrays[device];

            // Batched call to compute partial results for each tile.
            {
                trace::Block trace_block("slate::device::genorm");

                blas::Queue* queue = A.compute_queue(device, queue_index);

                blas::device_memcpy<scalar_t*>(a_dev_array, a_host_array,
                                    batch_count,
                                    blas::MemcpyKind::HostToDevice,
                                    *queue);

                for (int q = 0; q < 4; ++q) {
                    if (group_count[q] > 0) {
                        device::genorm(in_norm, scope,
                                       mb[q], nb[q],
                                       a_dev_array, lda[q],
                                       vals_dev_array, ldv,
                                       group_count[q], *queue);
                        a_dev_array += group_count[q];
                        vals_dev_array += group_count[q] * ldv;
                    }
                }

                vals_dev_array = vals_dev_arrays[device];

                blas::device_memcpy<real_t>(vals_host_array, vals_dev_array,
                                    batch_count*ldv,
                                    blas::MemcpyKind::DeviceToHost,
                                    *queue);

                queue->sync();
            }

            if (scope == NormScope::Matrix) {
                // Reduction over tiles to device result.
                if (in_norm == Norm::Max) {
                    devices_values[device] =
                        lapack::lange(in_norm, 1, batch_count, vals_host_array, 1);
                }
                else if (in_norm == Norm::Fro) {
                    for (int64_t k = 0; k < batch_count; ++k) {
                        combine_sumsq(devices_values[2*device + 0],
                                  devices_values[2*device + 1],
                                  vals_host_array[2*k + 0],
                                  vals_host_array[2*k + 1]);
                    }
                }
            }
        }
    }

    for (int device = 0; device < A.num_devices(); ++device) {
        blas::Queue* queue = A.compute_queue(device, queue_index);
        blas::device_free(a_dev_arrays[device], *queue);
        blas::device_free(vals_dev_arrays[device], *queue);
    }

    if (scope == NormScope::Matrix) {

        // Reduction over devices to local result.
        if (in_norm == Norm::Max) {
            *values = lapack::lange(in_norm,
                                    1, devices_values.size(),
                                    devices_values.data(), 1);
        }
        else if (in_norm == Norm::One) {

            for (int device = 0; device < A.num_devices(); ++device) {

                real_t* vals_host_array = vals_host_arrays[device].data();

                int64_t batch_count = 0;
                for (int q = 0; q < 4; ++q) {
                    int64_t nb = A.tileNb(jrange[q][0]);
                    for (int64_t i = irange[q][0]; i < irange[q][1]; ++i) {
                        for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                            if (A.tileIsLocal(i, j) &&
                                device == A.tileDevice(i, j))
                            {
                                blas::axpy(
                                    nb, 1.0,
                                    &vals_host_array[batch_count*ldv], 1,
                                    &values[j*ldv], 1);
                                ++batch_count;
                            }
                        }
                    }
                }
            }
        }
        else if (in_norm == Norm::Inf) {

            for (int device = 0; device < A.num_devices(); ++device) {

                real_t* vals_host_array = vals_host_arrays[device].data();

                int64_t batch_count = 0;
                for (int q = 0; q < 4; ++q) {
                    int64_t mb = A.tileMb(irange[q][0]);
                    for (int64_t i = irange[q][0]; i < irange[q][1]; ++i) {
                        for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                            if (A.tileIsLocal(i, j) &&
                                device == A.tileDevice(i, j))
                            {
                                blas::axpy(
                                    mb, 1.0,
                                    &vals_host_array[batch_count*ldv], 1,
                                    &values[i*ldv], 1);
                                ++batch_count;
                            }
                        }
                    }
                }
            }
        }
        else if (in_norm == Norm::Fro) {
            values[0] = 0;
            values[1] = 1;
            for (int device = 0; device < A.num_devices(); ++device) {
                combine_sumsq(values[0], values[1],
                          devices_values[2*device + 0],
                          devices_values[2*device + 1]);
            }
        }
    }
    else if (scope == NormScope::Columns) {

        if (in_norm == Norm::Max) {
            // Reduction over devices to local result.
            // todo: re-arrange loops to be able to issue omp tasks
            for (int device = 0; device < A.num_devices(); ++device) {

                real_t* vals_host_array = vals_host_arrays[device].data();

                int64_t batch_count = 0;
                for (int q = 0; q < 4; ++q) {
                    int64_t nb = A.tileNb(jrange[q][0]);
                    for (int64_t i = irange[q][0]; i < irange[q][1]; ++i) {
                        for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                            if (A.tileIsLocal(i, j) &&
                                device == A.tileDevice(i, j))
                            {
                                for (int k = 0; k < nb; ++k) {
                                    values[j*ldv + k] =
                                        max_nan(vals_host_array[batch_count*ldv + k],
                                                values[j*ldv + k]);
                                }
                                ++batch_count;
                            }
                        }
                    }
                }
            }
        }
        else {
            slate_not_implemented("The NormScope isn't yet supported.");
        }
    }
    else {
        slate_not_implemented("The NormScope isn't yet supported.");
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void norm<Target::HostTask, float>(
    Norm in_norm, NormScope scope, Matrix<float>&& A,
    float* values,
    int priority, int queue_index);

template
void norm<Target::HostNest, float>(
    Norm in_norm, NormScope scope, Matrix<float>&& A,
    float* values,
    int priority, int queue_index);

template
void norm<Target::Devices, float>(
    Norm in_norm, NormScope scope, Matrix<float>&& A,
    float* values,
    int priority, int queue_index);

// ----------------------------------------
template
void norm<Target::HostTask, double>(
    Norm in_norm, NormScope scope, Matrix<double>&& A,
    double* values,
    int priority, int queue_index);

template
void norm<Target::HostNest, double>(
    Norm in_norm, NormScope scope, Matrix<double>&& A,
    double* values,
    int priority, int queue_index);

template
void norm<Target::Devices, double>(
    Norm in_norm, NormScope scope, Matrix<double>&& A,
    double* values,
    int priority, int queue_index);

// ----------------------------------------
template
void norm< Target::HostTask, std::complex<float> >(
    Norm in_norm, NormScope scope, Matrix< std::complex<float> >&& A,
    float* values,
    int priority, int queue_index);

template
void norm< Target::HostNest, std::complex<float> >(
    Norm in_norm, NormScope scope, Matrix< std::complex<float> >&& A,
    float* values,
    int priority, int queue_index);

template
void norm< Target::Devices, std::complex<float> >(
    Norm in_norm, NormScope scope, Matrix< std::complex<float> >&& A,
    float* values,
    int priority, int queue_index);

// ----------------------------------------
template
void norm< Target::HostTask, std::complex<double> >(
    Norm in_norm, NormScope scope, Matrix< std::complex<double> >&& A,
    double* values,
    int priority, int queue_index);

template
void norm< Target::HostNest, std::complex<double> >(
    Norm in_norm, NormScope scope, Matrix< std::complex<double> >&& A,
    double* values,
    int priority, int queue_index);

template
void norm< Target::Devices, std::complex<double> >(
    Norm in_norm, NormScope scope, Matrix< std::complex<double> >&& A,
    double* values,
    int priority, int queue_index);

} // namespace internal
} // namespace slate
