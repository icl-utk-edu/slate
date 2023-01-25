// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/internal/device.hh"
#include "internal/internal_batch.hh"
#include "internal/internal.hh"
#include "slate/internal/util.hh"
#include "slate/BandMatrix.hh"
#include "internal/Tile_lapack.hh"
#include "slate/types.hh"

#include <vector>

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// General banded matrix norm.
/// Dispatches to target implementations.
///
/// @param[in] in_norm
/// - Norm::Max: values is dimension 1 and contains the local max.
/// - Norm::One: values is dimension n and contains the local column sum.
/// - Norm::Inf: values is dimension m and contains the local row sum.
/// - Norm::Fro: values is dimension 2 and contains the local scale and
///              sum-of-squares.
///
/// @ingroup norm_internal
///
template <Target target, typename scalar_t>
void norm(
    Norm in_norm, NormScope scope, BandMatrix<scalar_t>&& A,
    blas::real_type<scalar_t>* values,
    int priority, int queue_index)
{
    norm(internal::TargetType<target>(),
         in_norm, scope, A, values,
         priority, queue_index);
}

//------------------------------------------------------------------------------
/// General banded matrix norm.
/// Host OpenMP task implementation.
/// @ingroup norm_internal
///
template <typename scalar_t>
void norm(
    internal::TargetType<Target::HostTask>,
    Norm in_norm, NormScope scope, BandMatrix<scalar_t>& A,
    blas::real_type<scalar_t>* values,
    int priority, int queue_index)
{
    using blas::max;
    using blas::min;
    using real_t = blas::real_type<scalar_t>;

    // norms assume column major
    // todo: relax this assumption, a few cases need to be adjusted only
    const Layout layout = Layout::ColMajor;

    if (scope != NormScope::Matrix) {
        slate_not_implemented("The NormScope isn't yet supported.");
    }

    int64_t kl = A.lowerBandwidth();
    int64_t ku = A.upperBandwidth();

    // todo: initially, assume fixed size, square tiles for simplicity
    int64_t klt = ceildiv( kl, A.tileNb(0) );
    int64_t kut = ceildiv( ku, A.tileNb(0) );

    // i, j are tile row, tile col indices; ii, jj are row, col indices.
    //---------
    // max norm
    // max_{ii,jj} abs( A_{ii,jj} )
    if (in_norm == Norm::Max) {

        // Find max of each tile, append to tiles_maxima.
        std::vector<real_t> tiles_maxima;
        #pragma omp taskgroup
        for (int64_t j = 0; j < A.nt(); ++j) {
            int64_t i_begin = max(j - kut, 0);
            int64_t i_end   = min(j + klt + 1, A.mt());
            for (int64_t i = i_begin; i < i_end; ++i) {
                if (A.tileIsLocal(i, j)) {
                    #pragma omp task slate_omp_default_none \
                        shared( A, tiles_maxima ) \
                        priority(priority) firstprivate(i, j, layout, in_norm)
                    {
                        A.tileGetForReading(i, j, LayoutConvert(layout));
                        real_t tile_max;
                        genorm(in_norm, NormScope::Matrix, A(i, j), &tile_max);
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
        // j needs to be outer loop to get jj correct.
        std::vector<real_t> tiles_sums(A.n()*A.mt(), 0.0);
        int64_t jj = 0;
        #pragma omp taskgroup
        for (int64_t j = 0; j < A.nt(); ++j) {
            int64_t i_begin = max(j - kut, 0);
            int64_t i_end   = min(j + klt + 1, A.mt());
            for (int64_t i = i_begin; i < i_end; ++i) {
                if (A.tileIsLocal(i, j)) {
                    #pragma omp task slate_omp_default_none \
                        shared( A, tiles_sums ) \
                        firstprivate(i, j, layout, jj, in_norm) priority(priority)
                    {
                        A.tileGetForReading(i, j, LayoutConvert(layout));
                        genorm(in_norm, NormScope::Matrix, A(i, j), &tiles_sums[A.n()*i+jj]);
                    }
                }
            }
            jj += A.tileNb(j);
        }

        // Sum tile results into local results.
        // todo: This is currently a performance bottleneck.
        // Perhaps omp taskloop could be applied here.
        // Perhaps with chunking of A.nb().
        std::fill_n(values, A.n(), 0.0);
        for (int64_t i = 0; i < A.mt(); ++i)
            #pragma omp taskloop slate_omp_default_none \
                shared( A, tiles_sums, values ) \
                firstprivate(i) priority(priority)
            for (int64_t jj_ = 0; jj_ < A.n(); ++jj_)
                values[jj_] += tiles_sums[A.n()*i + jj_];
    }
    //---------
    // inf norm
    // max row sum = max_ii sum_jj abs( A_{ii,jj} )
    else if (in_norm == Norm::Inf) {

        // Sum each row within a tile.
        // i needs to be outer loop to get ii correct.
        std::vector<real_t> tiles_sums(A.m()*A.nt(), 0.0);
        int64_t ii = 0;
        #pragma omp taskgroup
        for (int64_t i = 0; i < A.mt(); ++i) {
            int64_t j_begin = max(i - klt, 0);
            int64_t j_end   = min(i + kut + 1, A.nt());
            for (int64_t j = j_begin; j < j_end; ++j) {
                if (A.tileIsLocal(i, j)) {
                    #pragma omp task slate_omp_default_none \
                        shared( A, tiles_sums ) \
                        firstprivate(i, j, ii, in_norm, layout) priority(priority)
                    {
                        A.tileGetForReading(i, j, LayoutConvert(layout));
                        genorm(in_norm, NormScope::Matrix, A(i, j), &tiles_sums[A.m()*j + ii]);
                    }
                }
            }
            ii += A.tileMb(i);
        }

        // Sum tile results into local results.
        // Right now it goes over the partial sums of the entire matrix,
        // with all the non-local sums being zero.
        // todo: Eventually this needs to be done like in the device code,
        // by summing up local contributions only.
        std::fill_n(values, A.m(), 0.0);
        for (int64_t j = 0; j < A.nt(); ++j) {
            #pragma omp taskloop slate_omp_default_none \
                shared( A, tiles_sums, values ) \
                firstprivate(j) priority(priority)
            for (int64_t ii_ = 0; ii_ < A.m(); ++ii_)
                values[ii_] += tiles_sums[A.m()*j + ii_];
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
        for (int64_t j = 0; j < A.nt(); ++j) {
            int64_t i_begin = max(j - kut, 0);
            int64_t i_end   = min(j + klt + 1, A.mt());
            for (int64_t i = i_begin; i < i_end; ++i) {
                if (A.tileIsLocal(i, j)) {
                    #pragma omp task slate_omp_default_none \
                        shared( A, values ) \
                        firstprivate(i, j, in_norm, layout) priority(priority)
                    {
                        A.tileGetForReading(i, j, LayoutConvert(layout));
                        real_t tile_values[2];
                        genorm(in_norm, NormScope::Matrix, A(i, j), tile_values);
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

//------------------------------------------------------------------------------
/// General banded matrix norm.
/// Host nested OpenMP implementation.
/// TODO: currently, this does only max norm.
/// @ingroup norm_internal
///
template <typename scalar_t>
void norm(
    internal::TargetType<Target::HostNest>,
    Norm in_norm, NormScope scope, BandMatrix<scalar_t>& A,
    blas::real_type<scalar_t>* values,
    int priority, int queue_index)
{
    using blas::max;
    using blas::min;
    using real_t = blas::real_type<scalar_t>;
    if (in_norm != Norm::Max)
        slate_not_implemented("The NormScope isn't yet supported.");

    // norms assume column major
    // todo: relax this assumption, a few cases need to be adjusted only
    const Layout layout = Layout::ColMajor;

    std::vector<real_t> tiles_maxima;

    int64_t A_mt = A.mt();
    int64_t A_nt = A.nt();

    int64_t kl = A.lowerBandwidth();
    int64_t ku = A.upperBandwidth();

    // todo: initially, assume fixed size, square tiles for simplicity
    int64_t klt = ceildiv( kl, A.tileNb(0) );
    int64_t kut = ceildiv( ku, A.tileNb(0) );

    // can't collapse loops due to dependencies
    #pragma omp parallel for schedule(dynamic, 1) slate_omp_default_none \
        shared(A, values) firstprivate(layout, A_nt, A_mt, klt, kut) \
        firstprivate(in_norm, tiles_maxima)
    for (int64_t j = 0; j < A_nt; ++j) {
        int64_t i_begin = max(j - kut, 0);
        int64_t i_end   = min(j + klt + 1, A_mt);
        for (int64_t i = i_begin; i < i_end; ++i) {
            if (A.tileIsLocal(i, j)) {
                A.tileGetForReading(i, j, LayoutConvert(layout));
                real_t tile_max;
                genorm(in_norm, NormScope::Matrix, A(i, j), &tile_max);
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

//------------------------------------------------------------------------------
/// General banded matrix norm.
/// GPU device implementation.
/// TODO
/// @ingroup norm_internal
///
template <typename scalar_t>
void norm(
    internal::TargetType<Target::Devices>,
    Norm in_norm, NormScope scope, BandMatrix<scalar_t>& A,
    blas::real_type<scalar_t>* values,
    int priority, int queue_index)
{
    using blas::max;
    using blas::min;
    using real_t = blas::real_type<scalar_t>;
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    // todo: relax this assumption, a few cases need to be adjusted only
    const Layout layout = Layout::ColMajor;

    assert(A.num_devices() > 0);

    std::vector<std::vector<scalar_t*> > a_host_arrays(A.num_devices());
    std::vector<std::vector<real_t> > vals_host_arrays(A.num_devices());

    std::vector<scalar_t**> a_dev_arrays(A.num_devices());
    std::vector<real_t*> vals_dev_arrays(A.num_devices());

    int64_t kl = A.lowerBandwidth();
    int64_t ku = A.upperBandwidth();

    // todo: initially, assume fixed size, square tiles for simplicity
    int64_t klt = ceildiv( kl, A.tileNb(0) );
    int64_t kut = ceildiv( ku, A.tileNb(0) );

    // devices_values used for max and Frobenius norms.
    std::vector<real_t> devices_values;

    int64_t ldv = 0;
    if (in_norm == Norm::Max) {
        ldv = 1;
        devices_values.resize(A.num_devices());
    }
    else if (in_norm == Norm::One) {
        ldv = A.tileNb(0);
    }
    else if (in_norm == Norm::Inf) {
        ldv = A.tileMb(0);
    }
    else if (in_norm == Norm::Fro) {
        ldv = 2;
        devices_values.resize(A.num_devices() * 2);
    }

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
    int64_t i_begin = 0;
    int64_t i_end   = 0;

    #pragma omp taskgroup
    for (int device = 0; device < A.num_devices(); ++device) {
        #pragma omp task slate_omp_default_none \
            shared(A, devices_values, vals_host_arrays, \
            vals_dev_arrays, jrange, irange, a_dev_arrays, a_host_arrays) \
            firstprivate(layout, in_norm, ldv, queue_index, device, i_end, i_begin, kut, klt) \
            priority(priority)
        {
            std::set<ij_tuple> A_tiles_set;

            for (int64_t j = 0; j < A.nt(); ++j) {
                i_begin = max(j - kut, 0);
                i_end   = min(j + klt + 1, A.mt());
                for (int64_t i = i_begin; i < i_end; ++i) {
                    if (A.tileIsLocal(i, j) && device == A.tileDevice(i, j))
                        A_tiles_set.insert({i, j});
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
                for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                    i_begin = max(j - kut, 0);
                    i_end   = min(j + klt + 1, A.mt());

                    i_begin = std::max( irange[q][0], i_begin );
                    i_end   = std::min( irange[q][1], i_end );

                    for (int64_t i = i_begin; i < i_end; ++i) {
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
                        device::genorm(in_norm, NormScope::Matrix,
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

    for (int device = 0; device < A.num_devices(); ++device) {
        blas::Queue* queue = A.compute_queue(device, queue_index);
        blas::device_free(a_dev_arrays[device], *queue);
        blas::device_free(vals_dev_arrays[device], *queue);
    }

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
                for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                    i_begin = max(j - kut, 0);
                    i_end   = min(j + klt + 1, A.mt());

                    i_begin = std::max( irange[q][0], i_begin );
                    i_end   = std::min( irange[q][1], i_end );
                    for (int64_t i = i_begin; i < i_end; ++i) {
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
                for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                    i_begin = max(j - kut, 0);
                    i_end   = min(j + klt + 1, A.mt());

                    i_begin = std::max( irange[q][0], i_begin );
                    i_end   = std::min( irange[q][1], i_end );
                    for (int64_t i = i_begin; i < i_end; ++i) {
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

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void norm<Target::HostTask, float>(
    Norm in_norm, NormScope scope, BandMatrix<float>&& A,
    float* values,
    int priority, int queue_index);

template
void norm<Target::HostNest, float>(
    Norm in_norm, NormScope scope, BandMatrix<float>&& A,
    float* values,
    int priority, int queue_index);

template
void norm<Target::Devices, float>(
    Norm in_norm, NormScope scope, BandMatrix<float>&& A,
    float* values,
    int priority, int queue_index);

// ----------------------------------------
template
void norm<Target::HostTask, double>(
    Norm in_norm, NormScope scope, BandMatrix<double>&& A,
    double* values,
    int priority, int queue_index);

template
void norm<Target::HostNest, double>(
    Norm in_norm, NormScope scope, BandMatrix<double>&& A,
    double* values,
    int priority, int queue_index);

template
void norm<Target::Devices, double>(
    Norm in_norm, NormScope scope, BandMatrix<double>&& A,
    double* values,
    int priority, int queue_index);

// ----------------------------------------
template
void norm< Target::HostTask, std::complex<float> >(
    Norm in_norm, NormScope scope, BandMatrix< std::complex<float> >&& A,
    float* values,
    int priority, int queue_index);

template
void norm< Target::HostNest, std::complex<float> >(
    Norm in_norm, NormScope scope, BandMatrix< std::complex<float> >&& A,
    float* values,
    int priority, int queue_index);

template
void norm< Target::Devices, std::complex<float> >(
    Norm in_norm, NormScope scope, BandMatrix< std::complex<float> >&& A,
    float* values,
    int priority, int queue_index);

// ----------------------------------------
template
void norm< Target::HostTask, std::complex<double> >(
    Norm in_norm, NormScope scope, BandMatrix< std::complex<double> >&& A,
    double* values,
    int priority, int queue_index);

template
void norm< Target::HostNest, std::complex<double> >(
    Norm in_norm, NormScope scope, BandMatrix< std::complex<double> >&& A,
    double* values,
    int priority, int queue_index);

template
void norm< Target::Devices, std::complex<double> >(
    Norm in_norm, NormScope scope, BandMatrix< std::complex<double> >&& A,
    double* values,
    int priority, int queue_index);

} // namespace internal
} // namespace slate
