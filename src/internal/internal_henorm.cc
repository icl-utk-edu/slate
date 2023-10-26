// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/internal/device.hh"
#include "internal/internal.hh"
#include "internal/internal_batch.hh"
#include "slate/internal/util.hh"
#include "slate/HermitianMatrix.hh"
#include "internal/Tile_lapack.hh"
#include "internal/Tile_henorm.hh"
#include "internal/Tile_synorm.hh"
#include "slate/types.hh"

#include <vector>

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Hermitian matrix norm.
/// Dispatches to target implementations.
///
/// @param[in] in_norm
/// - Norm::Max: values is dimension 1 and contains the local max.
/// - Norm::One: values is dimension n and contains the local column sum.
/// - Norm::Inf: for Hermitian, same as Norm::One.
/// - Norm::Fro: values is dimension 2 and contains the local scale and
///              sum-of-squares.
///
template <Target target, typename scalar_t>
void norm(
    Norm in_norm, NormScope scope, HermitianMatrix<scalar_t>&& A,
    blas::real_type<scalar_t>* values,
    int priority, int queue_index)
{
    norm(internal::TargetType<target>(),
         in_norm, scope, A, values,
         priority, queue_index);
}

//------------------------------------------------------------------------------
/// Hermitian matrix norm.
/// Host OpenMP task implementation.
/// @ingroup norm_internal
///
template <typename scalar_t>
void norm(
    internal::TargetType<Target::HostTask>,
    Norm in_norm, NormScope scope, HermitianMatrix<scalar_t>& A,
    blas::real_type<scalar_t>* values,
    int priority, int queue_index)
{
    using real_t = blas::real_type<scalar_t>;

    // norms assumes column major
    // todo: relax this assumption, a few cases need to be adjusted only
    const Layout layout = Layout::ColMajor;

    if (scope != NormScope::Matrix) {
        slate_not_implemented("The NormScope isn't yet supported.");
    }

    bool lower = (A.uploLogical() == Uplo::Lower);

    // i, j are tile row, tile col indices; ii, jj are row, col indices.
    //---------
    // max norm
    // max_{ii,jj} abs( A_{ii,jj} )
    if (in_norm == Norm::Max) {
        // Note: same code in slate::internal::trnorm( Norm::Max ).
        // Find max of each tile, append to tiles_maxima.
        std::vector<real_t> tiles_maxima;
        #pragma omp taskgroup
        for (int64_t j = 0; j < A.nt(); ++j) {
            // diagonal tile
            if (j < A.mt() && A.tileIsLocal(j, j)) {
                #pragma omp task slate_omp_default_none \
                    shared( A, tiles_maxima ) \
                    firstprivate(j, layout, in_norm) priority(priority)
                {
                    A.tileGetForReading(j, j, LayoutConvert(layout));
                    real_t tile_max;
                    henorm(in_norm, A(j, j), &tile_max);
                    #pragma omp critical
                    {
                        tiles_maxima.push_back(tile_max);
                    }
                }
            }
            // off-diagonal tiles
            if (lower) {
                for (int64_t i = j+1; i < A.mt(); ++i) {  // strictly lower
                    if (A.tileIsLocal(i, j)) {
                        #pragma omp task slate_omp_default_none \
                            shared( A, tiles_maxima ) \
                            firstprivate(i, j, layout, in_norm) priority(priority)
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
            else { // Uplo::Upper
                for (int64_t i = 0; i < j && i < A.mt(); ++i) {  // strictly upper
                    if (A.tileIsLocal(i, j)) {
                        #pragma omp task slate_omp_default_none \
                            shared( A, tiles_maxima ) \
                            firstprivate(i, j, layout, in_norm) priority(priority)
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
        }
        // end omp taskgroup

        // Find max of tiles_maxima.
        *values = lapack::lange(in_norm,
                                1, tiles_maxima.size(),
                                tiles_maxima.data(), 1);
    }
    //---------
    // one norm
    // max col sum = max_jj sum_ii abs( A_{ii,jj} )
    else if (in_norm == Norm::One || in_norm == Norm::Inf) {
        // Sum each column within a tile.
        std::vector<real_t> tiles_sums(A.n()*A.mt(), 0.0);
        int64_t jj = 0;
        #pragma omp taskgroup
        for (int64_t j = 0; j < A.nt(); ++j) {
            // diagonal tiles
            if (j < A.mt() && A.tileIsLocal(j, j)) {
                #pragma omp task slate_omp_default_none \
                    shared( A, tiles_sums ) \
                    firstprivate(j, jj, layout, in_norm) priority(priority)
                {
                    A.tileGetForReading(j, j, LayoutConvert(layout));
                    henorm(in_norm, A(j, j), &tiles_sums[A.n()*j + jj]);
                }
            }
            // off-diagonal tiles (same as synorm)
            if (lower) {
                int64_t ii = jj + A.tileNb(j);
                for (int64_t i = j+1; i < A.mt(); ++i) { // strictly lower
                    if (A.tileIsLocal(i, j)) {
                        #pragma omp task slate_omp_default_none \
                            shared( A, tiles_sums ) \
                            firstprivate(i, j, ii, jj, layout, in_norm) priority(priority)
                        {
                            A.tileGetForReading(i, j, LayoutConvert(layout));
                            synormOffdiag(in_norm, A(i, j),
                                          &tiles_sums[A.n()*i + jj],
                                          &tiles_sums[A.n()*j + ii]);
                        }
                    }
                    ii += A.tileMb(i);
                }
            }
            else { // Uplo::Upper
                int64_t ii = 0;
                for (int64_t i = 0; i < j && i < A.mt(); ++i) { // strictly upper
                    if (A.tileIsLocal(i, j)) {
                        #pragma omp task slate_omp_default_none \
                            shared( A, tiles_sums ) \
                            firstprivate(i, j, ii, jj, layout, in_norm) priority(priority)
                        {
                            A.tileGetForReading(i, j, LayoutConvert(layout));
                            synormOffdiag(in_norm, A(i, j),
                                          &tiles_sums[A.n()*i + jj],
                                          &tiles_sums[A.n()*j + ii]);
                        }
                    }
                    ii += A.tileMb(i);
                }
            }
            jj += A.tileNb(j);
        }
        // end omp taskgroup

        // Sum tile results into local results.
        // Summing up local contributions only.
        std::fill_n(values, A.n(), 0.0);
        int64_t nb0 = A.tileNb(0);
        int64_t mb0 = A.tileMb(0);
        // off-diagonal blocks
        for (int64_t j = 0; j < A.nt(); ++j) {
            for (int64_t i = 0; i < A.mt(); ++i) {
                int64_t nb = A.tileNb(j);
                int64_t mb = A.tileMb(i);
                if (A.tileIsLocal(i, j) &&
                    ( (  lower && i > j) ||
                      (! lower && i < j) ))
                {
                    // col sums
                    blas::axpy(
                        nb, 1.0,
                        &tiles_sums[A.n()*i + j*nb0 ], 1,
                        &values[j*nb0], 1);
                    // row sums
                    blas::axpy(
                        mb, 1.0,
                        &tiles_sums[A.m()*j + i*nb0 ], 1,
                        &values[i*mb0], 1);
                }
            }
        }

        // diagonal blocks
        for (int64_t j = 0; j < A.nt(); ++j) {
            int64_t nb = A.tileNb(j);
            if (A.tileIsLocal(j, j) ) {
                // col sums
                blas::axpy(
                    nb, 1.0,
                    &tiles_sums[A.n()*j + j*nb0 ], 1,
                    &values[j*nb0], 1);
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
        for (int64_t j = 0; j < A.nt(); ++j) {
            // diagonal tile
            if (j < A.mt() && A.tileIsLocal(j, j)) {
                A.tileGetForReading(j, j, LayoutConvert(layout));
                real_t tile_values[2];
                henorm(in_norm, A(j, j), tile_values);
                #pragma omp critical
                {
                    combine_sumsq(values[0], values[1],
                              tile_values[0], tile_values[1]);
                }
            }
            // off-diagonal tiles
            if (lower) {
                for (int64_t i = j+1; i < A.mt(); ++i) { // strictly lower
                    if (A.tileIsLocal(i, j)) {
                        #pragma omp task slate_omp_default_none \
                            shared( A, values ) \
                            firstprivate(i, j, layout, in_norm) priority(priority)
                        {
                            A.tileGetForReading(i, j, LayoutConvert(layout));
                            real_t tile_values[2];
                            genorm(in_norm, NormScope::Matrix, A(i, j), tile_values);
                            // double for symmetric entries
                            tile_values[1] *= 2;
                            #pragma omp critical
                            {
                                combine_sumsq(values[0], values[1],
                                          tile_values[0], tile_values[1]);
                            }
                        }
                    }
                }
            }
            else { // Uplo::Upper
                for (int64_t i = 0; i < j && i < A.mt(); ++i) { // strictly upper
                    if (A.tileIsLocal(i, j)) {
                        #pragma omp task slate_omp_default_none \
                            shared( A, values ) \
                            firstprivate(i, j, layout, in_norm) priority(priority)
                        {
                            A.tileGetForReading(i, j, LayoutConvert(layout));
                            real_t tile_values[2];
                            genorm(in_norm, NormScope::Matrix, A(i, j), tile_values);
                            // double for symmetric entries
                            tile_values[1] *= 2;
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
        // end omp taskgroup
    }
}

//------------------------------------------------------------------------------
/// Hermitian matrix norm.
/// Host nested OpenMP implementation.
/// @ingroup norm_internal
///
template <typename scalar_t>
void norm(
    internal::TargetType<Target::HostNest>,
    Norm in_norm, NormScope scope, HermitianMatrix<scalar_t>& A,
    blas::real_type<scalar_t>* values,
    int priority, int queue_index)
{
    slate_not_implemented("Target::HostNest isn't yet supported.");
}

//------------------------------------------------------------------------------
/// Hermitian matrix norm.
/// GPU device implementation.
/// @ingroup norm_internal
///
template <typename scalar_t>
void norm(
    internal::TargetType<Target::Devices>,
    Norm in_norm, NormScope scope, HermitianMatrix<scalar_t>& A,
    blas::real_type<scalar_t>* values,
    int priority, int queue_index)
{
    using real_t = blas::real_type<scalar_t>;

    // norms assume column major
    // todo: relax this assumption, a few cases need to be adjusted only
    const Layout layout = Layout::ColMajor;
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    if (scope != NormScope::Matrix) {
        slate_not_implemented("The NormScope isn't yet supported.");
    }

    bool lower = (A.uploLogical() == Uplo::Lower);

    assert(A.num_devices() > 0);

    std::vector<std::vector<real_t> > vals_host_arrays(A.num_devices());

    std::vector<real_t*> vals_dev_arrays(A.num_devices());

    // devices_values used for max and Frobenius norms.
    std::vector<real_t> devices_values;

    int64_t ldv = 0;
    if (in_norm == Norm::Max) {
        ldv = 1;
        devices_values.resize(A.num_devices());
    }
    else if (in_norm == Norm::One || in_norm == Norm::Inf) {
        for (int64_t j = 0; j < A.nt(); ++j) {
            ldv = std::max( ldv, A.tileNb(j) );
        }
        ldv *= 2;
    }
    else if (in_norm == Norm::Fro) {
        ldv = 2;
        devices_values.resize(A.num_devices() * 2);
    }

    for (int device = 0; device < A.num_devices(); ++device) {
        int64_t num_tiles = A.getMaxDeviceTiles(device);

        vals_host_arrays[device].resize(num_tiles*ldv);

        blas::Queue* queue = A.comm_queue(device);
        vals_dev_arrays[device] = blas::device_malloc<real_t>(num_tiles*ldv, *queue);
    }

    #pragma omp taskgroup
    for (int device = 0; device < A.num_devices(); ++device) {
        #pragma omp task slate_omp_default_none \
            shared( A, devices_values ) \
            shared(vals_host_arrays, vals_dev_arrays) \
            firstprivate(device, layout, lower, queue_index, in_norm, ldv) \
            priority(priority)
        {
            std::set<ij_tuple> A_tiles_set;

            for (int64_t i = 0; i < A.mt(); ++i) {
                for (int64_t j = 0; j < A.nt(); ++j) {
                    if (A.tileIsLocal(i, j) &&
                        device == A.tileDevice(i, j) &&
                        ( (  lower && i >= j) ||
                          (! lower && i <= j) ))
                    {
                        A_tiles_set.insert({i, j});
                    }
                }
            }
            A.tileGetForReading(A_tiles_set, device, LayoutConvert(layout));

            // Setup batched arguments.
            int64_t batch_size = A_tiles_set.size();
            scalar_t** a_array_host = A.array_host( device, queue_index );

            auto group_params = device_regions_build<true, 1, scalar_t>(
                                                    {A},
                                                    {a_array_host},
                                                    device );

            scalar_t** a_array_dev = A.array_device(device, queue_index);

            real_t* vals_host_array = vals_host_arrays[device].data();
            real_t* vals_dev_array = vals_dev_arrays[device];

            // Batched call to compute partial results for each tile.
            {
                trace::Block trace_block("slate::device::henorm");

                blas::Queue* queue = A.compute_queue(device, queue_index);

                blas::device_memcpy<scalar_t*>(a_array_dev, a_array_host,
                                    batch_size,
                                    blas::MemcpyKind::HostToDevice,
                                    *queue);
                real_t* vals_dev_array_group = vals_dev_array;
                for (size_t g = 0; g < group_params.size(); ++g) {
                    int64_t group_count = group_params[ g ].count;

                    if (group_params[ g ].is_diagonal) {
                        device::henorm(
                            in_norm, A.uploPhysical(), group_params[ g ].nb,
                            a_array_dev, group_params[ g ].ld[0],
                            vals_dev_array_group, ldv,
                            group_count, *queue );
                    }
                    else {
                        if (in_norm == Norm::One || in_norm == Norm::Inf) {
                            device::synormOffdiag(
                                in_norm,
                                group_params[ g ].mb, group_params[ g ].nb,
                                a_array_dev, group_params[ g ].ld[0],
                                vals_dev_array_group, ldv,
                                group_count, *queue );
                        }
                        else {
                            device::genorm(
                                in_norm, NormScope::Matrix,
                                group_params[ g ].mb, group_params[ g ].nb,
                                a_array_dev, group_params[ g ].ld[0],
                                vals_dev_array_group, ldv,
                                group_count, *queue );
                        }
                    }
                    a_array_dev += group_count;
                    vals_dev_array_group += group_count * ldv;
                    queue->sync();
                }

                blas::device_memcpy<real_t>(
                    vals_host_array, vals_dev_array,
                    batch_size*ldv,
                    blas::MemcpyKind::DeviceToHost,
                    *queue);

                queue->sync();
            }

            // Reduction over tiles to device result.
            if (in_norm == Norm::Max) {
                devices_values[device] =
                    lapack::lange(in_norm, 1, batch_size, vals_host_array, 1);
            }
            else if (in_norm == Norm::Fro) {
                int64_t cnt = 0;
                for (size_t g = 0; g < group_params.size(); ++g) {
                    int64_t group_count = group_params[ g ].count;

                    // double for symmetric entries in off-diagonal blocks
                    real_t mult = (group_params[ g ].is_diagonal) ? 1.0 : 2.0;

                    for (int64_t k = 0; k < group_count; ++k) {
                        combine_sumsq(
                            devices_values[2*device + 0],
                            devices_values[2*device + 1],
                            vals_host_array[2*cnt + 0],
                            vals_host_array[2*cnt + 1] * mult);
                        ++cnt;
                    }
                }
            }
        }
    }
    // end omp taskgroup

    for (int device = 0; device < A.num_devices(); ++device) {
        blas::Queue* queue = A.compute_queue(device, queue_index);
        blas::device_free(vals_dev_arrays[device], *queue);
    }

    // Reduction over devices to local result.
    if (in_norm == Norm::Max) {
        *values = lapack::lange(in_norm,
                                1, devices_values.size(),
                                devices_values.data(), 1);
    }
    else if (in_norm == Norm::One || in_norm == Norm::Inf) {
        auto irange = device_regions_range( true, A );
        auto jrange = device_regions_range( false, A );
        int64_t nb0 = A.tileNb(0);
        assert(A.tileNb(0) == A.tileMb(0));
        assert(A.n() == A.m());

        for (int device = 0; device < A.num_devices(); ++device) {

            real_t* vals_host_array = vals_host_arrays[device].data();

            int64_t batch_count = 0;
            for (size_t jj = 0; jj < jrange.size() - 1; ++jj) {
            for (size_t ii = 0; ii < irange.size() - 1; ++ii) {
                int64_t mb = A.tileMb( irange[ ii ] );
                int64_t nb = A.tileNb( jrange[ jj ] );
                for (int64_t j = jrange[ jj ]; j < jrange[ jj+1 ]; ++j) {
                for (int64_t i = irange[ ii ]; i < irange[ ii+1 ]; ++i) {
                    if (A.tileIsLocal( i, j ) && device == A.tileDevice( i, j )
                        && ((A.uplo() == Uplo::Lower && i > j) ||
                            (A.uplo() == Uplo::Upper && i < j))) {

                        // TODO this is broken for nonuniform block sizes
                        // col sums
                        blas::axpy(
                            nb, 1.0,
                            &vals_host_array[batch_count*ldv], 1,
                            &values[j*nb0], 1);
                        // row sums
                        blas::axpy(
                            mb, 1.0,
                            &vals_host_array[batch_count*ldv + nb], 1,
                            &values[i*nb0], 1);
                        ++batch_count;
                    }
                }} // for j,i

                int64_t ijstart = std::max(irange[ ii   ], jrange[ jj   ]);
                int64_t ijend   = std::min(irange[ ii+1 ], jrange[ jj+1 ]);
                for (int64_t ij = ijstart; ij < ijend; ++ij) {
                    if (A.tileIsLocal(ij, ij) &&
                        device == A.tileDevice(ij, ij))
                    {
                        blas::axpy(
                            nb, 1.0,
                            &vals_host_array[batch_count*ldv], 1,
                            &values[ij*nb0], 1);
                        ++batch_count;
                    }
                }
            }} // for jj,ii
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
    Norm in_norm, NormScope scope, HermitianMatrix<float>&& A,
    float* values,
    int priority, int queue_index);

template
void norm<Target::HostNest, float>(
    Norm in_norm, NormScope scope, HermitianMatrix<float>&& A,
    float* values,
    int priority, int queue_index);

template
void norm<Target::Devices, float>(
    Norm in_norm, NormScope scope, HermitianMatrix<float>&& A,
    float* values,
    int priority, int queue_index);

// ----------------------------------------
template
void norm<Target::HostTask, double>(
    Norm in_norm, NormScope scope, HermitianMatrix<double>&& A,
    double* values,
    int priority, int queue_index);

template
void norm<Target::HostNest, double>(
    Norm in_norm, NormScope scope, HermitianMatrix<double>&& A,
    double* values,
    int priority, int queue_index);

template
void norm<Target::Devices, double>(
    Norm in_norm, NormScope scope, HermitianMatrix<double>&& A,
    double* values,
    int priority, int queue_index);

// ----------------------------------------
template
void norm< Target::HostTask, std::complex<float> >(
    Norm in_norm, NormScope scope, HermitianMatrix< std::complex<float> >&& A,
    float* values,
    int priority, int queue_index);

template
void norm< Target::HostNest, std::complex<float> >(
    Norm in_norm, NormScope scope, HermitianMatrix< std::complex<float> >&& A,
    float* values,
    int priority, int queue_index);

template
void norm< Target::Devices, std::complex<float> >(
    Norm in_norm, NormScope scope, HermitianMatrix< std::complex<float> >&& A,
    float* values,
    int priority, int queue_index);

// ----------------------------------------
template
void norm< Target::HostTask, std::complex<double> >(
    Norm in_norm, NormScope scope, HermitianMatrix< std::complex<double> >&& A,
    double* values,
    int priority, int queue_index);

template
void norm< Target::HostNest, std::complex<double> >(
    Norm in_norm, NormScope scope, HermitianMatrix< std::complex<double> >&& A,
    double* values,
    int priority, int queue_index);

template
void norm< Target::Devices, std::complex<double> >(
    Norm in_norm, NormScope scope, HermitianMatrix< std::complex<double> >&& A,
    double* values,
    int priority, int queue_index);

} // namespace internal
} // namespace slate
