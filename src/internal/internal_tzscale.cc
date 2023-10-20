// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
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
#include "internal/internal_util.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Scale Trapezoid matrix entries by the real scalar numer/denom.
/// Dispatches to target implementations.
/// @ingroup scale_internal
///
template <Target target, typename scalar_t>
void scale(
    blas::real_type<scalar_t> numer, blas::real_type<scalar_t> denom,
    BaseTrapezoidMatrix<scalar_t>&& A,
    int priority, int queue_index)
{
    scale(internal::TargetType<target>(),
          numer, denom, A, priority, queue_index);
}

//------------------------------------------------------------------------------
/// Scale Trapezoid matrix entries by the real scalar numer/denom.
/// TODO handle transpose A case
/// Host OpenMP task implementation.
/// @ingroup scale_internal
///
template <typename scalar_t>
void scale(
    internal::TargetType<Target::HostTask>,
    blas::real_type<scalar_t> numer, blas::real_type<scalar_t> denom,
    BaseTrapezoidMatrix<scalar_t>& A,
    int priority, int queue_index)
{
    // trace::Block trace_block("scale");
    #pragma omp taskgroup
    if (A.uplo() == Uplo::Lower) {
        for (int64_t j = 0; j < A.nt(); ++j) {
            for (int64_t i = j; i < A.mt(); ++i) { // lower trapezoid
                if (A.tileIsLocal(i, j)) {
                    #pragma omp task slate_omp_default_none \
                        shared( A ) priority( priority ) \
                        firstprivate(i, j, numer, denom)
                    {
                        A.tileGetForWriting(i, j, LayoutConvert::None);
                        scale(numer, denom, A(i, j));
                    }
                }
            }
        }
    }
    else { // upper
        for (int64_t j = 0; j < A.nt(); ++j) {
            for (int64_t i = 0; i <= j && i < A.mt(); ++i) { // upper trapezoid
                if (A.tileIsLocal(i, j)) {
                    #pragma omp task slate_omp_default_none \
                        shared( A ) priority( priority ) \
                        firstprivate(i, j, numer, denom)
                    {
                        A.tileGetForWriting(i, j, LayoutConvert::None);
                        scale(numer, denom, A(i, j));
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void scale(internal::TargetType<Target::HostNest>,
           blas::real_type<scalar_t> numer, blas::real_type<scalar_t> denom,
           BaseTrapezoidMatrix<scalar_t>& A,
           int priority, int queue_index)
{
    slate_not_implemented("Target::HostNest isn't yet supported.");
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void scale(internal::TargetType<Target::HostBatch>,
           blas::real_type<scalar_t> numer, blas::real_type<scalar_t> denom,
           BaseTrapezoidMatrix<scalar_t>& A,
           int priority, int queue_index)
{
    slate_not_implemented("Target::HostBatch isn't yet supported.");
}

//------------------------------------------------------------------------------
/// Scale Trapezoid matrix entries by the real scalar numer/denom.
/// TODO handle transpose A case
/// GPU device implementation.
/// @ingroup scale_internal
///
template <typename scalar_t>
void scale(internal::TargetType<Target::Devices>,
           blas::real_type<scalar_t> numer, blas::real_type<scalar_t> denom,
           BaseTrapezoidMatrix<scalar_t>& A,
           int priority, int queue_index)
{
    using ij_tuple = typename BaseTrapezoidMatrix<scalar_t>::ij_tuple;

    #pragma omp taskgroup
    for (int device = 0; device < A.num_devices(); ++device) {
        #pragma omp task slate_omp_default_none priority( priority ) \
            shared( A ) firstprivate( device, queue_index, numer, denom )
        {
            // temporarily, convert both into same layout
            // todo: this is in-efficient, because both matrices may have same layout already
            //       and possibly wrong, because an input matrix is being altered
            // todo: best, handle directly through the CUDA kernels
            auto layout = LayoutConvert::ColMajor;
            std::set<ij_tuple> A_tiles_set;

            if (A.uplo() == Uplo::Lower) {
                for (int64_t j = 0; j < A.nt(); ++j) {
                    for (int64_t i = j; i < A.mt(); ++i) { // lower trapezoid
                        if (A.tileIsLocal(i, j) && device == A.tileDevice(i, j)) {
                            A_tiles_set.insert({i, j});
                        }
                    }
                }
            }
            else {  // upper
                for (int64_t j = 0; j < A.nt(); ++j) {
                    for (int64_t i = 0; i <= j && i < A.mt(); ++i) { // upper trapezoid
                        if (A.tileIsLocal(i, j) && device == A.tileDevice(i, j)) {
                            A_tiles_set.insert({i, j});
                        }
                    }
                }
            }
            A.tileGetForWriting( A_tiles_set, device, layout );

            int64_t batch_size = A_tiles_set.size();
            scalar_t** a_array_host = A.array_host( device, queue_index );

            auto group_params = device_regions_build<1, scalar_t>(
                                                    {A},
                                                    {a_array_host},
                                                    device );

            blas::Queue* queue = A.compute_queue( device, queue_index );

            scalar_t** a_array_dev = A.array_device( device, queue_index );
            blas::device_memcpy<scalar_t*>(
                a_array_dev, a_array_host, batch_size,
                blas::MemcpyKind::HostToDevice, *queue);

            for (size_t g = 0; g < group_params.size(); ++g) {
                int64_t group_count = group_params[ g ].count;

                if (group_params[ g ].is_diagonal) {
                    device::batch::tzscale(
                            A.uplo(),
                            group_params[ g ].mb, group_params[ g ].nb,
                            numer, denom, a_array_dev, group_params[ g ].ld[0],
                            group_count, *queue);
                }
                else {
                    device::batch::gescale(
                            group_params[ g ].mb, group_params[ g ].nb,
                            numer, denom, a_array_dev, group_params[ g ].ld[0],
                            group_count, *queue);
                }
                a_array_dev += group_count;
            }
            queue->sync();
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void scale<Target::HostTask, float>(
    float numer, float denom, BaseTrapezoidMatrix<float>&& A,
    int priority, int queue_index);

template
void scale<Target::HostNest, float>(
    float numer, float denom, BaseTrapezoidMatrix<float>&& A,
    int priority, int queue_index);

template
void scale<Target::HostBatch, float>(
    float numer, float denom, BaseTrapezoidMatrix<float>&& A,
    int priority, int queue_index);

template
void scale<Target::Devices, float>(
    float numer, float denom, BaseTrapezoidMatrix<float>&& A,
    int priority, int queue_index);

// ----------------------------------------
template
void scale<Target::HostTask, double>(
    double numer, double denom, BaseTrapezoidMatrix<double>&& A,
    int priority, int queue_index);

template
void scale<Target::HostNest, double>(
    double numer, double denom, BaseTrapezoidMatrix<double>&& A,
    int priority, int queue_index);

template
void scale<Target::HostBatch, double>(
    double numer, double denom, BaseTrapezoidMatrix<double>&& A,
    int priority, int queue_index);

template
void scale<Target::Devices, double>(
    double numer, double denom, BaseTrapezoidMatrix<double>&& A,
    int priority, int queue_index);

// ----------------------------------------
template
void scale< Target::HostTask, std::complex<float> >(
    float numer, float denom,
    BaseTrapezoidMatrix< std::complex<float> >&& A,
    int priority, int queue_index);

template
void scale< Target::HostNest, std::complex<float> >(
    float numer, float denom,
    BaseTrapezoidMatrix< std::complex<float> >&& A,
    int priority, int queue_index);

template
void scale< Target::HostBatch, std::complex<float> >(
    float numer, float denom,
    BaseTrapezoidMatrix< std::complex<float> >&& A,
    int priority, int queue_index);

template
void scale< Target::Devices, std::complex<float> >(
    float numer, float denom,
    BaseTrapezoidMatrix< std::complex<float> >&& A,
    int priority, int queue_index);

// ----------------------------------------
template
void scale< Target::HostTask, std::complex<double> >(
    double numer, double denom,
    BaseTrapezoidMatrix< std::complex<double> >&& A,
    int priority, int queue_index);

template
void scale< Target::HostNest, std::complex<double> >(
    double numer, double denom,
    BaseTrapezoidMatrix< std::complex<double> >&& A,
    int priority, int queue_index);

template
void scale< Target::HostBatch, std::complex<double> >(
    double numer, double denom,
    BaseTrapezoidMatrix< std::complex<double> >&& A,
    int priority, int queue_index);

template
void scale< Target::Devices, std::complex<double> >(
    double numer, double denom,
    BaseTrapezoidMatrix< std::complex<double> >&& A,
    int priority, int queue_index);

} // namespace internal
} // namespace slate
