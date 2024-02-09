// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/internal/device.hh"
#include "internal/internal_batch.hh"
#include "internal/internal.hh"
#include "slate/internal/util.hh"
#include "slate/Matrix.hh"
#include "slate/types.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// General matrix set.
/// Dispatches to target implementations.
/// @ingroup set_internal
///
template <Target target, typename scalar_t>
void set(
    scalar_t offdiag_value, scalar_t diag_value, Matrix<scalar_t>&& A,
    int priority, int queue_index)
{
    set(internal::TargetType<target>(),
        offdiag_value, diag_value, A, priority, queue_index);
}

//------------------------------------------------------------------------------
/// General matrix set.
/// TODO handle transpose A case
/// Host OpenMP task implementation.
/// @ingroup set_internal
///
template <typename scalar_t>
void set(
    internal::TargetType<Target::HostTask>,
    scalar_t offdiag_value, scalar_t diag_value, Matrix<scalar_t>& A,
    int priority, int queue_index)
{
    // trace::Block trace_block("set");
    #pragma omp taskgroup
    for (int64_t i = 0; i < A.mt(); ++i) {
        for (int64_t j = 0; j < A.nt(); ++j) {
            if (A.tileIsLocal(i, j)) {
                #pragma omp task slate_omp_default_none \
                    shared( A ) \
                    firstprivate(i, j, offdiag_value, diag_value) priority(priority)
                {
                    A.tileGetForWriting(i, j, LayoutConvert::None);
                    if (i == j)
                        A.at(i, j).set( offdiag_value, diag_value );
                    else
                        A.at(i, j).set( offdiag_value, offdiag_value );
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void set(internal::TargetType<Target::HostNest>,
         scalar_t offdiag_value, scalar_t diag_value, Matrix<scalar_t>& A,
         int priority, int queue_index)
{
    slate_not_implemented("Target::HostNest isn't yet supported.");
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void set(internal::TargetType<Target::HostBatch>,
         scalar_t offdiag_value, scalar_t diag_value, Matrix<scalar_t>& A,
         int priority, int queue_index)
{
    slate_not_implemented("Target::HostBatch isn't yet supported.");
}

//------------------------------------------------------------------------------
/// General matrix set.
/// TODO handle transpose A case
/// GPU device implementation.
/// @ingroup set_internal
///
template <typename scalar_t>
void set(internal::TargetType<Target::Devices>,
         scalar_t offdiag_value, scalar_t diag_value,
         Matrix<scalar_t>& A,
         int priority, int queue_index)
{
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    #pragma omp taskgroup
    for (int device = 0; device < A.num_devices(); ++device) {
        #pragma omp task slate_omp_default_none priority( priority ) \
            shared( A ) firstprivate( device, queue_index, offdiag_value, diag_value )
        {
            // Get local tiles for writing.
            // convert to column major layout to simplify lda's
            // todo: this is in-efficient because the diagonal is independant of layout
            // todo: best, handle directly through the CUDA kernels
            auto layout = LayoutConvert::ColMajor;
            std::set<ij_tuple> A_tiles_set;

            for (int64_t i = 0; i < A.mt(); ++i) {
                for (int64_t j = 0; j < A.nt(); ++j) {
                    if (A.tileIsLocal( i, j ) && device == A.tileDevice( i, j )) {
                        A_tiles_set.insert( { i, j } );
                    }
                }
            }
            A.tileGetForWriting( A_tiles_set, device, layout );

            int64_t batch_size = A_tiles_set.size();
            scalar_t** a_array_host = A.array_host( device, queue_index );

            // If offdiag == diag value, lump diag tiles with offdiag tiles
            // in one batch.
            bool diag_same = offdiag_value == diag_value;

            auto group_params = diag_same
                                   ? device_regions_build<true, 1, scalar_t, true>(
                                                {A}, {a_array_host}, device )
                                   : device_regions_build<true, 1, scalar_t, false>(
                                                {A}, {a_array_host}, device );
            blas::Queue* queue = A.compute_queue( device, queue_index );

            scalar_t** a_array_dev = A.array_device( device, queue_index );
            blas::device_memcpy<scalar_t*>(
                a_array_dev, a_array_host, batch_size, *queue );

            for (size_t g = 0; g < group_params.size(); ++g) {
                int64_t group_count = group_params[ g ].count;
                if (group_params[ g ].is_diagonal) {
                    device::batch::geset(
                        group_params[ g ].mb, group_params[ g ].nb,
                        offdiag_value, diag_value,
                        a_array_dev, group_params[ g ].ld[0],
                        group_count, *queue );
                }
                else {
                    device::batch::geset(
                        group_params[ g ].mb, group_params[ g ].nb,
                        offdiag_value, offdiag_value,
                        a_array_dev, group_params[ g ].ld[0],
                        group_count, *queue );
                }
                a_array_dev += group_count;
            }
            queue->sync();
        } // end task
    } // end for dev
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void set<Target::HostTask, float>(
    float offdiag_value, float diag_value,
    Matrix<float>&& A,
    int priority, int queue_index);

template
void set<Target::HostNest, float>(
    float offdiag_value, float diag_value,
    Matrix<float>&& A,
    int priority, int queue_index);

template
void set<Target::HostBatch, float>(
    float offdiag_value, float diag_value,
    Matrix<float>&& A,
    int priority, int queue_index);

template
void set<Target::Devices, float>(
    float offdiag_value, float diag_value,
    Matrix<float>&& A,
    int priority, int queue_index);

// ----------------------------------------
template
void set<Target::HostTask, double>(
    double offdiag_value, double diag_value,
    Matrix<double>&& A,
    int priority, int queue_index);

template
void set<Target::HostNest, double>(
    double offdiag_value, double diag_value,
    Matrix<double>&& A,
    int priority, int queue_index);

template
void set<Target::HostBatch, double>(
    double offdiag_value, double diag_value,
    Matrix<double>&& A,
    int priority, int queue_index);

template
void set<Target::Devices, double>(
    double offdiag_value, double diag_value,
    Matrix<double>&& A,
    int priority, int queue_index);

// ----------------------------------------
template
void set< Target::HostTask, std::complex<float> >(
    std::complex<float> offdiag_value, std::complex<float>  diag_value,
    Matrix< std::complex<float> >&& A,
    int priority, int queue_index);

template
void set< Target::HostNest, std::complex<float> >(
    std::complex<float> offdiag_value, std::complex<float>  diag_value,
    Matrix< std::complex<float> >&& A,
    int priority, int queue_index);

template
void set< Target::HostBatch, std::complex<float> >(
    std::complex<float> offdiag_value, std::complex<float>  diag_value,
    Matrix< std::complex<float> >&& A,
    int priority, int queue_index);

template
void set< Target::Devices, std::complex<float> >(
    std::complex<float> offdiag_value, std::complex<float>  diag_value,
    Matrix< std::complex<float> >&& A,
    int priority, int queue_index);

// ----------------------------------------
template
void set< Target::HostTask, std::complex<double> >(
    std::complex<double> offdiag_value, std::complex<double> diag_value,
    Matrix< std::complex<double> >&& A,
    int priority, int queue_index);

template
void set< Target::HostNest, std::complex<double> >(
    std::complex<double> offdiag_value, std::complex<double> diag_value,
    Matrix< std::complex<double> >&& A,
    int priority, int queue_index);

template
void set< Target::HostBatch, std::complex<double> >(
    std::complex<double> offdiag_value, std::complex<double> diag_value,
    Matrix< std::complex<double> >&& A,
    int priority, int queue_index);

template
void set< Target::Devices, std::complex<double> >(
    std::complex<double> offdiag_value, std::complex<double> diag_value,
    Matrix< std::complex<double> >&& A,
    int priority, int queue_index);

} // namespace internal
} // namespace slate
