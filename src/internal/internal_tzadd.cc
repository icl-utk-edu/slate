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

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Trapezoidal matrix add.
/// Dispatches to target implementations.
/// @ingroup add_internal
template <Target target, typename scalar_t>
void add(scalar_t alpha, BaseTrapezoidMatrix<scalar_t>&& A,
         scalar_t beta, BaseTrapezoidMatrix<scalar_t>&& B,
         int priority, int queue_index )
{
    add(internal::TargetType<target>(),
        alpha, A,
        beta,  B,
        priority, queue_index );
}

//------------------------------------------------------------------------------
/// Trapezoidal matrix add.
/// assumes A & B have same tile layout and dimensions, and have same distribution
/// TODO handle transpose A case
/// Host OpenMP task implementation.
/// @ingroup add_internal
template <typename scalar_t>
void add(internal::TargetType<Target::HostTask>,
           scalar_t alpha, BaseTrapezoidMatrix<scalar_t>& A,
           scalar_t beta, BaseTrapezoidMatrix<scalar_t>& B,
           int priority, int queue_index )
{
    // trace::Block trace_block("add");

    int64_t A_mt = A.mt();
    int64_t A_nt = A.nt();
    assert(A_mt == B.mt());
    assert(A_nt == B.nt());
    slate_error_if(A.uplo() != B.uplo());

    #pragma omp taskgroup
    if (B.uplo() == Uplo::Lower) {
        for (int64_t j = 0; j < A_nt; ++j) {
            for (int64_t i = j; i < A_mt; ++i) {
                if (B.tileIsLocal(i, j)) {
                    #pragma omp task slate_omp_default_none \
                        shared( A, B ) \
                        firstprivate( i, j, alpha, beta ) priority( priority )
                    {
                        A.tileGetForReading(i, j, LayoutConvert::None);
                        B.tileGetForWriting(i, j, LayoutConvert::None);
                        tile::add(
                            alpha, A(i, j),
                            beta,  B(i, j) );
                    }
                }
            }
        }
    }
    else { // upper
        for (int64_t j = 0; j < A.nt(); ++j) {
            for (int64_t i = 0; i <= j && i < A.mt(); ++i) {
                if (A.tileIsLocal(i, j)) {
                    #pragma omp task slate_omp_default_none \
                        shared( A, B ) \
                        firstprivate( i, j, alpha, beta ) priority( priority )
                    {
                        A.tileGetForReading(i, j, LayoutConvert::None);
                        B.tileGetForWriting(i, j, LayoutConvert::None);
                        tile::add(
                            alpha, A(i, j),
                            beta,  B(i, j) );
                    }
                }
            }
        }
    }
    // end omp taskgroup
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void add(internal::TargetType<Target::HostNest>,
           scalar_t alpha, BaseTrapezoidMatrix<scalar_t>& A,
           scalar_t beta, BaseTrapezoidMatrix<scalar_t>& B,
           int priority, int queue_index )
{
    slate_not_implemented("Target::HostNest isn't yet supported.");
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void add(internal::TargetType<Target::HostBatch>,
           scalar_t alpha, BaseTrapezoidMatrix<scalar_t>& A,
           scalar_t beta, BaseTrapezoidMatrix<scalar_t>& B,
           int priority, int queue_index )
{
    slate_not_implemented("Target::HostBatch isn't yet supported.");
}

//------------------------------------------------------------------------------
/// Trapezoidal matrix add.
/// assumes A & B have same tile layout and dimensions, and have same distribution
/// TODO handle transpose A case
/// GPU device implementation.
/// @ingroup add_internal
template <typename scalar_t>
void add(internal::TargetType<Target::Devices>,
           scalar_t alpha, BaseTrapezoidMatrix<scalar_t>& A,
           scalar_t beta, BaseTrapezoidMatrix<scalar_t>& B,
           int priority, int queue_index )
{
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;
    slate_error_if(A.uplo() != B.uplo());

    #pragma omp taskgroup
    for (int device = 0; device < B.num_devices(); ++device) {
        #pragma omp task slate_omp_default_none priority( priority ) \
            shared( A, B ) firstprivate( device, queue_index, alpha, beta )
        {
            // temporarily, convert both into same layout
            // todo: this is in-efficient, because both matrices may have same layout already
            //       and possibly wrong, because an input matrix is being altered
            // todo: best, handle directly through the CUDA kernels
            auto layout = Layout::ColMajor;
            std::set<ij_tuple> A_tiles_set, B_tiles_set;

            if (B.uplo() == Uplo::Lower) {
                for (int64_t j = 0; j < B.nt(); ++j) {
                    for (int64_t i = j; i < B.mt(); ++i) {
                        if (B.tileIsLocal(i, j) && device == B.tileDevice(i, j)) {
                            A_tiles_set.insert({i, j});
                            B_tiles_set.insert({i, j});
                        }
                    }
                }
            }
            else { // upper
                for (int64_t j = 0; j < B.nt(); ++j) {
                    for (int64_t i = 0; i <= j && i < B.mt(); ++i) {
                        if (B.tileIsLocal(i, j) && device == B.tileDevice(i, j)) {
                            A_tiles_set.insert({i, j});
                            B_tiles_set.insert({i, j});
                        }
                    }
                }
            }

            #pragma omp taskgroup
            {
                #pragma omp task slate_omp_default_none \
                    shared( A, A_tiles_set ) firstprivate( device, layout )
                {
                    A.tileGetForReading(A_tiles_set, device, LayoutConvert(layout));
                }
                #pragma omp task slate_omp_default_none \
                    shared( B, B_tiles_set ) firstprivate( device, layout )
                {
                    B.tileGetForWriting(B_tiles_set, device, LayoutConvert(layout));
                }
            }

            int64_t batch_size = A_tiles_set.size();
            scalar_t** a_array_host = B.array_host( device, queue_index );
            scalar_t** b_array_host = a_array_host + batch_size;

            auto group_params = device_regions_build<true, 2, scalar_t>(
                    {A, B},
                    {a_array_host, b_array_host},
                    device );

            scalar_t** a_array_dev = B.array_device( device, queue_index );
            scalar_t** b_array_dev = a_array_dev + batch_size;

            blas::Queue* queue = A.compute_queue(device, queue_index);

            blas::device_memcpy<scalar_t*>(
                a_array_dev, a_array_host, batch_size*2, *queue );

            for (size_t g = 0; g < group_params.size(); ++g) {
                int64_t group_count = group_params[ g ].count;
                if (group_params[ g ].is_diagonal) {
                    device::tzadd(
                            B.uplo(),
                            group_params[ g ].mb, group_params[ g ].nb,
                            alpha, a_array_dev, group_params[ g ].ld[0],
                            beta, b_array_dev, group_params[ g ].ld[1],
                            group_count, *queue);
                }
                else {
                    device::batch::geadd(
                            group_params[ g ].mb, group_params[ g ].nb,
                            alpha, a_array_dev, group_params[ g ].ld[0],
                            beta, b_array_dev, group_params[ g ].ld[1],
                            group_count, *queue);
                }
                a_array_dev += group_count;
                b_array_dev += group_count;
            }

            queue->sync();
        }
    }
    // end omp taskgroup
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void add<Target::HostTask, float>(
     float alpha, BaseTrapezoidMatrix<float>&& A,
     float beta, BaseTrapezoidMatrix<float>&& B,
     int priority, int queue_index );

template
void add<Target::HostNest, float>(
     float alpha, BaseTrapezoidMatrix<float>&& A,
     float beta, BaseTrapezoidMatrix<float>&& B,
     int priority, int queue_index );

template
void add<Target::HostBatch, float>(
     float alpha, BaseTrapezoidMatrix<float>&& A,
     float beta, BaseTrapezoidMatrix<float>&& B,
     int priority, int queue_index );

template
void add<Target::Devices, float>(
     float alpha, BaseTrapezoidMatrix<float>&& A,
     float beta, BaseTrapezoidMatrix<float>&& B,
     int priority, int queue_index );

// ----------------------------------------
template
void add<Target::HostTask, double>(
     double alpha, BaseTrapezoidMatrix<double>&& A,
     double beta, BaseTrapezoidMatrix<double>&& B,
     int priority, int queue_index );

template
void add<Target::HostNest, double>(
     double alpha, BaseTrapezoidMatrix<double>&& A,
     double beta, BaseTrapezoidMatrix<double>&& B,
     int priority, int queue_index );

template
void add<Target::HostBatch, double>(
     double alpha, BaseTrapezoidMatrix<double>&& A,
     double beta, BaseTrapezoidMatrix<double>&& B,
     int priority, int queue_index );

template
void add<Target::Devices, double>(
     double alpha, BaseTrapezoidMatrix<double>&& A,
     double beta, BaseTrapezoidMatrix<double>&& B,
     int priority, int queue_index );

// ----------------------------------------
template
void add< Target::HostTask, std::complex<float> >(
     std::complex<float> alpha, BaseTrapezoidMatrix< std::complex<float> >&& A,
     std::complex<float>  beta, BaseTrapezoidMatrix< std::complex<float> >&& B,
     int priority, int queue_index );

template
void add< Target::HostNest, std::complex<float> >(
     std::complex<float> alpha, BaseTrapezoidMatrix< std::complex<float> >&& A,
     std::complex<float>  beta, BaseTrapezoidMatrix< std::complex<float> >&& B,
     int priority, int queue_index );

template
void add< Target::HostBatch, std::complex<float> >(
     std::complex<float> alpha, BaseTrapezoidMatrix< std::complex<float> >&& A,
     std::complex<float>  beta, BaseTrapezoidMatrix< std::complex<float> >&& B,
     int priority, int queue_index );

template
void add< Target::Devices, std::complex<float> >(
     std::complex<float> alpha, BaseTrapezoidMatrix< std::complex<float> >&& A,
     std::complex<float>  beta, BaseTrapezoidMatrix< std::complex<float> >&& B,
     int priority, int queue_index );

// ----------------------------------------
template
void add< Target::HostTask, std::complex<double> >(
     std::complex<double> alpha, BaseTrapezoidMatrix< std::complex<double> >&& A,
     std::complex<double> beta, BaseTrapezoidMatrix< std::complex<double> >&& B,
     int priority, int queue_index );

template
void add< Target::HostNest, std::complex<double> >(
     std::complex<double> alpha, BaseTrapezoidMatrix< std::complex<double> >&& A,
     std::complex<double> beta, BaseTrapezoidMatrix< std::complex<double> >&& B,
     int priority, int queue_index );

template
void add< Target::HostBatch, std::complex<double> >(
     std::complex<double> alpha, BaseTrapezoidMatrix< std::complex<double> >&& A,
     std::complex<double> beta, BaseTrapezoidMatrix< std::complex<double> >&& B,
     int priority, int queue_index );

template
void add< Target::Devices, std::complex<double> >(
     std::complex<double> alpha, BaseTrapezoidMatrix< std::complex<double> >&& A,
     std::complex<double> beta, BaseTrapezoidMatrix< std::complex<double> >&& B,
     int priority, int queue_index );

} // namespace internal
} // namespace slate
