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
/// General matrix add.
/// Dispatches to target implementations.
/// @ingroup add_internal
///
/// todo: this function should just be named "add".
template <Target target, typename scalar_t>
void add(scalar_t alpha, Matrix<scalar_t>&& A,
         scalar_t beta, Matrix<scalar_t>&& B,
         int priority, int queue_index, Options const& opts)
{
    add(internal::TargetType<target>(),
        alpha, A,
        beta,  B,
        priority, queue_index, opts);
}

//------------------------------------------------------------------------------
/// General matrix add.
/// assumes A & B have same tile layout and dimensions, and have same distribution
/// TODO handle transpose A case
/// Host OpenMP task implementation.
/// @ingroup add_internal
///
/// todo: this function should just be named "add".
template <typename scalar_t>
void add(internal::TargetType<Target::HostTask>,
         scalar_t alpha, Matrix<scalar_t>& A,
         scalar_t beta, Matrix<scalar_t>& B,
         int priority, int queue_index, Options const& opts)
{
    // trace::Block trace_block("add");

    int64_t A_mt = A.mt();
    int64_t A_nt = A.nt();
    assert(A_mt == B.mt());
    assert(A_nt == B.nt());

    #pragma omp taskgroup
    for (int64_t i = 0; i < A_mt; ++i) {
        for (int64_t j = 0; j < A_nt; ++j) {
            if (B.tileIsLocal(i, j)) {
                #pragma omp task slate_omp_default_none \
                    shared( A, B ) \
                    firstprivate( i, j, alpha, beta )  priority(priority)
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

//------------------------------------------------------------------------------
/// todo: this function should just be named "add".
template <typename scalar_t>
void add(internal::TargetType<Target::HostNest>,
         scalar_t alpha, Matrix<scalar_t>& A,
         scalar_t beta, Matrix<scalar_t>& B,
         int priority, int queue_index, Options const& opts)
{
    slate_not_implemented("Target::HostNest isn't yet supported.");
}

//------------------------------------------------------------------------------
/// todo: this function should just be named "add".
template <typename scalar_t>
void add(internal::TargetType<Target::HostBatch>,
         scalar_t alpha, Matrix<scalar_t>& A,
         scalar_t beta, Matrix<scalar_t>& B,
         int priority, int queue_index, Options const& opts)
{
    slate_not_implemented("Target::HostBatch isn't yet supported.");
}

//------------------------------------------------------------------------------
/// General matrix add.
/// assumes A & B have same tile layout and dimensions, and have same distribution
/// TODO handle transpose A case
/// GPU device implementation.
/// @ingroup add_internal
///
/// todo: this function should just be named "add".
template <typename scalar_t>
void add(internal::TargetType<Target::Devices>,
         scalar_t alpha, Matrix<scalar_t>& A,
         scalar_t beta, Matrix<scalar_t>& B,
         int priority, int queue_index, Options const& opts)
{
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    #pragma omp taskgroup
    for (int device = 0; device < B.num_devices(); ++device) {
        #pragma omp task slate_omp_default_none priority( priority ) \
            shared( A, B ) \
            firstprivate( device, queue_index, beta, alpha )
        {
            // temporarily, convert both into same layout
            // todo: this is in-efficient, because both matrices may have same layout already
            //       and possibly wrong, because an input matrix is being altered
            // todo: best, handle directly through the CUDA kernels
            auto layout = Layout::ColMajor;
            std::set<ij_tuple> A_tiles_set, B_tiles_set;

            for (int64_t i = 0; i < B.mt(); ++i) {
                for (int64_t j = 0; j < B.nt(); ++j) {
                    if (B.tileIsLocal(i, j) && device == B.tileDevice(i, j)) {
                        A_tiles_set.insert({i, j});
                        B_tiles_set.insert({i, j});
                    }
                }
            }
            #pragma omp taskgroup
            {
                #pragma omp task slate_omp_default_none \
                    shared( A, A_tiles_set ) \
                    firstprivate(device, layout)
                {
                    A.tileGetForReading(A_tiles_set, device, LayoutConvert(layout));
                }
                #pragma omp task slate_omp_default_none \
                    shared( B, B_tiles_set ) \
                    firstprivate(device, layout)
                {
                    B.tileGetForWriting(B_tiles_set, device, LayoutConvert(layout));
                }
            }

            int64_t batch_size = A_tiles_set.size();
            scalar_t** a_array_host = B.array_host(device, queue_index);
            scalar_t** b_array_host = a_array_host + batch_size;

            auto group_params = device_regions_build<false, 2, scalar_t>(
                    {A, B},
                    {a_array_host, b_array_host},
                    device );

            scalar_t** a_array_dev = B.array_device(device, queue_index);
            scalar_t** b_array_dev = a_array_dev + batch_size;

            blas::Queue* queue = B.compute_queue(device, queue_index);

            blas::device_memcpy<scalar_t*>(a_array_dev, a_array_host,
                                batch_size*2,
                                blas::MemcpyKind::HostToDevice,
                                *queue);

            for (size_t g = 0; g < group_params.size(); ++g) {
                int64_t group_count = group_params[ g ].count;
                device::batch::geadd(
                        group_params[ g ].mb, group_params[ g ].nb,
                        alpha, a_array_dev, group_params[ g ].ld[0],
                        beta, b_array_dev, group_params[ g ].ld[1],
                        group_count, *queue);
                a_array_dev += group_count;
                b_array_dev += group_count;
            }

            queue->sync();
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
/// todo: these functions should just be named "add".
// ----------------------------------------
template
void add<Target::HostTask, float>(
     float alpha, Matrix<float>&& A,
     float beta, Matrix<float>&& B,
     int priority, int queue_index, Options const& opts);

template
void add<Target::HostNest, float>(
     float alpha, Matrix<float>&& A,
     float beta, Matrix<float>&& B,
     int priority, int queue_index, Options const& opts);

template
void add<Target::HostBatch, float>(
     float alpha, Matrix<float>&& A,
     float beta, Matrix<float>&& B,
     int priority, int queue_index, Options const& opts);

template
void add<Target::Devices, float>(
     float alpha, Matrix<float>&& A,
     float beta, Matrix<float>&& B,
     int priority, int queue_index, Options const& opts);

// ----------------------------------------
template
void add<Target::HostTask, double>(
     double alpha, Matrix<double>&& A,
     double beta, Matrix<double>&& B,
     int priority, int queue_index, Options const& opts);

template
void add<Target::HostNest, double>(
     double alpha, Matrix<double>&& A,
     double beta, Matrix<double>&& B,
     int priority, int queue_index, Options const& opts);

template
void add<Target::HostBatch, double>(
     double alpha, Matrix<double>&& A,
     double beta, Matrix<double>&& B,
     int priority, int queue_index, Options const& opts);

template
void add<Target::Devices, double>(
     double alpha, Matrix<double>&& A,
     double beta, Matrix<double>&& B,
     int priority, int queue_index, Options const& opts);

// ----------------------------------------
template
void add< Target::HostTask, std::complex<float> >(
     std::complex<float> alpha, Matrix< std::complex<float> >&& A,
     std::complex<float>  beta, Matrix< std::complex<float> >&& B,
     int priority, int queue_index, Options const& opts);

template
void add< Target::HostNest, std::complex<float> >(
     std::complex<float> alpha, Matrix< std::complex<float> >&& A,
     std::complex<float>  beta, Matrix< std::complex<float> >&& B,
     int priority, int queue_index, Options const& opts);

template
void add< Target::HostBatch, std::complex<float> >(
     std::complex<float> alpha, Matrix< std::complex<float> >&& A,
     std::complex<float>  beta, Matrix< std::complex<float> >&& B,
     int priority, int queue_index, Options const& opts);

template
void add< Target::Devices, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
    std::complex<float>  beta, Matrix< std::complex<float> >&& B,
    int priority, int queue_index, Options const& opts);

// ----------------------------------------
template
void add< Target::HostTask, std::complex<double> >(
     std::complex<double> alpha, Matrix< std::complex<double> >&& A,
     std::complex<double> beta, Matrix< std::complex<double> >&& B,
     int priority, int queue_index, Options const& opts);

template
void add< Target::HostNest, std::complex<double> >(
     std::complex<double> alpha, Matrix< std::complex<double> >&& A,
     std::complex<double> beta, Matrix< std::complex<double> >&& B,
     int priority, int queue_index, Options const& opts);

template
void add< Target::HostBatch, std::complex<double> >(
     std::complex<double> alpha, Matrix< std::complex<double> >&& A,
     std::complex<double> beta, Matrix< std::complex<double> >&& B,
     int priority, int queue_index, Options const& opts);

template
void add< Target::Devices, std::complex<double> >(
     std::complex<double> alpha, Matrix< std::complex<double> >&& A,
     std::complex<double> beta, Matrix< std::complex<double> >&& B,
     int priority, int queue_index, Options const& opts);

} // namespace internal
} // namespace slate
