// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/internal/device.hh"
#include "internal/internal_batch.hh"
#include "internal/internal.hh"
#include "slate/internal/util.hh"
#include "slate/Matrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/Tile_aux.hh"
#include "slate/types.hh"

namespace slate {

namespace device {

//------------------------------------------------------------------------------
// Overload for non-matching types, which throws an error.
//
template <typename x_scalar_t, typename y_scalar_t>
void transpose_batch(
    bool is_conj,
    int64_t m, int64_t n,
    x_scalar_t **dA_array,  int64_t lda,
    y_scalar_t **dAT_array, int64_t ldat,
    int64_t batch_count,
    blas::Queue& queue)
{
    using std::real;
    throw std::exception();  // not implemented
}

}  // namespace device

namespace internal {

//------------------------------------------------------------------------------
/// Copy and precision conversion.
/// Dispatches to target implementations.
/// @ingroup copy_internal
///
template <Target target, typename src_scalar_t, typename dst_scalar_t>
void copy( Matrix<src_scalar_t>&& A,
           Matrix<dst_scalar_t>&& B,
           int priority, int queue_index,
           Options const& opts )
{
    copy(internal::TargetType<target>(),
         A, B,
         priority, queue_index, opts);
}

//------------------------------------------------------------------------------
/// Copy and precision conversion.
/// assumes A & B have same tile layout and dimensions, and have same distribution
/// TODO handle transpose A case
/// Host OpenMP task implementation.
/// @ingroup copy_internal
///
template <typename src_scalar_t, typename dst_scalar_t>
void copy(internal::TargetType<Target::HostTask>,
          Matrix<src_scalar_t>& A,
          Matrix<dst_scalar_t>& B,
          int priority, int queue_index,
          Options const& opts )
{
    // trace::Block trace_block("copy");

    int64_t A_mt = A.mt();
    int64_t A_nt = A.nt();
    assert(A_mt == B.mt());
    assert(A_nt == B.nt());

    TileReleaseStrategy tile_release_strategy = get_option(
            opts, Option::TileReleaseStrategy, TileReleaseStrategy::All );

    bool call_tile_tick = tile_release_strategy == TileReleaseStrategy::Internal
                          || tile_release_strategy == TileReleaseStrategy::All;

    #pragma omp taskgroup
    for (int64_t i = 0; i < A_mt; ++i) {
        for (int64_t j = 0; j < A_nt; ++j) {
            if (B.tileIsLocal(i, j)) {
                #pragma omp task slate_omp_default_none \
                    shared( A, B ) firstprivate( i, j, HostNum, call_tile_tick ) \
                    priority(priority)
                {
                    A.tileGetForReading(i, j, LayoutConvert::None);
                    // tileAcquire() to avoid un-needed copy
                    B.tileAcquire(i, j, A.tileLayout(i, j));
                    B.tileModified(i, j, HostNum, true);
                    tile::gecopy( A(i, j), B(i, j) );
                    if (call_tile_tick) {
                        A.tileTick(i, j);
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
template <typename src_scalar_t, typename dst_scalar_t>
void copy(internal::TargetType<Target::HostNest>,
          Matrix<src_scalar_t>& A,
          Matrix<dst_scalar_t>& B,
          int priority, int queue_index,
          Options const& opts )
{
    slate_not_implemented("Target::HostNest isn't yet supported.");
}

//------------------------------------------------------------------------------
template <typename src_scalar_t, typename dst_scalar_t>
void copy(internal::TargetType<Target::HostBatch>,
          Matrix<src_scalar_t>& A,
          Matrix<dst_scalar_t>& B,
          int priority, int queue_index,
          Options const& opts )
{
    slate_not_implemented("Target::HostBatch isn't yet supported.");
}

//------------------------------------------------------------------------------
/// Copy and precision conversion.
/// Assumes A & B have same tile layout, dimensions, and distribution.
/// TODO: Inspect transposition?
/// GPU device implementation.
/// @ingroup copy_internal
///
template <typename src_scalar_t, typename dst_scalar_t>
void copy(internal::TargetType<Target::Devices>,
          Matrix<src_scalar_t>& A,
          Matrix<dst_scalar_t>& B,
          int priority, int queue_index,
          Options const& opts )
{
    using ij_tuple = typename BaseMatrix<src_scalar_t>::ij_tuple;

    TileReleaseStrategy tile_release_strategy = get_option(
            opts, Option::TileReleaseStrategy, TileReleaseStrategy::All );

    bool call_tile_tick = tile_release_strategy == TileReleaseStrategy::Internal
                          || tile_release_strategy == TileReleaseStrategy::All;

    #pragma omp taskgroup
    for (int device = 0; device < B.num_devices(); ++device) {
        #pragma omp task slate_omp_default_none priority( priority ) \
            shared( A, B ) firstprivate( device, queue_index, call_tile_tick )
        {
            std::set<ij_tuple> A_tiles_set;
            for (int64_t i = 0; i < B.mt(); ++i) {
                for (int64_t j = 0; j < B.nt(); ++j) {
                    if (B.tileIsLocal(i, j) && device == B.tileDevice(i, j)) {
                        A_tiles_set.insert({i, j});
                    }
                }
            }
            // no need to convert layout
            // TODO kernel assumes column major
            A.tileGetForReading(A_tiles_set, device, LayoutConvert::ColMajor);

            // no need to copy old values
            for (int64_t i = 0; i < B.mt(); ++i) {
                for (int64_t j = 0; j < B.nt(); ++j) {
                    if (B.tileIsLocal(i, j) && device == B.tileDevice(i, j)) {
                        B.tileAcquire(i, j, device, A.tileLayout(i, j, device));
                        B.tileModified(i, j, device, true);
                    }
                }
            }

            // Usually the output matrix (B) provides all the batch arrays.
            // Here we are using A, because of the possibly different types.
            src_scalar_t** a_array_host = A.array_host(device, queue_index);
            dst_scalar_t** b_array_host = B.array_host(device, queue_index);

            // Because A and B may be different types and C++ doesn't easily
            // support iterating over tuples.  We manually handle A
            std::vector<int64_t> lda;
            int64_t batch_count = 0;
            std::function<void(int64_t, int64_t, int64_t)>
            setup_A = [&] (int64_t group, int64_t i, int64_t j) {
                auto Aij = A( i, j, device );
                a_array_host[ batch_count ] = Aij.data();
                if (lda.size() == size_t(group)) {
                    lda.push_back( Aij.stride() );
                }
                else {
                    assert(lda.size() > size_t(group));
                    assert(lda[group] == Aij.stride());
                }
                ++batch_count;
            };
            auto group_params = device_regions_build<false, 1, dst_scalar_t>(
                                                    {B},
                                                    {b_array_host},
                                                    device,
                                                    setup_A );

            // Usually the output matrix (B) provides all the batch arrays.
            // Here we are using A, because of the different types.
            src_scalar_t** a_array_dev = A.array_device(device, queue_index);
            dst_scalar_t** b_array_dev = B.array_device(device, queue_index);

            blas::Queue* queue = B.compute_queue(device, queue_index);

            blas::device_memcpy<src_scalar_t*>(a_array_dev, a_array_host,
                                batch_count,
                                blas::MemcpyKind::HostToDevice,
                                *queue);

            blas::device_memcpy<dst_scalar_t*>(b_array_dev, b_array_host,
                                batch_count,
                                blas::MemcpyKind::HostToDevice,
                                *queue);

            bool is_trans = (A.op() != B.op());
            bool is_conj = false;
            if (is_trans) {
                is_conj = (A.op() == Op::ConjTrans || B.op() == Op::ConjTrans);
            }

            for (size_t g = 0; g < group_params.size(); ++g) {
                int64_t group_count = group_params[ g ].count;
                if (is_trans) {
                    device::transpose_batch(
                            is_conj,
                            group_params[ g ].mb, group_params[ g ].nb,
                            a_array_dev, lda[ g ],
                            b_array_dev, group_params[ g ].ld[0],
                            group_count, *queue);
                }
                else {
                    device::gecopy(
                            group_params[ g ].mb, group_params[ g ].nb,
                            a_array_dev, lda[ g ],
                            b_array_dev, group_params[ g ].ld[0],
                            group_count, *queue);
                }
                a_array_dev += group_count;
                b_array_dev += group_count;
            }

            queue->sync();

            if (call_tile_tick) {
                for (int64_t i = 0; i < B.mt(); ++i) {
                    for (int64_t j = 0; j < B.nt(); ++j) {
                        if (B.tileIsLocal(i, j) && device == B.tileDevice(i, j)) {
                            // erase tmp local and remote device tiles;
                            A.tileRelease(i, j, device);
                            // decrement life for remote tiles
                            A.tileTick(i, j);
                        }
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
//-----------------------------------------
// float => float
template
void copy<Target::HostTask, float, float>(
    Matrix<float>&& A, Matrix<float>&& B,
    int priority, int queue_index, Options const& opts );

template
void copy<Target::HostNest, float, float>(
    Matrix<float>&& A, Matrix<float>&& B,
    int priority, int queue_index, Options const& opts );

template
void copy<Target::HostBatch, float, float>(
    Matrix<float>&& A, Matrix<float>&& B,
    int priority, int queue_index, Options const& opts );

template
void copy<Target::Devices, float, float>(
    Matrix<float>&& A, Matrix<float>&& B,
    int priority, int queue_index, Options const& opts );

//-----------------------------------------
// float => double
template
void copy<Target::HostTask, float, double>(
    Matrix<float>&& A, Matrix<double>&& B,
    int priority, int queue_index, Options const& opts );

template
void copy<Target::HostNest, float, double>(
    Matrix<float>&& A, Matrix<double>&& B,
    int priority, int queue_index, Options const& opts );

template
void copy<Target::HostBatch, float, double>(
    Matrix<float>&& A, Matrix<double>&& B,
    int priority, int queue_index, Options const& opts );

template
void copy<Target::Devices, float, double>(
    Matrix<float>&& A, Matrix<double>&& B,
    int priority, int queue_index, Options const& opts );

//-----------------------------------------
// double => double
template
void copy<Target::HostTask, double, double>(
    Matrix<double>&& A, Matrix<double>&& B,
    int priority, int queue_index, Options const& opts );

template
void copy<Target::HostNest, double, double>(
    Matrix<double>&& A, Matrix<double>&& B,
    int priority, int queue_index, Options const& opts );

template
void copy<Target::HostBatch, double, double>(
    Matrix<double>&& A, Matrix<double>&& B,
    int priority, int queue_index, Options const& opts );

template
void copy<Target::Devices, double, double>(
    Matrix<double>&& A, Matrix<double>&& B,
    int priority, int queue_index, Options const& opts );

//-----------------------------------------
// double => float
template
void copy<Target::HostTask, double, float>(
    Matrix<double>&& A, Matrix<float>&& B,
    int priority, int queue_index, Options const& opts );

template
void copy<Target::HostNest, double, float>(
    Matrix<double>&& A, Matrix<float>&& B,
    int priority, int queue_index, Options const& opts );

template
void copy<Target::HostBatch, double, float>(
    Matrix<double>&& A, Matrix<float>&& B,
    int priority, int queue_index, Options const& opts );

template
void copy<Target::Devices, double, float>(
    Matrix<double>&& A, Matrix<float>&& B,
    int priority, int queue_index, Options const& opts );

//-----------------------------------------
// complex-float => complex-float
template
void copy< Target::HostTask, std::complex<float>, std::complex<float> >(
    Matrix< std::complex<float> >&& A, Matrix< std::complex<float> >&& B,
    int priority, int queue_index, Options const& opts );

template
void copy< Target::HostNest, std::complex<float>, std::complex<float> >(
    Matrix< std::complex<float> >&& A, Matrix< std::complex<float> >&& B,
    int priority, int queue_index, Options const& opts );

template
void copy< Target::HostBatch, std::complex<float>, std::complex<float> >(
    Matrix< std::complex<float> >&& A, Matrix< std::complex<float> >&& B,
    int priority, int queue_index, Options const& opts );

template
void copy< Target::Devices, std::complex<float>, std::complex<float> >(
    Matrix< std::complex<float> >&& A, Matrix< std::complex<float> >&& B,
    int priority, int queue_index, Options const& opts );

//-----------------------------------------
// complex-float => complex-double
template
void copy< Target::HostTask, std::complex<float>, std::complex<double> >(
    Matrix< std::complex<float> >&& A, Matrix< std::complex<double> >&& B,
    int priority, int queue_index, Options const& opts );

template
void copy< Target::HostNest, std::complex<float>, std::complex<double> >(
    Matrix< std::complex<float> >&& A, Matrix< std::complex<double> >&& B,
    int priority, int queue_index, Options const& opts );

template
void copy< Target::HostBatch, std::complex<float>, std::complex<double> >(
    Matrix< std::complex<float> >&& A, Matrix< std::complex<double> >&& B,
    int priority, int queue_index, Options const& opts );

template
void copy< Target::Devices, std::complex<float>, std::complex<double> >(
    Matrix< std::complex<float> >&& A, Matrix< std::complex<double> >&& B,
    int priority, int queue_index, Options const& opts );

//-----------------------------------------
// complex-double => complex-double
template
void copy< Target::HostTask, std::complex<double>, std::complex<double> >(
    Matrix< std::complex<double> >&& A, Matrix< std::complex<double> >&& B,
    int priority, int queue_index, Options const& opts );

template
void copy< Target::HostNest, std::complex<double>, std::complex<double> >(
    Matrix< std::complex<double> >&& A, Matrix< std::complex<double> >&& B,
    int priority, int queue_index, Options const& opts );

template
void copy< Target::HostBatch, std::complex<double>, std::complex<double> >(
    Matrix< std::complex<double> >&& A, Matrix< std::complex<double> >&& B,
    int priority, int queue_index, Options const& opts );

template
void copy< Target::Devices, std::complex<double>, std::complex<double> >(
    Matrix< std::complex<double> >&& A, Matrix< std::complex<double> >&& B,
    int priority, int queue_index, Options const& opts );

//-----------------------------------------
// complex-double => complex-float
template
void copy< Target::HostTask, std::complex<double>, std::complex<float> >(
    Matrix< std::complex<double> >&& A, Matrix< std::complex<float> >&& B,
    int priority, int queue_index, Options const& opts );

template
void copy< Target::Devices, std::complex<double>, std::complex<float> >(
    Matrix< std::complex<double> >&& A, Matrix< std::complex<float> >&& B,
    int priority, int queue_index, Options const& opts );

template
void copy< Target::HostNest, std::complex<double>, std::complex<float> >(
    Matrix< std::complex<double> >&& A, Matrix< std::complex<float> >&& B,
    int priority, int queue_index, Options const& opts );

template
void copy< Target::HostBatch, std::complex<double>, std::complex<float> >(
    Matrix< std::complex<double> >&& A, Matrix< std::complex<float> >&& B,
    int priority, int queue_index, Options const& opts );

//-----------------------------------------
// float => complex-float
template
void copy< Target::HostTask, float, std::complex<float> >(
    Matrix< float >&& A, Matrix< std::complex<float> >&& B,
    int priority, int queue_index, Options const& opts );

template
void copy< Target::Devices, float, std::complex<float> >(
    Matrix< float >&& A, Matrix< std::complex<float> >&& B,
    int priority, int queue_index, Options const& opts );

template
void copy< Target::HostNest, float, std::complex<float> >(
    Matrix< float >&& A, Matrix< std::complex<float> >&& B,
    int priority, int queue_index, Options const& opts );

template
void copy< Target::HostBatch, float, std::complex<float> >(
    Matrix< float >&& A, Matrix< std::complex<float> >&& B,
    int priority, int queue_index, Options const& opts );

//-----------------------------------------
// double => complex-double
template
void copy< Target::HostTask, double, std::complex<double> >(
    Matrix< double >&& A, Matrix< std::complex<double> >&& B,
    int priority, int queue_index, Options const& opts );

template
void copy< Target::Devices, double, std::complex<double> >(
    Matrix< double >&& A, Matrix< std::complex<double> >&& B,
    int priority, int queue_index, Options const& opts );

template
void copy< Target::HostNest, double, std::complex<double> >(
    Matrix< double >&& A, Matrix< std::complex<double> >&& B,
    int priority, int queue_index, Options const& opts );

template
void copy< Target::HostBatch, double, std::complex<double> >(
    Matrix< double >&& A, Matrix< std::complex<double> >&& B,
    int priority, int queue_index, Options const& opts );

} // namespace internal
} // namespace slate
