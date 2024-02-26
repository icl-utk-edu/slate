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
namespace internal {

//------------------------------------------------------------------------------
/// Copy and precision conversion.
/// Dispatches to target implementations.
/// @ingroup copy_internal
///
template <Target target, typename src_scalar_t, typename dst_scalar_t>
void copy(BaseTrapezoidMatrix<src_scalar_t>&& A,
          BaseTrapezoidMatrix<dst_scalar_t>&& B,
          int priority, int queue_index )
{
    copy(internal::TargetType<target>(),
         A, B,
         priority, queue_index );
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
          BaseTrapezoidMatrix<src_scalar_t>& A,
          BaseTrapezoidMatrix<dst_scalar_t>& B,
          int priority, int queue_index )
{
    // trace::Block trace_block("copy");

    slate_error_if(A.uplo() != B.uplo());
    bool lower = (B.uplo() == Uplo::Lower);

    assert(A.mt() == B.mt());
    assert(A.nt() == B.nt());

    #pragma omp taskgroup
    for (int64_t j = 0; j < B.nt(); ++j) {
        if (j < B.mt() && B.tileIsLocal(j, j)) {
            A.tileGetForReading(j, j, LayoutConvert::None);
            B.tileGetForWriting(j, j, LayoutConvert( A.tileLayout(j, j) ));
            tile::tzcopy( A(j, j), B(j, j) );
        }
        if (lower) {
            for (int64_t i = j+1; i < B.mt(); ++i) {
                if (B.tileIsLocal(i, j)) {
                    #pragma omp task slate_omp_default_none \
                        shared( A, B ) priority( priority ) \
                        firstprivate( i, j )
                    {
                        A.tileGetForReading(i, j, LayoutConvert::None);
                        B.tileAcquire(i, j, A.tileLayout(i, j));
                        B.tileModified(i, j, HostNum, true);
                        tile::gecopy( A(i, j), B(i, j) );
                    }
                }
            }
        }
        else { // Uplo::Upper
            for (int64_t i = 0; i < j && i < B.mt(); ++i) {
                if (B.tileIsLocal(i, j)) {
                    #pragma omp task slate_omp_default_none \
                        shared( A, B ) priority( priority ) \
                        firstprivate( i, j )
                    {
                        A.tileGetForReading(i, j, LayoutConvert::None);
                        B.tileAcquire(i, j, A.tileLayout(i, j));
                        B.tileModified(i, j, HostNum, true);
                        tile::gecopy( A(i, j), B(i, j) );
                    }
                }
            }
        }
    }
    // end omp taskgroup
}

//------------------------------------------------------------------------------
/// Copy and precision conversion.
/// assumes A & B have same tile layout and dimensions, and have same distribution
/// TODO: Inspect transposition?
/// GPU device implementation.
/// @ingroup copy_internal
///
template <typename src_scalar_t, typename dst_scalar_t>
void copy(internal::TargetType<Target::Devices>,
          BaseTrapezoidMatrix<src_scalar_t>& A,
          BaseTrapezoidMatrix<dst_scalar_t>& B,
          int priority, int queue_index )
{
    using ij_tuple = typename BaseMatrix<src_scalar_t>::ij_tuple;
    slate_error_if(A.uplo() != B.uplo());
    bool lower = (B.uplo() == Uplo::Lower);

    #pragma omp taskgroup
    for (int device = 0; device < B.num_devices(); ++device) {
        #pragma omp task slate_omp_default_none priority( priority ) \
            shared( A, B ) firstprivate( device, lower, queue_index )
        {
            std::set<ij_tuple> A_tiles, B_diag_tiles;
            for (int64_t i = 0; i < B.mt(); ++i) {
                for (int64_t j = 0; j < B.nt(); ++j) {
                    if (B.tileIsLocal(i, j) && device == B.tileDevice(i, j)
                        && (   (  lower && i >= j)
                            || (! lower && i <= j) ) )
                    {
                        A_tiles.insert( { i, j } );
                        if (i == j) {
                            B_diag_tiles.insert( { i, j } );
                        }
                        else {
                            B.tileAcquire( i, j, device, Layout::ColMajor );
                            B.tileModified( i, j, device, true );
                        }
                    }
                }
            }
            // For B, diagonal tiles must be fetched for writing;
            // off-diagonal tiles can be fetched for over-writing
            // (tileAcquire above).
            // TODO no need to conver layout of A but kernel assumes column major
            A.tileGetForReading( A_tiles, device, LayoutConvert::ColMajor );
            B.tileGetForWriting( B_diag_tiles, device, LayoutConvert::ColMajor );

            // Usually the output matrix (B) provides all the batch arrays.
            // Here we are using A, because of the different types.
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
            auto group_params = device_regions_build<true, 1, dst_scalar_t>(
                    {B},
                    {b_array_host},
                    device,
                    setup_A );

            // Usually the output matrix (B) provides all the batch arrays.
            // Here we are using A, because of the differen types.
            src_scalar_t** a_array_dev = A.array_device(device, queue_index);
            dst_scalar_t** b_array_dev = B.array_device(device, queue_index);

            blas::Queue* queue = A.compute_queue(device, queue_index);

            blas::device_memcpy<src_scalar_t*>(a_array_dev, a_array_host,
                                batch_count,
                                blas::MemcpyKind::HostToDevice,
                                *queue);

            blas::device_memcpy<dst_scalar_t*>(b_array_dev, b_array_host,
                                batch_count,
                                blas::MemcpyKind::HostToDevice,
                                *queue);

            for (size_t g = 0; g < group_params.size(); ++g) {
                int64_t group_count = group_params[ g ].count;
                if (group_params[ g ].is_diagonal) {
                    device::tzcopy(
                            B.uplo(),
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
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void copy<Target::HostTask, float, float>(
    BaseTrapezoidMatrix<float>&& A,
    BaseTrapezoidMatrix<float>&& B,
    int priority, int queue_index );

template
void copy<Target::HostTask, float, double>(
    BaseTrapezoidMatrix<float>&& A,
    BaseTrapezoidMatrix<double>&& B,
    int priority, int queue_index );

template
void copy<Target::Devices, float, float>(
    BaseTrapezoidMatrix<float>&& A,
    BaseTrapezoidMatrix<float>&& B,
    int priority, int queue_index );

template
void copy<Target::Devices, float, double>(
    BaseTrapezoidMatrix<float>&& A,
    BaseTrapezoidMatrix<double>&& B,
    int priority, int queue_index );

// ----------------------------------------
template
void copy<Target::HostTask, double, double>(
    BaseTrapezoidMatrix<double>&& A,
    BaseTrapezoidMatrix<double>&& B,
    int priority, int queue_index );

template
void copy<Target::HostTask, double, float>(
    BaseTrapezoidMatrix<double>&& A,
    BaseTrapezoidMatrix<float>&& B,
    int priority, int queue_index );

template
void copy<Target::Devices, double, double>(
    BaseTrapezoidMatrix<double>&& A,
    BaseTrapezoidMatrix<double>&& B,
    int priority, int queue_index );

template
void copy<Target::Devices, double, float>(
    BaseTrapezoidMatrix<double>&& A,
    BaseTrapezoidMatrix<float>&& B,
    int priority, int queue_index );

// ----------------------------------------
template
void copy< Target::HostTask, std::complex<float>, std::complex<float> >(
    BaseTrapezoidMatrix< std::complex<float> >&& A,
    BaseTrapezoidMatrix< std::complex<float> >&& B,
    int priority, int queue_index );

template
void copy< Target::HostTask, std::complex<float>, std::complex<double> >(
    BaseTrapezoidMatrix< std::complex<float> >&& A,
    BaseTrapezoidMatrix< std::complex<double> >&& B,
    int priority, int queue_index );

template
void copy< Target::Devices, std::complex<float>, std::complex<float>  >(
    BaseTrapezoidMatrix< std::complex<float> >&& A,
    BaseTrapezoidMatrix< std::complex<float> >&& B,
    int priority, int queue_index );

template
void copy< Target::Devices, std::complex<float>, std::complex<double>  >(
    BaseTrapezoidMatrix< std::complex<float> >&& A,
    BaseTrapezoidMatrix< std::complex<double> >&& B,
    int priority, int queue_index );

// ----------------------------------------
template
void copy< Target::HostTask, std::complex<double>, std::complex<double> >(
    BaseTrapezoidMatrix< std::complex<double> >&& A,
    BaseTrapezoidMatrix< std::complex<double> >&& B,
    int priority, int queue_index );

template
void copy< Target::HostTask, std::complex<double>, std::complex<float> >(
    BaseTrapezoidMatrix< std::complex<double> >&& A,
    BaseTrapezoidMatrix< std::complex<float> >&& B,
    int priority, int queue_index );

template
void copy< Target::Devices, std::complex<double>, std::complex<double> >(
    BaseTrapezoidMatrix< std::complex<double> >&& A,
    BaseTrapezoidMatrix< std::complex<double> >&& B,
    int priority, int queue_index );

template
void copy< Target::Devices, std::complex<double>, std::complex<float> >(
    BaseTrapezoidMatrix< std::complex<double> >&& A,
    BaseTrapezoidMatrix< std::complex<float> >&& B,
    int priority, int queue_index );

} // namespace internal
} // namespace slate
