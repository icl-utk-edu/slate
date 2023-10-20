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
/// Trapezoid matrix set.
/// Dispatches to target implementations.
/// @ingroup set_internal
///
template <Target target, typename scalar_t>
void set(
    scalar_t offdiag_value, scalar_t diag_value,
    BaseTrapezoidMatrix<scalar_t>&& A,
    int priority, int queue_index)
{
    set(internal::TargetType<target>(),
        offdiag_value, diag_value, A, priority, queue_index );
}

//------------------------------------------------------------------------------
/// Trapezoid matrix set.
/// TODO handle transpose A case
/// Host OpenMP task implementation.
/// @ingroup set_internal
///
template <typename scalar_t>
void set(
    internal::TargetType<Target::HostTask>,
    scalar_t offdiag_value, scalar_t diag_value,
    BaseTrapezoidMatrix<scalar_t>& A,
    int priority, int queue_index)
{
    // trace::Block trace_block("set");

    #pragma omp taskgroup
    if (A.uplo() == Uplo::Lower) {
        for (int64_t j = 0; j < A.nt(); ++j) {
            for (int64_t i = j; i < A.mt(); ++i) {  // lower trapezoid
                if (A.tileIsLocal(i, j)) {
                    #pragma omp task slate_omp_default_none \
                        shared( A ) priority( priority ) \
                        firstprivate( i, j, offdiag_value, diag_value )
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
    else { // upper
        for (int64_t j = 0; j < A.nt(); ++j) {
            for (int64_t i = 0; i <= j && i < A.mt(); ++i) {  // upper trapezoid
                if (A.tileIsLocal(i, j)) {
                    #pragma omp task slate_omp_default_none \
                        shared( A ) priority( priority ) \
                        firstprivate( i, j, offdiag_value, diag_value )
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
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void set(
    internal::TargetType<Target::HostNest>,
    scalar_t offdiag_value, scalar_t diag_value,
    BaseTrapezoidMatrix<scalar_t>& A,
    int priority, int queue_index)
{
    slate_not_implemented("Target::HostNest isn't yet supported.");
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void set(
    internal::TargetType<Target::HostBatch>,
    scalar_t offdiag_value, scalar_t diag_value,
    BaseTrapezoidMatrix<scalar_t>& A,
    int priority, int queue_index)
{
    slate_not_implemented("Target::HostBatch isn't yet supported.");
}

//------------------------------------------------------------------------------
/// Trapezoid matrix set.
/// TODO handle transpose A case
/// GPU device implementation.
/// @ingroup set_internal
///
template <typename scalar_t>
void set(
    internal::TargetType<Target::Devices>,
    scalar_t offdiag_value, scalar_t diag_value,
    BaseTrapezoidMatrix<scalar_t>& A,
    int priority, int queue_index)
{
    using ij_tuple = typename BaseTrapezoidMatrix<scalar_t>::ij_tuple;

    // Find ranges of matching mb's and ranges of matching nb's.
    std::vector< int64_t > irange = device_regions_range( true, A );
    std::vector< int64_t > jrange = device_regions_range( false, A );

    #pragma omp taskgroup
    for (int device = 0; device < A.num_devices(); ++device) {
        #pragma omp task priority( priority ) shared( A, irange, jrange ) \
            firstprivate( device, queue_index, offdiag_value, diag_value )
        {
            // temporarily, convert both into same layout
            // todo: this is in-efficient, because both matrices may have same layout already
            //       and possibly wrong, because an input matrix is being altered
            // todo: best, handle directly through the CUDA kernels
            auto layout = LayoutConvert::ColMajor;
            std::set<ij_tuple> A_tiles_set;

            if (A.uplo() == Uplo::Lower) {
                for (int64_t j = 0; j < A.nt(); ++j) {
                    for (int64_t i = j; i < A.mt(); ++i) {  // lower trapezoid
                        if (A.tileIsLocal( i, j ) && device == A.tileDevice( i, j )) {
                            A_tiles_set.insert( {i, j} );
                        }
                    }
                }
            }
            else { // upper
                for (int64_t j = 0; j < A.nt(); ++j) {
                    for (int64_t i = 0; i <= j && i < A.mt(); ++i) {  // upper trapezoid
                        if (A.tileIsLocal( i, j ) && device == A.tileDevice( i, j )) {
                            A_tiles_set.insert( {i, j} );
                        }
                    }
                }
            }
            A.tileGetForWriting( A_tiles_set, device, layout );

            scalar_t** a_array_host = A.array_host( device );
            scalar_t** a_array_dev  = A.array_device( device );

            // Build batch groups
            int64_t batch_count = 0;
            struct Params {
                int64_t count, mb, nb, lda;
                bool is_diagonal;
            };
            std::vector<Params> group_params;
            // Build batch groups for off-diagonal tiles,
            for (size_t jj = 0; jj < jrange.size() - 1; ++jj) {
            for (size_t ii = 0; ii < irange.size() - 1; ++ii) {
                Params group = { 0, -1, -1, -1, false };
                if (A.uplo() == Uplo::Lower) {
                    for (int64_t j = jrange[ jj ]; j < jrange[ jj+1 ]; ++j) {
                    for (int64_t i = std::max(irange[ ii ], j); i < irange[ ii+1 ]; ++i) {
                        if (i != j
                            && A.tileIsLocal(i, j)
                            && device == A.tileDevice(i, j)) {

                            auto Aij = A( i, j, device );
                            a_array_host[ batch_count ] = Aij.data();
                            if (group.count == 0) {
                                group.mb  = Aij.mb();
                                group.nb  = Aij.nb();
                                group.lda = Aij.stride();
                            }
                            else {
                                assert( group.mb  == Aij.mb() );
                                assert( group.nb  == Aij.nb() );
                                assert( group.lda == Aij.stride() );
                            }
                            ++group.count;
                            ++batch_count;
                        }
                    }} // for j,i
                }
                else { // A.uplo() == Uplo::Upper
                    for (int64_t j = jrange[ jj ]; j < jrange[ jj+1 ]; ++j) {
                    for (int64_t i = irange[ ii ]; i < irange[ ii+1 ] && i <= j; ++i) {
                        if (i != j
                            && A.tileIsLocal(i, j)
                            && device == A.tileDevice(i, j)) {

                            auto Aij = A( i, j, device );
                            a_array_host[ batch_count ] = Aij.data();
                            if (group.count == 0) {
                                group.mb  = Aij.mb();
                                group.nb  = Aij.nb();
                                group.lda = Aij.stride();
                            }
                            else {
                                assert( group.mb  == Aij.mb() );
                                assert( group.nb  == Aij.nb() );
                                assert( group.lda == Aij.stride() );
                            }
                            ++group.count;
                            ++batch_count;
                        }
                    }} // for j,i
                }
                if (group.count > 0) {
                    group_params.push_back( group );
                }
            }} // for jj,ii

            // Build batch groups for diagonal tiles,
            for (size_t jj = 0; jj < jrange.size() - 1; ++jj) {
            for (size_t ii = 0; ii < irange.size() - 1; ++ii) {
                Params group = { 0, -1, -1, -1, true };
                int64_t ijstart = std::max(irange[ ii   ], jrange[ jj   ]);
                int64_t ijend   = std::min(irange[ ii+1 ], jrange[ jj+1 ]);
                for (int64_t ij = ijstart; ij < ijend; ++ij) {
                    if (A.tileIsLocal( ij, ij )
                        && device == A.tileDevice( ij, ij ))
                    {
                        auto Aij = A( ij, ij, device );
                        a_array_host[ batch_count ] = Aij.data();
                        if (group.count == 0) {
                            group.mb  = Aij.mb();
                            group.nb  = Aij.nb();
                            group.lda = Aij.stride();
                        }
                        else {
                            assert( group.mb  == Aij.mb() );
                            assert( group.nb  == Aij.nb() );
                            assert( group.lda == Aij.stride() );
                        }
                        ++group.count;
                        ++batch_count;
                    }
                } // for ij
                if (group.count > 0) {
                    group_params.push_back( group );
                }
            }} // for jj,ii

            blas::Queue* queue = A.compute_queue(device, queue_index);

            blas::device_memcpy<scalar_t*>(
                a_array_dev, a_array_host, batch_count,
                blas::MemcpyKind::HostToDevice, *queue);

            for (size_t g = 0; g < group_params.size(); ++g) {
                int64_t group_count = group_params[ g ].count;

                if (group_params[ g ].is_diagonal) {
                    device::batch::tzset(
                        A.uplo(),
                        group_params[ g ].mb, group_params[ g ].nb,
                        offdiag_value, diag_value,
                        a_array_dev, group_params[ g ].lda,
                        group_count, *queue);
                }
                else {
                    device::batch::geset(
                        group_params[ g ].mb, group_params[ g ].nb,
                        offdiag_value, offdiag_value,
                        a_array_dev, group_params[ g ].lda,
                        group_count, *queue );
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
void set<Target::HostTask, float>(
    float offdiag_value, float diag_value,
    BaseTrapezoidMatrix<float>&& A,
    int priority, int queue_index);

template
void set<Target::HostNest, float>(
    float offdiag_value, float diag_value,
    BaseTrapezoidMatrix<float>&& A,
    int priority, int queue_index);

template
void set<Target::HostBatch, float>(
    float offdiag_value, float diag_value,
    BaseTrapezoidMatrix<float>&& A,
    int priority, int queue_index);

template
void set<Target::Devices, float>(
    float offdiag_value, float diag_value,
    BaseTrapezoidMatrix<float>&& A,
    int priority, int queue_index);

// ----------------------------------------
template
void set<Target::HostTask, double>(
    double offdiag_value, double diag_value,
    BaseTrapezoidMatrix<double>&& A,
    int priority, int queue_index);

template
void set<Target::HostNest, double>(
    double offdiag_value, double diag_value,
    BaseTrapezoidMatrix<double>&& A,
    int priority, int queue_index);

template
void set<Target::HostBatch, double>(
    double offdiag_value, double diag_value,
    BaseTrapezoidMatrix<double>&& A,
    int priority, int queue_index);

template
void set<Target::Devices, double>(
    double offdiag_value, double diag_value,
    BaseTrapezoidMatrix<double>&& A,
    int priority, int queue_index);

// ----------------------------------------
template
void set< Target::HostTask, std::complex<float> >(
    std::complex<float> offdiag_value, std::complex<float>  diag_value,
    BaseTrapezoidMatrix< std::complex<float> >&& A,
    int priority, int queue_index);

template
void set< Target::HostNest, std::complex<float> >(
    std::complex<float> offdiag_value, std::complex<float>  diag_value,
    BaseTrapezoidMatrix< std::complex<float> >&& A,
    int priority, int queue_index);

template
void set< Target::HostBatch, std::complex<float> >(
    std::complex<float> offdiag_value, std::complex<float>  diag_value,
    BaseTrapezoidMatrix< std::complex<float> >&& A,
    int priority, int queue_index);

template
void set< Target::Devices, std::complex<float> >(
    std::complex<float> offdiag_value, std::complex<float>  diag_value,
    BaseTrapezoidMatrix< std::complex<float> >&& A,
    int priority, int queue_index);

// ----------------------------------------
template
void set< Target::HostTask, std::complex<double> >(
    std::complex<double> offdiag_value, std::complex<double> diag_value,
    BaseTrapezoidMatrix< std::complex<double> >&& A,
    int priority, int queue_index);

template
void set< Target::HostNest, std::complex<double> >(
    std::complex<double> offdiag_value, std::complex<double> diag_value,
    BaseTrapezoidMatrix< std::complex<double> >&& A,
    int priority, int queue_index);

template
void set< Target::HostBatch, std::complex<double> >(
    std::complex<double> offdiag_value, std::complex<double> diag_value,
    BaseTrapezoidMatrix< std::complex<double> >&& A,
    int priority, int queue_index);

template
void set< Target::Devices, std::complex<double> >(
    std::complex<double> offdiag_value, std::complex<double> diag_value,
    BaseTrapezoidMatrix< std::complex<double> >&& A,
    int priority, int queue_index);

} // namespace internal
} // namespace slate
