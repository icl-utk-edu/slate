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

    int64_t mt = A.mt();
    int64_t nt = A.nt();

    // Find ranges of matching mb's.
    std::vector< int64_t > irange;
    int64_t last_mb = -1;
    for (int64_t i = 0; i < mt; ++i) {
        int64_t mb = A.tileMb( i );
        if (mb != last_mb) {
            last_mb = mb;
            irange.push_back( i );
        }
    }
    irange.push_back( mt );

    // Find ranges of matching nb's.
    std::vector< int64_t > jrange;
    int last_nb = -1;
    for (int64_t j = 0; j < nt; ++j) {
        int64_t nb = A.tileNb( j );
        if (nb != last_nb) {
            last_nb = nb;
            jrange.push_back( j );
        }
    }
    jrange.push_back( nt );

    #pragma omp taskgroup
    for (int device = 0; device < A.num_devices(); ++device) {
        #pragma omp task priority( priority ) \
            shared( A, irange, jrange ) \
            firstprivate( device, queue_index, offdiag_value, diag_value )
        {
            // Get local tiles for writing.
            // convert to column major layout to simplify lda's
            // todo: this is in-efficient because the diagonal is independant of layout
            // todo: best, handle directly through the CUDA kernels
            auto layout = LayoutConvert( Layout::ColMajor );
            std::set<ij_tuple> A_tiles_set;

            for (int64_t i = 0; i < A.mt(); ++i) {
                for (int64_t j = 0; j < A.nt(); ++j) {
                    if (A.tileIsLocal( i, j ) && device == A.tileDevice( i, j )) {
                        A_tiles_set.insert( { i, j } );
                    }
                }
            }
            A.tileGetForWriting( A_tiles_set, device, layout );

            scalar_t** a_array_host = A.array_host( device, queue_index );

            // If offdiag == diag value, lump diag tiles with offdiag tiles
            // in one batch.
            bool diag_same = offdiag_value == diag_value;

            // Build batch groups.
            int64_t batch_count = 0;
            struct Params {
                int64_t count, mb, nb, lda;
                scalar_t diag_value;
            };
            std::vector<Params> group_params;
            for (size_t jj = 0; jj < jrange.size() - 1; ++jj) {
            for (size_t ii = 0; ii < irange.size() - 1; ++ii) {
                Params group = { 0, -1, -1, -1, offdiag_value };
                for (int64_t j = jrange[ jj ]; j < jrange[ jj+1 ]; ++j) {
                for (int64_t i = irange[ ii ]; i < irange[ ii+1 ]; ++i) {
                    if ((diag_same || i != j)
                        && A.tileIsLocal( i, j )
                        && device == A.tileDevice( i, j ))
                    {
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
                }} // for j, i
                if (group.count > 0) {
                    group_params.push_back( group );
                }
            }} // for jj, ii

            // Build batch groups for diagonal tiles,
            // when offdiag_value != diag_value.
            if (! diag_same) {
                for (size_t jj = 0; jj < jrange.size() - 1; ++jj) {
                for (size_t ii = 0; ii < irange.size() - 1; ++ii) {
                    Params group = { 0, -1, -1, -1, diag_value };
                    // Diagonal tiles only in the intersection of irange and jrange
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
                }} // for jj, ii
            }

            blas::Queue* queue = A.compute_queue( device, queue_index );

            scalar_t** a_array_dev = A.array_device( device, queue_index );
            blas::device_memcpy<scalar_t*>(
                a_array_dev, a_array_host, batch_count,
                blas::MemcpyKind::HostToDevice, *queue);

            for (size_t g = 0; g < group_params.size(); ++g) {
                int64_t group_count = group_params[ g ].count;
                device::batch::geset(
                    group_params[ g ].mb,
                    group_params[ g ].nb,
                    offdiag_value, group_params[ g ].diag_value,
                    a_array_dev, group_params[ g ].lda,
                    group_count, *queue );
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
