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
/// Scale matrix entries by the real scalar numer/denom.
/// Dispatches to target implementations.
/// @ingroup scale_internal
///
template <Target target, typename scalar_t>
void scale(
    blas::real_type<scalar_t> numer, blas::real_type<scalar_t> denom,
    Matrix<scalar_t>&& A, int priority, int queue_index)
{
    scale(internal::TargetType<target>(),
          numer, denom, A, priority, queue_index);
}

//------------------------------------------------------------------------------
/// Scale matrix entries by the real scalar numer/denom.
/// TODO handle transpose A case
/// Host OpenMP task implementation.
/// @ingroup scale_internal
///
template <typename scalar_t>
void scale(
    internal::TargetType<Target::HostTask>,
    blas::real_type<scalar_t> numer, blas::real_type<scalar_t> denom,
    Matrix<scalar_t>& A, int priority, int queue_index)
{
    // trace::Block trace_block("scale");
    #pragma omp taskgroup
    for (int64_t i = 0; i < A.mt(); ++i) {
        for (int64_t j = 0; j < A.nt(); ++j) {
            if (A.tileIsLocal(i, j)) {
                #pragma omp task slate_omp_default_none \
                    shared( A ) \
                    firstprivate(i, j, numer, denom) priority(priority)
                {
                    A.tileGetForWriting(i, j, LayoutConvert::None);
                    scale(numer, denom, A(i, j));
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void scale(internal::TargetType<Target::HostNest>,
           blas::real_type<scalar_t> numer, blas::real_type<scalar_t> denom,
           Matrix<scalar_t>& A, int priority, int queue_index)
{
    slate_not_implemented("Target::HostNest isn't yet supported.");
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void scale(internal::TargetType<Target::HostBatch>,
           blas::real_type<scalar_t> numer, blas::real_type<scalar_t> denom,
           Matrix<scalar_t>& A, int priority, int queue_index)
{
    slate_not_implemented("Target::HostBatch isn't yet supported.");
}

//------------------------------------------------------------------------------
/// Scale matrix entries by the real scalar numer/denom.
/// TODO handle transpose A case
/// GPU device implementation.
/// @ingroup scale_internal
///
template <typename scalar_t>
void scale(internal::TargetType<Target::Devices>,
           blas::real_type<scalar_t> numer, blas::real_type<scalar_t> denom,
           Matrix<scalar_t>& A, int priority, int queue_index)
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
        #pragma omp task priority( priority ) shared( A, irange, jrange ) \
            firstprivate( device, queue_index, denom, numer )
        {
            // temporarily, convert both into same layout
            // todo: this is in-efficient, because both matrices may have same layout already
            //       and possibly wrong, because an input matrix is being altered
            // todo: best, handle directly through the CUDA kernels
            auto layout = LayoutConvert::ColMajor;
            std::set<ij_tuple> A_tiles_set;

            for (int64_t i = 0; i < A.mt(); ++i) {
                for (int64_t j = 0; j < A.nt(); ++j) {
                    if (A.tileIsLocal(i, j) && device == A.tileDevice(i, j)) {
                        A_tiles_set.insert({i, j});
                    }
                }
            }
            A.tileGetForWriting( A_tiles_set, device, layout );

            scalar_t** a_array_host = A.array_host( device, queue_index );

            int64_t batch_count = 0;
            struct Params {
                int64_t count, mb, nb, lda;
            };
            std::vector<Params> group_params;
            for (size_t jj = 0; jj < jrange.size() - 1; ++jj) {
            for (size_t ii = 0; ii < irange.size() - 1; ++ii) {
                Params group = { 0, -1, -1, -1 };
                for (int64_t j = jrange[ jj ]; j < jrange[ jj+1 ]; ++j) {
                for (int64_t i = irange[ ii ]; i < irange[ ii+1 ]; ++i) {
                    if (A.tileIsLocal( i, j ) && device == A.tileDevice( i, j )) {
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

            blas::Queue* queue = A.compute_queue( device, queue_index );

            scalar_t** a_array_dev = A.array_device( device, queue_index );
            blas::device_memcpy<scalar_t*>(
                a_array_dev, a_array_host, batch_count,
                blas::MemcpyKind::HostToDevice, *queue);

            for (size_t g = 0; g < group_params.size(); ++g) {
                int64_t group_count = group_params[ g ].count;
                device::batch::gescale(
                        group_params[ g ].mb, group_params[ g ].nb,
                        numer, denom, a_array_dev, group_params[ g ].lda,
                        group_count, *queue);
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
    float numer, float denom, Matrix<float>&& A,
    int priority, int queue_index);

template
void scale<Target::HostNest, float>(
    float numer, float denom, Matrix<float>&& A,
    int priority, int queue_index);

template
void scale<Target::HostBatch, float>(
    float numer, float denom, Matrix<float>&& A,
    int priority, int queue_index);

template
void scale<Target::Devices, float>(
    float numer, float denom, Matrix<float>&& A,
    int priority, int queue_index);

// ----------------------------------------
template
void scale<Target::HostTask, double>(
    double numer, double denom, Matrix<double>&& A,
    int priority, int queue_index);

template
void scale<Target::HostNest, double>(
    double numer, double denom, Matrix<double>&& A,
    int priority, int queue_index);

template
void scale<Target::HostBatch, double>(
    double numer, double denom, Matrix<double>&& A,
    int priority, int queue_index);

template
void scale<Target::Devices, double>(
    double numer, double denom, Matrix<double>&& A,
    int priority, int queue_index);

// ----------------------------------------
template
void scale< Target::HostTask, std::complex<float> >(
    float numer, float denom,
    Matrix< std::complex<float> >&& A,
    int priority, int queue_index);

template
void scale< Target::HostNest, std::complex<float> >(
    float numer, float denom,
    Matrix< std::complex<float> >&& A,
    int priority, int queue_index);

template
void scale< Target::HostBatch, std::complex<float> >(
    float numer, float denom,
    Matrix< std::complex<float> >&& A,
    int priority, int queue_index);

template
void scale< Target::Devices, std::complex<float> >(
    float numer, float  denom,
    Matrix< std::complex<float> >&& A,
    int priority, int queue_index);

// ----------------------------------------
template
void scale< Target::HostTask, std::complex<double> >(
    double numer, double denom,
    Matrix< std::complex<double> >&& A,
    int priority, int queue_index);

template
void scale< Target::HostNest, std::complex<double> >(
    double numer, double denom,
    Matrix< std::complex<double> >&& A,
    int priority, int queue_index);

template
void scale< Target::HostBatch, std::complex<double> >(
    double numer, double denom,
    Matrix< std::complex<double> >&& A,
    int priority, int queue_index);

template
void scale< Target::Devices, std::complex<double> >(
    double numer, double denom,
    Matrix< std::complex<double> >&& A,
    int priority, int queue_index);

} // namespace internal
} // namespace slate
