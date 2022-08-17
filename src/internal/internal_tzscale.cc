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
namespace device {

template <>
void tzscale(
    Uplo uplo,
    int64_t m, int64_t n,
    float numer, float denom,
    std::complex<float>** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue)
{
#if defined( BLAS_HAVE_CUBLAS )
    tzscale(uplo, m, n,
            numer, denom,
            (cuFloatComplex**) Aarray, lda,
            batch_count, queue);

#elif defined( BLAS_HAVE_ROCBLAS )
    tzscale(uplo, m, n,
            numer, denom,
            (hipFloatComplex**) Aarray, lda,
            batch_count, queue);
#endif
}

template <>
void tzscale(
    Uplo uplo,
    int64_t m, int64_t n,
    double numer, double denom,
    std::complex<double>** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue)
{
#if defined( BLAS_HAVE_CUBLAS )
    tzscale(uplo, m, n,
            numer, denom,
            (cuDoubleComplex**) Aarray, lda,
            batch_count, queue);

#elif defined( BLAS_HAVE_ROCBLAS )
    tzscale(uplo, m, n,
            numer, denom,
            (hipDoubleComplex**) Aarray, lda,
            batch_count, queue);
#endif
}

#if ! defined( SLATE_HAVE_DEVICE )
// Specializations to allow compilation without CUDA or HIP.
template <>
void tzscale(
    Uplo uplo,
    int64_t m, int64_t n,
    double numer, double denom,
    double** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue)
{
}

template <>
void tzscale(
    Uplo uplo,
    int64_t m, int64_t n,
    float numer, float denom,
    float** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue)
{
}
#endif // not SLATE_HAVE_DEVICE

} // namespace device


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

    // Define index ranges for regions of matrix.
    // Tiles in each region are all the same size.
    int64_t irange[4][2] = {
        { 0,        A.mt()-1 },
        { A.mt()-1, A.mt()   },
        { 0,        A.mt()-1 },
        { A.mt()-1, A.mt()   }
    };
    int64_t jrange[4][2] = {
        { 0,        A.nt()-1 },
        { 0,        A.nt()-1 },
        { A.nt()-1, A.nt()   },
        { A.nt()-1, A.nt()   }
    };

    #pragma omp taskgroup
    for (int device = 0; device < A.num_devices(); ++device) {
        #pragma omp task slate_omp_default_none \
            shared( A ) priority( priority ) \
            firstprivate(device, irange, jrange, queue_index, numer, denom)
        {
            // temporarily, convert both into same layout
            // todo: this is in-efficient, because both matrices may have same layout already
            //       and possibly wrong, because an input matrix is being altered
            // todo: best, handle directly through the CUDA kernels
            auto layout = Layout::ColMajor;
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
            A.tileGetForWriting(A_tiles_set, device, LayoutConvert(layout));

            scalar_t** a_array_host = A.array_host(device);

            int64_t batch_count = 0;
            int64_t mb[8], nb[8], lda[8], group_count[8];
            for (int q = 0; q < 4; ++q) {
                group_count[q] = 0;
                lda[q] = 0;
                mb[q] = A.tileMb(irange[q][0]);
                nb[q] = A.tileNb(jrange[q][0]);
                if (A.uplo() == Uplo::Lower) {
                    for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                        for (int64_t i = std::max(j, irange[q][0]); i < irange[q][1]; ++i) {
                            if (i != j && A.tileIsLocal(i, j) && device == A.tileDevice(i, j)) {
                                a_array_host[batch_count] = A(i, j, device).data();
                                lda[q] = A(i, j, device).stride();
                                ++group_count[q];
                                ++batch_count;
                            }
                        }
                    }
                }
                else { // upper
                    for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                        for (int64_t i = irange[q][0]; i < irange[q][1] && i <= j; ++i) {
                            if (i != j && A.tileIsLocal(i, j) && device == A.tileDevice(i, j)) {
                                a_array_host[batch_count] = A(i, j, device).data();
                                lda[q] = A(i, j, device).stride();
                                ++group_count[q];
                                ++batch_count;
                            }
                        }
                    }
                }
            }
            for (int q = 4; q < 8; ++q) {
                group_count[q] = 0;
                lda[q] = 0;
                mb[q] = A.tileMb(irange[q-4][0]);
                nb[q] = A.tileNb(jrange[q-4][0]);
                if (A.uplo() == Uplo::Lower) {
                    for (int64_t j = jrange[q-4][0]; j < jrange[q-4][1]; ++j) {
                        for (int64_t i = std::max(j, irange[q-4][0]); i < irange[q-4][1]; ++i) {
                            if (i == j && A.tileIsLocal(i, j) && device == A.tileDevice(i, j)) {
                                a_array_host[batch_count] = A(i, j, device).data();
                                lda[q] = A(i, j, device).stride();
                                ++group_count[q];
                                ++batch_count;
                            }
                        }
                    }
                }
                else { // upper
                    for (int64_t j = jrange[q-4][0]; j < jrange[q-4][1]; ++j) {
                        for (int64_t i = irange[q-4][0]; i < irange[q-4][1] && i <= j; ++i) {
                            if (i == j && A.tileIsLocal(i, j) && device == A.tileDevice(i, j)) {
                                a_array_host[batch_count] = A(i, j, device).data();
                                lda[q] = A(i, j, device).stride();
                                ++group_count[q];
                                ++batch_count;
                            }
                        }
                    }
                }
            }
            scalar_t** a_array_dev = A.array_device(device);

            blas::Queue* queue = A.compute_queue(device, queue_index);

            blas::device_memcpy<scalar_t*>(
                a_array_dev, a_array_host, batch_count,
                blas::MemcpyKind::HostToDevice, *queue);

            for (int q = 0; q < 4; ++q) {
                if (group_count[q] > 0) {
                    device::batch::gescale(mb[q], nb[q],
                                    numer, denom, a_array_dev, lda[q],
                                    group_count[q], *queue);
                    a_array_dev += group_count[q];
                }
            }
            for (int q = 4; q < 8; ++q) {
                if (group_count[q] > 0) {
                    device::tzscale(A.uplo(), mb[q], nb[q],
                                    numer, denom, a_array_dev, lda[q],
                                    group_count[q], *queue);
                    a_array_dev += group_count[q];
                }
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
