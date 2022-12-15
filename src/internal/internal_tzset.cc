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

//==============================================================================
// Specializations of device kernels to map std::complex => cuComplex, etc.,
// and define float, double versions if compiled without CUDA or HIP.
namespace device {

//------------------------------------------------------------------------------
// device single tile routine
template <>
void tzset(
    Uplo uplo,
    int64_t m, int64_t n,
    std::complex<float> const& offdiag_value, std::complex<float> const& diag_value,
    std::complex<float>* A, int64_t lda,
    blas::Queue& queue)
{
#if defined( BLAS_HAVE_CUBLAS )
    tzset(
        uplo, m, n,
        make_cuFloatComplex( offdiag_value.real(), offdiag_value.imag() ),
        make_cuFloatComplex( diag_value.real(), diag_value.imag() ),
        (cuFloatComplex*) A, lda,
        queue);

#elif defined( BLAS_HAVE_ROCBLAS )
    tzset(
        uplo, m, n,
        make_hipFloatComplex( offdiag_value.real(), offdiag_value.imag() ),
        make_hipFloatComplex( diag_value.real(), diag_value.imag() ),
        (hipFloatComplex*) A, lda,
        queue);
#endif
}

//----------------------------------------
template <>
void tzset(
    Uplo uplo,
    int64_t m, int64_t n,
    std::complex<double> const& offdiag_value, std::complex<double> const& diag_value,
    std::complex<double>* A, int64_t lda,
    blas::Queue& queue)
{
#if defined( BLAS_HAVE_CUBLAS )
    tzset(
        uplo, m, n,
        make_cuDoubleComplex( offdiag_value.real(), offdiag_value.imag() ),
        make_cuDoubleComplex( diag_value.real(), diag_value.imag() ),
        (cuDoubleComplex*) A, lda,
        queue);

#elif defined( BLAS_HAVE_ROCBLAS )
    tzset(
        uplo, m, n,
        make_hipDoubleComplex( offdiag_value.real(), offdiag_value.imag() ),
        make_hipDoubleComplex( diag_value.real(), diag_value.imag() ),
        (hipDoubleComplex*) A, lda,
        queue);
#endif
}

#if ! defined( SLATE_HAVE_DEVICE )
//----------------------------------------
// Specializations to allow compilation without CUDA or HIP.
template <>
void tzset(
    Uplo uplo,
    int64_t m, int64_t n,
    double const& offdiag_value, double const& diag_value,
    double* A, int64_t lda,
    blas::Queue& queue)
{
}

//----------------------------------------
template <>
void tzset(
    Uplo uplo,
    int64_t m, int64_t n,
    float const& offdiag_value, float const& diag_value,
    float* A, int64_t lda,
    blas::Queue& queue)
{
}
#endif // not SLATE_HAVE_DEVICE

//==============================================================================
namespace batch {

//------------------------------------------------------------------------------
// device::batch routine
template <>
void tzset(
    Uplo uplo,
    int64_t m, int64_t n,
    std::complex<float> const& offdiag_value, std::complex<float> const& diag_value,
    std::complex<float>** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue)
{
#if defined( BLAS_HAVE_CUBLAS )
    batch::tzset(
        uplo, m, n,
        make_cuFloatComplex( offdiag_value.real(), offdiag_value.imag() ),
        make_cuFloatComplex( diag_value.real(), diag_value.imag() ),
        (cuFloatComplex**) Aarray, lda,
        batch_count, queue);

#elif defined( BLAS_HAVE_ROCBLAS )
    batch::tzset(
        uplo, m, n,
        make_hipFloatComplex( offdiag_value.real(), offdiag_value.imag() ),
        make_hipFloatComplex( diag_value.real(), diag_value.imag() ),
        (hipFloatComplex**) Aarray, lda,
        batch_count, queue);
#endif
}

//----------------------------------------
template <>
void tzset(
    Uplo uplo,
    int64_t m, int64_t n,
    std::complex<double> const& offdiag_value, std::complex<double> const& diag_value,
    std::complex<double>** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue)
{
#if defined( BLAS_HAVE_CUBLAS )
    batch::tzset(
        uplo, m, n,
        make_cuDoubleComplex( offdiag_value.real(), offdiag_value.imag() ),
        make_cuDoubleComplex( diag_value.real(), diag_value.imag() ),
        (cuDoubleComplex**) Aarray, lda,
        batch_count, queue);

#elif defined( BLAS_HAVE_ROCBLAS )
    batch::tzset(
        uplo, m, n,
        make_hipDoubleComplex( offdiag_value.real(), offdiag_value.imag() ),
        make_hipDoubleComplex( diag_value.real(), diag_value.imag() ),
        (hipDoubleComplex**) Aarray, lda,
        batch_count, queue);
#endif
}

#if ! defined( SLATE_HAVE_DEVICE )
//----------------------------------------
// Specializations to allow compilation without CUDA or HIP.
template <>
void tzset(
    Uplo uplo,
    int64_t m, int64_t n,
    double const& offdiag_value, double const& diag_value,
    double** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue)
{
}

//----------------------------------------
template <>
void tzset(
    Uplo uplo,
    int64_t m, int64_t n,
    float const& offdiag_value, float const& diag_value,
    float** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue)
{
}
#endif // not SLATE_HAVE_DEVICE

} // namespace batch
} // namespace device


//==============================================================================
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

    // Define index ranges for regions of matrix.
    // Tiles in each region are all the same size.
    // Ranges begin : end are [ begin, end ), exclusive of end.
    // 0 is interior          [ 0 : mt-1, 0 : nt-1 ]
    // 1 is bottom row        [     mt-1, 0 : nt-1 ]
    // 2 is right  col        [ 0 : mt-1,     nt-1 ]
    // 3 is bottom-right tile [     mt-1,     nt-1 ]
    // 0-3 are for off-diagonal tiles.
    // 4-7 are the same as 0-3, respectively, but for diagonal tiles.
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
            firstprivate( device, irange, jrange, queue_index ) \
            firstprivate( offdiag_value, diag_value )
        {
            // temporarily, convert both into same layout
            // todo: this is in-efficient, because both matrices may have same layout already
            //       and possibly wrong, because an input matrix is being altered
            // todo: best, handle directly through the CUDA kernels
            auto layout = Layout::ColMajor;
            std::set<ij_tuple> A_tiles_set;

            if (A.uplo() == Uplo::Lower) {
                for (int64_t j = 0; j < A.nt(); ++j) {
                    for (int64_t i = j; i < A.mt(); ++i) {  // lower trapezoid
                        if (A.tileIsLocal(i, j) && device == A.tileDevice(i, j)) {
                            A_tiles_set.insert({i, j});
                        }
                    }
                }
            }
            else { // upper
                for (int64_t j = 0; j < A.nt(); ++j) {
                    for (int64_t i = 0; i <= j && i < A.mt(); ++i) {  // upper trapezoid
                        if (A.tileIsLocal(i, j) && device == A.tileDevice(i, j)) {
                            A_tiles_set.insert({i, j});
                        }
                    }
                }
            }
            A.tileGetForWriting(A_tiles_set, device, LayoutConvert(layout));

            scalar_t** a_array_host = A.array_host( device );
            scalar_t** a_array_dev  = A.array_device( device );

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
                            if (i != j
                                && A.tileIsLocal(i, j)
                                && device == A.tileDevice(i, j)) {

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
                            if (i != j
                                && A.tileIsLocal(i, j)
                                && device == A.tileDevice(i, j)) {

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
                            if (i == j
                                && A.tileIsLocal(i, j)
                                && device == A.tileDevice(i, j)) {

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
                            if (i == j
                                && A.tileIsLocal(i, j)
                                && device == A.tileDevice(i, j)) {

                                a_array_host[batch_count] = A(i, j, device).data();
                                lda[q] = A(i, j, device).stride();
                                ++group_count[q];
                                ++batch_count;
                            }
                        }
                    }
                }
            }

            blas::Queue* queue = A.compute_queue(device, queue_index);

            blas::device_memcpy<scalar_t*>(
                a_array_dev, a_array_host, batch_count,
                blas::MemcpyKind::HostToDevice, *queue);

            for (int q = 0; q < 4; ++q) {
                if (group_count[q] > 0) {
                    device::batch::geset(
                        mb[q], nb[q],
                        offdiag_value, offdiag_value,
                        a_array_dev, lda[q],
                        group_count[q], *queue);
                    a_array_dev += group_count[q];
                }
            }
            for (int q = 4; q < 8; ++q) {
                if (group_count[q] > 0) {
                    device::batch::tzset(
                        A.uplo(), mb[q], nb[q],
                        offdiag_value, diag_value,
                        a_array_dev, lda[q],
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
