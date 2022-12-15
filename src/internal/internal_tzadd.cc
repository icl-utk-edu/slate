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
void tzadd(
    Uplo uplo,
    int64_t m, int64_t n,
    std::complex<float> const& alpha, std::complex<float>** Aarray, int64_t lda,
    std::complex<float> const& beta, std::complex<float>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
#if defined( BLAS_HAVE_CUBLAS )
    tzadd(uplo, m, n,
          make_cuFloatComplex(alpha.real(), alpha.imag()),
          (cuFloatComplex**) Aarray, lda,
          make_cuFloatComplex(beta.real(), beta.imag()),
          (cuFloatComplex**) Barray, ldb,
          batch_count, queue);

#elif defined( BLAS_HAVE_ROCBLAS )
    tzadd(uplo, m, n,
          make_hipFloatComplex(alpha.real(), alpha.imag()),
          (hipFloatComplex**) Aarray, lda,
          make_hipFloatComplex(beta.real(), beta.imag()),
          (hipFloatComplex**) Barray, ldb,
          batch_count, queue);
#endif
}

template <>
void tzadd(
    Uplo uplo,
    int64_t m, int64_t n,
    std::complex<double> const& alpha, std::complex<double>** Aarray, int64_t lda,
    std::complex<double> const& beta, std::complex<double>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
#if defined( BLAS_HAVE_CUBLAS )
    tzadd(uplo, m, n,
          make_cuDoubleComplex(alpha.real(), alpha.imag()),
          (cuDoubleComplex**) Aarray, lda,
          make_cuDoubleComplex(beta.real(), beta.imag()),
          (cuDoubleComplex**) Barray, ldb,
          batch_count, queue);

#elif defined( BLAS_HAVE_ROCBLAS )
    tzadd(uplo, m, n,
          make_hipDoubleComplex(alpha.real(), alpha.imag()),
          (hipDoubleComplex**) Aarray, lda,
          make_hipDoubleComplex(beta.real(), beta.imag()),
          (hipDoubleComplex**) Barray, ldb,
          batch_count, queue);
#endif
}

#if ! defined( SLATE_HAVE_DEVICE )
// Specializations to allow compilation without CUDA or HIP.
template <>
void tzadd(
    Uplo uplo,
    int64_t m, int64_t n,
    double const& alpha, double** Aarray, int64_t lda,
    double const& beta, double** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
}

template <>
void tzadd(
    Uplo uplo,
    int64_t m, int64_t n,
    float const& alpha, float** Aarray, int64_t lda,
    float const& beta, float** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
}
#endif // not SLATE_HAVE_DEVICE

} // namespace device

namespace internal {

//------------------------------------------------------------------------------
/// Trapezoidal matrix add.
/// Dispatches to target implementations.
/// @ingroup add_internal
template <Target target, typename scalar_t>
void add(scalar_t alpha, BaseTrapezoidMatrix<scalar_t>&& A,
         scalar_t beta, BaseTrapezoidMatrix<scalar_t>&& B,
         int priority, int queue_index)
{
    add(internal::TargetType<target>(),
        alpha, A,
        beta,  B,
        priority, queue_index);
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
           int priority, int queue_index)
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
                        firstprivate(i, j, alpha, beta) priority(priority)
                    {
                        A.tileGetForReading(i, j, LayoutConvert::None);
                        B.tileGetForWriting(i, j, LayoutConvert::None);
                        tile::add(
                            alpha, A(i, j),
                            beta,  B(i, j) );
                        A.tileTick(i, j);
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
                        firstprivate(i, j, alpha, beta) priority(priority)
                    {
                        A.tileGetForReading(i, j, LayoutConvert::None);
                        B.tileGetForWriting(i, j, LayoutConvert::None);
                        tile::add(
                            alpha, A(i, j),
                            beta,  B(i, j) );
                        A.tileTick(i, j);
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
           int priority, int queue_index)
{
    slate_not_implemented("Target::HostNest isn't yet supported.");
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void add(internal::TargetType<Target::HostBatch>,
           scalar_t alpha, BaseTrapezoidMatrix<scalar_t>& A,
           scalar_t beta, BaseTrapezoidMatrix<scalar_t>& B,
           int priority, int queue_index)
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
           int priority, int queue_index)
{
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;
    slate_error_if(A.uplo() != B.uplo());

    // Define index ranges for regions of matrix.
    // Tiles in each region are all the same size.
    int64_t irange[4][2] = {
        { 0,        B.mt()-1 },
        { B.mt()-1, B.mt()   },
        { 0,        B.mt()-1 },
        { B.mt()-1, B.mt()   }
    };
    int64_t jrange[4][2] = {
        { 0,        B.nt()-1 },
        { 0,        B.nt()-1 },
        { B.nt()-1, B.nt()   },
        { B.nt()-1, B.nt()   }
    };

    #pragma omp taskgroup
    for (int device = 0; device < B.num_devices(); ++device) {
        #pragma omp task shared(A, B) priority(priority) \
            firstprivate(device, irange, jrange, queue_index, alpha, beta)
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
            scalar_t** a_array_host = B.array_host(device);
            scalar_t** b_array_host = a_array_host + batch_size;

            int64_t batch_count = 0;
            int64_t mb[8], nb[8], lda[8], ldb[8], group_count[8];
            for (int q = 0; q < 4; ++q) {
                group_count[q] = 0;
                lda[q] = 0;
                ldb[q] = 0;
                mb[q] = B.tileMb(irange[q][0]);
                nb[q] = B.tileNb(jrange[q][0]);
                if (B.uplo() == Uplo::Lower) {
                    for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                        for (int64_t i = std::max(j, irange[q][0]); i < irange[q][1]; ++i) {
                            if (i != j && B.tileIsLocal(i, j) && device == B.tileDevice(i, j)) {
                                a_array_host[batch_count] = A(i, j, device).data();
                                b_array_host[batch_count] = B(i, j, device).data();
                                lda[q] = A(i, j, device).stride();
                                ldb[q] = B(i, j, device).stride();
                                ++group_count[q];
                                ++batch_count;
                            }
                        }
                    }
                }
                else { // upper
                    for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                        for (int64_t i = irange[q][0]; i < irange[q][1] && i <= j; ++i) {
                            if (i != j && B.tileIsLocal(i, j) && device == B.tileDevice(i, j)) {
                                a_array_host[batch_count] = A(i, j, device).data();
                                b_array_host[batch_count] = B(i, j, device).data();
                                lda[q] = A(i, j, device).stride();
                                ldb[q] = B(i, j, device).stride();
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
                ldb[q] = 0;
                mb[q] = B.tileMb(irange[q-4][0]);
                nb[q] = B.tileNb(jrange[q-4][0]);
                if (B.uplo() == Uplo::Lower) {
                    for (int64_t j = jrange[q-4][0]; j < jrange[q-4][1]; ++j) {
                        for (int64_t i = std::max(j, irange[q-4][0]); i < irange[q-4][1]; ++i) {
                            if (i == j && B.tileIsLocal(i, j) && device == B.tileDevice(i, j)) {
                               a_array_host[batch_count] = A(i, j, device).data();
                               b_array_host[batch_count] = B(i, j, device).data();
                               lda[q] = A(i, j, device).stride();
                               ldb[q] = B(i, j, device).stride();
                               ++group_count[q];
                               ++batch_count;
                            }
                        }
                    }
                }
                else { //upper
                    for (int64_t j = jrange[q-4][0]; j < jrange[q-4][1]; ++j) {
                        for (int64_t i = irange[q-4][0]; i < irange[q-4][1] && i <= j; ++i) {
                            if (i == j && B.tileIsLocal(i, j) && device == B.tileDevice(i, j)) {
                               a_array_host[batch_count] = A(i, j, device).data();
                               b_array_host[batch_count] = B(i, j, device).data();
                               lda[q] = A(i, j, device).stride();
                               ldb[q] = B(i, j, device).stride();
                               ++group_count[q];
                               ++batch_count;
                            }
                        }
                    }
                }
            }

            slate_assert(batch_count == batch_size);

            scalar_t** a_array_dev = B.array_device(device);
            scalar_t** b_array_dev = a_array_dev + batch_size;

            blas::Queue* queue = A.compute_queue(device, queue_index);

            blas::device_memcpy<scalar_t*>(a_array_dev, a_array_host,
                                batch_count*2,
                                blas::MemcpyKind::HostToDevice,
                                *queue);

            for (int q = 0; q < 4; ++q) {
                if (group_count[q] > 0) {
                    device::batch::geadd(mb[q], nb[q],
                                  alpha, a_array_dev, lda[q],
                                  beta,  b_array_dev, ldb[q],
                                  group_count[q], *queue);
                    a_array_dev += group_count[q];
                    b_array_dev += group_count[q];
                }
            }
            for (int q = 4; q < 8; ++q) {
                if (group_count[q] > 0) {
                    device::tzadd(B.uplo(), mb[q], nb[q],
                                  alpha, a_array_dev, lda[q],
                                  beta,  b_array_dev, ldb[q],
                                  group_count[q], *queue);
                    a_array_dev += group_count[q];
                    b_array_dev += group_count[q];
                }
            }

            queue->sync();

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
    // end omp taskgroup
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void add<Target::HostTask, float>(
     float alpha, BaseTrapezoidMatrix<float>&& A,
     float beta, BaseTrapezoidMatrix<float>&& B,
     int priority, int queue_index);

template
void add<Target::HostNest, float>(
     float alpha, BaseTrapezoidMatrix<float>&& A,
     float beta, BaseTrapezoidMatrix<float>&& B,
     int priority, int queue_index);

template
void add<Target::HostBatch, float>(
     float alpha, BaseTrapezoidMatrix<float>&& A,
     float beta, BaseTrapezoidMatrix<float>&& B,
     int priority, int queue_index);

template
void add<Target::Devices, float>(
     float alpha, BaseTrapezoidMatrix<float>&& A,
     float beta, BaseTrapezoidMatrix<float>&& B,
     int priority, int queue_index);

// ----------------------------------------
template
void add<Target::HostTask, double>(
     double alpha, BaseTrapezoidMatrix<double>&& A,
     double beta, BaseTrapezoidMatrix<double>&& B,
     int priority, int queue_index);

template
void add<Target::HostNest, double>(
     double alpha, BaseTrapezoidMatrix<double>&& A,
     double beta, BaseTrapezoidMatrix<double>&& B,
     int priority, int queue_index);

template
void add<Target::HostBatch, double>(
     double alpha, BaseTrapezoidMatrix<double>&& A,
     double beta, BaseTrapezoidMatrix<double>&& B,
     int priority, int queue_index);

template
void add<Target::Devices, double>(
     double alpha, BaseTrapezoidMatrix<double>&& A,
     double beta, BaseTrapezoidMatrix<double>&& B,
     int priority, int queue_index);

// ----------------------------------------
template
void add< Target::HostTask, std::complex<float> >(
     std::complex<float> alpha, BaseTrapezoidMatrix< std::complex<float> >&& A,
     std::complex<float>  beta, BaseTrapezoidMatrix< std::complex<float> >&& B,
     int priority, int queue_index);

template
void add< Target::HostNest, std::complex<float> >(
     std::complex<float> alpha, BaseTrapezoidMatrix< std::complex<float> >&& A,
     std::complex<float>  beta, BaseTrapezoidMatrix< std::complex<float> >&& B,
     int priority, int queue_index);

template
void add< Target::HostBatch, std::complex<float> >(
     std::complex<float> alpha, BaseTrapezoidMatrix< std::complex<float> >&& A,
     std::complex<float>  beta, BaseTrapezoidMatrix< std::complex<float> >&& B,
     int priority, int queue_index);

template
void add< Target::Devices, std::complex<float> >(
    std::complex<float> alpha, BaseTrapezoidMatrix< std::complex<float> >&& A,
    std::complex<float>  beta, BaseTrapezoidMatrix< std::complex<float> >&& B,
    int priority, int queue_index);

// ----------------------------------------
template
void add< Target::HostTask, std::complex<double> >(
     std::complex<double> alpha, BaseTrapezoidMatrix< std::complex<double> >&& A,
     std::complex<double> beta, BaseTrapezoidMatrix< std::complex<double> >&& B,
     int priority, int queue_index);

template
void add< Target::HostNest, std::complex<double> >(
     std::complex<double> alpha, BaseTrapezoidMatrix< std::complex<double> >&& A,
     std::complex<double> beta, BaseTrapezoidMatrix< std::complex<double> >&& B,
     int priority, int queue_index);

template
void add< Target::HostBatch, std::complex<double> >(
     std::complex<double> alpha, BaseTrapezoidMatrix< std::complex<double> >&& A,
     std::complex<double> beta, BaseTrapezoidMatrix< std::complex<double> >&& B,
     int priority, int queue_index);

template
void add< Target::Devices, std::complex<double> >(
     std::complex<double> alpha, BaseTrapezoidMatrix< std::complex<double> >&& A,
     std::complex<double> beta, BaseTrapezoidMatrix< std::complex<double> >&& B,
     int priority, int queue_index);

} // namespace internal
} // namespace slate
