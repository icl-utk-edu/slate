// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
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

template <>
void gecopy(
    int64_t m, int64_t n,
    std::complex<float> const* const* Aarray, int64_t lda,
    std::complex<float>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
#if defined( BLAS_HAVE_CUBLAS )
    gecopy(m, n,
           (cuFloatComplex**) Aarray, lda,
           (cuFloatComplex**) Barray, ldb,
           batch_count, queue);

#elif defined( BLAS_HAVE_ROCBLAS )
    gecopy(m, n,
           (hipFloatComplex**) Aarray, lda,
           (hipFloatComplex**) Barray, ldb,
           batch_count, queue);
#endif
}

template <>
void gecopy(
    int64_t m, int64_t n,
    std::complex<float> const* const* Aarray, int64_t lda,
    std::complex<double>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
#if defined( BLAS_HAVE_CUBLAS )
    gecopy(m, n,
           (cuFloatComplex**) Aarray, lda,
           (cuDoubleComplex**) Barray, ldb,
           batch_count, queue);

#elif defined( BLAS_HAVE_ROCBLAS )
    gecopy(m, n,
           (hipFloatComplex**) Aarray, lda,
           (hipDoubleComplex**) Barray, ldb,
           batch_count, queue);
#endif
}

template <>
void gecopy(
    int64_t m, int64_t n,
    std::complex<double> const* const* Aarray, int64_t lda,
    std::complex<double>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
#if defined( BLAS_HAVE_CUBLAS )
    gecopy(m, n,
           (cuDoubleComplex**) Aarray, lda,
           (cuDoubleComplex**) Barray, ldb,
           batch_count, queue);

#elif defined( BLAS_HAVE_ROCBLAS )
    gecopy(m, n,
           (hipDoubleComplex**) Aarray, lda,
           (hipDoubleComplex**) Barray, ldb,
           batch_count, queue);
#endif
}

template <>
void gecopy(
    int64_t m, int64_t n,
    std::complex<double> const* const* Aarray, int64_t lda,
    std::complex<float>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
#if defined( BLAS_HAVE_CUBLAS )
    gecopy(m, n,
           (cuDoubleComplex**) Aarray, lda,
           (cuFloatComplex**) Barray, ldb,
           batch_count, queue);

#elif defined( BLAS_HAVE_ROCBLAS )
    gecopy(m, n,
           (hipDoubleComplex**) Aarray, lda,
           (hipFloatComplex**) Barray, ldb,
           batch_count, queue);
#endif
}

//---------------------------------------------------
#if ! defined( SLATE_HAVE_DEVICE )
// Specializations to allow compilation without CUDA or HIP.
template <>
void gecopy(
    int64_t m, int64_t n,
    double const* const* Aarray, int64_t lda,
    double** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
}

template <>
void gecopy(
    int64_t m, int64_t n,
    double const* const* Aarray, int64_t lda,
    float** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
}

template <>
void gecopy(
    int64_t m, int64_t n,
    float const* const* Aarray, int64_t lda,
    float** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
}
template <>
void gecopy(
    int64_t m, int64_t n,
    float const* const* Aarray, int64_t lda,
    double** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
}
#endif // not SLATE_HAVE_DEVICE

} // namespace device

namespace internal {

//------------------------------------------------------------------------------
/// Copy and precision conversion.
/// Dispatches to target implementations.
/// @ingroup copy_internal
///
template <Target target, typename src_scalar_t, typename dst_scalar_t>
void copy(Matrix<src_scalar_t>&& A,
          Matrix<dst_scalar_t>&& B,
          int priority, int queue_index)
{
    copy(internal::TargetType<target>(),
         A, B,
         priority, queue_index);
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
          int priority, int queue_index)
{
    // trace::Block trace_block("copy");

    int64_t A_mt = A.mt();
    int64_t A_nt = A.nt();
    assert(A_mt == B.mt());
    assert(A_nt == B.nt());

    #pragma omp taskgroup
    for (int64_t i = 0; i < A_mt; ++i) {
        for (int64_t j = 0; j < A_nt; ++j) {
            if (B.tileIsLocal(i, j)) {
                #pragma omp task slate_omp_default_none \
                    shared( A, B ) firstprivate( i, j, HostNum ) \
                    priority(priority)
                {
                    A.tileGetForReading(i, j, LayoutConvert::None);
                    // tileAcquire() to avoid un-needed copy
                    B.tileAcquire(i, j, A.tileLayout(i, j));
                    tile::gecopy( A(i, j), B(i, j) );
                    B.tileModified(i, j, HostNum, true);
                    A.tileTick(i, j);
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
          int priority, int queue_index)
{
    slate_not_implemented("Target::HostNest isn't yet supported.");
}

//------------------------------------------------------------------------------
template <typename src_scalar_t, typename dst_scalar_t>
void copy(internal::TargetType<Target::HostBatch>,
          Matrix<src_scalar_t>& A,
          Matrix<dst_scalar_t>& B,
          int priority, int queue_index)
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
          int priority, int queue_index)
{
    using ij_tuple = typename BaseMatrix<src_scalar_t>::ij_tuple;

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
        #pragma omp task slate_omp_default_none \
            shared( A, B ) \
            firstprivate(device, irange, jrange, queue_index) priority(priority)
        {
            std::set<ij_tuple> A_tiles_set;
            for (int64_t i = 0; i < B.mt(); ++i) {
                for (int64_t j = 0; j < B.nt(); ++j) {
                    if (B.tileIsLocal(i, j) && device == B.tileDevice(i, j)) {
                        A_tiles_set.insert({i, j});
                        // tileAcquire() instead to avoid un-needed copy
                        B.tileAcquire(i, j, device, Layout::ColMajor);
                        // copy local and remote tiles to CPU;
                        B.tileModified(i, j, device, true);
                    }
                }
            }
            // no need to convert layout.
            A.tileGetForReading(A_tiles_set, device, LayoutConvert::None);

            // Usually the output matrix (B) provides all the batch arrays.
            // Here we are using A, because of the possibly different types.
            src_scalar_t** a_array_host = A.array_host(device, queue_index);
            dst_scalar_t** b_array_host = B.array_host(device, queue_index);

            int64_t batch_count = 0;
            int64_t mb[4], nb[4], lda[4], ldb[4], group_count[4];
            for (int q = 0; q < 4; ++q) {
                group_count[q] = 0;
                lda[q] = 0;
                ldb[q] = 0;
                mb[q] = B.tileMb(irange[q][0]);
                nb[q] = B.tileNb(jrange[q][0]);
                for (int64_t i = irange[q][0]; i < irange[q][1]; ++i) {
                    for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                        if (B.tileIsLocal(i, j) && device == B.tileDevice(i, j)) {
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

            for (int q = 0; q < 4; ++q) {
                if (group_count[q] > 0) {
                    device::gecopy(mb[q], nb[q],
                                   a_array_dev, lda[q],
                                   b_array_dev, ldb[q],
                                   group_count[q], *queue);
                    a_array_dev += group_count[q];
                    b_array_dev += group_count[q];
                }
            }

            queue->sync();

            for (int64_t i = 0; i < B.mt(); ++i) {
                for (int64_t j = 0; j < B.nt(); ++j) {
                    if (B.tileIsLocal(i, j) && device == B.tileDevice(i, j)) {
                        B.tileModified(i, j, device);
                        // update output tile layout
                        // todo: what if extended?
                        B.tileLayout(i, j, device, A.tileLayout(i, j, device));
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

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void copy<Target::HostTask, float, float>(
    Matrix<float>&& A, Matrix<float>&& B,
    int priority, int queue_index);

template
void copy<Target::HostTask, float, double>(
    Matrix<float>&& A, Matrix<double>&& B,
    int priority, int queue_index);

template
void copy<Target::Devices, float, float>(
    Matrix<float>&& A, Matrix<float>&& B,
    int priority, int queue_index);

template
void copy<Target::Devices, float, double>(
    Matrix<float>&& A, Matrix<double>&& B,
    int priority, int queue_index);

// ----------------------------------------
template
void copy<Target::HostTask, double, double>(
    Matrix<double>&& A, Matrix<double>&& B,
    int priority, int queue_index);

template
void copy<Target::HostTask, double, float>(
    Matrix<double>&& A, Matrix<float>&& B,
    int priority, int queue_index);

template
void copy<Target::Devices, double, double>(
    Matrix<double>&& A, Matrix<double>&& B,
    int priority, int queue_index);

template
void copy<Target::Devices, double, float>(
    Matrix<double>&& A, Matrix<float>&& B,
    int priority, int queue_index);

// ----------------------------------------
template
void copy< Target::HostTask, std::complex<float>, std::complex<float> >(
    Matrix< std::complex<float> >&& A, Matrix< std::complex<float> >&& B,
    int priority, int queue_index);

template
void copy< Target::HostTask, std::complex<float>, std::complex<double> >(
    Matrix< std::complex<float> >&& A, Matrix< std::complex<double> >&& B,
    int priority, int queue_index);

template
void copy< Target::Devices, std::complex<float>, std::complex<float>  >(
    Matrix< std::complex<float> >&& A, Matrix< std::complex<float> >&& B,
    int priority, int queue_index);

template
void copy< Target::Devices, std::complex<float>, std::complex<double>  >(
    Matrix< std::complex<float> >&& A, Matrix< std::complex<double> >&& B,
    int priority, int queue_index);

// ----------------------------------------
template
void copy< Target::HostTask, std::complex<double>, std::complex<double> >(
    Matrix< std::complex<double> >&& A, Matrix< std::complex<double> >&& B,
    int priority, int queue_index);

template
void copy< Target::HostTask, std::complex<double>, std::complex<float> >(
    Matrix< std::complex<double> >&& A, Matrix< std::complex<float> >&& B,
    int priority, int queue_index);

template
void copy< Target::Devices, std::complex<double>, std::complex<double> >(
    Matrix< std::complex<double> >&& A, Matrix< std::complex<double> >&& B,
    int priority, int queue_index);

template
void copy< Target::Devices, std::complex<double>, std::complex<float> >(
    Matrix< std::complex<double> >&& A, Matrix< std::complex<float> >&& B,
    int priority, int queue_index);

// ----------------------------------------
template
void copy<Target::HostNest, float, float>(
    Matrix<float>&& A, Matrix<float>&& B,
    int priority, int queue_index);

template
void copy<Target::HostNest, float, double>(
    Matrix<float>&& A, Matrix<double>&& B,
    int priority, int queue_index);

template
void copy<Target::HostBatch, float, float>(
    Matrix<float>&& A, Matrix<float>&& B,
    int priority, int queue_index);

template
void copy<Target::HostBatch, float, double>(
    Matrix<float>&& A, Matrix<double>&& B,
    int priority, int queue_index);

// ----------------------------------------
template
void copy<Target::HostNest, double, double>(
    Matrix<double>&& A, Matrix<double>&& B,
    int priority, int queue_index);

template
void copy<Target::HostNest, double, float>(
    Matrix<double>&& A, Matrix<float>&& B,
    int priority, int queue_index);

template
void copy<Target::HostBatch, double, double>(
    Matrix<double>&& A, Matrix<double>&& B,
    int priority, int queue_index);

template
void copy<Target::HostBatch, double, float>(
    Matrix<double>&& A, Matrix<float>&& B,
    int priority, int queue_index);

// ----------------------------------------
template
void copy< Target::HostNest, std::complex<float>, std::complex<float> >(
    Matrix< std::complex<float> >&& A, Matrix< std::complex<float> >&& B,
    int priority, int queue_index);

template
void copy< Target::HostNest, std::complex<float>, std::complex<double> >(
    Matrix< std::complex<float> >&& A, Matrix< std::complex<double> >&& B,
    int priority, int queue_index);

template
void copy< Target::HostBatch, std::complex<float>, std::complex<float>  >(
    Matrix< std::complex<float> >&& A, Matrix< std::complex<float> >&& B,
    int priority, int queue_index);

template
void copy< Target::HostBatch, std::complex<float>, std::complex<double>  >(
    Matrix< std::complex<float> >&& A, Matrix< std::complex<double> >&& B,
    int priority, int queue_index);

// ----------------------------------------
template
void copy< Target::HostNest, std::complex<double>, std::complex<double> >(
    Matrix< std::complex<double> >&& A, Matrix< std::complex<double> >&& B,
    int priority, int queue_index);

template
void copy< Target::HostNest, std::complex<double>, std::complex<float> >(
    Matrix< std::complex<double> >&& A, Matrix< std::complex<float> >&& B,
    int priority, int queue_index);

template
void copy< Target::HostBatch, std::complex<double>, std::complex<double> >(
    Matrix< std::complex<double> >&& A, Matrix< std::complex<double> >&& B,
    int priority, int queue_index);

template
void copy< Target::HostBatch, std::complex<double>, std::complex<float> >(
    Matrix< std::complex<double> >&& A, Matrix< std::complex<float> >&& B,
    int priority, int queue_index);
} // namespace internal
} // namespace slate
