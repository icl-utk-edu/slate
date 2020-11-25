// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
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
void tzcopy(
    Uplo uplo,
    int64_t m, int64_t n,
    std::complex<float>** Aarray, int64_t lda,
    std::complex<float>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
#if !defined(SLATE_NO_CUDA)
    tzcopy(uplo,
           m, n,
           (cuFloatComplex**) Aarray, lda,
           (cuFloatComplex**) Barray, ldb,
           batch_count, queue);
#endif
}

template <>
void tzcopy(
    Uplo uplo,
    int64_t m, int64_t n,
    std::complex<float>** Aarray, int64_t lda,
    std::complex<double>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
#if !defined(SLATE_NO_CUDA)
    tzcopy(uplo,
           m, n,
           (cuFloatComplex**) Aarray, lda,
           (cuDoubleComplex**) Barray, ldb,
           batch_count, queue);
#endif
}

template <>
void tzcopy(
    Uplo uplo,
    int64_t m, int64_t n,
    std::complex<double>** Aarray, int64_t lda,
    std::complex<double>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
#if !defined(SLATE_NO_CUDA)
    tzcopy(uplo,
           m, n,
           (cuDoubleComplex**) Aarray, lda,
           (cuDoubleComplex**) Barray, ldb,
           batch_count, queue);
#endif
}

template <>
void tzcopy(
    Uplo uplo,
    int64_t m, int64_t n,
    std::complex<double>** Aarray, int64_t lda,
    std::complex<float>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
#if !defined(SLATE_NO_CUDA)
    tzcopy(uplo,
           m, n,
           (cuDoubleComplex**) Aarray, lda,
           (cuFloatComplex**) Barray, ldb,
           batch_count, queue);
#endif
}

//---------------------------------------------------
#if defined(SLATE_NO_CUDA)
// Specializations to allow compilation without CUDA.
template <>
void tzcopy(
    Uplo uplo,
    int64_t m, int64_t n,
    double** Aarray, int64_t lda,
    double** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
}

template <>
void tzcopy(
    Uplo uplo,
    int64_t m, int64_t n,
    double** Aarray, int64_t lda,
    float** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
}

template <>
void tzcopy(
    Uplo uplo,
    int64_t m, int64_t n,
    float** Aarray, int64_t lda,
    float** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
}
template <>
void tzcopy(
    Uplo uplo,
    int64_t m, int64_t n,
    float** Aarray, int64_t lda,
    double** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
}
#endif // not SLATE_NO_CUDA

} // namespace device

namespace internal {

//------------------------------------------------------------------------------
/// Copy and precision conversion.
/// Dispatches to target implementations.
/// @ingroup copy_internal
///
template <Target target, typename src_scalar_t, typename dst_scalar_t>
void copy(BaseTrapezoidMatrix<src_scalar_t>&& A,
          BaseTrapezoidMatrix<dst_scalar_t>&& B,
          int priority)
{
    copy(internal::TargetType<target>(),
         A, B,
         priority);
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
          int priority)
{
    // trace::Block trace_block("copy");

    slate_error_if(A.uplo() != B.uplo());
    bool lower = (B.uplo() == Uplo::Lower);

    assert(A.mt() == B.mt());
    assert(A.nt() == B.nt());

    for (int64_t j = 0; j < B.nt(); ++j) {
        if (j < B.mt() && B.tileIsLocal(j, j)) {
            A.tileGetForReading(j, j, LayoutConvert::None);
            B.tileGetForWriting(j, j, LayoutConvert::None);
            tzcopy(A(j, j), B(j, j));
            B.tileLayout(j, j, A.tileLayout(j, j));
            A.tileTick(j, j);
        }
        if (lower) {
            for (int64_t i = j+1; i < B.mt(); ++i) {
                if (B.tileIsLocal(i, j)) {
                    #pragma omp task shared(A, B) priority(priority)
                    {
                        A.tileGetForReading(i, j, LayoutConvert::None);
                        B.tileGetForWriting(i, j, LayoutConvert::None);
                        gecopy(A(i, j), B(i, j));
                        B.tileLayout(i, j, A.tileLayout(i, j));
                        A.tileTick(i, j);// TODO is this correct here?
                    }
                }
            }
        }
        else { // Uplo::Upper
            for (int64_t i = 0; i < j && i < B.mt(); ++i) {
                if (B.tileIsLocal(i, j)) {
                    #pragma omp task shared(A, B) priority(priority)
                    {
                        A.tileGetForReading(i, j, LayoutConvert::None);
                        B.tileGetForWriting(i, j, LayoutConvert::None);
                        gecopy(A(i, j), B(i, j));
                        B.tileLayout(i, j, A.tileLayout(i, j));
                        A.tileTick(i, j);// TODO is this correct here?
                    }
                }
            }
        }
    }

    #pragma omp taskwait
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
          int priority)
{
    using ij_tuple = typename BaseMatrix<src_scalar_t>::ij_tuple;
    slate_error_if(A.uplo() != B.uplo());
    bool lower = (B.uplo() == Uplo::Lower);

    // Define index ranges for quadrants of matrix.
    // Tiles in each quadrant are all the same size.
    int64_t irange[6][2] = {
        // off-diagonal
        { 0,        B.mt()-1 },
        { B.mt()-1, B.mt()   },
        { 0,        B.mt()-1 },
        { B.mt()-1, B.mt()   },
        // diagonal
        { 0,                          std::min(B.mt(), B.nt())-1 },
        { std::min(B.mt(), B.nt())-1, std::min(B.mt(), B.nt())   }
    };
    int64_t jrange[6][2] = {
        // off-diagonal
        { 0,        B.nt()-1 },
        { 0,        B.nt()-1 },
        { B.nt()-1, B.nt()   },
        { B.nt()-1, B.nt()   },
        // diagonal
        { 0,                          std::min(B.mt(), B.nt())-1 },
        { std::min(B.mt(), B.nt())-1, std::min(B.mt(), B.nt())   }
    };

    for (int device = 0; device < B.num_devices(); ++device) {
        #pragma omp task shared(A, B) priority(priority)
        {
            std::set<ij_tuple> A_tiles_set;
            for (int64_t i = 0; i < B.mt(); ++i) {
                for (int64_t j = 0; j < B.nt(); ++j) {
                    if (B.tileIsLocal(i, j) &&
                        device == B.tileDevice(i, j) &&
                        ( (  lower && i >= j) ||
                          (! lower && i <= j) ) )
                    {
                        A_tiles_set.insert({i, j});
                        // no need to convert layout
                        B.tileAcquire(i, j, device, A(i, j).layout());
                    }
                }
            }
            A.tileGetForReading(A_tiles_set, device, LayoutConvert::None);

            // Usually the output matrix (B) provides all the batch arrays.
            // Here we are using A, because of the differen types.
            src_scalar_t** a_array_host = A.array_host(device);
            dst_scalar_t** b_array_host = B.array_host(device);

            int64_t batch_count = 0;
            int64_t mb[6], nb[6], lda[6], ldb[6], group_count[6];
            // off-diagonal blocks
            for (int q = 0; q < 4; ++q) {
                group_count[q] = 0;
                lda[q] = 0;
                ldb[q] = 0;
                mb[q] = B.tileMb(irange[q][0]);
                nb[q] = B.tileNb(jrange[q][0]);
                for (int64_t i = irange[q][0]; i < irange[q][1]; ++i) {
                    for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                        if (B.tileIsLocal(i, j) &&
                            device == B.tileDevice(i, j) &&
                            ( (  lower && i > j) ||
                              (! lower && i < j) ) )
                        {
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
            // diagonal blocks
            for (int q = 4; q < 6; ++q) {
                group_count[q] = 0;
                lda[q] = 0;
                ldb[q] = 0;
                mb[q] = B.tileMb(irange[q][0]);
                nb[q] = B.tileNb(jrange[q][0]);
                for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                    if (B.tileIsLocal(j, j) &&
                        device == B.tileDevice(j, j) )
                    {
                        a_array_host[batch_count] = A(j, j, device).data();
                        b_array_host[batch_count] = B(j, j, device).data();
                        lda[q] = A(j, j, device).stride();
                        ldb[q] = B(j, j, device).stride();
                        ++group_count[q];
                        ++batch_count;
                    }
                }
            }

            // Usually the output matrix (B) provides all the batch arrays.
            // Here we are using A, because of the differen types.
            src_scalar_t** a_array_dev = A.array_device(device);
            dst_scalar_t** b_array_dev = B.array_device(device);

            const int batch_arrays_index = 0;
            blas::Queue* queue = A.queue(device, batch_arrays_index);
            blas::set_device(device);

            blas::device_memcpy<src_scalar_t*>((void*)a_array_dev, (void*)a_array_host,
                                batch_count,
                                cudaMemcpyHostToDevice,
                                *queue);

            blas::device_memcpy<dst_scalar_t*>((void*)b_array_dev, (void*)b_array_host,
                                batch_count,
                                cudaMemcpyHostToDevice,
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
            for (int q = 4; q < 6; ++q) {
                if (group_count[q] > 0) {
                    device::tzcopy(B.uplo(),
                                   mb[q], nb[q],
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
                    if (B.tileIsLocal(i, j) &&
                        device == B.tileDevice(i, j) &&
                        ( (  lower && i >= j) ||
                          (! lower && i <= j) ) )
                    {
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

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void copy<Target::HostTask, float, float>(
    BaseTrapezoidMatrix<float>&& A,
    BaseTrapezoidMatrix<float>&& B,
    int priority);

template
void copy<Target::HostTask, float, double>(
    BaseTrapezoidMatrix<float>&& A,
    BaseTrapezoidMatrix<double>&& B,
    int priority);

template
void copy<Target::Devices, float, float>(
    BaseTrapezoidMatrix<float>&& A,
    BaseTrapezoidMatrix<float>&& B,
    int priority);

template
void copy<Target::Devices, float, double>(
    BaseTrapezoidMatrix<float>&& A,
    BaseTrapezoidMatrix<double>&& B,
    int priority);

// ----------------------------------------
template
void copy<Target::HostTask, double, double>(
    BaseTrapezoidMatrix<double>&& A,
    BaseTrapezoidMatrix<double>&& B,
    int priority);

template
void copy<Target::HostTask, double, float>(
    BaseTrapezoidMatrix<double>&& A,
    BaseTrapezoidMatrix<float>&& B,
    int priority);

template
void copy<Target::Devices, double, double>(
    BaseTrapezoidMatrix<double>&& A,
    BaseTrapezoidMatrix<double>&& B,
    int priority);

template
void copy<Target::Devices, double, float>(
    BaseTrapezoidMatrix<double>&& A,
    BaseTrapezoidMatrix<float>&& B,
    int priority);

// ----------------------------------------
template
void copy< Target::HostTask, std::complex<float>, std::complex<float> >(
    BaseTrapezoidMatrix< std::complex<float> >&& A,
    BaseTrapezoidMatrix< std::complex<float> >&& B,
    int priority);

template
void copy< Target::HostTask, std::complex<float>, std::complex<double> >(
    BaseTrapezoidMatrix< std::complex<float> >&& A,
    BaseTrapezoidMatrix< std::complex<double> >&& B,
    int priority);

template
void copy< Target::Devices, std::complex<float>, std::complex<float>  >(
    BaseTrapezoidMatrix< std::complex<float> >&& A,
    BaseTrapezoidMatrix< std::complex<float> >&& B,
    int priority);

template
void copy< Target::Devices, std::complex<float>, std::complex<double>  >(
    BaseTrapezoidMatrix< std::complex<float> >&& A,
    BaseTrapezoidMatrix< std::complex<double> >&& B,
    int priority);

// ----------------------------------------
template
void copy< Target::HostTask, std::complex<double>, std::complex<double> >(
    BaseTrapezoidMatrix< std::complex<double> >&& A,
    BaseTrapezoidMatrix< std::complex<double> >&& B,
    int priority);

template
void copy< Target::HostTask, std::complex<double>, std::complex<float> >(
    BaseTrapezoidMatrix< std::complex<double> >&& A,
    BaseTrapezoidMatrix< std::complex<float> >&& B,
    int priority);

template
void copy< Target::Devices, std::complex<double>, std::complex<double> >(
    BaseTrapezoidMatrix< std::complex<double> >&& A,
    BaseTrapezoidMatrix< std::complex<double> >&& B,
    int priority);

template
void copy< Target::Devices, std::complex<double>, std::complex<float> >(
    BaseTrapezoidMatrix< std::complex<double> >&& A,
    BaseTrapezoidMatrix< std::complex<float> >&& B,
    int priority);

} // namespace internal
} // namespace slate
