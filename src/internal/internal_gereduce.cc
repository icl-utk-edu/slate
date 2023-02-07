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
void gereduce(
    int64_t m, int64_t n, int64_t mt,
    std::complex<float> alpha, std::complex<float>** Aarray, int64_t lda,
    std::complex<float> beta,  std::complex<float>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
#if defined( BLAS_HAVE_CUBLAS )
    gereduce(m, n, mt,
          make_cuFloatComplex(alpha.real(), alpha.imag()),
          (cuFloatComplex**) Aarray, lda,
          make_cuFloatComplex(beta.real(), beta.imag()),
          (cuFloatComplex**) Barray, ldb,
          batch_count, queue);

#elif defined( BLAS_HAVE_ROCBLAS )
    gereduce(m, n, mt,
          make_hipFloatComplex(alpha.real(), alpha.imag()),
          (hipFloatComplex**) Aarray, lda,
          make_hipFloatComplex(beta.real(), beta.imag()),
          (hipFloatComplex**) Barray, ldb,
          batch_count, queue);
#endif
}

template <>
void gereduce(
    int64_t m, int64_t n, int64_t mt,
    std::complex<double> alpha, std::complex<double>** Aarray, int64_t lda,
    std::complex<double> beta,  std::complex<double>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
#if defined( BLAS_HAVE_CUBLAS )
    gereduce(m, n, mt,
          make_cuDoubleComplex(alpha.real(), alpha.imag()),
          (cuDoubleComplex**) Aarray, lda,
          make_cuDoubleComplex(beta.real(), beta.imag()),
          (cuDoubleComplex**) Barray, ldb,
          batch_count, queue);

#elif defined( BLAS_HAVE_ROCBLAS )
    gereduce(m, n, mt,
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
void gereduce(
    int64_t m, int64_t n, int64_t mt,
    double alpha, double** Aarray, int64_t lda,
    double beta,  double** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
}

template <>
void gereduce(
    int64_t m, int64_t n, int64_t mt,
    float alpha, float** Aarray, int64_t lda,
    float beta,  float** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
}
#endif // not SLATE_HAVE_DEVICE

} // namespace device

namespace internal {

//------------------------------------------------------------------------------
/// General matrix reduction.
/// Compute A0 = (sum_i Ai), i = 1:mt-1.
/// Where A0 is the first row of tiles A(0, 1:nt-1),
/// and Ai is A(i, 1:nt-1).
/// Dispatches to target implementations.
/// @ingroup add_internal
///
template <Target target, typename scalar_t>
void gereduce(
         scalar_t alpha, Matrix<scalar_t>&& A,
         scalar_t beta,  Matrix<scalar_t>&& B,
         int priority, int queue_index)
{
    gereduce(internal::TargetType<target>(),
        alpha, A,
        beta,  B,
        priority, queue_index);
}

//------------------------------------------------------------------------------
/// General matrix reduction.
/// Host OpenMP task implementation.
/// @ingroup add_internal
///
template <typename scalar_t>
void gereduce(internal::TargetType<Target::HostTask>,
           scalar_t alpha, Matrix<scalar_t>& A,
           scalar_t beta,  Matrix<scalar_t>& B,
           int priority, int queue_index)
{
    int64_t A_mt = A.mt();
    int64_t A_nt = A.nt();

    #pragma omp taskgroup
    for (int64_t j = 0; j < A_nt; ++j) {
        if (B.tileIsLocal(0, j)) {
            #pragma omp task slate_omp_default_none \
                shared( A, B ) \
                firstprivate( j, alpha, beta, A_mt )  priority(priority)
            {
                for (int64_t i = 0; i < A_mt; ++i) {
                    if (A.tileIsLocal(i, j)) {
                        A.tileGetForReading(i, j, LayoutConvert::None);
                        B.tileGetForWriting(0, j, LayoutConvert::None);
                        tile::add(
                                alpha, A(i, j),
                                beta,  B(0, j) );
                        beta = 1.0;
                        A.tileTick(i, j);
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void gereduce(internal::TargetType<Target::HostNest>,
           scalar_t alpha, Matrix<scalar_t>& A,
           scalar_t beta,  Matrix<scalar_t>& B,
           int priority, int queue_index)
{
    gereduce( internal::TargetType<Target::HostTask>(),
        alpha, A, beta, B, priority, queue_index );
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void gereduce(internal::TargetType<Target::HostBatch>,
           scalar_t alpha, Matrix<scalar_t>& A,
           scalar_t beta,  Matrix<scalar_t>& B,
           int priority, int queue_index)
{
    gereduce( internal::TargetType<Target::HostTask>(),
        alpha, A, beta, B, priority, queue_index );
}

//------------------------------------------------------------------------------
/// General matrix reduction.
/// General matrix reduction.
/// GPU device implementation.
/// @ingroup add_internal
///
template <typename scalar_t>
void gereduce(internal::TargetType<Target::Devices>,
           scalar_t alpha, Matrix<scalar_t>& A,
           scalar_t beta,  Matrix<scalar_t>& B,
           int priority, int queue_index)
{
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    // two groups only
    // group for the submatrix (0: mt-1, 0: nt-2)
    // group for the last column (0:mt-1, nt-1)
    int64_t irange[2][2] = {
        { 0,        A.mt() },
        { 0,        A.mt() },
    };
    int64_t jrange[2][2] = {
        { 0,        A.nt()-1 },
        { A.nt()-1, A.nt()   },
    };

    #pragma omp taskgroup
    for (int device = 0; device < B.num_devices(); ++device) {
        #pragma omp task shared(A) \
            firstprivate(device, irange, jrange, queue_index, alpha) priority(priority)
        {
            auto layout = Layout::ColMajor;
            std::set<ij_tuple> A_tiles_set, B_tiles_set;

            // A will be reduced into B.
            // B = alpha x A + beta x B
            for (int64_t i = 0; i < A.mt(); ++i) {
                for (int64_t j = 0; j < A.nt(); ++j) {
                    if (B.tileIsLocal(0, j) && device == B.tileDevice(0, j)) {
                        if (A.tileIsLocal(i, j) && device == A.tileDevice(i, j)) {
                            A_tiles_set.insert({i, j});
                        }
                    }
                }
            }

            // B is one row of tiles.
            // B = alpha x A + beta x B
            for (int64_t j = 0; j < B.nt(); ++j) {
                if (B.tileIsLocal(0, j) && device == B.tileDevice(0, j)) {
                    B_tiles_set.insert({0, j});
                }
            }

            #pragma omp taskgroup
            {
                #pragma omp task slate_omp_default_none \
                    shared( A, A_tiles_set ) \
                    firstprivate(device, layout)
                {
                    A.tileGetForReading(A_tiles_set, device, LayoutConvert(layout));
                }
                #pragma omp task slate_omp_default_none \
                    shared( B, B_tiles_set ) \
                    firstprivate(device, layout)
                {
                    B.tileGetForWriting(B_tiles_set, device, LayoutConvert(layout));
                }
            }

            int64_t batch_size = B_tiles_set.size();
            scalar_t** b_array_host = B.array_host(device, queue_index);
            scalar_t** a_array_host = b_array_host + B_tiles_set.size();

            int64_t first_j = 0;
            int64_t batch_count = 0;
            int64_t batch_count_A = 0;
            int64_t mb[2], nb[2], lda[2], ldb[2], group_count[2], group_count_A[2];
            int64_t mt[2];
            // todo: change this to one group
            for (int q = 0; q < 2; ++q) {
                group_count[q] = 0;
                group_count_A[q] = 0;
                lda[q] = 0;
                ldb[q] = 0;
                mb[q] = B.tileMb(irange[q][0]);
                nb[q] = B.tileNb(jrange[q][0]);
                mt[q] = 0;

                for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                    if (B.tileIsLocal(0, j) && device == B.tileDevice(0, j)) {
                        b_array_host[batch_count] = B(0, j, device).data();
                        ++group_count[q];
                        ++batch_count;
                        ldb[q] = B(0, j, device).stride();
                    }
                }

                for (int64_t i = irange[q][0]; i < irange[q][1]; ++i) {
                    first_j = 0;
                    for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                        if (B.tileIsLocal(0, j) && device == B.tileDevice(0, j)) {
                            if (A.tileIsLocal(i, j) && device == A.tileDevice(i, j)) {
                                a_array_host[batch_count_A] = A(i, j, device).data();
                                lda[q] = A(i, j, device).stride();
                                ++group_count_A[q];
                                ++batch_count_A;
                                if (first_j == 0) {
                                    ++mt[q];
                                }
                                ++first_j;
                            }
                        }
                    }
                }
            }
            slate_assert(batch_count == batch_size);

            scalar_t** b_array_dev = B.array_device(device, queue_index);
            scalar_t** a_array_dev = b_array_dev + B_tiles_set.size();

            blas::Queue* queue = B.compute_queue(device, queue_index);
            blas::set_device( queue->device() );

            blas::device_memcpy<scalar_t*>(b_array_dev, b_array_host,
                                batch_count+A_tiles_set.size(),
                                blas::MemcpyKind::HostToDevice,
                                *queue);

            for (int q = 0; q < 2; ++q) {
                if (group_count[q] > 0 && mt[q] > 0) {
                    device::gereduce(mb[q], nb[q], mt[q],
                                  alpha, a_array_dev, lda[q],
                                  beta,  b_array_dev, ldb[q],
                                  group_count[q], *queue);
                    a_array_dev += group_count_A[q];
                    b_array_dev += group_count[q];
                }
            }

            queue->sync();

            for (int64_t i = 0; i < A.mt(); ++i) {
                for (int64_t j = 0; j < A.nt(); ++j) {
                    if (B.tileIsLocal(0, j) && device == B.tileDevice(0, j)) {
                        if (A.tileIsLocal(i, j) && device == A.tileDevice(i, j)) {
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
// ----------------------------------------
template
void gereduce<Target::HostTask, float>(
     float alpha, Matrix<float>&& A,
     float beta,  Matrix<float>&& B,
     int priority, int queue_index);

template
void gereduce<Target::HostNest, float>(
     float alpha, Matrix<float>&& A,
     float beta,  Matrix<float>&& B,
     int priority, int queue_index);

template
void gereduce<Target::HostBatch, float>(
     float alpha, Matrix<float>&& A,
     float beta,  Matrix<float>&& B,
     int priority, int queue_index);

template
void gereduce<Target::Devices, float>(
     float alpha, Matrix<float>&& A,
     float beta,  Matrix<float>&& B,
     int priority, int queue_index);

// ----------------------------------------
template
void gereduce<Target::HostTask, double>(
     double alpha, Matrix<double>&& A,
     double beta,  Matrix<double>&& B,
     int priority, int queue_index);

template
void gereduce<Target::HostNest, double>(
     double alpha, Matrix<double>&& A,
     double beta,  Matrix<double>&& B,
     int priority, int queue_index);

template
void gereduce<Target::HostBatch, double>(
     double alpha, Matrix<double>&& A,
     double beta,  Matrix<double>&& B,
     int priority, int queue_index);

template
void gereduce<Target::Devices, double>(
     double alpha, Matrix<double>&& A,
     double beta,  Matrix<double>&& B,
     int priority, int queue_index);

// ----------------------------------------
template
void gereduce< Target::HostTask, std::complex<float> >(
     std::complex<float> alpha, Matrix< std::complex<float> >&& A,
     std::complex<float> beta,  Matrix< std::complex<float> >&& B,
     int priority, int queue_index);

template
void gereduce< Target::HostNest, std::complex<float> >(
     std::complex<float> alpha, Matrix< std::complex<float> >&& A,
     std::complex<float> beta,  Matrix< std::complex<float> >&& B,
     int priority, int queue_index);

template
void gereduce< Target::HostBatch, std::complex<float> >(
     std::complex<float> alpha, Matrix< std::complex<float> >&& A,
     std::complex<float> beta,  Matrix< std::complex<float> >&& B,
     int priority, int queue_index);

template
void gereduce< Target::Devices, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
    std::complex<float> beta,  Matrix< std::complex<float> >&& B,
    int priority, int queue_index);

// ----------------------------------------
template
void gereduce< Target::HostTask, std::complex<double> >(
     std::complex<double> alpha, Matrix< std::complex<double> >&& A,
     std::complex<double> beta,  Matrix< std::complex<double> >&& B,
     int priority, int queue_index);

template
void gereduce< Target::HostNest, std::complex<double> >(
     std::complex<double> alpha, Matrix< std::complex<double> >&& A,
     std::complex<double> beta,  Matrix< std::complex<double> >&& B,
     int priority, int queue_index);

template
void gereduce< Target::HostBatch, std::complex<double> >(
     std::complex<double> alpha, Matrix< std::complex<double> >&& A,
     std::complex<double> beta,  Matrix< std::complex<double> >&& B,
     int priority, int queue_index);

template
void gereduce< Target::Devices, std::complex<double> >(
     std::complex<double> alpha, Matrix< std::complex<double> >&& A,
     std::complex<double> beta,  Matrix< std::complex<double> >&& B,
     int priority, int queue_index);

} // namespace internal
} // namespace slate
