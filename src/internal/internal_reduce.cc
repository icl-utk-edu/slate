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
void reduce(
    int64_t m, int64_t n, int64_t mt,
    std::complex<float> alpha, std::complex<float>** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue)
{
#if defined( BLAS_HAVE_CUBLAS )
    reduce(m, n, mt,
          make_cuFloatComplex(alpha.real(), alpha.imag()),
          (cuFloatComplex**) Aarray, lda,
          batch_count, queue);

#elif defined( BLAS_HAVE_ROCBLAS )
    reduce(m, n, mt,
          make_hipFloatComplex(alpha.real(), alpha.imag()),
          (hipFloatComplex**) Aarray, lda,
          batch_count, queue);
#endif
}

template <>
void reduce(
    int64_t m, int64_t n, int64_t mt,
    std::complex<double> alpha, std::complex<double>** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue)
{
#if defined( BLAS_HAVE_CUBLAS )
    reduce(m, n, mt,
          make_cuDoubleComplex(alpha.real(), alpha.imag()),
          (cuDoubleComplex**) Aarray, lda,
          batch_count, queue);

#elif defined( BLAS_HAVE_ROCBLAS )
    reduce(m, n, mt,
          make_hipDoubleComplex(alpha.real(), alpha.imag()),
          (hipDoubleComplex**) Aarray, lda,
          batch_count, queue);
#endif
}

#if ! defined( SLATE_HAVE_DEVICE )
// Specializations to allow compilation without CUDA or HIP.
template <>
void reduce(
    int64_t m, int64_t n, int64_t mt,
    double alpha, double** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue)
{
}

template <>
void reduce(
    int64_t m, int64_t n, int64_t mt,
    float alpha, float** Aarray, int64_t lda,
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
void reduce(std::vector<int64_t>& row_indices,
         scalar_t alpha, Matrix<scalar_t>&& A,
         int priority, int queue_index)
{
    reduce(internal::TargetType<target>(),
        row_indices,
        alpha, A,
        priority, queue_index);
}

//------------------------------------------------------------------------------
/// General matrix reduction.
/// Host OpenMP task implementation.
/// @ingroup add_internal
///
/// todo: this function should just be named "add".
template <typename scalar_t>
void reduce(internal::TargetType<Target::HostTask>,
           std::vector<int64_t>& row_indices,
           scalar_t alpha, Matrix<scalar_t>& A,
           int priority, int queue_index)
{
    // trace::Block trace_block("add");

    int64_t A_mt = A.mt();
    int64_t A_nt = A.nt();

    int64_t i0 = row_indices[0];
    #pragma omp taskgroup
    //for (int64_t i = 0; i < A_mt; ++i) {
    for (int64_t row = 1; row < int64_t(row_indices.size()); ++row) {
        int64_t i = row_indices[row];
        for (int64_t j = 0; j < A_nt; ++j) {
            if (A.tileIsLocal(i, j)) {
                //#pragma omp task slate_omp_default_none \
                    shared( A ) \
                    firstprivate(i, j, alpha)  priority(priority)
                {
                    A.tileGetForReading(i, j, LayoutConvert::None);
                    A.tileGetForWriting(i0, j, LayoutConvert::None);
                    tile::add(
                        alpha, A(i, j),
                        alpha, A(i0, j) );
                    A.tileTick(i, j);
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// todo: this function should just be named "add".
template <typename scalar_t>
void reduce(internal::TargetType<Target::HostNest>,
           std::vector<int64_t>& row_indices,
           scalar_t alpha, Matrix<scalar_t>& A,
           int priority, int queue_index)
{
    slate_not_implemented("Target::HostNest isn't yet supported.");
}

//------------------------------------------------------------------------------
/// todo: this function should just be named "add".
template <typename scalar_t>
void reduce(internal::TargetType<Target::HostBatch>,
           std::vector<int64_t>& row_indices,
           scalar_t alpha, Matrix<scalar_t>& A,
           int priority, int queue_index)
{
    slate_not_implemented("Target::HostBatch isn't yet supported.");
}

//------------------------------------------------------------------------------
/// General matrix add.
/// General matrix reduction.
/// GPU device implementation.
/// @ingroup add_internal
///
/// todo: this function should just be named "add".
template <typename scalar_t>
void reduce(internal::TargetType<Target::Devices>,
           std::vector<int64_t>& row_indices,
           scalar_t alpha, Matrix<scalar_t>& B,
           int priority, int queue_index)
{
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    // two groups only
    // group for the submatrix (0: mt-1, 0: nt-2)
    // group for the last column (0:mt-1, nt-1)
    int64_t irange[2][2] = {
        { 0,        B.mt() },
        { 0,        B.mt() },
    };
    int64_t jrange[2][2] = {
        { 0,        B.nt()-1 },
        { B.nt()-1, B.nt()   },
    };

    #pragma omp taskgroup
    for (int device = 0; device < B.num_devices(); ++device) {
        #pragma omp task shared(B) \
            firstprivate(device, irange, jrange, queue_index, alpha) priority(priority)
        {
            // temporarily, convert both into same layout
            // todo: this is in-efficient, because both matrices may have same layout already
            //       and possibly wrong, because an input matrix is being altered
            // todo: best, handle directly through the CUDA kernels
            auto layout = Layout::ColMajor;
            std::set<ij_tuple> B_tiles_set;

            //for (int64_t i = 0; i < B.mt(); ++i) {
            for (int64_t row = 0; row < int64_t(row_indices.size()); ++row) {
                int64_t i = row_indices[row];
                for (int64_t j = 0; j < B.nt(); ++j) {
                    if (B.tileIsLocal(i, j) && device == B.tileDevice(i, j)) {
                        //A_tiles_set.insert({i, j});
                        B_tiles_set.insert({i, j});
                    }
                }
            }
            #pragma omp taskgroup
            {
                #pragma omp task slate_omp_default_none \
                    shared( B, B_tiles_set ) \
                    firstprivate(device, layout)
                {
                    B.tileGetForWriting(B_tiles_set, device, LayoutConvert(layout));
                }
            }

            int64_t batch_size = B_tiles_set.size();
            scalar_t** b_array_host = B.array_host(device, queue_index);

            int64_t batch_count = 0;
            int64_t mb[2], nb[2], lda[2], ldb[2], group_count[2];
            int64_t mt[2];
            // todo: change this to one group
            for (int q = 0; q < 2; ++q) {
                group_count[q] = 0;
                lda[q] = 0;
                ldb[q] = 0;
                mt[q]  = 0;
                mb[q] = B.tileMb(irange[q][0]);
                nb[q] = B.tileNb(jrange[q][0]);
                //for (int64_t i = irange[q][0]; i < irange[q][1]; ++i) {
                for (int64_t row = 0; row < int64_t(row_indices.size()); ++row) {
                    int64_t i = row_indices[row];
                    for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                        if (B.tileIsLocal(i, j) && device == B.tileDevice(i, j)) {
                            b_array_host[batch_count] = B(i, j, device).data();
                            ldb[q] = B(i, j, device).stride();
                            ++group_count[q];
                            ++batch_count;
                        }
                    }
                    ++mt[q];
                }
            }
            slate_assert(batch_count == batch_size);

            scalar_t** b_array_dev = B.array_device(device, queue_index);

            blas::Queue* queue = B.compute_queue(device, queue_index);
            blas::set_device( queue->device() );


            blas::device_memcpy<scalar_t*>(b_array_dev, b_array_host,
                                batch_count,
                                blas::MemcpyKind::HostToDevice,
                                *queue);

            for (int q = 0; q < 2; ++q) {
                if (group_count[q] > 0 && mt[q] > 1) {
                    device::reduce(mb[q], nb[q], mt[q],
                                  alpha, b_array_dev, ldb[q],
                                  group_count[q], *queue);
                    b_array_dev += group_count[q];
                }
            }

            queue->sync();

            //for (int64_t i = 0; i < B.mt(); ++i) {
            for (int64_t row = 1; row < int64_t(row_indices.size()); ++row) {
                int64_t i = row_indices[row];
                for (int64_t j = 0; j < B.nt(); ++j) {
                    if (B.tileIsLocal(i, j) && device == B.tileDevice(i, j)) {
                        // erase tmp local and remote device tiles;
                        B.tileRelease(i, j, device);
                        // decrement life for remote tiles
                        B.tileTick(i, j);
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
/// todo: these functions should just be named "add".
// ----------------------------------------
template
void reduce<Target::HostTask, float>(
     std::vector<int64_t>& row_indices,
     float alpha, Matrix<float>&& A,
     int priority, int queue_index);

template
void reduce<Target::HostNest, float>(
     std::vector<int64_t>& row_indices,
     float alpha, Matrix<float>&& A,
     int priority, int queue_index);

template
void reduce<Target::HostBatch, float>(
     std::vector<int64_t>& row_indices,
     float alpha, Matrix<float>&& A,
     int priority, int queue_index);

template
void reduce<Target::Devices, float>(
     std::vector<int64_t>& row_indices,
     float alpha, Matrix<float>&& A,
     int priority, int queue_index);

// ----------------------------------------
template
void reduce<Target::HostTask, double>(
     std::vector<int64_t>& row_indices,
     double alpha, Matrix<double>&& A,
     int priority, int queue_index);

template
void reduce<Target::HostNest, double>(
     std::vector<int64_t>& row_indices,
     double alpha, Matrix<double>&& A,
     int priority, int queue_index);

template
void reduce<Target::HostBatch, double>(
     std::vector<int64_t>& row_indices,
     double alpha, Matrix<double>&& A,
     int priority, int queue_index);

template
void reduce<Target::Devices, double>(
     std::vector<int64_t>& row_indices,
     double alpha, Matrix<double>&& A,
     int priority, int queue_index);

// ----------------------------------------
template
void reduce< Target::HostTask, std::complex<float> >(
     std::vector<int64_t>& row_indices,
     std::complex<float> alpha, Matrix< std::complex<float> >&& A,
     int priority, int queue_index);

template
void reduce< Target::HostNest, std::complex<float> >(
     std::vector<int64_t>& row_indices,
     std::complex<float> alpha, Matrix< std::complex<float> >&& A,
     int priority, int queue_index);

template
void reduce< Target::HostBatch, std::complex<float> >(
     std::vector<int64_t>& row_indices,
     std::complex<float> alpha, Matrix< std::complex<float> >&& A,
     int priority, int queue_index);

template
void reduce< Target::Devices, std::complex<float> >(
    std::vector<int64_t>& row_indices,
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
    int priority, int queue_index);

// ----------------------------------------
template
void reduce< Target::HostTask, std::complex<double> >(
     std::vector<int64_t>& row_indices,
     std::complex<double> alpha, Matrix< std::complex<double> >&& A,
     int priority, int queue_index);

template
void reduce< Target::HostNest, std::complex<double> >(
     std::vector<int64_t>& row_indices,
     std::complex<double> alpha, Matrix< std::complex<double> >&& A,
     int priority, int queue_index);

template
void reduce< Target::HostBatch, std::complex<double> >(
     std::vector<int64_t>& row_indices,
     std::complex<double> alpha, Matrix< std::complex<double> >&& A,
     int priority, int queue_index);

template
void reduce< Target::Devices, std::complex<double> >(
     std::vector<int64_t>& row_indices,
     std::complex<double> alpha, Matrix< std::complex<double> >&& A,
     int priority, int queue_index);

} // namespace internal
} // namespace slate
