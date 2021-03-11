// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
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

#if defined(SLATE_NO_CUDA)
// Specializations to allow compilation without CUDA.
template <>
void tzset(
    int64_t m, int64_t n,
    double alpha, double beta,
    double** Aarray, int64_t lda,
    int64_t batch_count, cudaStream_t stream)
{
}

template <>
void tzset(
    int64_t m, int64_t n,
    float alpha, float beta,
    float** Aarray, int64_t lda,
    int64_t batch_count, cudaStream_t stream)
{
}
#endif // not SLATE_WITH_CUDA

} // namespace device

namespace internal {

//------------------------------------------------------------------------------
/// General matrix set.
/// Dispatches to target implementations.
/// @ingroup set_internal
///
template <Target target, typename scalar_t>
void set(scalar_t alpha, scalar_t beta, BaseTrapezoidMatrix<scalar_t>&& A, int priority)
{
    set(internal::TargetType<target>(),
        alpha, beta, A, priority);
}

//------------------------------------------------------------------------------
/// General matrix set.
/// TODO handle transpose A case
/// Host OpenMP task implementation.
/// @ingroup set_internal
///
template <typename scalar_t>
void set(internal::TargetType<Target::HostTask>,
         scalar_t alpha, scalar_t beta, BaseTrapezoidMatrix<scalar_t>& A, int priority)
{
    // trace::Block trace_block("set");

    if (A.uplo() == Uplo::Lower) {
        for (int64_t j = 0; j < A.nt(); ++j){
            for (int64_t i = j; i < A.mt(); ++i){  // lower trapezoid
                if (A.tileIsLocal(i, j)) {
                    #pragma omp task shared(A ) priority(priority)
                    {
                        A.tileGetForWriting(i, j, LayoutConvert::None);
                        if (i == j)
                            A.at(i, j).set(alpha, beta);
                        else
                            A.at(i, j).set(alpha, alpha);
                    }
                }
            }
        }
    }
    else { // upper
        for (int64_t j = 0; j < A.nt(); ++j) {
            for (int64_t i = 0; i <= j && i < A.mt(); ++i){  // upper trapezoid
                if (A.tileIsLocal(i, j)) {
                    #pragma omp task shared(A ) priority(priority)
                    {
                        A.tileGetForWriting(i, j, LayoutConvert::None);
                        if (i == j)
                            A.at(i, j).set(alpha, beta);
                        else
                            A.at(i, j).set(alpha, alpha);
                    }
                }
            }
        }
    }
    #pragma omp taskwait
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void set(internal::TargetType<Target::HostNest>,
         scalar_t alpha, scalar_t beta, BaseTrapezoidMatrix<scalar_t>& A, int priority)
{
    throw Exception("HostNest not yet implemented");
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void set(internal::TargetType<Target::HostBatch>,
         scalar_t alpha, scalar_t beta, BaseTrapezoidMatrix<scalar_t>& A, int priority)
{
    throw Exception("HostBatch not yet implemented");
}

//------------------------------------------------------------------------------
/// General matrix set.
/// TODO handle transpose A case
/// GPU device implementation.
/// @ingroup set_internal
///
template <typename scalar_t>
void set(internal::TargetType<Target::Devices>,
         scalar_t alpha, scalar_t beta, BaseTrapezoidMatrix<scalar_t>& A, int priority)
{
    using ij_tuple = typename BaseTrapezoidMatrix<scalar_t>::ij_tuple;

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

    for (int device = 0; device < A.num_devices(); ++device) {
        #pragma omp task shared(A) priority(priority)
        {
            // temporarily, convert both into same layout
            // todo: this is in-efficient, because both matrices may have same layout already
            //       and possibly wrong, because an input matrix is being altered
            // todo: best, handle directly through the CUDA kernels
            auto layout = Layout::ColMajor;
            std::set<ij_tuple> A_tiles_set;

            if (A.uplo() == Uplo::Lower) {
                for (int64_t j = 0; j < A.nt(); ++j){
                    for (int64_t i = j; i < A.mt(); ++i){  // lower trapezoid
                        if (A.tileIsLocal(i, j) && device == A.tileDevice(i, j)) {
                            A_tiles_set.insert({i, j});
                        }
                    }
                }
            }
            else { // upper
                for (int64_t j = 0; j < A.nt(); ++j) {
                    for (int64_t i = 0; i <= j && i < A.mt(); ++i){  // upper trapezoid
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
                            if (A.tileIsLocal(i, j) && device == A.tileDevice(i, j)) {
                                if (i != j) {
                                    a_array_host[batch_count] = A(i, j, device).data();
                                    lda[q] = A(i, j, device).stride();
                                    ++group_count[q];
                                    ++batch_count;
                                }
                            }
                        }
                    }
                }
                else { //upper
                    for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                        for (int64_t i = irange[q][0]; i < irange[q][1] && i <= j; ++i) {
                            if (A.tileIsLocal(i, j) && device == A.tileDevice(i, j)) {
                                if (i != j) {
                                    a_array_host[batch_count] = A(i, j, device).data();
                                    lda[q] = A(i, j, device).stride();
                                    ++group_count[q];
                                    ++batch_count;
                                }
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
                            if (A.tileIsLocal(i, j) && device == A.tileDevice(i, j)) {
                                if (i == j) {
                                    a_array_host[batch_count] = A(i, j, device).data();
                                    lda[q] = A(i, j, device).stride();
                                    ++group_count[q];
                                    ++batch_count;
                                }
                            }
                        }
                    }
                }
                else { //upper
                    for (int64_t j = jrange[q-4][0]; j < jrange[q-4][1]; ++j) {
                        for (int64_t i = irange[q-4][0]; i < irange[q-4][1] && i <= j; ++i) {
                            if (A.tileIsLocal(i, j) && device == A.tileDevice(i, j)) {
                                if (i == j) {
                                    a_array_host[batch_count] = A(i, j, device).data();
                                    lda[q] = A(i, j, device).stride();
                                    ++group_count[q];
                                    ++batch_count;
                                }
                            }
                        }
                    }
                }
            }

            scalar_t** a_array_dev = A.array_device(device);

            slate_cuda_call(cudaSetDevice(device));

            cudaStream_t stream = A.compute_stream(device);

            slate_cuda_call(
                cudaMemcpyAsync(a_array_dev, a_array_host,
                                sizeof(scalar_t*)*batch_count,
                                cudaMemcpyHostToDevice,
                                stream));

            for (int q = 0; q < 4; ++q) {
                if (group_count[q] > 0) {
                    device::geset(mb[q], nb[q],
                                  alpha, alpha, a_array_dev, lda[q],
                                  group_count[q], stream);
                    a_array_dev += group_count[q];
                }
            }
            for (int q = 4; q < 8; ++q) {
                if (group_count[q] > 0) {
                    device::geset(mb[q], nb[q],
                                  alpha, beta, a_array_dev, lda[q],
                                  group_count[q], stream);
                    a_array_dev += group_count[q];
                }
            }

            slate_cuda_call(cudaStreamSynchronize(stream));
        }
    }

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void set<Target::HostTask, float>(
    float alpha, float beta, BaseTrapezoidMatrix<float>&& A, int priority);

template
void set<Target::HostNest, float>(
    float alpha, float beta, BaseTrapezoidMatrix<float>&& A, int priority);

template
void set<Target::HostBatch, float>(
    float alpha, float beta, BaseTrapezoidMatrix<float>&& A, int priority);

template
void set<Target::Devices, float>(
    float alpha, float beta, BaseTrapezoidMatrix<float>&& A, int priority);

// ----------------------------------------
template
void set<Target::HostTask, double>(
    double alpha, double beta, BaseTrapezoidMatrix<double>&& A, int priority);

template
void set<Target::HostNest, double>(
    double alpha, double beta, BaseTrapezoidMatrix<double>&& A, int priority);

template
void set<Target::HostBatch, double>(
    double alpha, double beta, BaseTrapezoidMatrix<double>&& A, int priority);

template
void set<Target::Devices, double>(
    double alpha, double beta, BaseTrapezoidMatrix<double>&& A, int priority);

// ----------------------------------------
template
void set< Target::HostTask, std::complex<float> >(
    std::complex<float> alpha, std::complex<float>  beta,
    BaseTrapezoidMatrix< std::complex<float> >&& A, int priority);

template
void set< Target::HostNest, std::complex<float> >(
    std::complex<float> alpha, std::complex<float>  beta,
    BaseTrapezoidMatrix< std::complex<float> >&& A, int priority);

template
void set< Target::HostBatch, std::complex<float> >(
    std::complex<float> alpha, std::complex<float>  beta,
    BaseTrapezoidMatrix< std::complex<float> >&& A, int priority);

template
void set< Target::Devices, std::complex<float> >(
    std::complex<float> alpha, std::complex<float>  beta,
    BaseTrapezoidMatrix< std::complex<float> >&& A, int priority);

// ----------------------------------------
template
void set< Target::HostTask, std::complex<double> >(
    std::complex<double> alpha, std::complex<double> beta,
    BaseTrapezoidMatrix< std::complex<double> >&& A, int priority);

template
void set< Target::HostNest, std::complex<double> >(
    std::complex<double> alpha, std::complex<double> beta,
    BaseTrapezoidMatrix< std::complex<double> >&& A, int priority);

template
void set< Target::HostBatch, std::complex<double> >(
    std::complex<double> alpha, std::complex<double> beta,
    BaseTrapezoidMatrix< std::complex<double> >&& A, int priority);

template
void set< Target::Devices, std::complex<double> >(
    std::complex<double> alpha, std::complex<double> beta,
    BaseTrapezoidMatrix< std::complex<double> >&& A, int priority);

} // namespace internal
} // namespace slate
