// Copyright (c) 2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/internal/device.hh"
#include "slate/Matrix.hh"
#include "slate/types.hh"
#include "internal/DevVector.hh"
#include "internal/internal.hh"
#include "internal/Tile_trsm_addmod.hh"
#include "slate/Tile_blas.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Triangular solve matrix (multiple right-hand sides).
/// Dispatches to target implementations.
/// @ingroup trsm_internal
///
template <Target target, typename scalar_t>
void trsm_addmod(Side side, Uplo uplo, scalar_t alpha,
                 Matrix<scalar_t>&& A,
                 Matrix<scalar_t>&& U,
                 std::vector<blas::real_type<scalar_t>>&& S,
                 Matrix<scalar_t>&& B,
                 int64_t ib, int priority, Layout layout, int64_t queue_index,
                 Options const &opts)
{
    trsm_addmod(internal::TargetType<target>(),
                side, uplo, alpha, A, U, S, B,
                ib, priority, layout, queue_index, opts);
}

//------------------------------------------------------------------------------
/// Triangular solve matrix (multiple right-hand sides).
/// Host OpenMP task implementation.
/// @ingroup trsm_internal
///
template <typename scalar_t>
void trsm_addmod(internal::TargetType<Target::HostTask>,
                 Side side, Uplo uplo, scalar_t alpha,
                 Matrix<scalar_t>& A,
                 Matrix<scalar_t>& U,
                 std::vector<blas::real_type<scalar_t>>& S,
                 Matrix<scalar_t>& B,
                 int64_t ib, int priority, Layout layout, int64_t queue_index,
                 Options const &opts)
{
    assert(A.mt() == 1);
    assert(U.mt() == 1);

    if (B.numLocalTiles() > 0) {
        A.tileGetForReading(0, 0, LayoutConvert(layout));
        U.tileGetForReading(0, 0, LayoutConvert(layout));
    }

    // TODO figure out if the workspaces can be shared/reused
    #pragma omp taskgroup
    if (side == Side::Right) {
        assert(B.nt() == 1);
        for (int64_t i = 0; i < B.mt(); ++i) {
            if (B.tileIsLocal(i, 0)) {
                #pragma omp task slate_omp_default_none \
                    shared( A, U, S, B ) \
                    firstprivate(i, layout, side, uplo, ib) priority(priority)
                {
                    B.tileGetForWriting(i, 0, LayoutConvert(layout));
                    tile::trsm_addmod(ib, side, uplo, alpha,
                                      A(0, 0), U(0, 0), S, B(i, 0));
                    A.tileTick(0, 0);
                    U.tileTick(0, 0);
                }
            }
        }
    }
    else {
        assert(B.mt() == 1);
        for (int64_t j = 0; j < B.nt(); ++j) {
            if (B.tileIsLocal(0, j)) {
                #pragma omp task slate_omp_default_none \
                    shared( A, U, S, B ) \
                    firstprivate(j, layout, side, uplo, ib) priority(priority)
                {
                    B.tileGetForWriting(0, j, LayoutConvert(layout));
                    tile::trsm_addmod(ib, side, uplo, alpha,
                                      A(0, 0), U(0, 0), S, B(0, j));
                    A.tileTick(0, 0);
                    U.tileTick(0, 0);
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Triangular solve matrix (multiple right-hand sides).
/// Host nested OpenMP implementation.
/// @ingroup trsm_internal
///
template <typename scalar_t>
void trsm_addmod(internal::TargetType<Target::HostNest>,
                Side side, Uplo uplo, scalar_t alpha,
                Matrix<scalar_t>& A,
                Matrix<scalar_t>& U,
                std::vector<blas::real_type<scalar_t>>& S,
                Matrix<scalar_t>& B,
                int64_t ib, int priority, Layout layout, int64_t queue_index,
                Options const &opts)
{
    slate_not_implemented("Target::HostNest isn't yet supported.");
}

//------------------------------------------------------------------------------
/// Triangular solve matrix (multiple right-hand sides).
/// Host batched implementation.
/// @ingroup trsm_internal
///
template <typename scalar_t>
void trsm_addmod(internal::TargetType<Target::HostBatch>,
                Side side, Uplo uplo, scalar_t alpha,
                Matrix<scalar_t>& A,
                Matrix<scalar_t>& U,
                std::vector<blas::real_type<scalar_t>>& S,
                Matrix<scalar_t>& B,
                int64_t ib, int priority, Layout layout, int64_t queue_index,
                Options const &opts)
{
    slate_not_implemented("Target::Device isn't yet supported.");
}

//------------------------------------------------------------------------------
/// Triangular solve matrix (multiple right-hand sides).
/// GPU device batched cuBLAS implementation.
/// @ingroup trsm_internal
///
template <typename scalar_t>
void trsm_addmod(internal::TargetType<Target::Devices>,
                Side side, Uplo uplo, scalar_t alpha,
                Matrix<scalar_t>& A,
                Matrix<scalar_t>& U,
                std::vector<blas::real_type<scalar_t>>& S,
                Matrix<scalar_t>& B,
                int64_t ib, int priority, Layout layout, int64_t queue_index,
                Options const &opts)
{

    using blas::conj;
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;
    using real_t = blas::real_type<scalar_t>;

    TileReleaseStrategy tile_release_strategy = get_option(
            opts, Option::TileReleaseStrategy, TileReleaseStrategy::All );

    assert(B.num_devices() > 0);
    assert(A.mt() == 1);
    assert(B.uploPhysical() == Uplo::General);
    assert(A.mt() == A.nt());  // square
    assert(side == Side::Left ? A.mt() == B.mt() : A.mt() == B.nt());

    assert(B.op() == Op::NoTrans);
    assert(A.op() == Op::NoTrans);

    #pragma omp taskgroup
    for (int device = 0; device < B.num_devices(); ++device) {
        #pragma omp task shared(A, B) priority(priority) \
            firstprivate(device, side, layout, uplo) \
            firstprivate(tile_release_strategy, alpha, queue_index)
        {
            trace::Block trace_block("internal::trsm_addmod");
            std::set<ij_tuple> B_tiles_set;
            if (side == Side::Right) {
                for (int64_t i = 0; i < B.mt(); ++i) {
                    if (B.tileIsLocal(i, 0)
                        && device == B.tileDevice(i, 0))
                    {
                        B_tiles_set.insert({i, 0});
                    }
                }
            }
            else {
                for (int64_t j = 0; j < B.nt(); ++j) {
                    if (B.tileIsLocal(0, j)
                        && device == B.tileDevice(0, j))
                    {
                        B_tiles_set.insert({0, j});
                    }
                }
            }

            int64_t batch_size = B_tiles_set.size();
            if (batch_size > 0) {
                blas::Queue* queue = B.compute_queue(device, queue_index);
                assert(queue != nullptr);


                A.tileGetForReading(0, 0, device, LayoutConvert(layout));
                U.tileGetForReading(0, 0, device, LayoutConvert(layout));
                B.tileGetForWriting(B_tiles_set, device, LayoutConvert(layout));

                DevVector< real_t > dS;
                if (uplo == Uplo::Upper) {
                    dS.resize( S.size(), device, *queue );
                    blas::device_memcpy( dS.data(), S.data(), S.size(), *queue );
                }

                // interior col or row
                std::vector<scalar_t*> a_array0;
                std::vector<scalar_t*> u_array0;
                std::vector<real_t*>   s_array0;
                std::vector<scalar_t*> b_array0;
                a_array0.reserve( batch_size );
                b_array0.reserve( batch_size );
                if (uplo == Uplo::Lower) {
                    u_array0.reserve( batch_size );
                }
                else {
                    s_array0.reserve( batch_size );
                }

                // bottom-right tile
                // todo: replace batch trsm with plain trsm
                std::vector<scalar_t*> a_array1;
                std::vector<scalar_t*> u_array1;
                std::vector<real_t*>   s_array1;
                std::vector<scalar_t*> b_array1;

                int64_t lda0 = 0;
                int64_t ldu0 = 0;
                int64_t ldb0 = 0;
                int64_t lda1 = 0;
                int64_t ldu1 = 0;
                int64_t ldb1 = 0;

                int64_t mb0 = B.tileMb(0);
                int64_t nb0 = B.tileNb(0);
                int64_t mb1 = B.tileMb(B.mt()-1);
                int64_t nb1 = B.tileNb(B.nt()-1);

                if (side == Side::Right) {
                    for (int64_t i = 0; i < B.mt()-1; ++i) {
                        if (B.tileIsLocal(i, 0)
                            && device == B.tileDevice(i, 0))
                        {
                            a_array0.push_back( A(0, 0, device).data() );
                            b_array0.push_back( B(i, 0, device).data() );
                            lda0 = A(0, 0, device).stride();
                            ldb0 = B(i, 0, device).stride();
                            if (uplo == Uplo::Lower) {
                                u_array0.push_back( U(0, 0, device).data() );
                                ldu0 = U(0, 0, device).stride();
                            }
                            else {
                                s_array0.push_back( dS.data() );
                            }
                        }
                    }
                    {
                        int64_t i = B.mt()-1;
                        if (B.tileIsLocal(i, 0)
                            && device == B.tileDevice(i, 0))
                        {
                            a_array1.push_back( A(0, 0, device).data() );
                            b_array1.push_back( B(i, 0, device).data() );
                            lda1 = A(0, 0, device).stride();
                            ldb1 = B(i, 0, device).stride();
                            if (uplo == Uplo::Lower) {
                                u_array1.push_back( U(0, 0, device).data() );
                                ldu1 = U(0, 0, device).stride();
                            }
                            else {
                                s_array1.push_back( dS.data() );
                            }
                        }
                    }
                }
                else {
                    for (int64_t j = 0; j < B.nt()-1; ++j) {
                        if (B.tileIsLocal(0, j)
                            && device == B.tileDevice(0, j))
                        {
                            a_array0.push_back( A(0, 0, device).data() );
                            b_array0.push_back( B(0, j, device).data() );
                            lda0 = A(0, 0, device).stride();
                            ldb0 = B(0, j, device).stride();
                            if (uplo == Uplo::Lower) {
                                u_array0.push_back( U(0, 0, device).data() );
                                ldu0 = U(0, 0, device).stride();
                            }
                            else {
                                s_array0.push_back( dS.data() );
                            }
                        }
                    }
                    {
                        int64_t j = B.nt()-1;
                        if (B.tileIsLocal(0, j)
                            && device == B.tileDevice(0, j))
                        {
                            a_array1.push_back( A(0, 0, device).data() );
                            b_array1.push_back( B(0, j, device).data() );
                            lda1 = A(0, 0, device).stride();
                            ldb1 = B(0, j, device).stride();
                            if (uplo == Uplo::Lower) {
                                u_array1.push_back( U(0, 0, device).data() );
                                ldu1 = U(0, 0, device).stride();
                            }
                            else {
                                s_array1.push_back( dS.data() );
                            }
                        }
                    }
                }

                {
                    size_t total_batch = a_array0.size() + a_array1.size();
                    int64_t work_per_batch = std::max(mb0*nb0, mb1*nb1);

                    DevVector< scalar_t > dwork ( total_batch*work_per_batch, device, *queue );
                    std::vector<scalar_t*> workarray0 (a_array0.size());
                    std::vector<scalar_t*> workarray1 (a_array1.size());
                    scalar_t* dwork_ptr = dwork.data();
                    for (size_t i = 0; i < a_array0.size(); ++i) {
                        workarray0[i] = dwork_ptr + i*work_per_batch;
                    }
                    dwork_ptr += a_array0.size() * work_per_batch;
                    for (size_t i = 0; i < a_array1.size(); ++i) {
                        workarray1[i] = dwork_ptr + i*work_per_batch;
                    }

                    if (a_array0.size() > 0) {
                        device::batch_trsm_addmod(
                            layout, side, uplo, mb0, nb0, ib,
                            alpha, a_array0, lda0,
                                   u_array0, ldu0,
                                   s_array0,
                                   b_array0, ldb0,
                            workarray0,
                            a_array0.size(), *queue);
                    }

                    if (a_array1.size() > 0) {
                        device::batch_trsm_addmod(
                            layout, side, uplo, mb1, nb1, ib,
                            alpha, a_array1, lda1,
                                   u_array1, ldu1,
                                   s_array1,
                                   b_array1, ldb1,
                            workarray1,
                            a_array1.size(), *queue);
                    }

                    queue->sync();

                    // DevVector doesn't automatically release memory
                    dwork.clear(*queue);
                }

                if (tile_release_strategy == TileReleaseStrategy::Internal
                    || tile_release_strategy == TileReleaseStrategy::All) {

                    A.tileRelease(0, 0, device);
                    for (auto i = 0; i < batch_size; ++i) {
                        A.tileTick(0, 0);
                    }
                }

                // DevVector doesn't automatically release memory
                dS.clear( *queue );
            }
        }
    }
    // end omp taskgroup
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void trsm_addmod<Target::HostTask, float>(
    Side side, Uplo uplo, float alpha,
    Matrix<float>&& A,
    Matrix<float>&& U,
    std::vector<float>&& S,
    Matrix<float>&& B,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

template
void trsm_addmod<Target::HostNest, float>(
    Side side, Uplo uplo, float alpha,
    Matrix<float>&& A,
    Matrix<float>&& U,
    std::vector<float>&& S,
    Matrix<float>&& B,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

template
void trsm_addmod<Target::HostBatch, float>(
    Side side, Uplo uplo, float alpha,
    Matrix<float>&& A,
    Matrix<float>&& U,
    std::vector<float>&& S,
    Matrix<float>&& B,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

template
void trsm_addmod<Target::Devices, float>(
    Side side, Uplo uplo, float alpha,
    Matrix<float>&& A,
    Matrix<float>&& U,
    std::vector<float>&& S,
    Matrix<float>&& B,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

// ----------------------------------------
template
void trsm_addmod<Target::HostTask, double>(
    Side side, Uplo uplo, double alpha,
    Matrix<double>&& A,
    Matrix<double>&& U,
    std::vector<double>&& S,
    Matrix<double>&& B,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

template
void trsm_addmod<Target::HostNest, double>(
    Side side, Uplo uplo, double alpha,
    Matrix<double>&& A,
    Matrix<double>&& U,
    std::vector<double>&& S,
    Matrix<double>&& B,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

template
void trsm_addmod<Target::HostBatch, double>(
    Side side, Uplo uplo, double alpha,
    Matrix<double>&& A,
    Matrix<double>&& U,
    std::vector<double>&& S,
    Matrix<double>&& B,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

template
void trsm_addmod<Target::Devices, double>(
    Side side, Uplo uplo, double alpha,
    Matrix<double>&& A,
    Matrix<double>&& U,
    std::vector<double>&& S,
    Matrix<double>&& B,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

// ----------------------------------------
template
void trsm_addmod< Target::HostTask, std::complex<float> >(
    Side side, Uplo uplo, std::complex<float> alpha,
    Matrix<std::complex<float>>&& A,
    Matrix<std::complex<float>>&& U,
    std::vector<float>&& S,
    Matrix<std::complex<float>>&& B,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

template
void trsm_addmod< Target::HostNest, std::complex<float> >(
    Side side, Uplo uplo, std::complex<float> alpha,
    Matrix<std::complex<float>>&& A,
    Matrix<std::complex<float>>&& U,
    std::vector<float>&& S,
    Matrix<std::complex<float>>&& B,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

template
void trsm_addmod< Target::HostBatch, std::complex<float> >(
    Side side, Uplo uplo, std::complex<float> alpha,
    Matrix<std::complex<float>>&& A,
    Matrix<std::complex<float>>&& U,
    std::vector<float>&& S,
    Matrix<std::complex<float>>&& B,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

template
void trsm_addmod< Target::Devices, std::complex<float> >(
    Side side, Uplo uplo, std::complex<float> alpha,
    Matrix<std::complex<float>>&& A,
    Matrix<std::complex<float>>&& U,
    std::vector<float>&& S,
    Matrix<std::complex<float>>&& B,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

// ----------------------------------------
template
void trsm_addmod< Target::HostTask, std::complex<double> >(
    Side side, Uplo uplo, std::complex<double> alpha,
    Matrix<std::complex<double>>&& A,
    Matrix<std::complex<double>>&& U,
    std::vector<double>&& S,
    Matrix<std::complex<double>>&& B,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

template
void trsm_addmod< Target::HostNest, std::complex<double> >(
    Side side, Uplo uplo, std::complex<double> alpha,
    Matrix<std::complex<double>>&& A,
    Matrix<std::complex<double>>&& U,
    std::vector<double>&& S,
    Matrix<std::complex<double>>&& B,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

template
void trsm_addmod< Target::HostBatch, std::complex<double> >(
    Side side, Uplo uplo, std::complex<double> alpha,
    Matrix<std::complex<double>>&& A,
    Matrix<std::complex<double>>&& U,
    std::vector<double>&& S,
    Matrix<std::complex<double>>&& B,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

template
void trsm_addmod< Target::Devices, std::complex<double> >(
    Side side, Uplo uplo, std::complex<double> alpha,
    Matrix<std::complex<double>>&& A,
    Matrix<std::complex<double>>&& U,
    std::vector<double>&& S,
    Matrix<std::complex<double>>&& B,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

} // namespace internal
} // namespace slate
