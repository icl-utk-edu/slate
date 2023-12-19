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
                 Matrix<scalar_t>&& VT,
                 std::vector<blas::real_type<scalar_t>>&& S,
                 Matrix<scalar_t>&& B,
                 BlockFactor blockFactorType,
                 int64_t ib, int priority, Layout layout, int64_t queue_index,
                 Options const &opts)
{
    trsm_addmod(internal::TargetType<target>(),
                side, uplo, alpha, A, U, VT, S, B,
                blockFactorType, ib, priority, layout, queue_index, opts);
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
                 Matrix<scalar_t>& VT,
                 std::vector<blas::real_type<scalar_t>>& S,
                 Matrix<scalar_t>& B,
                 BlockFactor blockFactorType,
                 int64_t ib, int priority, Layout layout, int64_t queue_index,
                 Options const &opts)
{
    assert(A.mt() == 1);
    assert(U.mt() == 1);

    if (B.numLocalTiles() > 0) {
        A.tileGetForReading(0, 0, LayoutConvert(layout));
        if (uplo == Uplo::Lower) {
            U.tileGetForReading(0, 0, LayoutConvert(layout));
        }
        else {
            if (blockFactorType != BlockFactor::QR) {
                VT.tileGetForReading(0, 0, LayoutConvert(layout));
            }
        }
    }

    // TODO figure out if the workspaces can be shared/reused
    #pragma omp taskgroup
    if (side == Side::Right) {
        assert(B.nt() == 1);
        for (int64_t i = 0; i < B.mt(); ++i) {
            if (B.tileIsLocal(i, 0)) {
                #pragma omp task slate_omp_default_none \
                    shared( A, U, VT, S, B ) \
                    firstprivate(i, layout, side, uplo, ib) priority(priority)
                {
                    B.tileGetForWriting(i, 0, LayoutConvert(layout));
                    Tile<scalar_t> U_VT;
                    if (uplo == Uplo::Lower) {
                        U_VT = U(0, 0);
                    }
                    else {
                        if (blockFactorType == BlockFactor::QR) {
                            U_VT = A(0, 0); // Not used
                        }
                        else {
                            U_VT = VT(0, 0);
                        }
                    }
                    tile::trsm_addmod(blockFactorType, ib, side, uplo, alpha,
                                      A(0, 0), U_VT, U_VT, S, B(i, 0));
                    A.tileTick(0, 0);
                    if (uplo == Uplo::Lower) {
                        U.tileTick(0, 0);
                    }
                    else {
                        if (blockFactorType != BlockFactor::QR) {
                            VT.tileTick(0, 0);
                        }
                    }
                }
            }
        }
    }
    else {
        assert(B.mt() == 1);
        for (int64_t j = 0; j < B.nt(); ++j) {
            if (B.tileIsLocal(0, j)) {
                #pragma omp task slate_omp_default_none \
                    shared( A, U, VT, S, B ) \
                    firstprivate(j, layout, side, uplo, ib) priority(priority)
                {
                    B.tileGetForWriting(0, j, LayoutConvert(layout));
                    Tile<scalar_t> U_VT;
                    if (uplo == Uplo::Lower) {
                        U_VT = U(0, 0);
                    }
                    else {
                        if (blockFactorType == BlockFactor::QR) {
                            U_VT = A(0, 0); // Not used
                        }
                        else {
                            U_VT = VT(0, 0);
                        }
                    }
                    tile::trsm_addmod(blockFactorType, ib, side, uplo, alpha,
                                      A(0, 0), U_VT, U_VT, S, B(0, j));
                    A.tileTick(0, 0);
                    if (uplo == Uplo::Lower) {
                        U.tileTick(0, 0);
                    }
                    else {
                        if (blockFactorType != BlockFactor::QR) {
                            VT.tileTick(0, 0);
                        }
                    }
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
                 Matrix<scalar_t>& VT,
                 std::vector<blas::real_type<scalar_t>>& S,
                 Matrix<scalar_t>& B,
                 BlockFactor blockFactorType,
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
                 Matrix<scalar_t>& VT,
                 std::vector<blas::real_type<scalar_t>>& S,
                 Matrix<scalar_t>& B,
                 BlockFactor blockFactorType,
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
                 Matrix<scalar_t>& VT,
                 std::vector<blas::real_type<scalar_t>>& S,
                 Matrix<scalar_t>& B,
                 BlockFactor blockFactorType,
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
                if (uplo == Uplo::Lower) {
                    U.tileGetForReading(0, 0, device, LayoutConvert(layout));
                }
                else {
                    if (blockFactorType != BlockFactor::QR) {
                        VT.tileGetForReading(0, 0, device, LayoutConvert(layout));
                    }
                }
                B.tileGetForWriting(B_tiles_set, device, LayoutConvert(layout));

                real_t* dS_ptr;
                if (uplo == Uplo::Upper) {
                    dS_ptr = (real_t*)A.allocWorkspaceBuffer(device, S.size());
                    blas::device_memcpy( dS_ptr, S.data(), S.size(), *queue );
                }

                // interior col or row
                std::vector<scalar_t*> a_array0;
                std::vector<scalar_t*> u_array0;
                std::vector<scalar_t*> vt_array0;
                std::vector<real_t*>   s_array0;
                std::vector<scalar_t*> b_array0;
                a_array0.reserve( batch_size );
                b_array0.reserve( batch_size );
                if (uplo == Uplo::Lower) {
                    u_array0.reserve( batch_size );
                }
                else {
                    if (blockFactorType != BlockFactor::QR) {
                        vt_array0.reserve( batch_size );
                    }
                    s_array0.reserve( batch_size );
                }

                // bottom-right tile
                // todo: replace batch trsm with plain trsm
                std::vector<scalar_t*> a_array1;
                std::vector<scalar_t*> u_array1;
                std::vector<scalar_t*> vt_array1;
                std::vector<real_t*>   s_array1;
                std::vector<scalar_t*> b_array1;

                int64_t lda0 = 0;
                int64_t ldu0 = 0;
                int64_t ldvt0 = 0;
                int64_t ldb0 = 0;
                int64_t lda1 = 0;
                int64_t ldu1 = 0;
                int64_t ldvt1 = 0;
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
                                s_array0.push_back( dS_ptr );
                                if (blockFactorType != BlockFactor::QR) {
                                    vt_array0.push_back( VT(0, 0, device).data() );
                                    ldvt0 = VT(0, 0, device).stride();
                                }
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
                                s_array1.push_back( dS_ptr );
                                if (blockFactorType != BlockFactor::QR) {
                                    vt_array1.push_back( VT(0, 0, device).data() );
                                    ldvt1 = VT(0, 0, device).stride();
                                }
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
                                s_array0.push_back( dS_ptr );
                                if (blockFactorType != BlockFactor::QR) {
                                    vt_array0.push_back( VT(0, 0, device).data() );
                                    ldvt0 = VT(0, 0, device).stride();
                                }
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
                                s_array1.push_back( dS_ptr );
                                if (blockFactorType != BlockFactor::QR) {
                                    vt_array1.push_back( VT(0, 0, device).data() );
                                    ldvt1 = VT(0, 0, device).stride();
                                }
                            }
                        }
                    }
                }

                {
                    std::vector<scalar_t*> workarray0 (a_array0.size());
                    std::vector<scalar_t*> workarray1 (a_array1.size());
                    for (size_t i = 0; i < a_array0.size(); ++i) {
                        workarray0[i] = A.allocWorkspaceBuffer(device, mb0*nb0);
                    }
                    for (size_t i = 0; i < a_array1.size(); ++i) {
                        workarray1[i] = A.allocWorkspaceBuffer(device, mb1*nb1);
                    }

                    if (a_array0.size() > 0) {
                        device::batch_trsm_addmod(
                            blockFactorType,
                            layout, side, uplo, mb0, nb0, ib,
                            alpha, a_array0, lda0,
                                   u_array0, ldu0,
                                   vt_array0, ldvt0,
                                   s_array0,
                                   b_array0, ldb0,
                            workarray0,
                            a_array0.size(), *queue);
                    }

                    if (a_array1.size() > 0) {
                        device::batch_trsm_addmod(
                            blockFactorType,
                            layout, side, uplo, mb1, nb1, ib,
                            alpha, a_array1, lda1,
                                   u_array1, ldu1,
                                   vt_array1, ldvt1,
                                   s_array1,
                                   b_array1, ldb1,
                            workarray1,
                            a_array1.size(), *queue);
                    }

                    queue->sync();

                    // return workspace memory
                    for (size_t i = 0; i < a_array0.size(); ++i) {
                        A.freeWorkspaceBuffer(device, workarray0[i]);
                    }
                    for (size_t i = 0; i < a_array1.size(); ++i) {
                        A.freeWorkspaceBuffer(device, workarray1[i]);
                    }
                }

                if (tile_release_strategy == TileReleaseStrategy::Internal
                    || tile_release_strategy == TileReleaseStrategy::All) {

                    A.tileRelease(0, 0, device);
                    for (auto i = 0; i < batch_size; ++i) {
                        A.tileTick(0, 0);
                    }
                    if (uplo == Uplo::Lower) {
                        U.tileRelease(0, 0, device);
                        for (auto i = 0; i < batch_size; ++i) {
                            U.tileTick(0, 0);
                        }
                    }
                    else {
                        if (blockFactorType != BlockFactor::QR) {
                            VT.tileRelease(0, 0, device);
                            for (auto i = 0; i < batch_size; ++i) {
                                VT.tileTick(0, 0);
                            }
                        }
                    }
                }

                // return workspace memory
                if (uplo == Uplo::Upper) {
                    A.freeWorkspaceBuffer( device, (scalar_t*)dS_ptr );
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
void trsm_addmod<Target::HostTask, float>(
    Side side, Uplo uplo, float alpha,
    Matrix<float>&& A,
    Matrix<float>&& U,
    Matrix<float>&& VT,
    std::vector<float>&& S,
    Matrix<float>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

template
void trsm_addmod<Target::HostNest, float>(
    Side side, Uplo uplo, float alpha,
    Matrix<float>&& A,
    Matrix<float>&& U,
    Matrix<float>&& VT,
    std::vector<float>&& S,
    Matrix<float>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

template
void trsm_addmod<Target::HostBatch, float>(
    Side side, Uplo uplo, float alpha,
    Matrix<float>&& A,
    Matrix<float>&& U,
    Matrix<float>&& VT,
    std::vector<float>&& S,
    Matrix<float>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

template
void trsm_addmod<Target::Devices, float>(
    Side side, Uplo uplo, float alpha,
    Matrix<float>&& A,
    Matrix<float>&& U,
    Matrix<float>&& VT,
    std::vector<float>&& S,
    Matrix<float>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

// ----------------------------------------
template
void trsm_addmod<Target::HostTask, double>(
    Side side, Uplo uplo, double alpha,
    Matrix<double>&& A,
    Matrix<double>&& U,
    Matrix<double>&& VT,
    std::vector<double>&& S,
    Matrix<double>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

template
void trsm_addmod<Target::HostNest, double>(
    Side side, Uplo uplo, double alpha,
    Matrix<double>&& A,
    Matrix<double>&& U,
    Matrix<double>&& VT,
    std::vector<double>&& S,
    Matrix<double>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

template
void trsm_addmod<Target::HostBatch, double>(
    Side side, Uplo uplo, double alpha,
    Matrix<double>&& A,
    Matrix<double>&& U,
    Matrix<double>&& VT,
    std::vector<double>&& S,
    Matrix<double>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

template
void trsm_addmod<Target::Devices, double>(
    Side side, Uplo uplo, double alpha,
    Matrix<double>&& A,
    Matrix<double>&& U,
    Matrix<double>&& VT,
    std::vector<double>&& S,
    Matrix<double>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

// ----------------------------------------
template
void trsm_addmod< Target::HostTask, std::complex<float> >(
    Side side, Uplo uplo, std::complex<float> alpha,
    Matrix<std::complex<float>>&& A,
    Matrix<std::complex<float>>&& U,
    Matrix<std::complex<float>>&& VT,
    std::vector<float>&& S,
    Matrix<std::complex<float>>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

template
void trsm_addmod< Target::HostNest, std::complex<float> >(
    Side side, Uplo uplo, std::complex<float> alpha,
    Matrix<std::complex<float>>&& A,
    Matrix<std::complex<float>>&& U,
    Matrix<std::complex<float>>&& VT,
    std::vector<float>&& S,
    Matrix<std::complex<float>>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

template
void trsm_addmod< Target::HostBatch, std::complex<float> >(
    Side side, Uplo uplo, std::complex<float> alpha,
    Matrix<std::complex<float>>&& A,
    Matrix<std::complex<float>>&& U,
    Matrix<std::complex<float>>&& VT,
    std::vector<float>&& S,
    Matrix<std::complex<float>>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

template
void trsm_addmod< Target::Devices, std::complex<float> >(
    Side side, Uplo uplo, std::complex<float> alpha,
    Matrix<std::complex<float>>&& A,
    Matrix<std::complex<float>>&& U,
    Matrix<std::complex<float>>&& VT,
    std::vector<float>&& S,
    Matrix<std::complex<float>>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

// ----------------------------------------
template
void trsm_addmod< Target::HostTask, std::complex<double> >(
    Side side, Uplo uplo, std::complex<double> alpha,
    Matrix<std::complex<double>>&& A,
    Matrix<std::complex<double>>&& U,
    Matrix<std::complex<double>>&& VT,
    std::vector<double>&& S,
    Matrix<std::complex<double>>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

template
void trsm_addmod< Target::HostNest, std::complex<double> >(
    Side side, Uplo uplo, std::complex<double> alpha,
    Matrix<std::complex<double>>&& A,
    Matrix<std::complex<double>>&& U,
    Matrix<std::complex<double>>&& VT,
    std::vector<double>&& S,
    Matrix<std::complex<double>>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

template
void trsm_addmod< Target::HostBatch, std::complex<double> >(
    Side side, Uplo uplo, std::complex<double> alpha,
    Matrix<std::complex<double>>&& A,
    Matrix<std::complex<double>>&& U,
    Matrix<std::complex<double>>&& VT,
    std::vector<double>&& S,
    Matrix<std::complex<double>>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

template
void trsm_addmod< Target::Devices, std::complex<double> >(
    Side side, Uplo uplo, std::complex<double> alpha,
    Matrix<std::complex<double>>&& A,
    Matrix<std::complex<double>>&& U,
    Matrix<std::complex<double>>&& VT,
    std::vector<double>&& S,
    Matrix<std::complex<double>>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index,
    Options const &opts);

} // namespace internal
} // namespace slate