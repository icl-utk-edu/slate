// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/types.hh"
#include "slate/Tile_blas.hh"
#include "internal/internal.hh"
#include "internal/internal_batch.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Hermitian matrix multiply to update trailing matrix,
/// where A is a single tile.
/// If side = left,  B and C are each a single block row;
/// if side = right, B and C are each a single block col.
/// Unlike most BLAS operations, here op(B) and op(C) must be
/// both the same, either both NoTrans or both ConjTrans.
/// Dispatches to target implementations.
/// @ingroup hemm_internal
///
template <Target target, typename scalar_t>
void hemm(Side side,
          scalar_t alpha, HermitianMatrix<scalar_t>&& A,
                          Matrix<scalar_t>&& B,
          scalar_t beta,  Matrix<scalar_t>&& C,
          int priority, Options const& opts)
{
    // check dimensions
    assert(A.mt() == 1);
    assert(A.nt() == 1);
    if (side == Side::Left) {
        assert(B.mt() == 1);
        assert(C.mt() == 1);
        assert(B.nt() == C.nt());
    }
    else {
        assert(B.nt() == 1);
        assert(C.nt() == 1);
        assert(B.mt() == C.mt());
    }
    assert(B.op() == C.op());

    hemm(internal::TargetType<target>(),
         side,
         alpha, A, B,
         beta,  C,
         priority, opts);
}

//------------------------------------------------------------------------------
/// Hermitian matrix multiply to update trailing matrix.
/// Host OpenMP task implementation.
/// @ingroup hemm_internal
///
template <typename scalar_t>
void hemm(internal::TargetType<Target::HostTask>,
          Side side,
          scalar_t alpha, HermitianMatrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
          int priority, Options const& opts)
{
    // CPU uses ColMajor
    // todo: relax this assumption, by allowing Tile_blas.hh::hemm() to take layout param
    // todo: optimize for the number of layout conversions,
    //       by watching 'layout' and 'C(i, j).layout()'
    const Layout layout = Layout::ColMajor;

    TileReleaseStrategy tile_release_strategy = get_option(
            opts, Option::TileReleaseStrategy, TileReleaseStrategy::All );

    bool call_tile_tick = tile_release_strategy == TileReleaseStrategy::Internal
                          || tile_release_strategy == TileReleaseStrategy::All;

    int err = 0;
    #pragma omp taskgroup
    if (side == Side::Left) {
        for (int64_t j = 0; j < C.nt(); ++j) {
            if (C.tileIsLocal(0, j)) {
                #pragma omp task slate_omp_default_none \
                    shared( A, B, C, err ) \
                    firstprivate(j, layout, side, alpha, beta, call_tile_tick) \
                    priority(priority)
                {
                    try {
                        A.tileGetForReading(0, 0, LayoutConvert(layout));
                        B.tileGetForReading(0, j, LayoutConvert(layout));
                        C.tileGetForWriting(0, j, LayoutConvert(layout));
                        tile::hemm(
                            side,
                            alpha, A(0, 0), B(0, j),
                            beta,  C(0, j) );
                        if (call_tile_tick) {
                            // todo: should tileRelease()?
                            A.tileTick(0, 0);
                            B.tileTick(0, j);
                        }
                    }
                    catch (std::exception& e) {
                        err = __LINE__;
                    }
                }
            }
        }
    }
    else {
        // side == Right
        for (int64_t i = 0; i < C.mt(); ++i) {
            if (C.tileIsLocal(i, 0)) {
                #pragma omp task slate_omp_default_none \
                    shared( A, B, C, err ) \
                    firstprivate(i, layout, side, alpha, beta, call_tile_tick) \
                    priority(priority)
                {
                    try {
                        A.tileGetForReading(0, 0, LayoutConvert(layout));
                        B.tileGetForReading(i, 0, LayoutConvert(layout));
                        C.tileGetForWriting(i, 0, LayoutConvert(layout));
                        tile::hemm(
                            side,
                            alpha, A(0, 0), B(i, 0),
                            beta,  C(i, 0) );
                        if (call_tile_tick) {
                            // todo: should tileRelease()?
                            A.tileTick(0, 0);
                            B.tileTick(i, 0);
                        }
                    }
                    catch (std::exception& e) {
                        err = __LINE__;
                    }
                }
            }
        }
    }

    if (err)
        throw std::exception();
}

//------------------------------------------------------------------------------
/// Hermitian matrix multiply to update trailing matrix.
/// Host nested OpenMP implementation.
/// @ingroup hemm_internal
///
template <typename scalar_t>
void hemm(internal::TargetType<Target::HostNest>,
          Side side,
          scalar_t alpha, HermitianMatrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
          int priority, Options const& opts)
{
    // CPU uses ColMajor
    // todo: relax this assumption, by allowing Tile_blas.hh::hemm() to take layout param
    // todo: optimize for the number of layout conversions,
    //       by watching 'layout' and 'C(i, j).layout()'
    const Layout layout = Layout::ColMajor;

    int err = 0;
    if (side == Side::Left) {
        #pragma omp parallel for schedule(dynamic, 1) slate_omp_default_none \
            shared(A, B, C, err) firstprivate(layout, side, alpha, beta)
        for (int64_t j = 0; j < C.nt(); ++j) {
            if (C.tileIsLocal(0, j)) {
                try {
                    A.tileGetForReading(0, 0, LayoutConvert(layout));
                    B.tileGetForReading(0, j, LayoutConvert(layout));
                    C.tileGetForWriting(0, j, LayoutConvert(layout));
                    tile::hemm(
                        side,
                        alpha, A(0, 0), B(0, j),
                        beta,  C(0, j) );
                    // todo: should tileRelease()?
                    A.tileTick(0, 0);
                    B.tileTick(0, j);
                }
                catch (std::exception& e) {
                    err = __LINE__;
                }
            }
        }
    }
    else {
        // side == Right
        #pragma omp parallel for schedule(dynamic, 1) slate_omp_default_none \
            shared(A, B, C, err) firstprivate(layout, side, alpha, beta)
        for (int64_t i = 0; i < C.mt(); ++i) {
            if (C.tileIsLocal(i, 0)) {
                try {
                    A.tileGetForReading(0, 0, LayoutConvert(layout));
                    B.tileGetForReading(i, 0, LayoutConvert(layout));
                    C.tileGetForWriting(i, 0, LayoutConvert(layout));
                    tile::hemm(
                        side,
                        alpha, A(0, 0), B(i, 0),
                        beta,  C(i, 0) );
                    // todo: should tileRelease()?
                    A.tileTick(0, 0);
                    B.tileTick(i, 0);
                }
                catch (std::exception& e) {
                    err = __LINE__;
                }
            }
        }
    }

    if (err)
        throw std::exception();
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void hemm< Target::HostTask, float >(
    Side side,
    float alpha, HermitianMatrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  Matrix<float>&& C,
    int priority, Options const& opts);

template
void hemm<Target::HostNest, float>(
    Side side,
    float alpha, HermitianMatrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  Matrix<float>&& C,
    int priority, Options const& opts);

// ----------------------------------------
template
void hemm<Target::HostTask, double>(
    Side side,
    double alpha, HermitianMatrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  Matrix<double>&& C,
    int priority, Options const& opts);

template
void hemm<Target::HostNest, double>(
    Side side,
    double alpha, HermitianMatrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  Matrix<double>&& C,
    int priority, Options const& opts);

// ----------------------------------------
template
void hemm< Target::HostTask, std::complex<float> >(
    Side side,
    std::complex<float> alpha, HermitianMatrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  Matrix< std::complex<float> >&& C,
    int priority, Options const& opts);

template
void hemm< Target::HostNest, std::complex<float> >(
    Side side,
    std::complex<float> alpha, HermitianMatrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  Matrix< std::complex<float> >&& C,
    int priority, Options const& opts);

// ----------------------------------------
template
void hemm< Target::HostTask, std::complex<double> >(
    Side side,
    std::complex<double> alpha, HermitianMatrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  Matrix< std::complex<double> >&& C,
    int priority, Options const& opts);

template
void hemm< Target::HostNest, std::complex<double> >(
    Side side,
    std::complex<double> alpha, HermitianMatrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  Matrix< std::complex<double> >&& C,
    int priority, Options const& opts);

} // namespace internal
} // namespace slate
