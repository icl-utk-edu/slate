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
/// @ingroup hemmA_internal
///
template <Target target, typename scalar_t>
void hemmA(Side side,
           scalar_t alpha, HermitianMatrix<scalar_t>&& A,
                                    Matrix<scalar_t>&& B,
           scalar_t beta,  Matrix<scalar_t>&& C,
           int priority)
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

    hemmA(internal::TargetType<target>(),
          side,
          alpha, A,
                 B,
          beta,  C,
          priority);
}

//------------------------------------------------------------------------------
/// Hermitian matrix multiply to update trailing matrix.
/// Host OpenMP task implementation.
/// @ingroup hemmA_internal
///
template <typename scalar_t>
void hemmA(internal::TargetType<Target::HostTask>,
           Side side,
           scalar_t alpha, HermitianMatrix<scalar_t>& A,
                           Matrix<scalar_t>& B,
           scalar_t beta,  Matrix<scalar_t>& C,
           int priority)
{
    // CPU uses ColMajor
    // todo: relax this assumption, by allowing Tile_blas.hh::hemm() to take layout param
    // todo: optimize for the number of layout conversions,
    //       by watching 'layout' and 'C(i, j).layout()'
    const Layout layout = Layout::ColMajor;

    // 1 block tile for now
    assert(A.nt() == B.mt());
    assert(A.mt() == C.mt());

    int err = 0;
    #pragma omp taskgroup
    for (int64_t i = 0; i < A.mt(); ++i) {
        for (int64_t j = 0; j < A.nt(); ++j) {
            if (A.tileIsLocal(i, j)) {
                #pragma omp task slate_omp_default_none \
                    shared( A, B, C, err ) \
                    firstprivate(i, j, layout, side, alpha, beta) priority(priority)
                {
                    try {
                        A.tileGetForReading(i, j, LayoutConvert(layout));
                        for (int64_t k = 0; k < B.nt(); ++k) {
                            B.tileGetForReading(j, k, LayoutConvert(layout));
                            C.tileGetForWriting(i, k, LayoutConvert(layout));
                            tile::hemm(
                                side,
                                alpha, A(i, j), B(j, k),
                                beta,  C(i, k) );

                            // todo: should tileRelease()?
                            A.tileTick(i, j);
                            B.tileTick(j, k);
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
/// @ingroup hemmA_internal
///
template <typename scalar_t>
void hemmA(internal::TargetType<Target::HostNest>,
           Side side,
           scalar_t alpha, HermitianMatrix<scalar_t>& A,
                           Matrix<scalar_t>& B,
           scalar_t beta,  Matrix<scalar_t>& C,
           int priority)
{
    // CPU uses ColMajor
    // todo: relax this assumption, by allowing Tile_blas.hh::hemm() to take layout param
    // todo: optimize for the number of layout conversions,
    //       by watching 'layout' and 'C(i, j).layout()'
    const Layout layout = Layout::ColMajor;

    int err = 0;
    if (side == Side::Left) {
        #pragma omp parallel for schedule(dynamic, 1) slate_omp_default_none \
            shared(A, B, C, err) firstprivate(side, layout, alpha, beta)
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
            shared(A, B, C, err) firstprivate(side, layout, alpha, beta)
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
void hemmA< Target::HostTask, float >(
    Side side,
    float alpha, HermitianMatrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  Matrix<float>&& C,
    int priority);

template
void hemmA<Target::HostNest, float>(
    Side side,
    float alpha, HermitianMatrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  Matrix<float>&& C,
    int priority);

// ----------------------------------------
template
void hemmA<Target::HostTask, double>(
    Side side,
    double alpha, HermitianMatrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  Matrix<double>&& C,
    int priority);

template
void hemmA<Target::HostNest, double>(
    Side side,
    double alpha, HermitianMatrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  Matrix<double>&& C,
    int priority);

// ----------------------------------------
template
void hemmA< Target::HostTask, std::complex<float> >(
    Side side,
    std::complex<float> alpha, HermitianMatrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  Matrix< std::complex<float> >&& C,
    int priority);

template
void hemmA< Target::HostNest, std::complex<float> >(
    Side side,
    std::complex<float> alpha, HermitianMatrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  Matrix< std::complex<float> >&& C,
    int priority);

// ----------------------------------------
template
void hemmA< Target::HostTask, std::complex<double> >(
    Side side,
    std::complex<double> alpha, HermitianMatrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  Matrix< std::complex<double> >&& C,
    int priority);

template
void hemmA< Target::HostNest, std::complex<double> >(
    Side side,
    std::complex<double> alpha, HermitianMatrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  Matrix< std::complex<double> >&& C,
    int priority);

} // namespace internal
} // namespace slate
