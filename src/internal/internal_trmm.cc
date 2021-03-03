// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/TriangularMatrix.hh"
#include "slate/types.hh"
#include "slate/Tile_blas.hh"
#include "internal/internal.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Triangular matrix multiply.
/// Dispatches to target implementations.
/// @ingroup trmm_internal
///
template <Target target, typename scalar_t>
void trmm(Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>&& A,
                                    Matrix<scalar_t>&& B,
          int priority)
{
    trmm(internal::TargetType<target>(),
         side,
         alpha, A,
                B,
         priority);
}

//------------------------------------------------------------------------------
/// Triangular matrix multiply.
/// Host OpenMP task implementation.
/// @ingroup trmm_internal
///
template <typename scalar_t>
void trmm(internal::TargetType<Target::HostTask>,
          Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>& A,
                                    Matrix<scalar_t>& B,
          int priority)
{
    // CPU assumes column major
    // todo: relax this assumption, by allowing Tile_blas.hh::trmm() to take layout param
    // todo: optimize for the number of layout conversions,
    //       by watching 'layout' and 'C(i, j).layout()'
    const Layout layout = Layout::ColMajor;

    assert(A.mt() == 1);

    // alternatively, if (side == right), (conj)-transpose both A and B,
    // then assume side == left; see slate::trmm
    if (side == Side::Right) {
        assert(B.nt() == 1);
        for (int64_t i = 0; i < B.mt(); ++i) {
            if (B.tileIsLocal(i, 0)) {
                #pragma omp task shared(A, B)
                {
                    A.tileGetForReading(0, 0, LayoutConvert(layout));
                    B.tileGetForWriting(i, 0, LayoutConvert(layout));
                    trmm(side, A.diag(),
                         alpha, A(0, 0),
                                B(i, 0));
                    // todo: should tileRelease()?
                    A.tileTick(0, 0);
                }
            }
        }
    }
    else {
        assert(B.mt() == 1);
        for (int64_t j = 0; j < B.nt(); ++j) {
            if (B.tileIsLocal(0, j)) {
                #pragma omp task shared(A, B)
                {
                    A.tileGetForReading(0, 0, LayoutConvert(layout));
                    B.tileGetForWriting(0, j, LayoutConvert(layout));
                    trmm(side, A.diag(),
                         alpha, A(0, 0),
                                B(0, j));
                    // todo: should tileRelease()?
                    A.tileTick(0, 0);
                }
            }
        }
    }

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
/// Triangular matrix multiply.
/// Host nested OpenMP implementation.
/// @ingroup trmm_internal
///
template <typename scalar_t>
void trmm(internal::TargetType<Target::HostNest>,
          Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>& A,
                                    Matrix<scalar_t>& B,
          int priority)
{
    slate_not_implemented("Not available yet");
}

//------------------------------------------------------------------------------
/// Triangular matrix multiply.
/// Host batched implementation.
/// @ingroup trmm_internal
///
template <typename scalar_t>
void trmm(internal::TargetType<Target::HostBatch>,
          Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>& A,
                                    Matrix<scalar_t>& B,
          int priority)
{
    slate_not_implemented("Not available yet");
}
//------------------------------------------------------------------------------
/// Triangular matrix multiply.
/// GPU device batched cuBLAS implementation.
/// @ingroup trmm_internal
///
template <typename scalar_t>
void trmm(internal::TargetType<Target::Devices>,
          Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>& A,
                                    Matrix<scalar_t>& B,
          int priority)
{
    printf("%s %d NOT IMPLEMENTED YET\n", __FILE__, __LINE__);
    // CPU assumes column major
    // todo: relax this assumption, by allowing Tile_blas.hh::trmm() to take layout param
    // todo: optimize for the number of layout conversions,
    //       by watching 'layout' and 'C(i, j).layout()'
    const Layout layout = Layout::ColMajor;

    assert(A.mt() == 1);

    // alternatively, if (side == right), (conj)-transpose both A and B,
    // then assume side == left; see slate::trmm
    if (side == Side::Right) {
        assert(B.nt() == 1);
        for (int64_t i = 0; i < B.mt(); ++i) {
            if (B.tileIsLocal(i, 0)) {
                #pragma omp task shared(A, B)
                {
                    A.tileGetForReading(0, 0, LayoutConvert(layout));
                    B.tileGetForWriting(i, 0, LayoutConvert(layout));
                    trmm(side, A.diag(),
                         alpha, A(0, 0),
                                B(i, 0));
                    // todo: should tileRelease()?
                    A.tileTick(0, 0);
                }
            }
        }
    }
    else {
        assert(B.mt() == 1);
        for (int64_t j = 0; j < B.nt(); ++j) {
            if (B.tileIsLocal(0, j)) {
                #pragma omp task shared(A, B)
                {
                    A.tileGetForReading(0, 0, LayoutConvert(layout));
                    B.tileGetForWriting(0, j, LayoutConvert(layout));
                    trmm(side, A.diag(),
                         alpha, A(0, 0),
                                B(0, j));
                    // todo: should tileRelease()?
                    A.tileTick(0, 0);
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
void trmm<Target::HostTask, float>(
    Side side,
    float alpha, TriangularMatrix<float>&& A,
                           Matrix<float>&& B,
    int priority);

// ----------------------------------------
template
void trmm<Target::HostTask, double>(
    Side side,
    double alpha, TriangularMatrix<double>&& A,
                            Matrix<double>&& B,
    int priority);

// ----------------------------------------
template
void trmm< Target::HostTask, std::complex<float> >(
    Side side,
    std::complex<float> alpha, TriangularMatrix< std::complex<float> >&& A,
                                         Matrix< std::complex<float> >&& B,
    int priority);

// ----------------------------------------
template
void trmm< Target::HostTask, std::complex<double> >(
    Side side,
    std::complex<double> alpha, TriangularMatrix< std::complex<double> >&& A,
                                          Matrix< std::complex<double> >&& B,
    int priority);

// ----------------------------------------
template
void trmm<Target::HostNest, float>(
    Side side,
    float alpha, TriangularMatrix<float>&& A,
                           Matrix<float>&& B,
    int priority);

// ----------------------------------------
template
void trmm<Target::HostNest, double>(
    Side side,
    double alpha, TriangularMatrix<double>&& A,
                            Matrix<double>&& B,
    int priority);

// ----------------------------------------
template
void trmm< Target::HostNest, std::complex<float> >(
    Side side,
    std::complex<float> alpha, TriangularMatrix< std::complex<float> >&& A,
                                         Matrix< std::complex<float> >&& B,
    int priority);

// ----------------------------------------
template
void trmm< Target::HostNest, std::complex<double> >(
    Side side,
    std::complex<double> alpha, TriangularMatrix< std::complex<double> >&& A,
                                          Matrix< std::complex<double> >&& B,
    int priority);

// ----------------------------------------
template
void trmm<Target::HostBatch, float>(
    Side side,
    float alpha, TriangularMatrix<float>&& A,
                           Matrix<float>&& B,
    int priority);

// ----------------------------------------
template
void trmm<Target::HostBatch, double>(
    Side side,
    double alpha, TriangularMatrix<double>&& A,
                            Matrix<double>&& B,
    int priority);

// ----------------------------------------
template
void trmm< Target::HostBatch, std::complex<float> >(
    Side side,
    std::complex<float> alpha, TriangularMatrix< std::complex<float> >&& A,
                                         Matrix< std::complex<float> >&& B,
    int priority);

// ----------------------------------------
template
void trmm< Target::HostBatch, std::complex<double> >(
    Side side,
    std::complex<double> alpha, TriangularMatrix< std::complex<double> >&& A,
                                          Matrix< std::complex<double> >&& B,
    int priority);

// ----------------------------------------
template
void trmm<Target::Devices, float>(
    Side side,
    float alpha, TriangularMatrix<float>&& A,
                           Matrix<float>&& B,
    int priority);

// ----------------------------------------
template
void trmm<Target::Devices, double>(
    Side side,
    double alpha, TriangularMatrix<double>&& A,
                            Matrix<double>&& B,
    int priority);

// ----------------------------------------
template
void trmm< Target::Devices, std::complex<float> >(
    Side side,
    std::complex<float> alpha, TriangularMatrix< std::complex<float> >&& A,
                                         Matrix< std::complex<float> >&& B,
    int priority);

// ----------------------------------------
template
void trmm< Target::Devices, std::complex<double> >(
    Side side,
    std::complex<double> alpha, TriangularMatrix< std::complex<double> >&& A,
                                          Matrix< std::complex<double> >&& B,
    int priority);


} // namespace internal
} // namespace slate
