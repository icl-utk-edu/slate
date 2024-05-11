// Copyright (c) 2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/types.hh"
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
void trsmA_addmod(Side side, Uplo uplo, scalar_t alpha,
                  Matrix<scalar_t>&& A,
                  Matrix<scalar_t>&& U,
                  Matrix<scalar_t>&& VT,
                  std::vector<blas::real_type<scalar_t>>&& S,
                  Matrix<scalar_t>&& B,
                  BlockFactor blockFactorType,
                  int64_t ib, int priority, Layout layout, int64_t queue_index)
{
    trsmA_addmod(internal::TargetType<target>(),
                 side, uplo, alpha, A, U, VT, S, B,
                 blockFactorType, ib, priority, layout, queue_index);
}

//------------------------------------------------------------------------------
/// Triangular solve matrix (multiple right-hand sides).
/// Host OpenMP task implementation.
/// @ingroup trsm_internal
///
template <typename scalar_t>
void trsmA_addmod(internal::TargetType<Target::HostTask>,
                  Side side, Uplo uplo, scalar_t alpha,
                  Matrix<scalar_t>& A,
                  Matrix<scalar_t>& U,
                  Matrix<scalar_t>& VT,
                  std::vector<blas::real_type<scalar_t>>& S,
                  Matrix<scalar_t>& B,
                  BlockFactor blockFactorType,
                  int64_t ib, int priority, Layout layout, int64_t queue_index)
{
    // CPU assumes column major
    // todo: relax this assumption, by allowing Tile_blas.hh::trsm()
    //       to take layout param
    // todo: optimize for the number of layout conversions,
    //       by watching 'layout' and 'B(i, j).layout()'
    assert(layout == Layout::ColMajor);
    assert(U.mt() == 1);
    assert(VT.mt() == 1);
    assert(A.mt() == 1);
    assert(A.tileIsLocal(0,0) == U.tileIsLocal(0,0));
    assert(A.tileIsLocal(0,0) == VT.tileIsLocal(0,0));

    if (A.tileIsLocal(0, 0)) {
        A .tileGetForReading(0, 0, LayoutConvert(layout));
        if (uplo == Uplo::Lower) {
            U.tileGetForReading(0, 0, LayoutConvert(layout));
        }
        else {
            VT.tileGetForReading(0, 0, LayoutConvert(layout));
        }
    }

    #pragma omp taskgroup
    if (side == Side::Right) {
        assert(B.nt() == 1);
        if (A.tileIsLocal(0, 0)) {
            for (int64_t i = 0; i < B.mt(); ++i) {
                #pragma omp task slate_omp_default_none \
                    shared( A, U, VT, S, B ) \
                    firstprivate(i, layout, side, uplo, ib) priority(priority)
                {
                    B.tileGetForWriting(i, 0, LayoutConvert(layout));
                    tile::trsm_addmod(blockFactorType, ib, side, uplo, alpha,
                                      A(0, 0), U(0, 0), VT(0, 0), S, B(i, 0));
                }
            }
        }
    }
    else {
        assert(B.mt() == 1);
        if (A.tileIsLocal(0, 0)) {
            for (int64_t j = 0; j < B.nt(); ++j) {
                #pragma omp task slate_omp_default_none \
                    shared( A, U, VT, S, B ) \
                    firstprivate(j, layout, side, uplo, ib) priority(priority)
                {
                    B.tileGetForWriting(0, j, LayoutConvert(layout));
                    tile::trsm_addmod(blockFactorType, ib, side, uplo, alpha,
                                      A(0, 0), U(0, 0), VT(0, 0), S, B(0, j));
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
void trsmA_addmod(internal::TargetType<Target::HostNest>,
                  Side side, Uplo uplo, scalar_t alpha,
                  Matrix<scalar_t>& A,
                  Matrix<scalar_t>& U,
                  Matrix<scalar_t>& VT,
                  std::vector<blas::real_type<scalar_t>>& S,
                  Matrix<scalar_t>& B,
                  BlockFactor blockFactorType,
                  int64_t ib, int priority, Layout layout, int64_t queue_index)
{
    slate_not_implemented("Target::HostNest isn't yet supported.");
}

//------------------------------------------------------------------------------
/// Triangular solve matrix (multiple right-hand sides).
/// Host batched implementation.
/// @ingroup trsm_internal
///
template <typename scalar_t>
void trsmA_addmod(internal::TargetType<Target::HostBatch>,
                  Side side, Uplo uplo, scalar_t alpha,
                  Matrix<scalar_t>& A,
                  Matrix<scalar_t>& U,
                  Matrix<scalar_t>& VT,
                  std::vector<blas::real_type<scalar_t>>& S,
                  Matrix<scalar_t>& B,
                  BlockFactor blockFactorType,
                  int64_t ib, int priority, Layout layout, int64_t queue_index)
{
    slate_not_implemented("Target::HostBatch isn't yet supported.");
}

//------------------------------------------------------------------------------
/// Triangular solve matrix (multiple right-hand sides).
/// GPU device batched cuBLAS implementation.
/// @ingroup trsm_internal
///
template <typename scalar_t>
void trsmA_addmod(internal::TargetType<Target::Devices>,
                  Side side, Uplo uplo, scalar_t alpha,
                  Matrix<scalar_t>& A,
                  Matrix<scalar_t>& U,
                  Matrix<scalar_t>& VT,
                  std::vector<blas::real_type<scalar_t>>& S,
                  Matrix<scalar_t>& B,
                  BlockFactor blockFactorType,
                  int64_t ib, int priority, Layout layout, int64_t queue_index)
{
    slate_not_implemented("Target::Devices isn't yet supported.");
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void trsmA_addmod<Target::HostTask, float>(
    Side side, Uplo uplo, float alpha,
    Matrix<float>&& A,
    Matrix<float>&& U,
    Matrix<float>&& VT,
    std::vector<float>&& S,
    Matrix<float>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index);

template
void trsmA_addmod<Target::HostNest, float>(
    Side side, Uplo uplo, float alpha,
    Matrix<float>&& A,
    Matrix<float>&& U,
    Matrix<float>&& VT,
    std::vector<float>&& S,
    Matrix<float>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index);

template
void trsmA_addmod<Target::HostBatch, float>(
    Side side, Uplo uplo, float alpha,
    Matrix<float>&& A,
    Matrix<float>&& U,
    Matrix<float>&& VT,
    std::vector<float>&& S,
    Matrix<float>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index);

template
void trsmA_addmod<Target::Devices, float>(
    Side side, Uplo uplo, float alpha,
    Matrix<float>&& A,
    Matrix<float>&& U,
    Matrix<float>&& VT,
    std::vector<float>&& S,
    Matrix<float>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index);

// ----------------------------------------
template
void trsmA_addmod<Target::HostTask, double>(
    Side side, Uplo uplo, double alpha,
    Matrix<double>&& A,
    Matrix<double>&& U,
    Matrix<double>&& VT,
    std::vector<double>&& S,
    Matrix<double>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index);

template
void trsmA_addmod<Target::HostNest, double>(
    Side side, Uplo uplo, double alpha,
    Matrix<double>&& A,
    Matrix<double>&& U,
    Matrix<double>&& VT,
    std::vector<double>&& S,
    Matrix<double>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index);

template
void trsmA_addmod<Target::HostBatch, double>(
    Side side, Uplo uplo, double alpha,
    Matrix<double>&& A,
    Matrix<double>&& U,
    Matrix<double>&& VT,
    std::vector<double>&& S,
    Matrix<double>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index);

template
void trsmA_addmod<Target::Devices, double>(
    Side side, Uplo uplo, double alpha,
    Matrix<double>&& A,
    Matrix<double>&& U,
    Matrix<double>&& VT,
    std::vector<double>&& S,
    Matrix<double>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index);

// ----------------------------------------
template
void trsmA_addmod< Target::HostTask, std::complex<float> >(
    Side side, Uplo uplo, std::complex<float> alpha,
    Matrix<std::complex<float>>&& A,
    Matrix<std::complex<float>>&& U,
    Matrix<std::complex<float>>&& VT,
    std::vector<float>&& S,
    Matrix<std::complex<float>>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index);

template
void trsmA_addmod< Target::HostNest, std::complex<float> >(
    Side side, Uplo uplo, std::complex<float> alpha,
    Matrix<std::complex<float>>&& A,
    Matrix<std::complex<float>>&& U,
    Matrix<std::complex<float>>&& VT,
    std::vector<float>&& S,
    Matrix<std::complex<float>>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index);

template
void trsmA_addmod< Target::HostBatch, std::complex<float> >(
    Side side, Uplo uplo, std::complex<float> alpha,
    Matrix<std::complex<float>>&& A,
    Matrix<std::complex<float>>&& U,
    Matrix<std::complex<float>>&& VT,
    std::vector<float>&& S,
    Matrix<std::complex<float>>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index);

template
void trsmA_addmod< Target::Devices, std::complex<float> >(
    Side side, Uplo uplo, std::complex<float> alpha,
    Matrix<std::complex<float>>&& A,
    Matrix<std::complex<float>>&& U,
    Matrix<std::complex<float>>&& VT,
    std::vector<float>&& S,
    Matrix<std::complex<float>>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index);

// ----------------------------------------
template
void trsmA_addmod< Target::HostTask, std::complex<double> >(
    Side side, Uplo uplo, std::complex<double> alpha,
    Matrix<std::complex<double>>&& A,
    Matrix<std::complex<double>>&& U,
    Matrix<std::complex<double>>&& VT,
    std::vector<double>&& S,
    Matrix<std::complex<double>>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index);

template
void trsmA_addmod< Target::HostNest, std::complex<double> >(
    Side side, Uplo uplo, std::complex<double> alpha,
    Matrix<std::complex<double>>&& A,
    Matrix<std::complex<double>>&& U,
    Matrix<std::complex<double>>&& VT,
    std::vector<double>&& S,
    Matrix<std::complex<double>>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index);

template
void trsmA_addmod< Target::HostBatch, std::complex<double> >(
    Side side, Uplo uplo, std::complex<double> alpha,
    Matrix<std::complex<double>>&& A,
    Matrix<std::complex<double>>&& U,
    Matrix<std::complex<double>>&& VT,
    std::vector<double>&& S,
    Matrix<std::complex<double>>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index);

template
void trsmA_addmod< Target::Devices, std::complex<double> >(
    Side side, Uplo uplo, std::complex<double> alpha,
    Matrix<std::complex<double>>&& A,
    Matrix<std::complex<double>>&& U,
    Matrix<std::complex<double>>&& VT,
    std::vector<double>&& S,
    Matrix<std::complex<double>>&& B,
    BlockFactor blockFactorType,
    int64_t ib, int priority, Layout layout, int64_t queue_index);

} // namespace internal
} // namespace slate
