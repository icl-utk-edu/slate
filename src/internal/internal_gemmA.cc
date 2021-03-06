// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/types.hh"
#include "slate/Tile_blas.hh"
#include "internal/internal.hh"
#include "internal/internal_batch.hh"

#ifdef SLATE_WITH_MKL
    #include <mkl_cblas.h>
#else
    #include <cblas.h>
#endif

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// General matrix multiply for a left-looking update.
/// Does computation where the A tiles are local.
/// If needed (e.g., not local), inserts C tiles and sets beta to 0.
/// On output, partial products Cik exist distributed wherever Aij is local,
/// for all i = 0 : A.mt, j = 0 : A.nt, k=.
/// Dispatches to target implementations.
/// In the complex case,
/// if $op(C)$ is transpose, then $op(A)$ and $op(B)$ cannot be conjTranspose;
/// if $op(C)$ is conjTranspose, then $op(A)$ and $op(B)$ cannot be transpose.
/// @ingroup gemm_internal
///
template <Target target, typename scalar_t>
void gemmA(scalar_t alpha, Matrix<scalar_t>&& A,
                           Matrix<scalar_t>&& B,
           scalar_t beta,  Matrix<scalar_t>&& C,
           Layout layout, int priority)
{
    if (C.is_complex &&
        ((C.op() == Op::Trans &&
         (A.op() == Op::ConjTrans || B.op() == Op::ConjTrans)) ||
         (C.op() == Op::ConjTrans &&
         (A.op() == Op::Trans || B.op() == Op::Trans))))
    {
        throw std::exception();
    }

    gemmA(internal::TargetType<target>(),
          alpha, A,
                 B,
          beta,  C,
          layout, priority);
}

//------------------------------------------------------------------------------
/// General matrix multiply for a left-looking update.
/// This routine consists of two steps:  the management of the memory
/// and the actual computation.
/// Note that this routine may insert some tiles that must be erased later.
/// The original intent was to erase them when calling the ListReduce routine.
/// First step:
///   It iterates over the tiles of A and gets the needed tiles of B and C.
///   In the case where the corresponding tiles of C do not exist, it
///   acquires them. It means these tiles are created and inserted as workspace.
///   It is expected that they are erased either by the calling routine or
///   through the call to ListReduce.
/// Second step:
///   As soon as the data is ready, the routine calls gemm. However, the beta
///   is used once and only in the case where the current tile of C existed
///   prior the call to this routine. Otherwise, the beta value is replaced
///   by zero.
///   In any case, the internal value beta_j is set to one to allow the
///   accumulation in the tile.
/// Host OpenMP task implementation.
/// @ingroup gemm_internal
///
template <typename scalar_t>
void gemmA(internal::TargetType<Target::HostTask>,
           scalar_t alpha, Matrix<scalar_t>& A,
                           Matrix<scalar_t>& B,
           scalar_t beta,  Matrix<scalar_t>& C,
           Layout layout, int priority)
{
    // check dimensions
    assert(A.nt() == B.mt());
    assert(A.mt() == C.mt());

    int err   = 0;
    // This assumes that if a tile has to be acquired, then all tiles
    // have to be acquired
    // TODO make it a matrix of the C tiles involved c.TileAcquire(i, k)
    int c_tile_acquired = 0;
    #pragma omp taskgroup
    for (int64_t i = 0; i < A.mt(); ++i) {
        for (int64_t j = 0; j < A.nt(); ++j) {
            if (A.tileIsLocal(i, j)) {
                #pragma omp task slate_omp_default_none \
                    shared( A, B, C, err, c_tile_acquired ) \
                    firstprivate(i, j, layout) priority(priority)
                {
                    try {
                        A.tileGetForReading(i, j, LayoutConvert(layout));
                        for (int64_t k = 0; k < B.nt(); ++k) {
                            B.tileGetForReading(j, k, LayoutConvert(layout));

                            if (C.tileIsLocal(i, k)) {
                                C.tileGetForWriting(i, k, LayoutConvert(layout));
                            }
                            else {
                                if (! C.tileExists(i, k)) {
                                    c_tile_acquired = 1;
                                    #pragma omp critical
                                    {
                                        C.tileAcquire( i, k, HostNum, layout );
                                    }
                                }
                            }
                        }
                    }
                    catch (std::exception& e) {
                        err = __LINE__;
                    }
                }
            }
        }
    }

    #pragma omp taskgroup
    for (int64_t i = 0; i < A.mt(); ++i) {
        #pragma omp task slate_omp_default_none \
            shared( A, B, C, err ) \
            firstprivate(i, alpha, beta, c_tile_acquired) priority(priority)
        {
            try {

                scalar_t beta_j;
                for (int64_t k = 0; k < B.nt(); ++k) {
                    if (! c_tile_acquired || C.tileIsLocal(i, k)) {
                        beta_j = beta;
                    }
                    else {
                        beta_j = scalar_t(0.0);
                    }
                    bool Cik_modified = false;
                    for (int64_t j = 0; j < A.nt(); ++j) {
                        if (A.tileIsLocal(i, j)) {
                            gemm(alpha,  A(i, j),
                                         B(j, k),
                                 beta_j, C(i, k));

                            beta_j = scalar_t(1.0);

                            A.tileTick(i, j);
                            B.tileTick(j, k);
                            Cik_modified = true;
                        }
                    }
                    if (Cik_modified)
                        // mark this tile modified
                        C.tileModified(i, k);
                }
            }
            catch (std::exception& e) {
                err = __LINE__;
            }
        }
    }

    if (err)
        slate_error(std::string("Error in omp-task line: ")+std::to_string(err));
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void gemmA<Target::HostTask, float>(
    float alpha, Matrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  Matrix<float>&& C,
    Layout layout, int priority);

template
void gemmA<Target::HostTask, double>(
    double alpha, Matrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  Matrix<double>&& C,
    Layout layout, int priority);

template
void gemmA< Target::HostTask, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  Matrix< std::complex<float> >&& C,
    Layout layout, int priority);

template
void gemmA< Target::HostTask, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  Matrix< std::complex<double> >&& C,
    Layout layout, int priority);

} // namespace internal
} // namespace slate
