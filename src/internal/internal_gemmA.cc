// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
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
/// General matrix multiply for a left-looking update,
/// where B and C are single block columns.
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
/// General matrix multiply for a left-looking update,
/// where B and C are single block columns.
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
    assert(B.nt() == 1);
    assert(C.nt() == 1);
    assert(A.nt() == B.mt());
    assert(A.mt() == C.mt());

    int err = 0;
    for (int64_t i = 0; i < A.mt(); ++i) {
        for (int64_t j = 0; j < A.nt(); ++j) {
            if (A.tileIsLocal(i, j)) {
                #pragma omp task shared(A, B, C, err) priority(priority)
                {
                    try {
                        A.tileGetForReading(i, j, LayoutConvert(layout));
                        B.tileGetForReading(j, 0, LayoutConvert(layout));

                        if (C.tileIsLocal(i, 0)) {
                            C.tileGetForWriting(i, 0, LayoutConvert(layout));
                        }
                        else {
                            #pragma omp critical
                            {
                                C.tileAcquire(i, 0, C.hostNum(), layout);
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

    #pragma omp taskwait

    for (int64_t i = 0; i < A.mt(); ++i) {
        #pragma omp task shared(A, B, C, err) priority(priority)
        {
            try {

                scalar_t beta_j;
                if (C.tileIsLocal(i, 0))
                    beta_j = beta;
                else
                    beta_j = scalar_t(0.0);
                bool Ci0_modified = false;
                for (int64_t j = 0; j < A.nt(); ++j) {
                    if (A.tileIsLocal(i, j)) {
                        gemm(alpha,  A(i, j),
                                     B(j, 0),
                             beta_j, C(i, 0));

                        beta_j = scalar_t(1.0);

                        A.tileTick(i, j);
                        B.tileTick(j, 0);
                        Ci0_modified = true;
                    }
                }
                if (Ci0_modified)
                    // mark this tile modified
                    C.tileModified(i, 0);
            }
            catch (std::exception& e) {
                err = __LINE__;
            }
        }
    }

    #pragma omp taskwait

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
