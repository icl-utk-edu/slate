// Copyright (c) 2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_TILE_TRSM_ADDMOD_HH
#define SLATE_TILE_TRSM_ADDMOD_HH

#include "internal/internal.hh"
#include "slate/Tile.hh"
#include "slate/types.hh"

#include <blas.hh>
#include <lapack.hh>

namespace slate {
namespace tile {

template <typename scalar_t>
scalar_t* matrix_offset(Layout layout, scalar_t* A, int64_t lda, int64_t i, int64_t j)
{
    if (layout == Layout::ColMajor) {
        return A + i + j*lda;
    }
    else {
        return A + i*lda + j;
    }
}

template <typename scalar_t>
void diag_solve(Layout layout, Side side, int64_t mb, int64_t nb,
                blas::real_type<scalar_t>* S,
                scalar_t* B, int64_t ldb,
                scalar_t* C, int64_t ldc)
{
    if (side == Side::Left) {
        if (layout == Layout::ColMajor) {
            assert(ldb >= mb);
            assert(ldc >= mb);
            for (int64_t j = 0; j < nb; j++) {
                for (int64_t i = 0; i < mb; i++) {
                    C[i + j*ldc] = B[i + j*ldb] / S[i];
                }
            }
        }
        else {
            assert(ldb >= nb);
            assert(ldc >= nb);
            for (int64_t i = 0; i < mb; i++) {
                for (int64_t j = 0; j < nb; j++) {
                    C[i*ldc + j] = B[i*ldb + j] / S[i];
                }
            }
        }
    }
    else {
        if (layout == Layout::ColMajor) {
            assert(ldb >= mb);
            assert(ldc >= mb);
            for (int64_t j = 0; j < nb; j++) {
                for (int64_t i = 0; i < mb; i++) {
                    C[i + j*ldc] = B[i + j*ldb] / S[j];
                }
            }
        }
        else {
            assert(ldb >= nb);
            assert(ldc >= nb);
            for (int64_t i = 0; i < mb; i++) {
                for (int64_t j = 0; j < nb; j++) {
                    C[i*ldc + j] = B[i*ldb + j] / S[j];
                }
            }
        }
    }
}

template<class scalar_t>
void lacpy_layout(Layout layout, int64_t mb, int64_t nb,
                  scalar_t A, int64_t lda, scalar_t B, int64_t ldb) {
    if (layout == Layout::ColMajor) {
        lapack::lacpy(lapack::MatrixType::General, mb, nb,
                      A, lda, B, ldb);
    }
    else {
        lapack::lacpy(lapack::MatrixType::General, nb, mb,
                      A, lda, B, ldb);
    }
}

template <typename scalar_t>
void trsm_addmod_recur_lower_left(int64_t ib, Layout layout,
                                  int64_t mb, int64_t nb,
                                  scalar_t alpha,
                                  scalar_t* A, int64_t lda,
                                  scalar_t* U, int64_t ldu,
                                  blas::real_type<scalar_t>* S,
                                  scalar_t* B, int64_t ldb,
                                  scalar_t* work, int64_t ldwork)

{
    scalar_t one = 1.0;
    scalar_t zero = 0.0;

    if (mb <= ib) {
        blas::gemm(layout,
                   Op::ConjTrans, Op::NoTrans,
                   mb, nb, mb,
                   alpha, U, ldu,
                          B, ldb,
                   zero,  work, ldwork);

        lacpy_layout(layout, mb, nb,
                     work, ldwork, B, ldb);
    }
    else {
        int64_t m1 = (((mb-1)/ib)/2+1) * ib; // half the tiles, rounded up
        int64_t m2 = mb-m1;

        trsm_addmod_recur_lower_left(ib, layout, m1, nb, alpha,
                A, lda,
                U, ldu,
                S,
                B, ldb,
                work, ldwork);

        blas::gemm(layout, Op::NoTrans, Op::NoTrans, m2, nb, m1,
                -one,  matrix_offset(layout, A, lda, m1, 0), lda,
                       matrix_offset(layout, B, ldb, 0,  0), ldb,
                alpha, matrix_offset(layout, B, ldb, m1, 0), ldb);

        trsm_addmod_recur_lower_left(ib, layout, m2, nb, one,
                matrix_offset(layout, A, lda, m1, m1), lda,
                matrix_offset(layout, U, ldu, m1, m1), ldu,
                S + m1,
                matrix_offset(layout, B, ldu, m1, 0),  ldb,
                work, ldwork);
    }
}

template <typename scalar_t>
void trsm_addmod_recur_upper_left(int64_t ib, Layout layout,
                                  int64_t mb, int64_t nb,
                                  scalar_t alpha,
                                  scalar_t* A, int64_t lda,
                                  scalar_t* U, int64_t ldu,
                                  blas::real_type<scalar_t>* S,
                                  scalar_t* B, int64_t ldb,
                                  scalar_t* work, int64_t ldwork)

{
    scalar_t one = 1.0;
    scalar_t zero = 0.0;

    if (mb <= ib) {
        diag_solve(layout, Side::Left, mb, nb,
                   S, B, ldb, work, ldwork);

        blas::gemm(layout,
                   Op::ConjTrans, Op::NoTrans,
                   mb, nb, mb,
                   alpha, A, lda,
                          work, ldwork,
                   zero,  B, ldb);
    }
    else {
        int64_t m1 = (((mb-1)/ib+1)/2) * ib; // half the tiles, rounded down
        int64_t m2 = mb-m1;

        trsm_addmod_recur_upper_left(ib, layout, m2, nb, alpha,
                matrix_offset(layout, A, lda, m1, m1), lda,
                matrix_offset(layout, U, ldu, m1, m1), ldu,
                S + m1,
                matrix_offset(layout, B, ldu, m1, 0),  ldb,
                work, ldwork);

        blas::gemm(layout, Op::NoTrans, Op::NoTrans, m1, nb, m2,
                -one,  matrix_offset(layout, A, lda, 0,  m1), lda,
                       matrix_offset(layout, B, ldb, m1, 0),  ldb,
                alpha, matrix_offset(layout, B, ldb, 0,  0),  ldb);

        trsm_addmod_recur_upper_left(ib, layout, m1, nb, one,
                A, lda,
                U, ldu,
                S,
                B, ldb,
                work, ldwork);
    }
}

template <typename scalar_t>
void trsm_addmod_recur_lower_right(int64_t ib, Layout layout,
                                   int64_t mb, int64_t nb,
                                   scalar_t alpha,
                                   scalar_t* A, int64_t lda,
                                   scalar_t* U, int64_t ldu,
                                   blas::real_type<scalar_t>* S,
                                   scalar_t* B, int64_t ldb,
                                   scalar_t* work, int64_t ldwork)

{
    scalar_t one = 1.0;
    scalar_t zero = 0.0;

    if (nb <= ib) {
        // halt recursion
        blas::gemm(layout,
                   Op::NoTrans, Op::ConjTrans,
                   mb, nb, nb,
                   alpha, B, ldb,
                          U, ldu,
                   zero,  work, ldwork);

        lacpy_layout(layout, mb, nb,
                     work, ldwork, B, ldb);
    }
    else {
        // recurse
        int64_t n1 = (((nb-1)/ib+1)/2) * ib; // half the tiles, rounded down
        int64_t n2 = nb-n1;

        trsm_addmod_recur_lower_right(ib, layout, mb, n2, alpha,
                matrix_offset(layout, A, lda, n1, n1), lda,
                matrix_offset(layout, U, ldu, n1, n1), ldu,
                S + n1,
                matrix_offset(layout, B, ldu, 0,  n1),  ldb,
                work, ldwork);

        blas::gemm(layout, Op::NoTrans, Op::NoTrans, mb, n1, n2,
                -one,  matrix_offset(layout, B, ldb, 0,  n1), ldb,
                       matrix_offset(layout, A, lda, n1, 0),  lda,
                alpha, matrix_offset(layout, B, ldb, 0,  0),  ldb);

        trsm_addmod_recur_lower_right(ib, layout, mb, n1, one,
                A, lda,
                U, ldu,
                S,
                B, ldb,
                work, ldwork);
    }
}

template <typename scalar_t>
void trsm_addmod_recur_upper_right(int64_t ib, Layout layout,
                                   int64_t mb, int64_t nb,
                                   scalar_t alpha,
                                   scalar_t* A, int64_t lda,
                                   scalar_t* U, int64_t ldu,
                                   blas::real_type<scalar_t>* S,
                                   scalar_t* B, int64_t ldb,
                                   scalar_t* work, int64_t ldwork)

{
    scalar_t one = 1.0;
    scalar_t zero = 0.0;
    if (nb <= ib) {
        // halt recursion
        blas::gemm(layout,
                   Op::NoTrans, Op::ConjTrans,
                   mb, nb, nb,
                   alpha, B, ldb,
                          A, lda,
                   zero,  work, ldwork);

        diag_solve(layout, Side::Right, mb, nb,
                   S, work, ldwork, B, ldb);
    }
    else {
        // recurse
        int64_t n1 = (((nb-1)/ib)/2+1) * ib; // half the tiles, rounded up
        int64_t n2 = nb-n1;

        trsm_addmod_recur_upper_right(ib, layout, mb, n1, alpha,
                A, lda,
                U, ldu,
                S,
                B, ldb,
                work, ldwork);

        blas::gemm(layout, Op::NoTrans, Op::NoTrans, mb, n2, n1,
                -one,  matrix_offset(layout, B, ldb, 0,  0), ldb,
                       matrix_offset(layout, A, lda, 0, n1), lda,
                alpha, matrix_offset(layout, B, ldb, 0, n1), ldb);

        trsm_addmod_recur_upper_right(ib, layout, mb, n2, one,
                matrix_offset(layout, A, lda, n1, n1), lda,
                matrix_offset(layout, U, ldu, n1, n1), ldu,
                S + n1,
                matrix_offset(layout, B, ldb, 0, n1),  ldb,
                work, ldwork);
    }
}


template <typename scalar_t>
void trsm_addmod(int64_t ib, Side side, Uplo uplo, scalar_t alpha,
                 Tile<scalar_t> A,
                 Tile<scalar_t> U,
                 std::vector<blas::real_type<scalar_t>> S,
                 Tile<scalar_t> B)
{
    slate_assert(U.uploPhysical() == Uplo::General);
    slate_assert(B.uploPhysical() == Uplo::General);
    slate_assert(U.layout() == B.layout());

    slate_assert(A.mb() == A.nb());
    slate_assert(U.mb() == U.nb());
    slate_assert(A.mb() == int64_t(S.size()));
    slate_assert(A.mb() == U.mb());

    int64_t mb = B.mb(), nb = B.nb();

    // TODO this allocation can be made smaller
    int64_t work_stride = B.layout() == Layout::ColMajor ? mb : nb;
    std::vector<scalar_t> work_vect(mb*nb);
    scalar_t* work = work_vect.data();

    if (uplo == Uplo::Lower) {
        if (side == Side::Right) {
            // Lower, Right
            slate_assert(U.mb() == nb);

            trsm_addmod_recur_lower_right(ib, B.layout(), mb, nb, alpha,
                                          A.data(), A.stride(),
                                          U.data(), U.stride(),
                                          S.data(),
                                          B.data(), B.stride(),
                                          work, work_stride);
        }
        else {
            // Lower, Left
            slate_assert(U.mb() == mb);

            trsm_addmod_recur_lower_left(ib, B.layout(), mb, nb, alpha,
                                         A.data(), A.stride(),
                                         U.data(), U.stride(),
                                         S.data(),
                                         B.data(), B.stride(),
                                         work, work_stride);
        }
    }
    else {
        if (side == Side::Right) {
            // Upper, Right
            slate_assert(A.mb() == nb);

            trsm_addmod_recur_upper_right(ib, B.layout(), mb, nb, alpha,
                                          A.data(), A.stride(),
                                          U.data(), U.stride(),
                                          S.data(),
                                          B.data(), B.stride(),
                                          work, work_stride);
        }
        else {
            // Upper, Left
            slate_assert(A.mb() == mb);

            trsm_addmod_recur_upper_left(ib, B.layout(), mb, nb, alpha,
                                         A.data(), A.stride(),
                                         U.data(), U.stride(),
                                         S.data(),
                                         B.data(), B.stride(),
                                         work, work_stride);
        }
    }
}


} // namespace internal
} // namespace slate
#endif // SLATE_TILE_TRSM_ADDMOD_HH
