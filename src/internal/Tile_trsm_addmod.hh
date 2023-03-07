// Copyright (c) 2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_TILE_TRSM_ADDMOD_HH
#define SLATE_TILE_TRSM_ADDMOD_HH

#include "internal/internal.hh"
#include "slate/Tile.hh"
#include "slate/types.hh"
#include "slate/enums.hh"

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

template <BlockFactor factorType, typename scalar_t>
void trsm_addmod_recur_lower_left(int64_t ib, Layout layout,
                                  int64_t mb, int64_t nb,
                                  scalar_t alpha,
                                  scalar_t* A, int64_t lda,
                                  scalar_t* U, int64_t ldu,
                                  scalar_t* B, int64_t ldb,
                                  scalar_t* work, int64_t ldwork)

{
    const scalar_t one = 1.0;
    [[maybe_unused]]
    const scalar_t zero = 0.0;

    if (mb <= ib) {
        // halt recursion
        if constexpr (factorType == BlockFactor::SVD
                      || factorType == BlockFactor::QLP
                      || factorType == BlockFactor::QRCP
                      || factorType == BlockFactor::QR) {
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
            slate_not_implemented( "Block factorization not implemented" );
        }
    }
    else {
        int64_t m1 = (((mb-1)/ib)/2+1) * ib; // half the tiles, rounded up
        int64_t m2 = mb-m1;

        trsm_addmod_recur_lower_left<factorType>(ib, layout, m1, nb, alpha,
                A, lda,
                U, ldu,
                B, ldb,
                work, ldwork);

        blas::gemm(layout, Op::NoTrans, Op::NoTrans, m2, nb, m1,
                -one,  matrix_offset(layout, A, lda, m1, 0), lda,
                       matrix_offset(layout, B, ldb, 0,  0), ldb,
                alpha, matrix_offset(layout, B, ldb, m1, 0), ldb);

        trsm_addmod_recur_lower_left<factorType>(ib, layout, m2, nb, one,
                matrix_offset(layout, A, lda, m1, m1), lda,
                matrix_offset(layout, U, ldu, m1, m1), ldu,
                matrix_offset(layout, B, ldb, m1, 0),  ldb,
                work, ldwork);
    }
}

template <BlockFactor factorType, typename scalar_t>
void trsm_addmod_recur_upper_left(int64_t ib, Layout layout,
                                  int64_t mb, int64_t nb,
                                  scalar_t alpha,
                                  scalar_t* A, int64_t lda,
                                  scalar_t* VT, int64_t ldvt,
                                  blas::real_type<scalar_t>* S,
                                  scalar_t* B, int64_t ldb,
                                  scalar_t* work, int64_t ldwork)

{
    const scalar_t one = 1.0;
    [[maybe_unused]]
    const scalar_t zero = 0.0;

    if (mb <= ib) {
        // halt recursion
        if constexpr (factorType == BlockFactor::SVD) {
            diag_solve(layout, Side::Left, mb, nb,
                       S, B, ldb, work, ldwork);

            blas::gemm(layout,
                       Op::ConjTrans, Op::NoTrans,
                       mb, nb, mb,
                       alpha, VT, ldvt,
                              work, ldwork,
                       zero,  B, ldb);
        }
        else if constexpr (factorType == BlockFactor::QLP
                           || factorType == BlockFactor::QRCP) {
            auto uplo = factorType == BlockFactor::QLP ? Uplo::Lower : Uplo::Upper;
            blas::trsm(layout, Side::Left, uplo,
                       Op::NoTrans, Diag::NonUnit, mb, nb,
                       alpha, A, lda, B, ldb);

            blas::gemm(layout,
                       Op::ConjTrans, Op::NoTrans,
                       mb, nb, mb,
                       alpha, VT, ldvt,
                              B, ldb,
                       zero,  work, ldwork);
            lacpy_layout(layout, mb, nb,
                         work, ldwork, B, ldb);
        }
        else if constexpr (factorType == BlockFactor::QR) {
            blas::trsm(layout, Side::Left, Uplo::Upper,
                       Op::NoTrans, Diag::NonUnit, mb, nb,
                       alpha, A, lda, B, ldb);
        }
        else {
            slate_not_implemented( "Block factorization not implemented" );
        }
    }
    else {
        int64_t m1 = (((mb-1)/ib+1)/2) * ib; // half the tiles, rounded down
        int64_t m2 = mb-m1;

        trsm_addmod_recur_upper_left<factorType>(ib, layout, m2, nb, alpha,
                matrix_offset(layout, A, lda, m1, m1), lda,
                matrix_offset(layout, VT, ldvt, m1, m1), ldvt,
                S + m1,
                matrix_offset(layout, B, ldb, m1, 0),  ldb,
                work, ldwork);

        blas::gemm(layout, Op::NoTrans, Op::NoTrans, m1, nb, m2,
                -one,  matrix_offset(layout, A, lda, 0,  m1), lda,
                       matrix_offset(layout, B, ldb, m1, 0),  ldb,
                alpha, matrix_offset(layout, B, ldb, 0,  0),  ldb);

        trsm_addmod_recur_upper_left<factorType>(ib, layout, m1, nb, one,
                A, lda,
                VT, ldvt,
                S,
                B, ldb,
                work, ldwork);
    }
}

template <BlockFactor factorType, typename scalar_t>
void trsm_addmod_recur_lower_right(int64_t ib, Layout layout,
                                   int64_t mb, int64_t nb,
                                   scalar_t alpha,
                                   scalar_t* A, int64_t lda,
                                   scalar_t* U, int64_t ldu,
                                   scalar_t* B, int64_t ldb,
                                   scalar_t* work, int64_t ldwork)

{
    const scalar_t one = 1.0;
    [[maybe_unused]]
    const scalar_t zero = 0.0;

    if (nb <= ib) {
        // halt recursion
        if constexpr (factorType == BlockFactor::SVD
                      || factorType == BlockFactor::QLP
                      || factorType == BlockFactor::QRCP
                      || factorType == BlockFactor::QR) {
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
            slate_not_implemented( "Block factorization not implemented" );
        }
    }
    else {
        // recurse
        int64_t n1 = (((nb-1)/ib+1)/2) * ib; // half the tiles, rounded down
        int64_t n2 = nb-n1;

        trsm_addmod_recur_lower_right<factorType>(ib, layout, mb, n2, alpha,
                matrix_offset(layout, A, lda, n1, n1), lda,
                matrix_offset(layout, U, ldu, n1, n1), ldu,
                matrix_offset(layout, B, ldu, 0,  n1),  ldb,
                work, ldwork);

        blas::gemm(layout, Op::NoTrans, Op::NoTrans, mb, n1, n2,
                -one,  matrix_offset(layout, B, ldb, 0,  n1), ldb,
                       matrix_offset(layout, A, lda, n1, 0),  lda,
                alpha, matrix_offset(layout, B, ldb, 0,  0),  ldb);

        trsm_addmod_recur_lower_right<factorType>(ib, layout, mb, n1, one,
                A, lda,
                U, ldu,
                B, ldb,
                work, ldwork);
    }
}

template <BlockFactor factorType, typename scalar_t>
void trsm_addmod_recur_upper_right(int64_t ib, Layout layout,
                                   int64_t mb, int64_t nb,
                                   scalar_t alpha,
                                   scalar_t* A, int64_t lda,
                                   scalar_t* VT, int64_t ldvt,
                                   blas::real_type<scalar_t>* S,
                                   scalar_t* B, int64_t ldb,
                                   scalar_t* work, int64_t ldwork)

{
    const scalar_t one = 1.0;
    [[maybe_unused]]
    const scalar_t zero = 0.0;

    if (nb <= ib) {
        // halt recursion
        if constexpr (factorType == BlockFactor::SVD) {
            blas::gemm(layout,
                       Op::NoTrans, Op::ConjTrans,
                       mb, nb, nb,
                       alpha, B, ldb,
                              VT, ldvt,
                       zero,  work, ldwork);

            diag_solve(layout, Side::Right, mb, nb,
                       S, work, ldwork, B, ldb);
        }
        else if constexpr (factorType == BlockFactor::QLP
                           || factorType == BlockFactor::QRCP) {
            blas::gemm(layout,
                       Op::NoTrans, Op::ConjTrans,
                       mb, nb, nb,
                       alpha, B, ldb,
                              VT, ldvt,
                       zero,  work, ldwork);
            lacpy_layout(layout, mb, nb,
                         work, ldwork, B, ldb);

            auto uplo = factorType == BlockFactor::QLP ? Uplo::Lower : Uplo::Upper;
            blas::trsm(Layout::ColMajor, Side::Right, uplo,
                       Op::NoTrans, Diag::NonUnit, mb, nb,
                       alpha, A, lda, B, ldb);
        }
        else if constexpr (factorType == BlockFactor::QR) {
            blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper,
                       Op::NoTrans, Diag::NonUnit, mb, nb,
                       alpha, A, lda, B, ldb);
        }
        else {
            slate_not_implemented( "Block factorization not implemented" );
        }
    }
    else {
        // recurse
        int64_t n1 = (((nb-1)/ib)/2+1) * ib; // half the tiles, rounded up
        int64_t n2 = nb-n1;

        trsm_addmod_recur_upper_right<factorType>(ib, layout, mb, n1, alpha,
                A, lda,
                VT, ldvt,
                S,
                B, ldb,
                work, ldwork);

        blas::gemm(layout, Op::NoTrans, Op::NoTrans, mb, n2, n1,
                -one,  matrix_offset(layout, B, ldb, 0,  0), ldb,
                       matrix_offset(layout, A, lda, 0, n1), lda,
                alpha, matrix_offset(layout, B, ldb, 0, n1), ldb);

        trsm_addmod_recur_upper_right<factorType>(ib, layout, mb, n2, one,
                matrix_offset(layout, A, lda, n1, n1), lda,
                matrix_offset(layout, VT, ldvt, n1, n1), ldvt,
                S + n1,
                matrix_offset(layout, B, ldb, 0, n1),  ldb,
                work, ldwork);
    }
}


template <BlockFactor factorType, typename scalar_t>
void trsm_addmod_helper(int64_t ib, Side side, Uplo uplo, scalar_t alpha,
                 Tile<scalar_t> A,
                 Tile<scalar_t> U,
                 Tile<scalar_t> VT,
                 std::vector<blas::real_type<scalar_t>>& S,
                 Tile<scalar_t> B)
{
    slate_assert(U.uploPhysical() == Uplo::General);
    slate_assert(VT.uploPhysical() == Uplo::General);
    slate_assert(B.uploPhysical() == Uplo::General);
    slate_assert(U.layout() == B.layout());
    slate_assert(VT.layout() == B.layout());

    slate_assert(A.mb() == A.nb());
    slate_assert(U.mb() == U.nb());
    slate_assert(A.mb() == U.mb());

    blas::real_type<scalar_t>* S_data = nullptr;
    if (factorType == BlockFactor::SVD) {
        slate_assert(A.mb() == int64_t(S.size()));
        S_data = S.data();
    }

    int64_t mb = B.mb(), nb = B.nb();

    // TODO this allocation can be made smaller
    int64_t work_stride = B.layout() == Layout::ColMajor ? mb : nb;
    std::vector<scalar_t> work_vect(mb*nb);
    scalar_t* work = work_vect.data();

    if (uplo == Uplo::Lower) {
        if (side == Side::Right) {
            // Lower, Right
            slate_assert(U.mb() == nb);

            trsm_addmod_recur_lower_right<factorType>(ib, B.layout(), mb, nb, alpha,
                                          A.data(), A.stride(),
                                          U.data(), U.stride(),
                                          B.data(), B.stride(),
                                          work, work_stride);
        }
        else {
            // Lower, Left
            slate_assert(U.mb() == mb);

            trsm_addmod_recur_lower_left<factorType>(ib, B.layout(), mb, nb, alpha,
                                         A.data(), A.stride(),
                                         U.data(), U.stride(),
                                         B.data(), B.stride(),
                                         work, work_stride);
        }
    }
    else {
        if (side == Side::Right) {
            // Upper, Right
            slate_assert(A.mb() == nb);

            trsm_addmod_recur_upper_right<factorType>(ib, B.layout(), mb, nb, alpha,
                                          A.data(), A.stride(),
                                         VT.data(),VT.stride(),
                                          S_data,
                                          B.data(), B.stride(),
                                          work, work_stride);
        }
        else {
            // Upper, Left
            slate_assert(A.mb() == mb);

            trsm_addmod_recur_upper_left<factorType>(ib, B.layout(), mb, nb, alpha,
                                         A.data(), A.stride(),
                                        VT.data(),VT.stride(),
                                         S_data,
                                         B.data(), B.stride(),
                                         work, work_stride);
        }
    }
}

template<typename scalar_t>
void trsm_addmod(BlockFactor factorType, int64_t ib, Side side, Uplo uplo, scalar_t alpha,
                 Tile<scalar_t> A,
                 Tile<scalar_t> U,
                 Tile<scalar_t> VT,
                 std::vector<blas::real_type<scalar_t>>& S,
                 Tile<scalar_t> B)
{
    if (factorType == BlockFactor::SVD) {
        trsm_addmod_helper<BlockFactor::SVD>(ib, side, uplo, alpha, A, U, VT, S, B);
    }
    else if (factorType == BlockFactor::QLP) {
        trsm_addmod_helper<BlockFactor::QLP>(ib, side, uplo, alpha, A, U, VT, S, B);
    }
    else if (factorType == BlockFactor::QRCP) {
        trsm_addmod_helper<BlockFactor::QRCP>(ib, side, uplo, alpha, A, U, VT, S, B);
    }
    else if (factorType == BlockFactor::QR) {
        trsm_addmod_helper<BlockFactor::QR>(ib, side, uplo, alpha, A, U, VT, S, B);
    }
    else {
        slate_not_implemented( "Block factorization not implemented" );
    }
}


} // namespace internal
} // namespace slate
#endif // SLATE_TILE_TRSM_ADDMOD_HH
