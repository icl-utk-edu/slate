// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_TILE_BLAS_HH
#define SLATE_TILE_BLAS_HH

#include <blas.hh>

#include "slate/Tile.hh"
#include "slate/internal/util.hh"
#include "slate/internal/device.hh"

#include <list>

namespace slate {

namespace tile {

//------------------------------------------------------------------------------
/// General matrix multiply: $op(C) = \alpha op(A) op(B) + \beta C$.
/// Use transpose() or conjTranspose() to set $op(A)$, $op(B)$, and $op(C)$.
/// In the complex case,
/// if $op(C)$ is transpose, then $op(A)$ and $op(B)$ cannot be conjTranspose;
/// if $op(C)$ is conjTranspose, then $op(A)$ and $op(B)$ cannot be transpose.
/// @ingroup gemm_tile
///
template <typename scalar_t>
void gemm(
    scalar_t alpha, Tile<scalar_t> const& A,
                    Tile<scalar_t> const& B,
    scalar_t beta,  Tile<scalar_t>& C)
{
    trace::Block trace_block("blas::gemm");

    using blas::conj;

    slate_assert(A.uploPhysical() == Uplo::General);
    slate_assert(B.uploPhysical() == Uplo::General);
    slate_assert(C.uploPhysical() == Uplo::General);
    slate_assert(C.mb() == A.mb());  // m
    slate_assert(C.nb() == B.nb());  // n
    slate_assert(A.nb() == B.mb());  // k
    slate_assert(A.layout() == C.layout());
    slate_assert(B.layout() == C.layout());

    if (C.op() == Op::NoTrans) {
        // C = opA(A) opB(B) + C
        blas::gemm(C.layout(),
                   A.op(), B.op(),
                   C.mb(), C.nb(), A.nb(),
                   alpha, A.data(), A.stride(),
                          B.data(), B.stride(),
                   beta,  C.data(), C.stride());
    }
    else {
        // opC is Trans or ConjTrans
        // opC(C) = opA(A)) opB(B) + opC(C) becomes
        // C = opC(opA(A) opB(B)) + C = opC(opB(B)) opC(opA(A)) + C
        // invert opA, opB if possible; swap A <=> B; swap m <=> n
        Op opA;
        if (A.op() == Op::NoTrans)
            opA = C.op();
        else if (A.op() == C.op() || C.is_real) {
            // A and C are both Trans or both ConjTrans;
            // Trans == ConjTrans if real
            opA = Op::NoTrans;
        }
        else
            throw std::exception();

        Op opB;
        if (B.op() == Op::NoTrans)
            opB = C.op();
        else if (B.op() == C.op() || C.is_real) {
            // B and C are both Trans or both ConjTrans;
            // Trans == ConjTrans if real
            opB = Op::NoTrans;
        }
        else
            throw std::exception();

        if (C.op() == Op::ConjTrans) {
            alpha = conj(alpha);
            beta  = conj(beta);
        }

        blas::gemm(C.layout(),
                   opB, opA,
                   C.nb(), C.mb(), A.nb(),
                   alpha, B.data(), B.stride(),
                          A.data(), A.stride(),
                   beta,  C.data(), C.stride());
    }
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup gemm_tile
///
template <typename scalar_t>
void gemm(
    scalar_t alpha, Tile<scalar_t> const&& A,
                    Tile<scalar_t> const&& B,
    scalar_t beta,  Tile<scalar_t>&& C)
{
    gemm(alpha, A, B, beta, C);
}

//------------------------------------------------------------------------------
///
/// @ingroup gemv_tile
///
template <typename scalar_t>
void gemv(scalar_t alpha, Tile<scalar_t> const& A,
                          scalar_t const* x,
          scalar_t beta,  scalar_t* y)
{
//  trace::Block trace_block("blas::gemv");

    assert(A.uploPhysical() == Uplo::General);

    if (A.op() == Op::NoTrans) {
        // y = Ax+y
        blas::gemv(A.layout(),
                   A.op(),
                   A.mb(), A.nb(),
                   alpha, A.data(), A.stride(),
                          x, 1,
                   beta,  y, 1);
    }
    else {
        // y' = xA'+y'
        blas::gemv(A.layout(),
                   A.op(),
                   A.nb(), A.mb(),
                   alpha, A.data(), A.stride(),
                          x, 1,
                   beta,  y, 1);
    }
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup gemv_tile
///
template <typename scalar_t>
void gemv(scalar_t alpha, Tile<scalar_t> const&& A,
                          scalar_t const* x,
          scalar_t beta,  scalar_t* y)
{
    gemv(alpha, A, x, beta, y);
}

//------------------------------------------------------------------------------
///
/// @ingroup ger_tile
///
template <typename scalar_t>
void ger(scalar_t alpha, scalar_t const* x,
                         scalar_t const* y,
                         Tile<scalar_t>& A)
{
//  trace::Block trace_block("blas::ger");

    assert(A.uploPhysical() == Uplo::General);

    if (A.op() == Op::NoTrans) {
        // A = xy'+A
        blas::ger(A.layout(),
                  A.mb(), A.nb(),
                  alpha, x, 1,
                         y, 1,
                         A.data(), A.stride());
    }
    else {
        // A' = yx'+A'
        blas::ger(A.layout(),
                  A.nb(), A.mb(),
                  alpha, y, 1,
                         x, 1,
                         A.data(), A.stride());
    }
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup gemv_tile
///
template <typename scalar_t>
void ger(scalar_t alpha, scalar_t const* x,
                         scalar_t const* y,
                         Tile<scalar_t>&& A)
{
    ger(alpha, x, y, A);
}

//------------------------------------------------------------------------------
/// Hermitian matrix multiply: $C = \alpha A op(B) + \beta op(C)$
///                         or $C = \alpha op(B) A + \beta op(C)$,
/// where $A$ is Hermitian.
/// Unlike most BLAS operations, here op(B) and op(C) must be
/// both the same, either both NoTrans or both ConjTrans.
/// @ingroup hemm_tile
///
template <typename scalar_t>
void hemm(
    Side side,
    scalar_t alpha, Tile<scalar_t> const& A,
                    Tile<scalar_t> const& B,
    scalar_t beta,  Tile<scalar_t>& C)
{
    trace::Block trace_block("blas::hemm");

    using blas::conj;

    assert(A.mb() == A.nb());  // square
    assert(B.mb() == C.mb());
    assert(B.nb() == C.nb());
    if (side == Side::Left)
        assert(A.mb() == B.mb());
    else
        assert(A.mb() == B.nb());
    assert(B.op() == C.op());
    assert(A.op() != Op::Trans);
    assert(B.op() != Op::Trans);

    // A.op can be ignored, since A == A^T
    if (B.op() == Op::NoTrans) {
        blas::hemm(blas::Layout::ColMajor,
                   side, A.uploPhysical(),
                   C.mb(), C.nb(),
                   alpha, A.data(), A.stride(),
                          B.data(), B.stride(),
                   beta,  C.data(), C.stride());
    }
    else {
        // ConjTrans
        // undo transpose by swapping left <=> right, m <=> n, conj alpha & beta
        side = (side == Side::Left ? Side::Right : Side::Left);
        blas::hemm(blas::Layout::ColMajor,
                   side, A.uploPhysical(),
                   C.nb(), C.mb(),
                   conj(alpha), A.data(), A.stride(),
                                B.data(), B.stride(),
                   conj(beta),  C.data(), C.stride());
    }
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup hemm_tile
///
template <typename scalar_t>
void hemm(
    Side side,
    scalar_t alpha, Tile<scalar_t> const&& A,
                    Tile<scalar_t> const&& B,
    scalar_t beta,  Tile<scalar_t>&& C)
{
    hemm(side, alpha, A, B, beta, C);
}

//------------------------------------------------------------------------------
/// Hermitian rank-k update: $C = \alpha op(A) op(A)^H + \beta C$.
/// Use conjTranspose to set $op(A)$.
/// In the complex case, C cannot be transpose.
/// @ingroup herk_tile
///
// Allowing C^T would require two conjugations: conj( conj(C) + A*A^H ).
template <typename scalar_t>
void herk(
    blas::real_type<scalar_t> alpha, Tile<scalar_t> const& A,
    blas::real_type<scalar_t> beta,  Tile<scalar_t>& C)
{
    trace::Block trace_block("blas::herk");

    assert(A.uploPhysical() == Uplo::General);
    assert(C.mb() == C.nb());  // square
    assert(C.mb() == A.mb());  // n
    if (C.is_complex && C.op() == Op::Trans)
        throw std::exception();

    blas::herk(blas::Layout::ColMajor,
               C.uploPhysical(), A.op(),
               C.nb(), A.nb(),
               alpha, A.data(), A.stride(),
               beta,  C.data(), C.stride());
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup herk_tile
///
template <typename scalar_t>
void herk(
    blas::real_type<scalar_t> alpha, Tile<scalar_t> const&& A,
    blas::real_type<scalar_t> beta,  Tile<scalar_t>&& C)
{
    herk(alpha, A, beta, C);
}

//------------------------------------------------------------------------------
/// Hermitian rank-2k update:
///     $C = \alpha op(A) op(B)^T + \alpha op(B) op(A)^T + \beta C$.
/// Use transpose or conjTranspose to set $op(A)$ and $op(B)$.
/// In the complex case, C cannot be transpose.
/// @ingroup her2k_tile
///
// Allowing C^H would require two conjugations: conj( conj(C) + A*A^T ).
template <typename scalar_t>
void her2k(
    scalar_t alpha,                 Tile<scalar_t> const& A,
                                    Tile<scalar_t> const& B,
    blas::real_type<scalar_t> beta, Tile<scalar_t>& C)
{
    trace::Block trace_block("blas::her2k");

    using blas::conj;

    assert(A.op() == B.op());
    assert(A.uploPhysical() == Uplo::General);
    assert(B.uploPhysical() == Uplo::General);
    assert(C.mb() == C.nb());  // square
    assert(C.mb() == A.mb());  // n
    assert(C.mb() == B.mb());  // n
    if (C.is_complex && C.op() == Op::Trans)
        throw std::exception();

    blas::her2k(blas::Layout::ColMajor,
                C.uploPhysical(), A.op(),
                C.nb(), A.nb(),
                alpha, A.data(), A.stride(),
                       B.data(), B.stride(),
                beta,  C.data(), C.stride());
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup her2k_tile
///
template <typename scalar_t>
void her2k(
    scalar_t alpha,                 Tile<scalar_t> const&& A,
                                    Tile<scalar_t> const&& B,
    blas::real_type<scalar_t> beta, Tile<scalar_t>&& C)
{
    her2k(alpha, A, B, beta, C);
}

//------------------------------------------------------------------------------
///
/// @ingroup her2_tile
///
template <typename scalar_t>
void her2(scalar_t alpha, scalar_t const* x,
                          scalar_t const* y,
                          Tile<scalar_t>& A)
{
//  trace::Block trace_block("blas::her2");

    blas::her2(A.layout(),
               A.uploPhysical(),
               A.nb(),
               alpha, x, 1,
                      y, 1,
                      A.data(), A.stride());
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup her2_tile
///
template <typename scalar_t>
void her2(scalar_t alpha, scalar_t const* x,
                          scalar_t const* y,
                          Tile<scalar_t>&& A)
{
    her2(alpha, x, y, A);
}

//------------------------------------------------------------------------------
/// Symmetric matrix multiply: $C = \alpha A op(B) + \beta op(C)$
///                         or $C = \alpha op(B) A + \beta op(C)$,
/// where $A$ is symmetric.
/// Unlike most BLAS operations, here op(B) and op(C) must be
/// both the same, either both NoTrans or both Trans.
/// @ingroup symm_tile
///
template <typename scalar_t>
void symm(
    Side side,
    scalar_t alpha, Tile<scalar_t> const& A,
                    Tile<scalar_t> const& B,
    scalar_t beta,  Tile<scalar_t>& C)
{
    trace::Block trace_block("blas::symm");

    using blas::conj;

    assert(A.mb() == A.nb());  // square
    assert(B.mb() == C.mb());
    assert(B.nb() == C.nb());
    if (side == Side::Left)
        assert(A.mb() == B.mb());
    else
        assert(A.mb() == B.nb());
    assert(B.op() == C.op());
    assert(A.op() != Op::ConjTrans);
    assert(B.op() != Op::ConjTrans);

    // A.op can be ignored, since A == A^T
    if (B.op() == Op::NoTrans) {
        blas::symm(blas::Layout::ColMajor,
                   side, A.uploPhysical(),
                   C.mb(), C.nb(),
                   alpha, A.data(), A.stride(),
                          B.data(), B.stride(),
                   beta,  C.data(), C.stride());
    }
    else {
        // Trans
        // undo transpose by swapping left <=> right, m <=> n
        side = (side == Side::Left ? Side::Right : Side::Left);
        blas::symm(blas::Layout::ColMajor,
                   side, A.uploPhysical(),
                   C.nb(), C.mb(),
                   alpha, A.data(), A.stride(),
                          B.data(), B.stride(),
                   beta,  C.data(), C.stride());
    }
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup symm_tile
///
template <typename scalar_t>
void symm(
    Side side,
    scalar_t alpha, Tile<scalar_t> const&& A,
                    Tile<scalar_t> const&& B,
    scalar_t beta,  Tile<scalar_t>&& C)
{
    symm(side, alpha, A, B, beta, C);
}

//------------------------------------------------------------------------------
///
/// @ingroup symv_tile
///
template <typename scalar_t>
void symv(scalar_t alpha, Tile<scalar_t> const& A,
                          scalar_t const* x,
          scalar_t beta,  scalar_t* y)
{
//  trace::Block trace_block("blas::symv");

    assert(A.mb() == A.nb());  // square

    blas::symv(blas::Layout::ColMajor,
               A.uploPhysical(),
               A.nb(),
               alpha, A.data(), A.stride(),
                      x, 1,
               beta,  y, 1);
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup symm_tile
///
template <typename scalar_t>
void symv(scalar_t alpha, Tile<scalar_t> const&& A,
                          scalar_t const* x,
          scalar_t beta,  scalar_t* y)
{
    symv(alpha, A, x, beta, y);
}

//------------------------------------------------------------------------------
///
/// @ingroup symv_tile
///
template <typename scalar_t>
void hemv(scalar_t alpha, Tile<scalar_t> const& A,
                          scalar_t const* x,
          scalar_t beta,  scalar_t* y)
{
//  trace::Block trace_block("blas::symv");

    assert(A.mb() == A.nb());  // square

    blas::hemv(blas::Layout::ColMajor,
               A.uploPhysical(),
               A.nb(),
               alpha, A.data(), A.stride(),
                      x, 1,
               beta,  y, 1);
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup symm_tile
///
template <typename scalar_t>
void hemv(scalar_t alpha, Tile<scalar_t> const&& A,
                          scalar_t const* x,
          scalar_t beta,  scalar_t* y)
{
    hemv(alpha, A, x, beta, y);
}

//------------------------------------------------------------------------------
/// Symmetric rank-k update: $C = \alpha op(A) op(A)^T + \beta C$.
/// Use transpose or conjTranspose to set $op(A)$.
/// In the complex case, C cannot be conjTranspose.
/// @ingroup syrk_tile
///
// Allowing C^H would require two conjugations: conj( conj(C) + A*A^T ).
template <typename scalar_t>
void syrk(
    scalar_t alpha, Tile<scalar_t> const& A,
    scalar_t beta,  Tile<scalar_t>& C)
{
    trace::Block trace_block("blas::syrk");

    using blas::conj;

    assert(A.uploPhysical() == Uplo::General);
    assert(C.mb() == C.nb());  // square
    assert(C.mb() == A.mb());  // n
    if (C.is_complex && C.op() == Op::ConjTrans)
        throw std::exception();

    blas::syrk(blas::Layout::ColMajor,
               C.uploPhysical(), A.op(),
               C.nb(), A.nb(),
               alpha, A.data(), A.stride(),
               beta,  C.data(), C.stride());
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup syrk_tile
///
template <typename scalar_t>
void syrk(
    scalar_t alpha, Tile<scalar_t> const&& A,
    scalar_t beta,  Tile<scalar_t>&& C)
{
    syrk(alpha, A, beta, C);
}

//------------------------------------------------------------------------------
/// Symmetric rank-2k update:
///     $C = \alpha op(A) op(B)^T + \alpha op(B) op(A)^T + \beta C$.
/// Use transpose or conjTranspose to set $op(A)$ and $op(B)$.
/// In the complex case, C cannot be conjTranspose.
/// @ingroup syr2k_tile
///
// Allowing C^H would require two conjugations: conj( conj(C) + A*A^T ).
template <typename scalar_t>
void syr2k(
    scalar_t alpha, Tile<scalar_t> const& A,
                    Tile<scalar_t> const& B,
    scalar_t beta,  Tile<scalar_t>& C)
{
    trace::Block trace_block("blas::syr2k");

    using blas::conj;

    assert(A.op() == B.op());
    assert(A.uploPhysical() == Uplo::General);
    assert(B.uploPhysical() == Uplo::General);
    assert(C.mb() == C.nb());  // square
    assert(C.mb() == A.mb());  // n
    assert(C.mb() == B.mb());  // n
    if (C.is_complex && C.op() == Op::ConjTrans)
        throw std::exception();

    blas::syr2k(blas::Layout::ColMajor,
                C.uploPhysical(), A.op(),
                C.nb(), A.nb(),
                alpha, A.data(), A.stride(),
                       B.data(), B.stride(),
                beta,  C.data(), C.stride());
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup syr2k_tile
///
template <typename scalar_t>
void syr2k(
    scalar_t alpha, Tile<scalar_t> const&& A,
                    Tile<scalar_t> const&& B,
    scalar_t beta,  Tile<scalar_t>&& C)
{
    syr2k(alpha, A, B, beta, C);
}

//------------------------------------------------------------------------------
/// Triangular matrix-matrix multiply:
///     $B = \alpha op(A) B$ or
///     $B = \alpha B op(A)$
/// where $A$ is triangular.
/// @ingroup trmm_tile
///
template <typename scalar_t>
void trmm(
    Side side, Diag diag,
    scalar_t alpha, Tile<scalar_t> const& A,
                    Tile<scalar_t>& B)
{
    trace::Block trace_block("blas::trmm");

    using blas::conj;

    assert(B.uploPhysical() == Uplo::General);
    assert(A.mb() == A.nb());  // square
    assert(side == Side::Left ? A.mb() == B.mb()    // m
                              : A.mb() == B.nb());  // n
    if (B.op() == Op::NoTrans) {
        blas::trmm(blas::Layout::ColMajor,
                   side, A.uploPhysical(), A.op(), diag,
                   B.mb(), B.nb(),
                   alpha, A.data(), A.stride(),
                          B.data(), B.stride());
    }
    else {
        if (A.is_complex && A.op() != Op::NoTrans && A.op() != B.op())
            throw std::exception();

        // switch op(A) <=> op(B), side left <=> right, m <=> n
        Side side2 = (side == Side::Left ? Side::Right : Side::Left);
        Op opA;
        if (A.op() == Op::NoTrans)
            opA = B.op();
        else if (A.op() == B.op() || A.is_real) {
            // A and B are both Trans or both ConjTrans;
            // Trans == ConjTrans if real
            opA = Op::NoTrans;
        }
        else
            throw std::exception();

        if (B.op() == Op::ConjTrans)
            alpha = conj(alpha);

        blas::trmm(blas::Layout::ColMajor,
                   side2, A.uploPhysical(), opA, diag,
                   B.nb(), B.mb(),
                   alpha, A.data(), A.stride(),
                          B.data(), B.stride());
    }
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup trmm_tile
///
template <typename scalar_t>
void trmm(
    Side side, Diag diag,
    scalar_t alpha, Tile<scalar_t> const&& A,
                    Tile<scalar_t>&& B)
{
    trmm(side, diag, alpha, A, B);
}

//------------------------------------------------------------------------------
/// Triangular solve: $B = \alpha op(A)^{-1} B$ or $B = \alpha B op(A)^{-1}$.
/// Use transpose/conjTranspose to set op(A). uplo is set in the tile.
/// In the complex case,
/// if $op(B)$ is transpose, then $op(A)$ cannot be conjTranspose;
/// if $op(B)$ is conjTranspose, then $op(A)$ cannot be transpose.
/// @ingroup trsm_tile
///
template <typename scalar_t>
void trsm(
    Side side, Diag diag,
    scalar_t alpha, Tile<scalar_t> const& A,
                    Tile<scalar_t>& B)
{
    trace::Block trace_block("blas::trsm");

    using blas::conj;

    assert(B.uploPhysical() == Uplo::General);
    assert(A.mb() == A.nb());  // square
    assert(side == Side::Left ? A.mb() == B.mb()    // m
                              : A.mb() == B.nb());  // n
    if (B.op() == Op::NoTrans) {
        blas::trsm(blas::Layout::ColMajor,
                   side, A.uploPhysical(), A.op(), diag,
                   B.mb(), B.nb(),
                   alpha, A.data(), A.stride(),
                          B.data(), B.stride());
    }
    else {
        if (A.is_complex && A.op() != Op::NoTrans && A.op() != B.op())
            throw std::exception();

        // switch op(A) <=> op(B), side left <=> right, m <=> n
        Side side2 = (side == Side::Left ? Side::Right : Side::Left);
        Op opA;
        if (A.op() == Op::NoTrans)
            opA = B.op();
        else if (A.op() == B.op() || A.is_real) {
            // A and B are both Trans or both ConjTrans;
            // Trans == ConjTrans if real
            opA = Op::NoTrans;
        }
        else
            throw std::exception();

        if (B.op() == Op::ConjTrans)
            alpha = conj(alpha);

        blas::trsm(blas::Layout::ColMajor,
                   side2, A.uploPhysical(), opA, diag,
                   B.nb(), B.mb(),
                   alpha, A.data(), A.stride(),
                          B.data(), B.stride());
    }
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup trsm_tile
///
template <typename scalar_t>
void trsm(
    Side side, Diag diag,
    scalar_t alpha, Tile<scalar_t> const&& A,
                    Tile<scalar_t>&& B)
{
    trsm(side, diag, alpha, A, B);
}

//------------------------------------------------------------------------------
/// Scale by a constant: $A = \alpha A$.
/// @ingroup scale_tile
///
template <typename scalar_t>
void scale(
    scalar_t alpha, Tile<scalar_t>& A)
{
    trace::Block trace_block("blas::scale");

    using blas::conj;
    if (A.op() == Op::ConjTrans)
        alpha = conj(alpha);

    int64_t col_inc = A.colIncrement();
    int64_t row_inc = A.rowIncrement();
    scalar_t* A00 = &A.at(0, 0);
    if (col_inc == 1) {
        // one column at a time
        for (int64_t j = 0; j < A.nb(); ++j)
            blas::scal(A.mb(), alpha, &A00[j*row_inc], col_inc);
    }
    else {
        // one row at a time
        for (int64_t i = 0; i < A.mb(); ++i)
            blas::scal(A.nb(), alpha, &A00[i*col_inc], row_inc);
    }
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup scale_tile
///
template <typename scalar_t>
void scale(
    scalar_t alpha, Tile<scalar_t>&& A)
{
    scale(alpha, A);
}

//------------------------------------------------------------------------------
/// Computes $Y = \alpha X + Y$.
/// @ingroup geadd_tile
///
// todo: rename add?
template <typename scalar_t>
void add(
    scalar_t alpha, Tile<scalar_t> const& X, Tile<scalar_t>& Y )
{
    trace::Block trace_block("blas::add");

    // todo: relax these assumptions, by adjusting the loops below
    assert(X.op() == Y.op());
    assert(X.uploPhysical() == Uplo::General);
    assert(Y.uploPhysical() == Uplo::General);

    int64_t y_col_inc = Y.colIncrement();
    int64_t y_row_inc = Y.rowIncrement();
    scalar_t* Y00 = &Y.at(0, 0);
    int64_t x_col_inc = X.colIncrement();
    int64_t x_row_inc = X.rowIncrement();
    const scalar_t* X00 = &X.at(0, 0);

    if (y_col_inc == 1) {
        // one column of y at a time
        int64_t m = std::min(X.mb(), Y.mb());
        for (int64_t j = 0; j < std::min(X.nb(), Y.nb()); ++j) {
            blas::axpy( m, alpha,
                        &X00[j*x_row_inc], x_col_inc,
                        &Y00[j*y_row_inc], y_col_inc );
        }
    }
    else {
        // one row of y at a time
        int64_t n = std::min(X.nb(), Y.nb());
        for (int64_t i = 0; i < std::min(X.mb(), Y.mb()); ++i) {
            blas::axpy( n, alpha,
                        &X00[i*x_col_inc], x_row_inc,
                        &Y00[i*y_col_inc], y_row_inc );
        }
    }
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup geadd_tile
///
// todo: rename add?
template <typename scalar_t>
void add(
    scalar_t alpha, Tile<scalar_t> const& X, Tile<scalar_t>&& Y )
{
    add( alpha, X, Y );
}

//------------------------------------------------------------------------------
/// Computes $Y = \alpha X + \beta Y$.
/// @ingroup geadd_tile
///
// todo: rename add?
template <typename scalar_t>
void add(
    scalar_t alpha, Tile<scalar_t> const& X,
    scalar_t beta,  Tile<scalar_t>& Y )
{
    // trace::Block trace_block("blas::add");

    assert(X.op() == Y.op() && X.mb() == Y.mb() && X.nb() == Y.nb());
    assert(X.uploPhysical() == Y.uploPhysical());

    int64_t y_col_inc = Y.colIncrement();
    int64_t y_row_inc = Y.rowIncrement();
    scalar_t* Y00 = &Y.at(0, 0);
    int64_t x_col_inc = X.colIncrement();
    int64_t x_row_inc = X.rowIncrement();
    const scalar_t* X00 = &X.at(0, 0);

    // Process uplo --> col/row, then scale y=b*y and add y=ax+y
    if (X.uploPhysical() == Uplo::General) {
        if (y_col_inc == 1) {
            // one column of y at a time
            int64_t m = std::min(X.mb(), Y.mb());
            for (int64_t j = 0; j < std::min(X.nb(), Y.nb()); ++j) {
                blas::scal( m, beta,
                            &Y00[j*y_row_inc], y_col_inc );
                blas::axpy( m, alpha,
                            &X00[j*x_row_inc], x_col_inc,
                            &Y00[j*y_row_inc], y_col_inc );
            }
        }
        else {
            // one row of y at a time
            int64_t n = std::min(X.nb(), Y.nb());
            for (int64_t i = 0; i < std::min(X.mb(), Y.mb()); ++i) {
                blas::scal( n, beta,
                            &Y00[i*y_col_inc], y_row_inc );
                blas::axpy( n, alpha,
                            &X00[i*x_col_inc], x_row_inc,
                            &Y00[i*y_col_inc], y_row_inc );
            }
        }
    }
    else if (X.uploPhysical() == Uplo::Lower) {
        int64_t m = std::min(X.mb(), Y.mb());
        int64_t n = std::min(X.nb(), Y.nb());
        if (y_col_inc == 1) {
            // one column of y at a time
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = j; i < m; ++i) {
                    Y00[i+j*y_row_inc] = beta * Y00[i+j*y_row_inc] + alpha * X00[i+j*x_row_inc];
                }
            }
        }
        else {
            // one row of y at a time
            slate_not_implemented("add uplo == Lower cannot process by row, only by column");
        }
    }
    else if (X.uploPhysical() == Uplo::Upper) {
        int64_t m = std::min(X.mb(), Y.mb());
        int64_t n = std::min(X.nb(), Y.nb());
        if (y_col_inc == 1) {
            // one column of y at a time
            if (m > n) {
                for (int64_t j = 0; j < n; ++j) {
                    for (int64_t i = 0; i <= j; ++i) {
                        Y00[i+j*y_row_inc] = beta * Y00[i+j*y_row_inc] + alpha * X00[i+j*x_row_inc];
                    }
                }
            }
            else {
                for (int64_t j = 0; j < n; ++j) {
                    for (int64_t i = 0; i <= j && i < m; ++i) {
                        Y00[i+j*y_row_inc] = beta * Y00[i+j*y_row_inc] + alpha * X00[i+j*x_row_inc];
                    }
                }
            }
        }
        else {
            // one row of y at a time
            slate_not_implemented("add uplo == Upper cannot process by row, only by column");
        }
    }
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup geadd_tile
///
// todo: rename add?
template <typename scalar_t>
void add(
    scalar_t alpha, Tile<scalar_t> const& X,
    scalar_t beta,  Tile<scalar_t>&& Y )
{
    add( alpha, X, beta, Y );
}

} // namespace tile

} // namespace slate

#endif // SLATE_TILE_BLAS_HH
