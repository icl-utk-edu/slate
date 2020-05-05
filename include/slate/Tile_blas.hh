//------------------------------------------------------------------------------
// Copyright (c) 2017, University of Tennessee
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the University of Tennessee nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF TENNESSEE BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//------------------------------------------------------------------------------
// This research was supported by the Exascale Computing Project (17-SC-20-SC),
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
//------------------------------------------------------------------------------
// For assistance with SLATE, email <slate-user@icl.utk.edu>.
// You can also join the "SLATE User" Google group by going to
// https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user,
// signing in with your Google credentials, and then clicking "Join group".
//------------------------------------------------------------------------------

#ifndef SLATE_TILE_BLAS_HH
#define SLATE_TILE_BLAS_HH

#include <blas.hh>

#include "slate/Tile.hh"
#include "slate/internal/util.hh"
#include "slate/internal/device.hh"

#include <list>

namespace slate {

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

    const int64_t col_inc = A.colIncrement();
    const int64_t row_inc = A.rowIncrement();
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
/// Swap a partial row of two local tiles:
///     A[ i1, j_offset : j_offset+n-1 ] and
///     B[ i2, j_offset : j_offset+n-1 ].
/// Either or both tiles can be transposed to swap columns.
/// @ingroup swap_tile
///
template <typename scalar_t>
void swapLocalRow(
    int64_t j_offset, int64_t n,
    Tile<scalar_t>& A, int64_t i1,
    Tile<scalar_t>& B, int64_t i2)
{
    // todo: size assertions, quick return
    if (n <= 0) return;

    blas::swap(n, &A.at(i1,j_offset), A.rowIncrement(),
                  &B.at(i2,j_offset), B.rowIncrement());
}

//--------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup swap_tile
///
template <typename scalar_t>
void swapLocalRow(
    int64_t j_offset, int64_t n,
    Tile<scalar_t>&& A, int64_t i1,
    Tile<scalar_t>&& B, int64_t i2)
{
    swapLocalRow(j_offset, n, A, i1, B, i2);
}

//------------------------------------------------------------------------------
/// Swap a partial row, A[ i, j : j+n-1 ], with another MPI process.
/// The tile can be transposed to swap a column.
/// @ingroup swap_tile
///
template <typename scalar_t>
void swapRemoteRow(
    int64_t j, int64_t n,
    Tile<scalar_t>& A, int64_t i,
    int other_rank, MPI_Comm mpi_comm, int tag = 0)
{
    // todo: size assertions, quick return
    if (n <= 0) return;

    std::vector<scalar_t> local_row(n);
    std::vector<scalar_t> other_row(n);

    // todo: Perhaps create an MPI type and let MPI pack it?
    blas::copy(n, &A.at(i, j), A.rowIncrement(), &local_row[0], 1);

    MPI_Sendrecv(
        local_row.data(), n, mpi_type<scalar_t>::value, other_rank, tag,
        other_row.data(), n, mpi_type<scalar_t>::value, other_rank, tag,
        mpi_comm, MPI_STATUS_IGNORE);

    blas::copy(n, &other_row[0], 1, &A.at(i, j), A.rowIncrement());
}

//--------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup swap_tile
///
template <typename scalar_t>
void swapRemoteRow(
    int64_t j, int64_t n,
    Tile<scalar_t>&& A, int64_t i,
    int other_rank, MPI_Comm mpi_comm, int tag = 0)
{
    swapRemoteRow(j, n, A, i, other_rank, mpi_comm, tag);
}

//------------------------------------------------------------------------------
/// Swap a partial row, A[ i, j : j+n-1 ], on a GPU device,
/// with another MPI process.
/// The tile must be row-major, and cannot be transposed.
/// @ingroup swap_tile
///
// todo: implement with a GPUDirect call
template <typename scalar_t>
void swapRemoteRowDevice(
    int64_t j, int64_t n,
    int device, Tile<scalar_t>& A, int64_t i,
    int other_rank, MPI_Comm mpi_comm, cudaStream_t stream, int tag = 0)
{
    std::vector<scalar_t> local_row(n);
    std::vector<scalar_t> other_row(n);

    // todo: this assumes row is contiguous on GPU, right? Add asserts.
    slate_cuda_call(cudaSetDevice(device));
    slate_cuda_call(cudaMemcpyAsync(local_row.data(), &A.at(i, j),
                                    sizeof(scalar_t)*n, cudaMemcpyDeviceToHost,
                                    stream));
    slate_cuda_call(cudaStreamSynchronize(stream));

    MPI_Sendrecv(
        local_row.data(), n, mpi_type<scalar_t>::value, other_rank, tag,
        other_row.data(), n, mpi_type<scalar_t>::value, other_rank, tag,
        mpi_comm, MPI_STATUS_IGNORE);

    slate_cuda_call(cudaMemcpyAsync(&A.at(i, j), other_row.data(),
                                    sizeof(scalar_t)*n, cudaMemcpyHostToDevice,
                                    stream));
    slate_cuda_call(cudaStreamSynchronize(stream));
}

//--------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup swap_tile
///
template <typename scalar_t>
void swapRemoteRowDevice(
    int64_t j, int64_t n,
    int device, Tile<scalar_t>&& A, int64_t i,
    int other_rank, MPI_Comm mpi_comm, cudaStream_t stream, int tag = 0)
{
    swapRemoteRowDevice(j, n, device, A, i, other_rank, mpi_comm, stream, tag);
}

//------------------------------------------------------------------------------
/// Swap one element, A(i, j), with another MPI process.
/// @ingroup swap_tile
///
template <typename scalar_t>
void swapRemoteElement(
    Tile<scalar_t>& A, int64_t i, int64_t j,
    int other_rank, MPI_Comm mpi_comm, int tag = 0)
{
    scalar_t local_element = A(i, j);
    scalar_t other_element;

    MPI_Sendrecv(
        &local_element, 1, mpi_type<scalar_t>::value, other_rank, tag,
        &other_element, 1, mpi_type<scalar_t>::value, other_rank, tag,
        mpi_comm, MPI_STATUS_IGNORE);

    A.at(i, j) = other_element;
}

//--------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup swap_tile
///
template <typename scalar_t>
void swapRemoteElement(
    Tile<scalar_t>&& A, int64_t i, int64_t j,
    int other_rank, MPI_Comm mpi_comm, int tag = 0)
{
    swapRemoteElement(A, i, j, other_rank, mpi_comm, tag);
}

//------------------------------------------------------------------------------
/// Computes $Y = \alpha X + Y$.
/// @ingroup geadd_tile
///
// todo: rename add?
template <typename scalar_t>
void axpy(scalar_t alpha, Tile<scalar_t> const& X, Tile<scalar_t>& Y)
{
    trace::Block trace_block("blas::axpy");

    // todo: relax these assumptions, by adjusting the loops below
    assert(X.op() == Y.op());
    assert(X.uploPhysical() == Uplo::General);
    assert(Y.uploPhysical() == Uplo::General);

    const int64_t y_col_inc = Y.colIncrement();
    const int64_t y_row_inc = Y.rowIncrement();
    scalar_t* Y00 = &Y.at(0, 0);
    const int64_t x_col_inc = X.colIncrement();
    const int64_t x_row_inc = X.rowIncrement();
    const scalar_t* X00 = &X.at(0, 0);

    if (y_col_inc == 1) {
        // one column of y at a time
        int64_t m = std::min(X.mb(), Y.mb());
        for (int64_t j = 0; j < std::min(X.nb(), Y.nb()); ++j)
            blas::axpy(m, alpha, &X00[j*x_row_inc], x_col_inc,
                                 &Y00[j*y_row_inc], y_col_inc);
    }
    else {
        // one row of y at a time
        int64_t n = std::min(X.nb(), Y.nb());
        for (int64_t i = 0; i < std::min(X.mb(), Y.mb()); ++i)
            blas::axpy(n, alpha, &X00[i*x_col_inc], x_row_inc,
                                 &Y00[i*y_col_inc], y_row_inc);
    }
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup geadd_tile
///
// todo: rename add?
template <typename scalar_t>
void axpy(scalar_t alpha, Tile<scalar_t> const& X, Tile<scalar_t>&& Y)
{
    axpy(alpha, X, Y);
}

//------------------------------------------------------------------------------
/// Computes $Y = \alpha X + \beta Y$.
/// @ingroup geadd_tile
///
// todo: rename add?
template <typename scalar_t>
void axpby(scalar_t alpha, Tile<scalar_t> const& X,
           scalar_t beta,  Tile<scalar_t>& Y)
{
    // trace::Block trace_block("blas::axpby");

    // TODO should be able to loosen these restrictions
    assert(X.op() == Y.op());
    assert(X.uploPhysical() == Uplo::General);
    assert(Y.uploPhysical() == Uplo::General);

    const int64_t y_col_inc = Y.colIncrement();
    const int64_t y_row_inc = Y.rowIncrement();
    scalar_t* Y00 = &Y.at(0, 0);
    const int64_t x_col_inc = X.colIncrement();
    const int64_t x_row_inc = X.rowIncrement();
    const scalar_t* X00 = &X.at(0, 0);

    // Process by col/row, scale y=b*y then add y=ax+y
    if (Y.colIncrement()==1) {
        // one column of y at a time
        int64_t m = std::min(X.mb(), Y.mb());
        for (int64_t j = 0; j < std::min(X.nb(), Y.nb()); ++j) {
            blas::scal(m, beta,  &Y00[j*y_row_inc], y_col_inc);
            blas::axpy(m, alpha, &X00[j*x_row_inc], x_col_inc,
                                 &Y00[j*y_row_inc], y_col_inc);
        }
    }
    else {
        // one row of y at a time
        int64_t n = std::min(X.nb(), Y.nb());
        for (int64_t i = 0; i < std::min(X.mb(), Y.mb()); ++i) {
            blas::scal(n, beta,  &Y00[i*y_col_inc], y_row_inc);
            blas::axpy(n, alpha, &X00[i*x_col_inc], x_row_inc,
                                 &Y00[i*y_col_inc], y_row_inc);
        }
    }
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup geadd_tile
///
// todo: rename add?
template <typename scalar_t>
void axpby(scalar_t alpha, Tile<scalar_t> const& X,
           scalar_t beta,  Tile<scalar_t>&& Y)
{
    axpby(alpha, X, beta, Y);
}

} // namespace slate

#endif // SLATE_TILE_BLAS_HH
