#ifndef SLATE_TILE_BLAS_HH
#define SLATE_TILE_BLAS_HH

#include <blas.hh>

#include "slate_Tile.hh"

namespace slate {

///=============================================================================
// Tile BLAS

///-----------------------------------------------------------------------------
/// \brief
/// General matrix multiply: $op(C) = \alpha op(A) op(B) + \beta C$.
/// Use transpose() or conj_transpose() to set $op(A)$, $op(B)$, and $op(C)$.
/// In the complex case,
/// if $op(C)$ is transpose, then $op(A)$ and $op(B)$ cannot be conj_transpose;
/// if $op(C)$ is conj_transpose, then $op(A)$ and $op(B)$ cannot be transpose.
template <typename scalar_t>
void gemm(
    scalar_t alpha, Tile<scalar_t> const& A,
                    Tile<scalar_t> const& B,
    scalar_t beta,  Tile<scalar_t>& C)
{
    trace::Block trace_block("blas::gemm");

    using blas::conj;

    assert(A.uplo() == Uplo::General);
    assert(B.uplo() == Uplo::General);
    assert(C.uplo() == Uplo::General);
    assert(C.mb() == A.mb());  // m
    assert(C.nb() == B.nb());  // n
    assert(A.nb() == B.mb());  // k
    if (C.op() == Op::NoTrans) {
        // C = opA(A) opB(B) + C
        blas::gemm(blas::Layout::ColMajor,
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
        else if (A.op() == C.op() || C.is_real)
            // A and C are both Trans or both ConjTrans; Trans == ConjTrans if real
            opA = Op::NoTrans;
        else
            throw std::exception();

        Op opB;
        if (B.op() == Op::NoTrans)
            opB = C.op();
        else if (B.op() == C.op() || C.is_real)
            // B and C are both Trans or both ConjTrans; Trans == ConjTrans if real
            opB = Op::NoTrans;
        else
            throw std::exception();

        if (C.op() == Op::ConjTrans) {
            alpha = conj( alpha );
            beta  = conj( beta );
        }

        blas::gemm(blas::Layout::ColMajor,
                   opB, opA,
                   C.nb(), C.mb(), A.nb(),
                   alpha, B.data(), B.stride(),
                          A.data(), A.stride(),
                   beta,  C.data(), C.stride());
    }
}

///----------------------------------------
/// Converts rvalue refs to lvalue refs.
template <typename scalar_t>
void gemm(
    scalar_t alpha, Tile<scalar_t> const&& A,
                    Tile<scalar_t> const&& B,
    scalar_t beta,  Tile<scalar_t>&& C)
{
    gemm( alpha, A, B, beta, C );
}

///-----------------------------------------------------------------------------
/// \brief
/// Cholesky factorization of tile: $L L^H = A$ or $U^H U = A$.
/// uplo is set in the tile.
template <typename scalar_t>
int64_t potrf(Tile<scalar_t>& A)
{
    trace::Block trace_block("lapack::potrf");

    return lapack::potrf(A.uplo(),
                         A.nb(),
                         A.data(), A.stride());
}

///----------------------------------------
/// Converts rvalue refs to lvalue refs.
template <typename scalar_t>
int64_t potrf(Tile<scalar_t>&& A)
{
    return potrf( A );
}

///-----------------------------------------------------------------------------
/// \brief
/// Symmetric matrix multiply: $C = \alpha A op(B) + \beta op(C)$
///                         or $C = \alpha op(B) A + \beta op(C)$,
/// where $A$ is symmetric.
/// Unlike most BLAS operations, here op(B) and op(C) must be
/// both the same, either both NoTrans or both Trans.
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
    assert( B.op() == C.op() );
    assert( A.op() != Op::ConjTrans );
    assert( B.op() != Op::ConjTrans );

    // A.op can be ignored, since A == A^T
    if (B.op() == Op::NoTrans) {
        blas::symm(blas::Layout::ColMajor,
                   side, A.uplo(),
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
                   side, A.uplo(),
                   C.nb(), C.mb(),
                   alpha, A.data(), A.stride(),
                          B.data(), B.stride(),
                   beta,  C.data(), C.stride());
    }
}

///----------------------------------------
/// Converts rvalue refs to lvalue refs.
template <typename scalar_t>
void symm(
    Side side,
    scalar_t alpha, Tile<scalar_t> const&& A,
                    Tile<scalar_t> const&& B,
    scalar_t beta,  Tile<scalar_t>&& C)
{
    symm( side, alpha, A, B, beta, C );
}

///-----------------------------------------------------------------------------
/// \brief
/// Symmetric rank-k update: $C = \alpha op(A) op(A)^T + \beta C$.
/// Use transpose or conj_transpose to set $op(A)$.
/// In the complex case, C cannot be conj_transpose.
// Allowing C^H would require two conjugations: conj( conj(C) + A*A^T ).
template <typename scalar_t>
void syrk(
    scalar_t alpha, Tile<scalar_t> const& A,
    scalar_t beta,  Tile<scalar_t>& C)
{
    trace::Block trace_block("blas::syrk");

    using blas::conj;

    assert(A.uplo() == Uplo::General);
    assert(C.mb() == C.nb());  // square
    assert(C.mb() == A.mb());  // n
    if (C.is_complex && C.op() == Op::ConjTrans)
        throw std::exception();

    blas::syrk(blas::Layout::ColMajor,
               C.uplo(), A.op(),
               C.nb(), A.nb(),
               alpha, A.data(), A.stride(),
               beta,  C.data(), C.stride());
}

///----------------------------------------
/// Converts rvalue refs to lvalue refs.
template <typename scalar_t>
void syrk(
    scalar_t alpha, Tile<scalar_t> const&& A,
    scalar_t beta,  Tile<scalar_t>&& C)
{
    syrk( alpha, A, beta, C );
}

///-----------------------------------------------------------------------------
/// \brief
/// Symmetric rank-2k update: $C = \alpha op(A) op(B)^T + \alpha op(B) op(A)^T + \beta C$.
/// Use transpose or conj_transpose to set $op(A)$ and $op(B)$.
/// In the complex case, C cannot be conj_transpose.
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
    assert(A.uplo() == Uplo::General);
    assert(B.uplo() == Uplo::General);
    assert(C.mb() == C.nb());  // square
    assert(C.mb() == A.mb());  // n
    assert(C.mb() == B.mb());  // n
    if (C.is_complex && C.op() == Op::ConjTrans)
        throw std::exception();

    blas::syr2k(blas::Layout::ColMajor,
                C.uplo(), A.op(),
                C.nb(), A.nb(),
                alpha, A.data(), A.stride(),
                       B.data(), B.stride(),
                beta,  C.data(), C.stride());
}

///----------------------------------------
/// Converts rvalue refs to lvalue refs.
template <typename scalar_t>
void syr2k(
    scalar_t alpha, Tile<scalar_t> const&& A,
                    Tile<scalar_t> const&& B,
    scalar_t beta,  Tile<scalar_t>&& C)
{
    syr2k( alpha, A, B, beta, C );
}

///-----------------------------------------------------------------------------
/// \brief
/// Hermitian rank-k update: $C = \alpha op(A) op(A)^H + \beta C$.
/// Use conj_transpose to set $op(A)$.
/// In the real case, C cannot be transpose.
// Allowing C^T would require two conjugations: conj( conj(C) + A*A^H ).
template <typename scalar_t>
void herk(
    blas::real_type<scalar_t> alpha, Tile<scalar_t> const& A,
    blas::real_type<scalar_t> beta,  Tile<scalar_t>& C)
{
    trace::Block trace_block("blas::herk");

    assert(A.uplo() == Uplo::General);
    assert(C.mb() == C.nb());  // square
    assert(C.mb() == A.mb());  // n
    if (C.is_complex && C.op() == Op::Trans)
        throw std::exception();

    blas::herk(blas::Layout::ColMajor,
               C.uplo(), A.op(),
               C.nb(), A.nb(),
               alpha, A.data(), A.stride(),
               beta,  C.data(), C.stride());
}

///----------------------------------------
/// Converts rvalue refs to lvalue refs.
template <typename scalar_t>
void herk(
    blas::real_type<scalar_t> alpha, Tile<scalar_t> const&& A,
    blas::real_type<scalar_t> beta,  Tile<scalar_t>&& C)
{
    herk( alpha, A, beta, C );
}

///-----------------------------------------------------------------------------
/// \brief
template <typename scalar_t>
void trmm(
    Side side, Diag diag,
    scalar_t alpha, Tile<scalar_t> const& A,
                    Tile<scalar_t>& B)
{
    trace::Block trace_block("blas::trmm");

    using blas::conj;

    assert(B.uplo() == Uplo::General);
    assert(A.mb() == A.nb());  // square
    assert(side == Side::Left ? A.mb() == B.mb()    // m
                              : A.mb() == B.nb());  // n
    if (B.op() == Op::NoTrans) {
        blas::trmm(blas::Layout::ColMajor,
                   side, A.uplo(), A.op(), diag,
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
        else if (A.op() == B.op() || A.is_real)
            // A and B are both Trans or both ConjTrans; Trans == ConjTrans if real
            opA = Op::NoTrans;
        else
            throw std::exception();

        if (B.op() == Op::ConjTrans) {
            alpha = conj( alpha );
        }

        blas::trmm(blas::Layout::ColMajor,
                   side2, A.uplo(), opA, diag,
                   B.nb(), B.mb(),
                   alpha, A.data(), A.stride(),
                          B.data(), B.stride());
    }
}

///----------------------------------------
/// Converts rvalue refs to lvalue refs.
template <typename scalar_t>
void trmm(
    Side side, Diag diag,
    scalar_t alpha, Tile<scalar_t> const&& A,
                    Tile<scalar_t>&& B)
{
    trmm( side, diag, alpha, A, B );
}

///-----------------------------------------------------------------------------
/// \brief
/// Triangular solve: $B = \alpha op(A)^{-1} B$ or $B = \alpha B op(A)^{-1}$.
/// Use transpose/conj_transpose to set op(A). uplo is set in the tile.
/// In the complex case,
/// if $op(B)$ is transpose, then $op(A)$ cannot be conj_transpose;
/// if $op(B)$ is conj_transpose, then $op(A)$ cannot be transpose.
template <typename scalar_t>
void trsm(
    Side side, Diag diag,
    scalar_t alpha, Tile<scalar_t> const& A,
                    Tile<scalar_t>& B)
{
    trace::Block trace_block("blas::trsm");

    using blas::conj;

    assert(B.uplo() == Uplo::General);
    assert(A.mb() == A.nb());  // square
    assert(side == Side::Left ? A.mb() == B.mb()    // m
                              : A.mb() == B.nb());  // n
    if (B.op() == Op::NoTrans) {
        blas::trsm(blas::Layout::ColMajor,
                   side, A.uplo(), A.op(), diag,
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
        else if (A.op() == B.op() || A.is_real)
            // A and B are both Trans or both ConjTrans; Trans == ConjTrans if real
            opA = Op::NoTrans;
        else
            throw std::exception();

        if (B.op() == Op::ConjTrans) {
            alpha = conj( alpha );
        }

        blas::trsm(blas::Layout::ColMajor,
                   side2, A.uplo(), opA, diag,
                   B.nb(), B.mb(),
                   alpha, A.data(), A.stride(),
                          B.data(), B.stride());
    }
}

///----------------------------------------
/// Converts rvalue refs to lvalue refs.
template <typename scalar_t>
void trsm(
    Side side, Diag diag,
    scalar_t alpha, Tile<scalar_t> const&& A,
                    Tile<scalar_t>&& B)
{
    trsm( side, diag, alpha, A, B );
}

} // namespace slate

#endif        //  #ifndef SLATE_TILE_BLAS_HH
