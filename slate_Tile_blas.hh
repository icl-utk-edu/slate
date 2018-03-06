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
        else if (A.op() == C.op() || std::is_arithmetic< scalar_t >::value)
            // A and C are both Trans or both ConjTrans;
            // Trans and ConjTrans are identical if scalar_t is real
            opA = Op::NoTrans;
        else
            throw std::exception();

        Op opB;
        if (B.op() == Op::NoTrans)
            opB = C.op();
        else if (B.op() == C.op() || std::is_arithmetic< scalar_t >::value)
            // B and C are both Trans or both ConjTrans;
            // Trans and ConjTrans are identical if scalar_t is real
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
    if (is_complex< scalar_t >::value && C.op() == Op::ConjTrans)
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
/// Hermitian rank-k update: $C = \alpha op(A) op(A)^H + \beta C$.
/// Use conj_transpose to set $op(A)$.
/// In the real case, C cannot be transpose.
// Allowing C^T would require two conjugations: conj( conj(C) + A*A^H ).
template <typename scalar_t>
void herk(
    typename blas::traits<scalar_t>::real_t alpha, Tile<scalar_t> const& A,
    typename blas::traits<scalar_t>::real_t beta,  Tile<scalar_t>& C)
{
    trace::Block trace_block("blas::herk");

    assert(A.uplo() == Uplo::General);
    assert(C.mb() == C.nb());  // square
    assert(C.mb() == A.mb());  // n
    if (is_complex< scalar_t >::value && C.op() == Op::Trans)
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
    typename blas::traits<scalar_t>::real_t alpha, Tile<scalar_t> const&& A,
    typename blas::traits<scalar_t>::real_t beta,  Tile<scalar_t>&& C)
{
    herk( alpha, A, beta, C );
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
        if (is_complex< scalar_t >::value &&
            A.op() != Op::NoTrans && A.op() != B.op())
            throw std::exception();

        // switch op(A) <=> op(B), side left <=> right, m <=> n
        Side side2 = (side == Side::Left ? Side::Right : Side::Left);
        Op opA;
        if (A.op() == Op::NoTrans)
            opA = B.op();
        else if (A.op() == B.op() || std::is_arithmetic< scalar_t >::value)
            // A and B are both Trans or both ConjTrans;
            // Trans and ConjTrans are identical if scalar_t is real
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
