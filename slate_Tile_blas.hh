#ifndef SLATE_TILE_BLAS_HH
#define SLATE_TILE_BLAS_HH

#include <blas.hh>

#include "slate_Tile.hh"

namespace slate {

///=============================================================================
// Tile BLAS

///-----------------------------------------------------------------------------
/// \brief
/// General matrix multiply: $C = \alpha op(A) op(B) + \beta C$.
/// Use transpose/conj_transpose to set $op(A)$ and $op(B)$.
template <typename scalar_t>
void gemm(
    scalar_t alpha, Tile<scalar_t> const& A,
                    Tile<scalar_t> const& B,
    scalar_t beta,  Tile<scalar_t>& C)
{
    trace::Block trace_block("blas::gemm");

    assert(A.uplo() == blas::Uplo::General);
    assert(B.uplo() == blas::Uplo::General);
    assert(C.uplo() == blas::Uplo::General);
    assert(C.op() == blas::Op::NoTrans);  // todo: row-major
    assert(C.mb() == A.mb());  // m
    assert(C.nb() == B.nb());  // n
    assert(A.nb() == B.mb());  // k
    blas::gemm(blas::Layout::ColMajor,
               A.op(), B.op(),
               C.mb(), C.nb(), A.nb(),
               alpha, A.data(), A.stride(),
                      B.data(), B.stride(),
               beta,  C.data(), C.stride());
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
void potrf(Tile<scalar_t>& A)
{
    trace::Block trace_block("lapack::potrf");

    assert(A.op() == blas::Op::NoTrans);  // todo: row-major
    lapack::potrf(A.uplo(),
                  A.nb(),
                  A.data(), A.stride());
}

///----------------------------------------
/// Converts rvalue refs to lvalue refs.
template <typename scalar_t>
void potrf(Tile<scalar_t>&& A)
{
    potrf( A );
}

///-----------------------------------------------------------------------------
/// \brief
/// Symmetric rank-k update: $C = \alpha op(A) op(A)^T + \beta C$.
/// Use transpose or conj_transpose to set $op(A)$.
template <typename scalar_t>
void syrk(
    scalar_t alpha, Tile<scalar_t> const& A,
    scalar_t beta,  Tile<scalar_t>& C)
{
    trace::Block trace_block("blas::syrk");

    assert(A.uplo() == blas::Uplo::General);
    assert(C.mb() == C.nb());  // square
    assert(C.mb() == A.mb());  // n
    assert(C.op() == blas::Op::NoTrans);  // todo: row-major
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
/// Hermitian rank-k update: $C = \alpha op(A) op(A)^T + \beta C$.
/// Use conj_transpose to set $op(A)$.
template <typename scalar_t>
void herk(
    typename blas::traits<scalar_t>::real_t alpha, Tile<scalar_t> const& A,
    typename blas::traits<scalar_t>::real_t beta,  Tile<scalar_t>& C)
{
    trace::Block trace_block("blas::herk");

    assert(A.uplo() == blas::Uplo::General);
    assert(C.mb() == C.nb());  // square
    assert(C.mb() == A.mb());  // n
    assert(C.op() == blas::Op::NoTrans);  // todo: row-major
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
template <typename scalar_t>
void trsm(
    blas::Side side, blas::Diag diag,
    scalar_t alpha, Tile<scalar_t> const& A,
                    Tile<scalar_t>& B)
{
    trace::Block trace_block("blas::trsm");

    assert(B.uplo() == blas::Uplo::General);
    assert(B.op() == blas::Op::NoTrans);  // todo: row-major
    assert(A.mb() == A.nb());  // square
    assert(side == blas::Side::Left ? A.mb() == B.mb()    // m
                                    : A.mb() == B.nb());  // n
    blas::trsm(blas::Layout::ColMajor,
               side, A.uplo(), A.op(), diag,
               B.mb(), B.nb(),
               alpha, A.data(), A.stride(),
                      B.data(), B.stride());
}

///----------------------------------------
/// Converts rvalue refs to lvalue refs.
template <typename scalar_t>
void trsm(
    blas::Side side, blas::Diag diag,
    scalar_t alpha, Tile<scalar_t> const&& A,
                    Tile<scalar_t>&& B)
{
    trsm( side, diag, alpha, A, B );
}

} // namespace slate

#endif        //  #ifndef SLATE_TILE_BLAS_HH
