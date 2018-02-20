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
    trace::Block trace_block(trace::Color::MediumAquamarine);

    assert(A.uplo() == blas::Uplo::General);
    assert(B.uplo() == blas::Uplo::General);
    assert(C.uplo() == blas::Uplo::General);
    assert(C.op() == blas::Op::NoTrans);  // todo: row-major
    assert(C.m() == A.m());  // m
    assert(C.n() == B.n());  // n
    assert(A.n() == B.m());  // k
    blas::gemm(blas::Layout::ColMajor,
               A.op(), B.op(),
               C.m(), C.n(), A.n(),
               alpha, A.data(), A.stride(),
                      B.data(), B.stride(),
               beta,  C.data(), C.stride());
}

///-----------------------------------------------------------------------------
/// \brief
/// Cholesky factorization of tile: $L L^H = A$ or $U^H U = A$.
/// uplo is set in the tile.
template <typename scalar_t>
void potrf(Tile<scalar_t>& A)
{
    trace::Block trace_block(trace::Color::RosyBrown);

    assert(A.op() == blas::Op::NoTrans);  // todo: row-major
    lapack::potrf(A.uplo(),
                  A.n(),
                  A.data(), A.stride());
}

///-----------------------------------------------------------------------------
/// \brief
/// Symmetric rank-k update: $C = \alpha op(A) op(A)^T + \beta C$.
/// Use transpose/conj_transpose to set $op(A)$.
template <typename scalar_t>
void syrk(
    scalar_t alpha, Tile<scalar_t> const& A,
    scalar_t beta,  Tile<scalar_t>& C)
{
    trace::Block trace_block(trace::Color::CornflowerBlue);

    assert(A.uplo() == blas::Uplo::General);
    assert(C.m() == C.n());  // square
    assert(C.m() == A.m());  // n
    assert(C.op() == blas::Op::NoTrans);  // todo: row-major
    blas::syrk(blas::Layout::ColMajor,
               C.uplo(), A.op(),
               C.n(), A.n(),
               alpha, A.data(), A.stride(),
               beta,  C.data(), C.stride());
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
    trace::Block trace_block(trace::Color::MediumPurple);

    assert(B.uplo() == blas::Uplo::General);
    assert(B.op() == blas::Op::NoTrans);  // todo: row-major
    assert(A.m() == A.n());  // square
    assert(side == blas::Side::Left ? A.m() == B.m()    // m
                                    : A.m() == B.n());  // n
    blas::trsm(blas::Layout::ColMajor,
               side, A.uplo(), A.op(), diag,
               B.m(), B.n(),
               alpha, A.data(), A.stride(),
                      B.data(), B.stride());
}

} // namespace slate

#endif        //  #ifndef SLATE_TILE_BLAS_HH
