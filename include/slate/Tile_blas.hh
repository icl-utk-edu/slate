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
/// Use transpose() or conj_transpose() to set $op(A)$, $op(B)$, and $op(C)$.
/// In the complex case,
/// if $op(C)$ is transpose, then $op(A)$ and $op(B)$ cannot be conj_transpose;
/// if $op(C)$ is conj_transpose, then $op(A)$ and $op(B)$ cannot be transpose.
/// @ingroup gemm_tile
///
template <typename scalar_t>
void gemm(
    scalar_t alpha, Tile<scalar_t> const& A,
                    Tile<scalar_t> const& B,
    scalar_t beta,  Tile<scalar_t>& C)
{
    trace::Block trace_block("blas::gemm");

    // assumes column major for now
    // todo: relax this assumption
    const blas::Layout layout = blas::Layout::ColMajor;

    using blas::conj;

    assert(A.uploPhysical() == Uplo::General);
    assert(B.uploPhysical() == Uplo::General);
    assert(C.uploPhysical() == Uplo::General);
    assert(C.mb() == A.mb());  // m
    assert(C.nb() == B.nb());  // n
    assert(A.nb() == B.mb());  // k
    assert(A.layout() == layout);
    assert(B.layout() == layout);
    assert(C.layout() == layout);

    if (C.op() == Op::NoTrans) {
        // C = opA(A) opB(B) + C
        blas::gemm(layout,
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

        blas::gemm(layout,
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
/// Use conj_transpose to set $op(A)$.
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
/// Use transpose or conj_transpose to set $op(A)$ and $op(B)$.
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
/// Symmetric rank-k update: $C = \alpha op(A) op(A)^T + \beta C$.
/// Use transpose or conj_transpose to set $op(A)$.
/// In the complex case, C cannot be conj_transpose.
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
/// Use transpose or conj_transpose to set $op(A)$ and $op(B)$.
/// In the complex case, C cannot be conj_transpose.
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
/// Use transpose/conj_transpose to set op(A). uplo is set in the tile.
/// In the complex case,
/// if $op(B)$ is transpose, then $op(A)$ cannot be conj_transpose;
/// if $op(B)$ is conj_transpose, then $op(A)$ cannot be transpose.
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
    for (int64_t j = 0; j < A.nb(); ++j)
        for (int64_t i = 0; i < A.mb(); ++i)
            A.at(i, j) *= alpha;
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
/// Swap rows or columns of two local tiles, depending on op().
/// @ingroup swap_tile
///
template <typename scalar_t>
void swap(int64_t j_offs, int64_t n,
          Tile<scalar_t>& A, int64_t i1,
          Tile<scalar_t>& B, int64_t i2)
{
    // todo: size assertions
    for (int64_t j = j_offs; j < j_offs+n; ++j)
        std::swap(A.at(i1, j), B.at(i2, j));
}

//--------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup swap_tile
///
template <typename scalar_t>
void swap(int64_t j_offs, int64_t n,
          Tile<scalar_t>&& A, int64_t i1,
          Tile<scalar_t>&& B, int64_t i2)
{
    swap(j_offs, n, A, i1, B, i2);
}

//------------------------------------------------------------------------------
/// Swap rows or columns with another process, depending on op().
/// @ingroup swap_tile
///
template <typename scalar_t>
void swap(int64_t j, int64_t n,
          Tile<scalar_t>& A, int64_t i,
          int other_rank, MPI_Comm mpi_comm, int tag = 0)
{
    std::vector<scalar_t> local_row(n);
    std::vector<scalar_t> other_row(n);

    for (int64_t k = 0; k < n; ++k)
        local_row[k] = A(i, j+k);

    MPI_Sendrecv(
        local_row.data(), n, mpi_type<scalar_t>::value, other_rank, tag,
        other_row.data(), n, mpi_type<scalar_t>::value, other_rank, tag,
        mpi_comm, MPI_STATUS_IGNORE);

    for (int64_t k = 0; k < n; ++k)
         A.at(i, j+k) = other_row[k];
}

//--------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup swap_tile
///
template <typename scalar_t>
void swap(int64_t j, int64_t n,
          Tile<scalar_t>&& A, int64_t i,
          int other_rank, MPI_Comm mpi_comm, int tag = 0)
{
    swap(j, n, A, i, other_rank, mpi_comm, tag);
}

//------------------------------------------------------------------------------
/// Swap rows or columns with another process, depending on op().
/// @ingroup swap_tile
///
/// todo: implement with a GPUDirect call
template <typename scalar_t>
void swap(int64_t j, int64_t n,
          int device, Tile<scalar_t>& A, int64_t i,
          int other_rank, MPI_Comm mpi_comm, cudaStream_t stream, int tag = 0)
{
    std::vector<scalar_t> local_row(n);
    std::vector<scalar_t> other_row(n);

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
void swap(int64_t j, int64_t n,
          int device, Tile<scalar_t>&& A, int64_t i,
          int other_rank, MPI_Comm mpi_comm, cudaStream_t stream, int tag = 0)
{
    swap(j, n, device, A, i, other_rank, mpi_comm, stream, tag);
}

//------------------------------------------------------------------------------
/// Swap one element with another process.
/// @ingroup swap_tile
///
template <typename scalar_t>
void swap(Tile<scalar_t>& A, int64_t i, int64_t j,
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
void swap(Tile<scalar_t>&& A, int64_t i, int64_t j,
          int other_rank, MPI_Comm mpi_comm, int tag = 0)
{
    swap(A, i, j, other_rank, mpi_comm, tag);
}

//------------------------------------------------------------------------------
/// Computes $Y = \alpha X + Y$.
/// @ingroup geadd_tile
///
template <typename scalar_t>
void axpy(scalar_t alpha, Tile<scalar_t> const& X, Tile<scalar_t>& Y)
{
    trace::Block trace_block("blas::axpy");

    // todo: relax these assumptions, by adjusting the loops below
    assert(X.op() == Y.op());
    assert(X.uploPhysical() == Uplo::General);
    assert(Y.uploPhysical() == Uplo::General);

    for (int64_t i = 0; i < std::min(X.mb(), Y.mb()); ++i)
        for (int64_t j = 0; j < std::min(X.nb(), Y.nb()); ++j)
            Y.at(i, j) += alpha*X(i, j);
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup geadd_tile
///
template <typename scalar_t>
void axpy(scalar_t alpha, Tile<scalar_t> const& X, Tile<scalar_t>&& Y)
{
    axpy(alpha, X, Y);
}

//------------------------------------------------------------------------------
/// Computes $Y = \alpha X + \beta Y$.
/// @ingroup geadd_tile
///
template <typename scalar_t>
void axpby(scalar_t alpha, Tile<scalar_t> const& X,
           scalar_t beta, Tile<scalar_t>& Y)
{
    // trace::Block trace_block("blas::axpby");

    // TODO should be able to loosen these restrictions
    assert(X.op() == Y.op());
    assert(X.uploPhysical() == Uplo::General);
    assert(Y.uploPhysical() == Uplo::General);

    for (int64_t i = 0; i < std::min(X.mb(), Y.mb()); ++i)
        for (int64_t j = 0; j < std::min(X.nb(), Y.nb()); ++j)
            Y.at(i, j) = alpha*X(i, j) + beta*Y(i, j);
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup geadd_tile
///
template <typename scalar_t>
void axpby(scalar_t alpha, Tile<scalar_t> const& X,
           scalar_t beta, Tile<scalar_t>&& Y)
{
    axpby(alpha, X, beta, Y);
}

//------------------------------------------------------------------------------
/// Copy and precision conversion.
/// TODO: Move functiona that are not really BLAS to Tile_aux.hh.
/// @ingroup copy_tile
///
template <typename src_scalar_t, typename dst_scalar_t>
void gecopy(Tile<src_scalar_t> const& A, Tile<dst_scalar_t>& B)
{
//  trace::Block trace_block("aux::copy");

    assert(A.mb() == B.mb());
    assert(A.nb() == B.nb());

    for (int64_t j = 0; j < B.nb(); ++j)
        for (int64_t i = 0; i < B.mb(); ++i)
            B.at(i, j) = A.at(i, j);
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup copy_tile
///
template <typename src_scalar_t, typename dst_scalar_t>
void gecopy(Tile<src_scalar_t> const&& A, Tile<dst_scalar_t>&& B)
{
    gecopy(A, B);
}

//------------------------------------------------------------------------------
/// Copy and precision conversion.
/// TODO: Move functiona that are not really BLAS to Tile_aux.hh.
/// @ingroup copy_tile
///
template <typename src_scalar_t, typename dst_scalar_t>
void tzcopy(Tile<src_scalar_t> const& A, Tile<dst_scalar_t>& B)
{
//  trace::Block trace_block("aux::copy");

    // TODO: Can be loosened?
    assert(A.uplo() != Uplo::General);
    assert(B.uplo() == A.uplo());

    assert(A.op() == Op::NoTrans);
    assert(B.op() == A.op());

    assert(A.mb() == B.mb());
    assert(A.nb() == B.nb());

    for (int64_t j = 0; j < B.nb(); ++j) {
        if (j < B.mb()) {
            B.at(j, j) = A.at(j, j);
        }
        if (B.uplo() == Uplo::Lower) {
            for (int64_t i = j; i < B.mb(); ++i) {
                B.at(i, j) = A.at(i, j);
            }
        }
        else {
            for (int64_t i = 0; i <= j && i < B.mb(); ++i) {
                B.at(i, j) = A.at(i, j);
            }
        }
    }
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup copy_tile
///
template <typename src_scalar_t, typename dst_scalar_t>
void tzcopy(Tile<src_scalar_t> const&& A, Tile<dst_scalar_t>&& B)
{
    tzcopy(A, B);
}

//------------------------------------------------------------------------------
/// In-place conversion between column and row-major layout for square tiles.
/// Takes a pointer to the original tile in MatrixStorage, instead of a
/// reference to a copy of the tile, in order to adjust the tile's layout flag.
/// @ingroup convert_tile
///
template <typename scalar_t>
void convert_layout(Tile<scalar_t>* X)
{
    trace::Block trace_block("slate::convert_layout");
    assert(X->mb() == X->nb());

    for (int64_t j = 0; j < X->nb(); ++j) {
        for (int64_t i = 0; i < j; ++i) { // upper
            std::swap(X->at(i, j), X->at(j, i));
        }
    }

    X->layout(X->layout() == Layout::RowMajor ? Layout::ColMajor
                                              : Layout::RowMajor);
}

//------------------------------------------------------------------------------
/// In-place conversion between column and row-major layout for square tiles.
/// Takes a pointer to the original tile in MatrixStorage, instead of a
/// reference to a copy of the tile, in order to adjust the tile's layout flag.
/// @ingroup convert_tile
///
template <typename scalar_t>
void convert_layout(Tile<scalar_t>* X, cudaStream_t stream)
{
    trace::Block trace_block("slate::device::transpose");
    assert(X->mb() == X->nb());

    device::transpose(X->mb(), X->data(), X->stride(), stream);
    slate_cuda_call(
        cudaStreamSynchronize(stream));

    X->layout(X->layout() == Layout::RowMajor ? Layout::ColMajor
                                              : Layout::RowMajor);
}

} // namespace slate

#endif // SLATE_TILE_BLAS_HH
