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

#include "slate.hh"
#include "slate_Matrix.hh"
#include "slate_types.hh"
#include "slate_Tile_tpmqrt.hh"
#include "internal/slate_internal.hh"
#include "internal/slate_internal_util.hh"
#include "aux/slate_Debug.hh"

namespace slate {
namespace internal {

///-----------------------------------------------------------------------------
/// Distributed QR triangle-triangle factorization of column of tiles.
/// Each rank has one triangular tile, the result of local geqrf panel.
/// Dispatches to target implementations.
template <Target target, typename scalar_t>
void unmqr(Side side, Op op,
           Matrix<scalar_t>&& A,
           Matrix<scalar_t>&& T,
           Matrix<scalar_t>&& C,
           Matrix<scalar_t>&& W)
{
    unmqr<target>(internal::TargetType<target>(),
          side, op, A, T, C, W);
}

///-----------------------------------------------------------------------------
/// Distributed QR triangle-triangle factorization, host implementation.
/// Assumes A and T are single block-column
/// Assumes W and C have same dimensions and distribution
template <Target target, typename scalar_t>
void unmqr(internal::TargetType<target>,
           Side side, Op op,
           Matrix<scalar_t>& A,
           Matrix<scalar_t>& T,
           Matrix<scalar_t>& C,
           Matrix<scalar_t>& W)
{
    int64_t A_mt = A.mt();
    int64_t C_mt = C.mt();
    int64_t C_nt = C.nt();

    assert(A_mt >= 1);
    assert(A.nt() == 1);
    assert(C_mt == A_mt);
    assert(C_nt >= 1);
    assert(W.nt() == C_nt);

    // Build a list of local tile's row indices in current matrix C.
    std::vector<int64_t> row_indices;

    for (int64_t i = 0; i < C_mt; ++i) {
        for (int64_t j = 0; j < C_nt; ++j) {
            if (C.tileIsLocal(i, j)) {
                row_indices.push_back(i);
                break;
            }
        }
    }

    if (row_indices.size() < 1)
        return;

    // Verification step
    // Build a set of panel tiles ranks correspoding to my row_indices
    // std::set<int64_t> panel_ranks;
    // for (auto r: row_indices) {
    //     panel_ranks.insert(A.tileRank(r, 0));
    // }
    // assert(panel_ranks.size() == 1);

    // this rank's top most row in this column of A holding the triangular tile
    int64_t r_top = row_indices[0];
    assert(r_top < A_mt);
    assert(r_top >= 0);

    // pick one row of W matching the local matrix top row distribution
    auto Wr = W.sub(r_top, r_top, 0, C_nt-1);
    for (int64_t j = 0; j < Wr.nt(); ++j) {
        if(Wr.tileIsLocal(0, j)){
            Wr.tileInsert(0, j);
        }
    }

    // Q = I - V x T x V**H

    if (side == Side::Left) {

        // op(Q) = I - V x op(T) x V**H

        auto T00 = T.sub(r_top, r_top, 0, 0);
        auto T0 = TriangularMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, T00);
        if (op != Op::NoTrans) {
            T0 = conj_transpose(T0);
        }
        // V = A(:,0), V is a one block column

        // Need to compute:
        // op(Q) x C = C - V x op(T) x V**H x C

        // C = |C1|
        //     |C2|
        // C1 = C(0,:)
        auto C1 = C.sub(r_top, r_top, 0, C_nt-1);
        // C2 = C(1:,:)

        // V = |V1|
        //     |V2|
        // V1 = V(0)
        auto A00 = A.sub(r_top, r_top, 0,0);
        auto V1 = TriangularMatrix<scalar_t>(Uplo::Lower, Diag::Unit, A00);
        auto V1T = conj_transpose(V1);
        // V2 = V(1:)

        // --------------------
        // op(Q) x C = C - V x op(T) x (V**H x C)
        // W = V**H x C
        // W <- C1
        C1.tileGetAllForWriting(C1.hostNum());// todo: issue omp tasks for copy to host
        Wr.copy(C1);

        internal::trmm<Target::HostTask, scalar_t>(
                        Side::Left,
                        scalar_t(1.0), std::move(V1T),
                                       std::move(Wr));

        if (row_indices.size() > 1) {
            // W <- GEMM(V2T, C2, W)
            for (int64_t ri = 1; ri < int64_t(row_indices.size()); ++ri) {
                int64_t row = row_indices[ri];
                auto ViT = conj_transpose(A.sub(row, row, 0, 0));
                auto Ci = C.sub(row, row, 0, C_nt-1);
                if (target == Target::Devices) {
                    Ci.tileGetAndHoldAllOnDevices();// todo: release the hold later
                }
                internal::gemm<target>(
                        scalar_t(1.0), std::move(ViT),
                                       std::move(Ci),
                        scalar_t(1.0), std::move(Wr));
            }
        }

        // --------------------
        // op(Q) x C = C - V x (op(T) x W)
        // W <- TRMM(T0,W)
        internal::trmm<Target::HostTask, scalar_t>(
                        Side::Left,
                        scalar_t(1.0), std::move(T0),
                                       std::move(Wr));

        // --------------------
        // op(Q) x C = C - V x W
        if (row_indices.size() > 1) {
            // C2 <- GEMM(V2, W, C2)
            internal::gemm<target>(
                    scalar_t(-1.0), A.sub(row_indices[1], A_mt-1, 0, 0),
                                    std::move(Wr),
                    scalar_t(1.0),  C.sub(row_indices[1], C_mt-1, 0, C_nt-1));
        }
        // W <- TRMM(V1,W)
        internal::trmm<Target::HostTask, scalar_t>(
                        Side::Left,
                        scalar_t(-1.0), std::move(V1),
                                        std::move(Wr));
        // C1 <- GEADD(W, C1)
        internal::geadd<Target::HostTask>(
                        scalar_t(1.0), std::move(Wr),
                        scalar_t(1.0), std::move(C1));

    }else
    if (side == Side::Right){
        // TODO
    }

    // free workspace
    for (int j = 0; j < Wr.nt(); ++j){
        if(Wr.tileIsLocal(0, j)){
            Wr.tileErase(0, j);
        }
    }

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void unmqr<Target::HostTask, float>(
    Side side, Op op,
    Matrix<float>&& A,
    Matrix<float>&& T,
    Matrix<float>&& C,
    Matrix<float>&& W);

template
void unmqr<Target::HostNest, float>(
    Side side, Op op,
    Matrix<float>&& A,
    Matrix<float>&& T,
    Matrix<float>&& C,
    Matrix<float>&& W);

template
void unmqr<Target::HostBatch, float>(
    Side side, Op op,
    Matrix<float>&& A,
    Matrix<float>&& T,
    Matrix<float>&& C,
    Matrix<float>&& W);

template
void unmqr<Target::Devices, float>(
    Side side, Op op,
    Matrix<float>&& A,
    Matrix<float>&& T,
    Matrix<float>&& C,
    Matrix<float>&& W);

// ----------------------------------------
template
void unmqr<Target::HostTask, double>(
    Side side, Op op,
    Matrix<double>&& A,
    Matrix<double>&& T,
    Matrix<double>&& C,
    Matrix<double>&& W);

template
void unmqr<Target::HostNest, double>(
    Side side, Op op,
    Matrix<double>&& A,
    Matrix<double>&& T,
    Matrix<double>&& C,
    Matrix<double>&& W);

template
void unmqr<Target::HostBatch, double>(
    Side side, Op op,
    Matrix<double>&& A,
    Matrix<double>&& T,
    Matrix<double>&& C,
    Matrix<double>&& W);

template
void unmqr<Target::Devices, double>(
    Side side, Op op,
    Matrix<double>&& A,
    Matrix<double>&& T,
    Matrix<double>&& C,
    Matrix<double>&& W);

// ----------------------------------------
template
void unmqr< Target::HostTask, std::complex<float> >(
    Side side, Op op,
    Matrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& T,
    Matrix< std::complex<float> >&& C,
    Matrix< std::complex<float> >&& W);

template
void unmqr< Target::HostNest, std::complex<float> >(
    Side side, Op op,
    Matrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& T,
    Matrix< std::complex<float> >&& C,
    Matrix< std::complex<float> >&& W);

template
void unmqr< Target::HostBatch, std::complex<float> >(
    Side side, Op op,
    Matrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& T,
    Matrix< std::complex<float> >&& C,
    Matrix< std::complex<float> >&& W);

template
void unmqr< Target::Devices, std::complex<float> >(
    Side side, Op op,
    Matrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& T,
    Matrix< std::complex<float> >&& C,
    Matrix< std::complex<float> >&& W);

// ----------------------------------------
template
void unmqr< Target::HostTask, std::complex<double> >(
    Side side, Op op,
    Matrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& T,
    Matrix< std::complex<double> >&& C,
    Matrix< std::complex<double> >&& W);

template
void unmqr< Target::HostNest, std::complex<double> >(
    Side side, Op op,
    Matrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& T,
    Matrix< std::complex<double> >&& C,
    Matrix< std::complex<double> >&& W);

template
void unmqr< Target::HostBatch, std::complex<double> >(
    Side side, Op op,
    Matrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& T,
    Matrix< std::complex<double> >&& C,
    Matrix< std::complex<double> >&& W);

template
void unmqr< Target::Devices, std::complex<double> >(
    Side side, Op op,
    Matrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& T,
    Matrix< std::complex<double> >&& C,
    Matrix< std::complex<double> >&& W);

} // namespace internal
} // namespace slate
