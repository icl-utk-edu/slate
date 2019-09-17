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

#include "slate/Matrix.hh"
#include "slate/types.hh"
#include "internal/Tile_tpmqrt.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Multiply matrix by Q from local QR factorization.
/// @ingroup geqrf_internal
///
template <Target target, typename scalar_t>
void unmqr(Side side, Op op,
           Matrix<scalar_t>&& V,
           Matrix<scalar_t>&& T,
           Matrix<scalar_t>&& C,
           Matrix<scalar_t>&& W)
{
    unmqr<target>(internal::TargetType<target>(),
          side, op, V, T, C, W);
}

//------------------------------------------------------------------------------
/// Multiply matrix by Q from local QR factorization.
/// C = op(Q) C for side = left, or
/// C = C op(Q) for side = right.
/// Assumes V and T are each a single block-column.
/// Assumes W and C have the same dimensions and distribution.
/// This corresponds to larfb( ..., direct=Forward, storev=Columnwise, ... ).
/// This does not include applying the distributed triangle-triangle reductions.
/// @ingroup geqrf_internal
///
template <Target target, typename scalar_t>
void unmqr(internal::TargetType<target>,
           Side side, Op op,
           Matrix<scalar_t>  V,  // pass by value, not reference, for slicing
           Matrix<scalar_t>& T,
           Matrix<scalar_t>& C,
           Matrix<scalar_t>& W)
{
    const scalar_t one = 1;

    int64_t mt = C.mt();
    int64_t nt = C.nt();

    assert(mt >= 1);
    assert(nt >= 1);
    assert(V.mt() == mt);
    assert(V.nt() == 1);
    assert(W.mt() == mt);
    assert(W.nt() == nt);

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    // Build a list of row indices that have local tiles in current matrix C.
    std::vector<int64_t> local_indices;
    for (int64_t i = 0; i < mt; ++i) {
        for (int64_t j = 0; j < nt; ++j) {
            if (C.tileIsLocal(i, j)) {
                local_indices.push_back(i);
                break;
            }
        }
    }
    if (local_indices.size() < 1)
        return;

    // This rank's first (top-most) local row of V holds the triangular tile.
    int64_t first = local_indices[0];
    assert(first < mt);
    assert(first >= 0);

    // Get corresponding row of W to match the local matrix distribution.
    auto Wr = W.sub(first, first, 0, nt-1);
    Wr.insertLocalTiles();

    if (side == Side::Left) {
        // Compute
        // op(Q) C = (I - V op(T) V^H) C = C - V op(T) V^H C
        // in three major steps:
        // 1. W = V^H C
        // 2. W = op(T) W
        // 3. C = C - V W

        // V = [ V0  ]  triangular part (nb-by-nb)
        //     [ V0b ]  rectangular part, only if V0 is trapezoid (mb > nb)
        //     [ V1  ]  remaining tiles
        // Example: m = 6, n = 3, nb = 4, V0 tile is trapezoid (4x3):
        // V = [ .     ]  V0
        //     [ . .   ]
        //     [ . . . ]
        //     [- - - -]
        //     [ . . . ]  V0b
        //     [-------]
        //     [ . . . ]  V1
        //     [ . . . ]
        auto V0 = V.sub(first, first, 0, 0);
        int64_t mb = V0.tileMb(0);
        int64_t nb = V0.tileNb(0);

        auto T0 = T.sub(first, first, 0, 0);

        // C = [ C0  ]
        //     [ C0b ]  only if V0 is trapezoid
        //     [ C1  ]
        auto C0 = C.sub(first, first, 0, nt-1);
        // todo: issue omp tasks for copy to host
        C0.tileGetAllForWriting(C0.hostNum(), LayoutConvert(layout));

        // Householder vectors are only first min( mb, nb ) columns in lower
        // triangular part of V. If V0 tile is short (mb < nb), slice V to
        // first mb columns, and T to mb-by-mb. This can happen when V0 is
        // the bottom block row.
        // Example: m = 3, n = 5, nb = 5, V0 tile is wide trapezoid (3x5):
        // V0 = [ .     |     ]
        //      [ . .   |     ]
        //      [ . . . |     ]
        if (mb < nb) {
            V = V.slice(0, V.m()-1, 0, mb-1);  // first mb cols
            V0 = V.sub(first, first, 0, 0);
            T0 = T0.slice(0, mb-1, 0, mb-1);  // first mb-by-mb part
            nb = mb;
        }

        // If V0 tile is a tall trapezoid, slice V0 into triangular and
        // rectangular parts, and slice T, C, and Wr correspondingly.
        bool trapezoid = (mb > nb);
        Matrix<scalar_t> V0b, C0b;
        if (trapezoid) {
            int64_t n = C0.n();
            V0b = V0.slice(nb, mb-1, 0, nb-1);  // last mb - nb rows
            V0  = V0.slice(0,  nb-1, 0, nb-1);  // first nb rows
            T0  = T0.slice(0,  nb-1, 0, nb-1);  // first nb-by-nb part
            C0b = C0.slice(nb, mb-1, 0, n-1);   // last mb - nb rows
            C0  = C0.slice(0,  nb-1, 0, n-1);   // first nb rows
            Wr  = Wr.slice(0,  nb-1, 0, n-1);   // first nb rows
        }

        // Interpret as triangular matrices.
        auto V0tr = TriangularMatrix<scalar_t>(Uplo::Lower, Diag::Unit, V0);
        auto T0tr = TriangularMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, T0);
        if (op != Op::NoTrans) {
            T0tr = conj_transpose(T0tr);
        }

        // --------------------
        // 1. W = V^H C

        // W <- C0
        // W <- V0^H W
        internal::copy(std::move(C0), std::move(Wr));
        internal::trmm<Target::HostTask>(
                Side::Left,
                one, conj_transpose(V0tr),
                     std::move(Wr));

        if (trapezoid) {
            // W <- V0b^H C0b + W
            internal::gemm<Target::HostTask>(
                    one, conj_transpose(V0b),
                         std::move(C0b),
                    one, std::move(Wr),
                    layout);
        }

        // W <- V1^H C1 + W
        for (int64_t i = 1; i < int64_t(local_indices.size()); ++i) {
            int64_t row = local_indices[i];
            auto Ci = C.sub(row, row, 0, nt-1);
            if (target == Target::Devices) {
                // todo: release the hold later
                Ci.tileGetAndHoldAllOnDevices(LayoutConvert(layout));
            }
            internal::gemm<target>(
                    one, conj_transpose(V.sub(row, row, 0, 0)),
                         std::move(Ci),
                    one, std::move(Wr),
                    layout);
        }

        // --------------------
        // 2. W <- op(T0) W; op is already applied to T0tr.
        internal::trmm<Target::HostTask>(
                Side::Left,
                one, std::move(T0tr),
                     std::move(Wr));

        // --------------------
        // 3. C = C - V W
        if (local_indices.size() > 1) {
            // C1 <- C1 - V1 W
            internal::gemm<target>(
                    -one, V.sub(local_indices[1], mt-1, 0, 0),
                          std::move(Wr),
                    one,  C.sub(local_indices[1], mt-1, 0, nt-1),
                    layout);
        }

        if (trapezoid) {
            // C0b <- C0b - V0b W
            internal::gemm<Target::HostTask>(
                    -one, std::move(V0b),
                          std::move(Wr),
                    one,  std::move(C0b),
                    layout);
        }

        // W <- V0 W
        internal::trmm<Target::HostTask>(
                Side::Left,
                one, std::move(V0tr),
                     std::move(Wr));
        // C0 <- C0 - W
        internal::geadd<Target::HostTask>(
                -one, std::move(Wr),
                one,  std::move(C0));
    }
    else if (side == Side::Right) {
        // TODO
    }

    // free workspace
    // todo: Wr.clear();
    for (int j = 0; j < Wr.nt(); ++j) {
        if (Wr.tileIsLocal(0, j)) {
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
    Matrix<float>&& V,
    Matrix<float>&& T,
    Matrix<float>&& C,
    Matrix<float>&& W);

template
void unmqr<Target::HostNest, float>(
    Side side, Op op,
    Matrix<float>&& V,
    Matrix<float>&& T,
    Matrix<float>&& C,
    Matrix<float>&& W);

template
void unmqr<Target::HostBatch, float>(
    Side side, Op op,
    Matrix<float>&& V,
    Matrix<float>&& T,
    Matrix<float>&& C,
    Matrix<float>&& W);

template
void unmqr<Target::Devices, float>(
    Side side, Op op,
    Matrix<float>&& V,
    Matrix<float>&& T,
    Matrix<float>&& C,
    Matrix<float>&& W);

// ----------------------------------------
template
void unmqr<Target::HostTask, double>(
    Side side, Op op,
    Matrix<double>&& V,
    Matrix<double>&& T,
    Matrix<double>&& C,
    Matrix<double>&& W);

template
void unmqr<Target::HostNest, double>(
    Side side, Op op,
    Matrix<double>&& V,
    Matrix<double>&& T,
    Matrix<double>&& C,
    Matrix<double>&& W);

template
void unmqr<Target::HostBatch, double>(
    Side side, Op op,
    Matrix<double>&& V,
    Matrix<double>&& T,
    Matrix<double>&& C,
    Matrix<double>&& W);

template
void unmqr<Target::Devices, double>(
    Side side, Op op,
    Matrix<double>&& V,
    Matrix<double>&& T,
    Matrix<double>&& C,
    Matrix<double>&& W);

// ----------------------------------------
template
void unmqr< Target::HostTask, std::complex<float> >(
    Side side, Op op,
    Matrix< std::complex<float> >&& V,
    Matrix< std::complex<float> >&& T,
    Matrix< std::complex<float> >&& C,
    Matrix< std::complex<float> >&& W);

template
void unmqr< Target::HostNest, std::complex<float> >(
    Side side, Op op,
    Matrix< std::complex<float> >&& V,
    Matrix< std::complex<float> >&& T,
    Matrix< std::complex<float> >&& C,
    Matrix< std::complex<float> >&& W);

template
void unmqr< Target::HostBatch, std::complex<float> >(
    Side side, Op op,
    Matrix< std::complex<float> >&& V,
    Matrix< std::complex<float> >&& T,
    Matrix< std::complex<float> >&& C,
    Matrix< std::complex<float> >&& W);

template
void unmqr< Target::Devices, std::complex<float> >(
    Side side, Op op,
    Matrix< std::complex<float> >&& V,
    Matrix< std::complex<float> >&& T,
    Matrix< std::complex<float> >&& C,
    Matrix< std::complex<float> >&& W);

// ----------------------------------------
template
void unmqr< Target::HostTask, std::complex<double> >(
    Side side, Op op,
    Matrix< std::complex<double> >&& V,
    Matrix< std::complex<double> >&& T,
    Matrix< std::complex<double> >&& C,
    Matrix< std::complex<double> >&& W);

template
void unmqr< Target::HostNest, std::complex<double> >(
    Side side, Op op,
    Matrix< std::complex<double> >&& V,
    Matrix< std::complex<double> >&& T,
    Matrix< std::complex<double> >&& C,
    Matrix< std::complex<double> >&& W);

template
void unmqr< Target::HostBatch, std::complex<double> >(
    Side side, Op op,
    Matrix< std::complex<double> >&& V,
    Matrix< std::complex<double> >&& T,
    Matrix< std::complex<double> >&& C,
    Matrix< std::complex<double> >&& W);

template
void unmqr< Target::Devices, std::complex<double> >(
    Side side, Op op,
    Matrix< std::complex<double> >&& V,
    Matrix< std::complex<double> >&& T,
    Matrix< std::complex<double> >&& C,
    Matrix< std::complex<double> >&& W);

} // namespace internal
} // namespace slate
