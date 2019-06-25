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
    int64_t mt = C.mt();
    int64_t nt = C.nt();

    assert(mt >= 1);
    assert(nt >= 1);
    assert(V.mt() == mt);
    assert(V.nt() == 1);
    assert(W.nt() == nt);

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    // Build a list of local tile's row indices in current matrix C.
    std::vector<int64_t> row_indices;
    for (int64_t i = 0; i < mt; ++i) {
        for (int64_t j = 0; j < nt; ++j) {
            if (C.tileIsLocal(i, j)) {
                row_indices.push_back(i);
                break;
            }
        }
    }
    if (row_indices.size() < 1)
        return;

    // this rank's top-most row in this column of V holding the triangular tile
    int64_t r_top = row_indices[0];
    assert(r_top < mt);
    assert(r_top >= 0);

    // pick one row of W matching the local matrix top row distribution
    auto Wr = W.sub(r_top, r_top, 0, nt-1);
    Wr.insertLocalTiles();

    if (side == Side::Left) {
        // Need to compute:
        // op(Q) C = (I - V op(T) V^H) C = C - V op(T) V^H C

        // V = | V0  |  triangular part (nb x nb)
        //     | V0b |  rectangular part, only if V0 is trapezoid (mb > nb)
        //     | V1  |  full tiles
        auto V0 = V.sub(r_top, r_top, 0, 0);
        int64_t mb = V0(0, 0).mb();
        int64_t nb = V0(0, 0).nb();

        auto T0 = T.sub(r_top, r_top, 0, 0);

        // C = | C0  |
        //     | C0b |  only if V0 is trapezoid
        //     | C1  |
        auto C0 = C.sub(r_top, r_top, 0, nt-1);
        C0.tileGetAllForWriting(C0.hostNum(), LayoutConvert(layout));// todo: issue omp tasks for copy to host

        // Householder vectors are only min( mb, nb ) columns in lower
        // triangular part of V.
        // If V0 tile is wide, slice V to first mb columns, and T to mb x mb.
        if (mb < nb) {
            V = V.slice(0, V.m()-1, 0, mb-1);  // first mb cols
            V0 = V.sub(r_top, r_top, 0, 0);
            T0 = T0.slice(0, mb-1, 0, mb-1);  // first mb x mb part
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
            T0  = T0.slice(0,  nb-1, 0, nb-1);  // first nb x nb part
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
        // op(Q) C = C - V op(T) (V^H C)
        // W = V^H C

        // W <- C0
        // W <- V0^H W
        Wr.copy(C0);
        internal::trmm<Target::HostTask>(
                Side::Left,
                scalar_t(1.0), conj_transpose(V0tr),
                               std::move(Wr));

        if (trapezoid) {
            // W <- V0b^H C0b
            internal::gemm<Target::HostTask>(
                    scalar_t(1.0), conj_transpose(V0b),
                                   std::move(C0b),
                    scalar_t(1.0), std::move(Wr),
                    layout);
        }

        // W <- V1^H C1 + W
        for (int64_t ri = 1; ri < int64_t(row_indices.size()); ++ri) {
            int64_t row = row_indices[ri];
            auto Ci = C.sub(row, row, 0, nt-1);
            if (target == Target::Devices) {
                // todo: release the hold later
                Ci.tileGetAndHoldAllOnDevices(LayoutConvert(layout));
            }
            internal::gemm<target>(
                    scalar_t(1.0), conj_transpose(V.sub(row, row, 0, 0)),
                                   std::move(Ci),
                    scalar_t(1.0), std::move(Wr),
                    layout);
        }

        // --------------------
        // op(Q) C = C - V (op(T) W)
        // W <- T0^{op} W
        internal::trmm<Target::HostTask>(
                Side::Left,
                scalar_t(1.0), std::move(T0tr),
                               std::move(Wr));

        // --------------------
        // op(Q) C = C - V W
        if (row_indices.size() > 1) {
            // C1 <- C1 - V1 W
            internal::gemm<target>(
                    scalar_t(-1.0), V.sub(row_indices[1], mt-1, 0, 0),
                                    std::move(Wr),
                    scalar_t(1.0),  C.sub(row_indices[1], mt-1, 0, nt-1),
                    layout);
        }

        if (trapezoid) {
            // C0b <- C0b - V0b W
            internal::gemm<Target::HostTask>(
                    scalar_t(-1.0), std::move(V0b),
                                    std::move(Wr),
                    scalar_t(1.0),  std::move(C0b),
                    layout);
        }

        // W <- V0 W
        internal::trmm<Target::HostTask>(
                Side::Left,
                scalar_t(1.0), std::move(V0tr),
                               std::move(Wr));
        // C0 <- C0 - W
        internal::geadd<Target::HostTask>(
                scalar_t(-1.0), std::move(Wr),
                scalar_t(1.0),  std::move(C0));
    }
    else if (side == Side::Right) {
        // TODO
    }

    // free workspace
    Wr.clear();

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
