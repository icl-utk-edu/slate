// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/types.hh"
#include "internal/Tile_tpmlqt.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Multiply matrix by Q from local LQ factorization.
/// @ingroup gelqf_internal
///
template <Target target, typename scalar_t>
void unmlq(Side side, Op op,
           Matrix<scalar_t>&& V,
           Matrix<scalar_t>&& T,
           Matrix<scalar_t>&& C,
           Matrix<scalar_t>&& W,
           int priority, int64_t queue_index )
{
    unmlq<target>(internal::TargetType<target>(),
          side, op, V, T, C, W, priority, queue_index );
}

//------------------------------------------------------------------------------
/// Multiply matrix by Q from local LQ factorization.
/// C = op(Q) C for side = left, or
/// C = C op(Q) for side = right.
/// Assumes V and T are each a single block-row.
/// Assumes W and C have the same dimensions and distribution.
/// This corresponds to larfb( ..., direct=Forward, storev=Rowwise, ... ).
/// This does not include applying the distributed triangle-triangle reductions.
/// @ingroup gelqf_internal
///
template <Target target, typename scalar_t>
void unmlq(internal::TargetType<target>,
           Side side, Op op,
           Matrix<scalar_t>  V,  // pass by value, not reference, for slicing
           Matrix<scalar_t>& T,
           Matrix<scalar_t>& C,
           Matrix<scalar_t>& W,
           int priority, int64_t queue_index )
{
    const scalar_t one = 1;

    int64_t mt = C.mt();
    int64_t nt = C.nt();

    assert(mt >= 1);
    assert(nt >= 1);
    assert(V.mt() == 1);
    assert(W.mt() == mt);
    assert(W.nt() == nt);

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    if (side == Side::Left) {
        //----------------------------------------
        // Multiply by Q on left:
        // op(Q) C = (I - V^H op(T) V) C = C - V^H op(T) V C
        // in three major steps:
        // 1. W = V C
        // 2. W = op(T) W
        // 3. C = C - V^H W

        assert(V.nt() == mt);

        // Build a list of row indices that have local tiles in matrix C.
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

        // This rank's first (top-most) local row of V holds the triangular tile.
        int64_t first = row_indices[0];
        assert(first < mt);
        assert(first >= 0);

        // Get corresponding row of W to match the local matrix distribution.
        auto Wr = W.sub(first, first, 0, nt-1);
        Wr.insertLocalTiles(target);

        // V = [ V0  V0b  V1 ]
        // V0  is triangular part (mb-by-mb)
        // V0b is rectangular part, non-empty only if V0 is trapezoid (nb > mb)
        // V1  is remaining tiles
        // Example: m = 3, n = 6, nb = 4, V0 tile is trapezoid (3x4):
        // V = [ . . . | . | . . ]
        //     [   . . | . | . . ]
        //     [     . | . | . . ]
        //       V0     V0b  V1
        auto V0 = V.sub(0, 0, first, first);
        int64_t mb = V0.tileMb(0);
        int64_t nb = V0.tileNb(0);

        auto T0 = T.sub(0, 0, first, first);
        // Use first mb-by-mb or nb-by-nb portion.
        T0 = T0.slice(0, std::min(mb, nb)-1, 0, std::min(mb, nb)-1);

        // C = [ C0  ]
        //     [ C0b ]  non-empty only if V0 is trapezoid
        //     [ C1  ]
        auto C0 = C.sub(first, first, 0, nt-1);
        // todo: issue omp tasks for copy to host
        C0.tileGetAllForWriting( HostNum, LayoutConvert(layout) );

        // Householder vectors are only first min( mb, nb ) rows in upper
        // triangular part of V. If V0 tile is tall (mb > nb), slice V to
        // first nb rows, and T to nb-by-nb. This can happen when V0 is
        // the right block col.
        // Example: m = 5, n = 3, nb = 5, V0 tile is tall trapezoid (5x3):
        // V0 = [ . . . ]
        //      [   . . ]
        //      [     . ]
        //      [- - - -]
        //      [       ]
        //      [       ]
        if (mb > nb) {
            V = V.slice(0, nb-1, 0, V.n()-1);  // first nb rows
            V0 = V.sub(0, 0, first, first);
            //T0 = T0.slice(0, nb-1, 0, nb-1);  // first nb-by-nb part
            mb = nb;
        }

        // If V0 tile is a wide trapezoid, slice V0 into triangular and
        // rectangular parts, and slice T, C, and Wr correspondingly.
        bool trapezoid = (mb < nb);
        Matrix<scalar_t> V0b, C0b;
        if (trapezoid) {
            int64_t n = C0.n();
            V0b = V0.slice(0, mb-1, mb, nb-1);  // last nb - mb cols
            V0  = V0.slice(0, mb-1, 0,  mb-1);  // first mb cols
            T0  = T0.slice(0, mb-1, 0,  mb-1);  // first mb-by-mb part
            C0b = C0.slice(mb, nb-1, 0, n-1);   // last nb - mb rows
            C0  = C0.slice(0,  mb-1, 0, n-1);   // first mb rows
            Wr  = Wr.slice(0,  mb-1, 0, n-1);   // first mb rows
        }

        // Interpret as triangular matrices.
        auto V0tr = TriangularMatrix<scalar_t>(Uplo::Upper, Diag::Unit, V0);
        auto T0tr = TriangularMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, T0);
        if (op == Op::NoTrans) {
            T0tr = conj_transpose( T0tr );
        }

        // --------------------
        // 1. W = V C

        // W <- C0
        // W <- V0 W
        internal::copy<target>(
                std::move(C0), std::move(Wr),
                priority, queue_index );
        internal::trmm<target>(
                Side::Left,
                one, std::move(V0tr),
                     std::move(Wr),
                priority, queue_index );

        if (trapezoid) {
            // W <- V0b C0b + W
            internal::gemm<target>(
                    one, std::move(V0b),
                         std::move(C0b),
                    one, std::move(Wr),
                    layout,
                    priority, queue_index );
        }

        // W <- V1 C1 + W
        for (int64_t i = 1; i < int64_t(row_indices.size()); ++i) {
            int64_t row = row_indices[i];
            auto Ci = C.sub(row, row, 0, nt-1);
            if (target == Target::Devices) {
                // todo: release the hold later
                Ci.tileGetAndHoldAllOnDevices(LayoutConvert(layout));
            }
            internal::gemm<target>(
                    one, V.sub(0, 0, row, row),
                         std::move(Ci),
                    one, std::move(Wr),
                    layout,
                    priority, queue_index );
        }

        // --------------------
        // 2. W <- op(T0) W; op is already applied to T0tr.
        internal::trmm<target>(
                Side::Left,
                one, std::move(T0tr),
                     std::move(Wr),
                priority, queue_index );

        // --------------------
        // 3. C = C - V^H W
        if (row_indices.size() > 1) {
            // C1 <- C1 - V1^H W
            internal::gemm<target>(
                    -one, conj_transpose( V.sub( 0, 0, row_indices[1], mt-1 ) ),
                          std::move(Wr),
                    one,  C.sub(row_indices[1], mt-1, 0, nt-1),
                    layout,
                    priority, queue_index );
        }

        if (trapezoid) {
            // C0b <- C0b - V0b^H W
            internal::gemm<target>(
                    -one, conj_transpose( std::move( V0b ) ),
                          std::move(Wr),
                    one,  std::move(C0b),
                    layout,
                    priority, queue_index );
        }

        // W <- V0^H W
        internal::trmm<target>(
                Side::Left,
                one, conj_transpose( V0tr ),
                     std::move(Wr),
                priority, queue_index );

        // C0 <- C0 - W
        internal::add<target>(
                -one, std::move(Wr),
                one,  std::move(C0),
                priority, queue_index );

        // free workspace
        // todo: Wr.clear();
        for (int64_t j = 0; j < Wr.nt(); ++j) {
            if (Wr.tileIsLocal(0, j)) {
                Wr.tileErase(0, j, AllDevices);
            }
        }
    }
    else {
        //----------------------------------------
        // Multiply by Q on right:
        // C op(Q) = C (I - V^H op(T) V) = C - C V^H op(T) V
        // in three major steps:
        // 1. W = C V^H
        // 2. W = W op(T)
        // 3. C = C - W V

        assert(V.nt() == nt);

        // Build a list of col indices that have local tiles in matrix C.
        std::vector<int64_t> col_indices;
        for (int64_t j = 0; j < nt; ++j) {
            for (int64_t i = 0; i < mt; ++i) {
                if (C.tileIsLocal(i, j)) {
                    col_indices.push_back(j);
                    break;
                }
            }
        }
        if (col_indices.size() < 1)
            return;

        // This rank's first (left-most) local col of V holds the triangular tile.
        int64_t first = col_indices[0];
        assert(first < nt);
        assert(first >= 0);

        // Get corresponding col of W to match the local matrix distribution.
        auto Wc = W.sub(0, mt-1, first, first);
        Wc.insertLocalTiles(target);

        // V = [ V0  V0b  V1 ]
        // V0  is triangular part (mb-by-mb)
        // V0b is rectangular part, non-empty only if V0 is trapezoid (nb > mb)
        // V1  is remaining tiles
        // Example: m = 3, n = 6, nb = 4, V0 tile is trapezoid (3x4):
        // V = [ . . . | . | . . ]
        //     [   . . | . | . . ]
        //     [     . | . | . . ]
        //       V0     V0b  V1
        auto V0 = V.sub(0, 0, first, first);
        int64_t mb = V0.tileMb(0);
        int64_t nb = V0.tileNb(0);

        auto T0 = T.sub(0, 0, first, first);
        // Use first mb-by-mb or nb-by-nb portion.
        T0 = T0.slice(0, std::min(mb, nb)-1, 0, std::min(mb, nb)-1);

        // C = [ C0  C0b  C1 ]
        //       C0b is non-empty only if V0 is trapezoid
        auto C0 = C.sub(0, mt-1, first, first);
        // todo: issue omp tasks for copy to host
        C0.tileGetAllForWriting( HostNum, LayoutConvert(layout) );

        // Householder vectors are only first min( mb, nb ) rows in upper
        // triangular part of V. If V0 tile is tall (mb > nb), slice V to
        // first nb rows, and T to nb-by-nb. This can happen when V0 is
        // the right block col.
        // Example: m = 5, n = 3, nb = 5, V0 tile is tall trapezoid (5x3):
        // V0 = [ . . . ]
        //      [   . . ]
        //      [     . ]
        //      [- - - -]
        //      [       ]
        //      [       ]
        if (mb > nb) {
            V = V.slice(0, nb-1, 0, V.n()-1);  // first nb rows
            V0 = V.sub(0, 0, first, first);
            //T0 = T0.slice(0, nb-1, 0, nb-1);  // first nb-by-nb part
            mb = nb;
        }

        // If V0 tile is a wide trapezoid, slice V0 into triangular and
        // rectangular parts, and slice T, C, and Wc correspondingly.
        bool trapezoid = (mb < nb);
        Matrix<scalar_t> V0b, C0b;
        if (trapezoid) {
            int64_t m = C0.m();
            V0b = V0.slice(0, mb-1, mb, nb-1);  // last nb - mb cols
            V0  = V0.slice(0, mb-1, 0,  mb-1);  // first mb cols
            T0  = T0.slice(0, mb-1, 0,  mb-1);  // first mb-by-mb part
            C0b = C0.slice(0, m-1,  mb, nb-1);  // last nb - mb cols
            C0  = C0.slice(0, m-1,  0,  mb-1);  // first mb cols
            Wc  = Wc.slice(0, m-1,  0,  mb-1);  // first mb cols
        }

        // Interpret as triangular matrices.
        auto V0tr = TriangularMatrix<scalar_t>(Uplo::Upper, Diag::Unit, V0);
        auto T0tr = TriangularMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, T0);
        if (op == Op::NoTrans) {
            T0tr = conj_transpose( T0tr );
        }

        // --------------------
        // 1. W = C V^H

        // W <- C0
        // W <- W V0^H
        internal::copy<target>(
                std::move(C0), std::move(Wc),
                priority, queue_index );
        internal::trmm<target>(
                Side::Right,
                one, conj_transpose( V0tr ),
                     std::move(Wc),
                priority, queue_index );

        if (trapezoid) {
            // W <- C0b V0b^H + W
            internal::gemm<target>(
                    one, std::move(C0b),
                         conj_transpose( V0b ),
                    one, std::move(Wc),
                    layout,
                    priority, queue_index );
        }

        // W <- C1 V1^H + W
        for (int64_t i = 1; i < int64_t(col_indices.size()); ++i) {
            int64_t col = col_indices[i];
            auto Ci = C.sub(0, mt-1, col, col);
            if (target == Target::Devices) {
                // todo: release the hold later
                Ci.tileGetAndHoldAllOnDevices(LayoutConvert(layout));
            }
            internal::gemm<target>(
                    one, std::move(Ci),
                         conj_transpose( V.sub( 0, 0, col, col ) ),
                    one, std::move(Wc),
                    layout,
                    priority, queue_index );
        }

        // --------------------
        // 2. W <- W op(T0); op is already applied to T0tr.
        internal::trmm<target>(
                Side::Right,
                one, std::move(T0tr),
                     std::move(Wc),
                priority, queue_index );

        // --------------------
        // 3. C = C - W V
        if (col_indices.size() > 1) {
            // C1 <- C1 - W V1
            internal::gemm<target>(
                    -one, std::move(Wc),
                          V.sub(0, 0,    col_indices[1], nt-1),
                    one,  C.sub(0, mt-1, col_indices[1], nt-1),
                    layout,
                    priority, queue_index );
        }

        if (trapezoid) {
            // C0b <- C0b - W V0b
            internal::gemm<target>(
                    -one, std::move(Wc),
                          std::move(V0b),
                    one,  std::move(C0b),
                    layout,
                    priority, queue_index );
        }

        // W <- W V0
        internal::trmm<target>(
                Side::Right,
                one, std::move(V0tr),
                     std::move(Wc),
                priority, queue_index );

        // C0 <- C0 - W
        internal::add<target>(
                -one, std::move(Wc),
                one,  std::move(C0),
                priority, queue_index );

        // free workspace
        // todo: Wc.clear();
        for (int64_t i = 0; i < Wc.mt(); ++i) {
            if (Wc.tileIsLocal(i, 0)) {
                Wc.tileErase(i, 0, AllDevices);
            }
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void unmlq<Target::HostTask, float>(
    Side side, Op op,
    Matrix<float>&& V,
    Matrix<float>&& T,
    Matrix<float>&& C,
    Matrix<float>&& W,
    int priority, int64_t queue_index );

template
void unmlq<Target::HostNest, float>(
    Side side, Op op,
    Matrix<float>&& V,
    Matrix<float>&& T,
    Matrix<float>&& C,
    Matrix<float>&& W,
    int priority, int64_t queue_index );

template
void unmlq<Target::HostBatch, float>(
    Side side, Op op,
    Matrix<float>&& V,
    Matrix<float>&& T,
    Matrix<float>&& C,
    Matrix<float>&& W,
    int priority, int64_t queue_index );

template
void unmlq<Target::Devices, float>(
    Side side, Op op,
    Matrix<float>&& V,
    Matrix<float>&& T,
    Matrix<float>&& C,
    Matrix<float>&& W,
    int priority, int64_t queue_index );

// ----------------------------------------
template
void unmlq<Target::HostTask, double>(
    Side side, Op op,
    Matrix<double>&& V,
    Matrix<double>&& T,
    Matrix<double>&& C,
    Matrix<double>&& W,
    int priority, int64_t queue_index );

template
void unmlq<Target::HostNest, double>(
    Side side, Op op,
    Matrix<double>&& V,
    Matrix<double>&& T,
    Matrix<double>&& C,
    Matrix<double>&& W,
    int priority, int64_t queue_index );

template
void unmlq<Target::HostBatch, double>(
    Side side, Op op,
    Matrix<double>&& V,
    Matrix<double>&& T,
    Matrix<double>&& C,
    Matrix<double>&& W,
    int priority, int64_t queue_index );

template
void unmlq<Target::Devices, double>(
    Side side, Op op,
    Matrix<double>&& V,
    Matrix<double>&& T,
    Matrix<double>&& C,
    Matrix<double>&& W,
    int priority, int64_t queue_index );

// ----------------------------------------
template
void unmlq< Target::HostTask, std::complex<float> >(
    Side side, Op op,
    Matrix< std::complex<float> >&& V,
    Matrix< std::complex<float> >&& T,
    Matrix< std::complex<float> >&& C,
    Matrix< std::complex<float> >&& W,
    int priority, int64_t queue_index );

template
void unmlq< Target::HostNest, std::complex<float> >(
    Side side, Op op,
    Matrix< std::complex<float> >&& V,
    Matrix< std::complex<float> >&& T,
    Matrix< std::complex<float> >&& C,
    Matrix< std::complex<float> >&& W,
    int priority, int64_t queue_index );

template
void unmlq< Target::HostBatch, std::complex<float> >(
    Side side, Op op,
    Matrix< std::complex<float> >&& V,
    Matrix< std::complex<float> >&& T,
    Matrix< std::complex<float> >&& C,
    Matrix< std::complex<float> >&& W,
    int priority, int64_t queue_index );

template
void unmlq< Target::Devices, std::complex<float> >(
    Side side, Op op,
    Matrix< std::complex<float> >&& V,
    Matrix< std::complex<float> >&& T,
    Matrix< std::complex<float> >&& C,
    Matrix< std::complex<float> >&& W,
    int priority, int64_t queue_index );

// ----------------------------------------
template
void unmlq< Target::HostTask, std::complex<double> >(
    Side side, Op op,
    Matrix< std::complex<double> >&& V,
    Matrix< std::complex<double> >&& T,
    Matrix< std::complex<double> >&& C,
    Matrix< std::complex<double> >&& W,
    int priority, int64_t queue_index );

template
void unmlq< Target::HostNest, std::complex<double> >(
    Side side, Op op,
    Matrix< std::complex<double> >&& V,
    Matrix< std::complex<double> >&& T,
    Matrix< std::complex<double> >&& C,
    Matrix< std::complex<double> >&& W,
    int priority, int64_t queue_index );

template
void unmlq< Target::HostBatch, std::complex<double> >(
    Side side, Op op,
    Matrix< std::complex<double> >&& V,
    Matrix< std::complex<double> >&& T,
    Matrix< std::complex<double> >&& C,
    Matrix< std::complex<double> >&& W,
    int priority, int64_t queue_index );

template
void unmlq< Target::Devices, std::complex<double> >(
    Side side, Op op,
    Matrix< std::complex<double> >&& V,
    Matrix< std::complex<double> >&& T,
    Matrix< std::complex<double> >&& C,
    Matrix< std::complex<double> >&& W,
    int priority, int64_t queue_index );

} // namespace internal
} // namespace slate
