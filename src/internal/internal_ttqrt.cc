// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/types.hh"
#include "internal/Tile_tpqrt.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Distributed QR triangle-triangle factorization of column of tiles.
/// Each rank has one triangular tile, the result of local geqrf panel.
/// Dispatches to target implementations.
/// @ingroup geqrf_internal
///
template <Target target, typename scalar_t>
void ttqrt(Matrix<scalar_t>&& A,
           Matrix<scalar_t>&& T)
{
    ttqrt(internal::TargetType<target>(),
          A, T);
}

//------------------------------------------------------------------------------
/// Distributed QR triangle-triangle factorization, host implementation.
/// Assumes panel tiles reside on host.
/// @ingroup geqrf_internal
///
template <typename scalar_t>
void ttqrt(internal::TargetType<Target::HostTask>,
           Matrix<scalar_t>& A,
           Matrix<scalar_t>& T)
{
    // Assumes column major
    const Layout layout = Layout::ColMajor;

    int64_t A_mt = A.mt();

    // Find ranks in this column.
    std::set<int> ranks_set;
    A.sub(0, A_mt-1, 0, 0).getRanks(&ranks_set);

    // Find each rank's first (top-most) row in this column,
    // which is the triangular tile resulting from local geqrf panel.
    std::vector< std::pair<int, int64_t> > rank_rows;
    rank_rows.reserve(ranks_set.size());
    for (int r: ranks_set) {
        for (int64_t i = 0; i < A_mt; ++i) {
            if (A.tileRank(i, 0) == r) {
                rank_rows.push_back({r, i});
                break;
            }
        }
    }
    // Sort rank_rows by row.
    std::sort(rank_rows.begin(), rank_rows.end(), compareSecond<int, int64_t>);

    // Find this rank.
    int index;
    for (index = 0; index < int(rank_rows.size()); ++index) {
        if (rank_rows[index].first == A.mpiRank())
            break;
    }

    if (index < int(rank_rows.size())) {
        // This rank has a tile in this column, at row i.
        int64_t i = rank_rows[index].second;
        int nranks = rank_rows.size();
        int nlevels = int( ceil( log2( nranks ) ) );

        // Example: 2D cyclic, p = 7, q = 1, column k = 9
        //                                           Levels
        //               { rank, row }        index  L=0  L=1  L=2
        // rank_rows = [ {    2,   9 },    // 0      src  src  src
        //               {    3,  10 },    // 1      dst   |    |
        //                                 //              |    |
        //               {    4,  11 },    // 2      src  dst   |
        //               {    5,  12 },    // 3      dst        |
        //                                 //                   |
        //               {    6,  13 },    // 4      src  src  dst
        //               {    0,  14 },    // 5      dst   |
        //                                 //              |
        //               {    1,  15 } ];  // 6       x   dst
        // src-dst pairs indicate tiles that are factored together.
        //
        // Two triangular tiles are factored with tpqrt on dst rank,
        // with the resulting triangular tile sent back to src rank.
        // Each rank occurs once as dst, except for index 0
        // (here, rank_row {2, 9}), which is always src, never dst.
        // For each pair, the Householder vectors V overwrite the bottom tile,
        // A(i, 0) on dst. The T matrix is also stored on dst.
        int step = 1;
        for (int level = 0; level < nlevels; ++level) {
            if (index % (2*step) == 0) {
                if (index + step < nranks) {
                    // Send tile to dst, then receive updated tile back.
                    int dst = rank_rows[ index + step ].first;
                    A.tileSend(i, 0, dst);
                    A.tileRecv(i, 0, dst, layout);
                }
            }
            else {
                // Receive tile from src.
                int     src   = rank_rows[ index - step ].first;
                int64_t i_src = rank_rows[ index - step ].second;
                A.tileRecv(i_src, 0, src, layout);

                A.tileGetForWriting(i, 0, LayoutConvert(layout));

                // Factor tiles, which eliminates local tile A(i, 0).
                T.tileInsert(i, 0);
                T(i, 0).set(0);
                int64_t l = std::min(A.tileMb(i), A.tileNb(0));
                tpqrt(l, A(i_src, 0), A(i, 0), T(i, 0));

                T.tileModified(i, 0);

                // Send updated tile back. This rank is done!
                A.tileSend(i_src, 0, src);
                A.tileTick(i_src, 0);
                break;
            }
            step *= 2;
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void ttqrt<Target::HostTask, float>(
    Matrix<float>&& A,
    Matrix<float>&& T);

// ----------------------------------------
template
void ttqrt<Target::HostTask, double>(
    Matrix<double>&& A,
    Matrix<double>&& T);

// ----------------------------------------
template
void ttqrt< Target::HostTask, std::complex<float> >(
    Matrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& T);

// ----------------------------------------
template
void ttqrt< Target::HostTask, std::complex<double> >(
    Matrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& T);

} // namespace internal
} // namespace slate
