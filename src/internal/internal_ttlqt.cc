// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/types.hh"
#include "internal/Tile_tplqt.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Distributed LQ triangle-triangle factorization of row of tiles.
/// Each rank has one triangular tile, the result of local gelqf panel.
/// Dispatches to target implementations.
/// @ingroup gelqf_internal
///
template <Target target, typename scalar_t>
void ttlqt(Matrix<scalar_t>&& A,
           Matrix<scalar_t>&& T)
{
    ttlqt(internal::TargetType<target>(),
          A, T);
}

//------------------------------------------------------------------------------
/// Distributed LQ triangle-triangle factorization, host implementation.
/// Assumes panel tiles reside on host.
/// @ingroup gelqf_internal
///
template <typename scalar_t>
void ttlqt(internal::TargetType<Target::HostTask>,
           Matrix<scalar_t>& A,
           Matrix<scalar_t>& T)
{
    // Assumes column major
    const Layout layout = Layout::ColMajor;

    int64_t A_nt = A.nt();

    // Find ranks in this row.
    std::set<int> ranks_set;
    A.sub(0, 0, 0, A_nt-1).getRanks(&ranks_set);

    // Find each rank's first (left-most) col in this row,
    // which is the triangular tile resulting from local gelqf panel.
    std::vector< std::pair<int, int64_t> > rank_cols;
    rank_cols.reserve(ranks_set.size());
    for (int r: ranks_set) {
        for (int64_t j = 0; j < A_nt; ++j) {
            if (A.tileRank(0, j) == r) {
                rank_cols.push_back({r, j});
                break;
            }
        }
    }
    // Sort rank_cols by col.
    std::sort(rank_cols.begin(), rank_cols.end(), compareSecond<int, int64_t>);

    // Find this rank.
    int index;
    for (index = 0; index < int(rank_cols.size()); ++index) {
        if (rank_cols[index].first == A.mpiRank())
            break;
    }

    if (index < int(rank_cols.size())) {
        // This rank has a tile in this row, at col j.
        int64_t j = rank_cols[index].second;
        int nranks = rank_cols.size();
        int nlevels = int( ceil( log2( nranks ) ) );

        // Example: 2D cyclic, p = 7, q = 1, row k = 9
        //                                           Levels
        //               { rank, col }        index  L=0  L=1  L=2
        // rank_cols = [ {    2,   9 },    // 0      src  src  src
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
        // Two triangular tiles are factored with tplqt on dst rank,
        // with the resulting triangular tile sent back to src rank.
        // Each rank occurs once as dst, except for index 0
        // (here, rank_col {2, 9}), which is always src, never dst.
        // For each pair, the Householder vectors V overwrite the right tile,
        // A(0, j) on dst. The T matrix is also stored on dst.
        int step = 1;
        for (int level = 0; level < nlevels; ++level) {
            if (index % (2*step) == 0) {
                if (index + step < nranks) {
                    // Send tile to dst, then receive updated tile back.
                    int dst = rank_cols[ index + step ].first;
                    A.tileSend(0, j, dst);
                    A.tileRecv(0, j, dst, layout);
                }
            }
            else {
                // Receive tile from src.
                int     src   = rank_cols[ index - step ].first;
                int64_t j_src = rank_cols[ index - step ].second;
                A.tileRecv(0, j_src, src, layout);

                A.tileGetForWriting(0, j, LayoutConvert(layout));

                // Factor tiles, which eliminates local tile A(j, 0).
                T.tileInsert(0, j);
                T( 0, j ).set( 0 );
                int64_t l = std::min(A.tileMb(0), A.tileNb(j));
                tplqt(l, A(0, j_src), A(0, j), T(0, j));

                T.tileModified(0, j);

                // Send updated tile back. This rank is done!
                A.tileSend(0, j_src, src);
                A.tileTick(0, j_src);
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
void ttlqt<Target::HostTask, float>(
    Matrix<float>&& A,
    Matrix<float>&& T);

// ----------------------------------------
template
void ttlqt<Target::HostTask, double>(
    Matrix<double>&& A,
    Matrix<double>&& T);

// ----------------------------------------
template
void ttlqt< Target::HostTask, std::complex<float> >(
    Matrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& T);

// ----------------------------------------
template
void ttlqt< Target::HostTask, std::complex<double> >(
    Matrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& T);

} // namespace internal
} // namespace slate
