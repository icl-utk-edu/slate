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
#include "internal/Tile_tpmlqt.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Distributed multiply matrix by Q from LQ triangle-triangle factorization of
/// row of tiles.
/// Dispatches to target implementations.
/// todo: This assumes A and T have already been communicated as needed.
/// However, it necesarily handles communication for C.
/// Tag is used in gelqf to differentiate communication for look-ahead panel
/// from rest of trailing matrix.
/// @ingroup gelqf_internal
///
template <Target target, typename scalar_t>
void ttmlq(Side side, Op op,
           Matrix<scalar_t>&& A,
           Matrix<scalar_t>&& T,
           Matrix<scalar_t>&& C,
           int tag)
{
    ttmlq(internal::TargetType<target>(),
          side, op, A, T, C, tag);
}

//------------------------------------------------------------------------------
/// Distributed multiply matrix by Q from LQ triangle-triangle factorization of
/// row of tiles, host implementation.
/// @ingroup gelqf_internal
///
template <typename scalar_t>
void ttmlq(internal::TargetType<Target::HostTask>,
           Side side, Op op,
           Matrix<scalar_t>& A,
           Matrix<scalar_t>& T,
           Matrix<scalar_t>& C,
           int tag)
{
    // Assumes column major
    const Layout layout = Layout::ColMajor;

    int64_t A_nt = A.nt();
    assert(A.mt() == 1);
    if (side == Side::Left)
        assert(A_nt == C.mt());
    else
        assert(A_nt == C.nt());

    // Find ranks in this row of A.
    std::set<int> ranks_set;
    A.getRanks(&ranks_set);

    // Find each rank's first (left-most) col in this row of A,
    // which is the triangular tile resulting from local gelqf panel.
    std::vector< std::pair<int, int64_t> > rank_indices;
    rank_indices.reserve(ranks_set.size());
    for (int r: ranks_set) {
        for (int64_t j = 0; j < A_nt; ++j) {
            if (A.tileRank(0, j) == r) {
                rank_indices.push_back({r, j});
                break;
            }
        }
    }
    // Sort rank_indices by index.
    std::sort(rank_indices.begin(), rank_indices.end(),
              compareSecond<int, int64_t>);

    int nranks = rank_indices.size();
    int nlevels = int( ceil( log2( nranks ) ) );

    // Apply reduction tree.
    // If Left, NoTrans or Right, Trans, apply descending from root to leaves,
    // i.e., in reverse order of how they were created.
    // If Left, Trans or Right, NoTrans, apply ascending from leaves to root,
    // i.e., in same order as they were created.
    // Example for A.mt == 8.
    // Leaves:
    //     ttqrt( a0, a1 )
    //     ttqrt( a2, a3 )
    //     ttqrt( a4, a5 )
    //     ttqrt( a6, a7 )
    // Next level:
    //     ttqrt( a0, a2 )
    //     ttqrt( a4, a6 )
    // Root:
    //     ttqrt( a0, a4 )
    bool descend = (side == Side::Left) == (op == Op::NoTrans);
    int step;
    if (descend)
        step = pow(2, nlevels - 1);
    else
        step = 1;

    for (int level = 0; level < nlevels; ++level) {
        for (int index = 0; index < nranks; index += step) {
            if (side == Side::Left) {
                int64_t i = rank_indices[ index ].second;
                // At each level, scan rows of C for local tiles.
                // TODO: the j loop can be parallelized, but care needs to be
                // taken so that MPI makes progress.
                // todo: for better performance, split into three tasks:
                //      - send-receive task,
                //      - update task-loop,
                //      - send-receive task
                for (int64_t j = 0; j < C.nt(); ++j) {
                    if (C.tileIsLocal(i, j)) {
                        if (index % (2*step) == 0) {
                            if (index + step < nranks) {
                                // Send tile to dst, then receive updated tile back.
                                int64_t i_dst = rank_indices[ index + step ].second;
                                int dst = C.tileRank(i_dst, j);
                                C.tileSend(i, j, dst, tag);
                                C.tileRecv(i, j, dst, layout, tag);
                            }
                        }
                        else {
                            // Receive tile from src.
                            int64_t i_src = rank_indices[ index - step ].second;
                            int     src   = C.tileRank(i_src, j);
                            C.tileRecv(i_src, j, src, layout, tag);

                            A.tileGetForReading(0, i, LayoutConvert(layout));
                            T.tileGetForReading(0, i, LayoutConvert(layout));
                            C.tileGetForWriting(i, j, LayoutConvert(layout));

                            // Apply Q
                            // todo: Handle wide A0j as in ttmqr.
                            tpmlqt(side, op, std::min( A.tileMb(0), A.tileNb(i) ),
                                   A(0, i), T(0, i),
                                   C(i_src, j), C(i, j));

                            // todo: should tileRelease()?
                            A.tileTick(0, i);
                            T.tileTick(0, i);
                            // Send updated tile back.
                            C.tileSend(i_src, j, src, tag);
                            C.tileTick(i_src, j);
                        }
                    }
                }
            }
            else { // Right
                int64_t j = rank_indices[ index ].second;
                // At each level, scan cols of C for local tiles.
                // TODO: the i loop can be parallelized
                for (int64_t i = 0; i < C.mt(); ++i) {
                    if (C.tileIsLocal(i, j)) {
                        if (index % (2*step) == 0) {
                            if (index + step < nranks) {
                                // Send tile to dst, then receive updated tile back.
                                int64_t j_dst = rank_indices[ index + step ].second;
                                int dst = C.tileRank(i, j_dst);
                                C.tileSend(i, j, dst, tag);
                                C.tileRecv(i, j, dst, layout, tag);
                            }
                        }
                        else {
                            // Receive tile from src.
                            int64_t j_src = rank_indices[ index - step ].second;
                            int     src   = C.tileRank(i, j_src);
                            C.tileRecv(i, j_src, src, layout, tag);

                            A.tileGetForReading(0, j, LayoutConvert(layout));
                            T.tileGetForReading(0, j, LayoutConvert(layout));
                            C.tileGetForWriting(i, j, LayoutConvert(layout));

                            // Apply Q.
                            // todo: Handle wide A0j as in ttmqr.
                            tpmlqt(side, op, std::min( A.tileMb(0), A.tileNb(j) ),
                                   A(0, j), T(0, j),
                                   C(i, j_src), C(i, j));

                            // todo: should tileRelease()?
                            A.tileTick(0, j);
                            T.tileTick(0, j);
                            // Send updated tile back.
                            C.tileSend(i, j_src, src, tag);
                            C.tileTick(i, j_src);
                        }
                    }
                }
            }
        }
        if (descend)
            step /= 2;
        else
            step *= 2;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void ttmlq<Target::HostTask, float>(
    Side side, Op op,
    Matrix<float>&& A,
    Matrix<float>&& T,
    Matrix<float>&& C,
    int tag);

// ----------------------------------------
template
void ttmlq<Target::HostTask, double>(
    Side side, Op op,
    Matrix<double>&& A,
    Matrix<double>&& T,
    Matrix<double>&& C,
    int tag);

// ----------------------------------------
template
void ttmlq< Target::HostTask, std::complex<float> >(
    Side side, Op op,
    Matrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& T,
    Matrix< std::complex<float> >&& C,
    int tag);

// ----------------------------------------
template
void ttmlq< Target::HostTask, std::complex<double> >(
    Side side, Op op,
    Matrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& T,
    Matrix< std::complex<double> >&& C,
    int tag);

} // namespace internal
} // namespace slate
