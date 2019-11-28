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

    int64_t k_end;
    int64_t i, j, i1, j1, i_dst, j_dst;

    if (side == Side::Left) {
        k_end = C.nt();
    }
    else {
        k_end = C.mt();
    }

    for (int level = 0; level < nlevels; ++level) {
        for (int index = 0; index < nranks; index += step) {
            int64_t rank_ind = rank_indices[ index ].second;
            // if (side == left), scan rows of C for local tiles;
            // if (side == right), scan cols of C for local tiles
            // Three for-loops: 1) send, receive 2) update 3) receive, send
            for (int64_t k = 0; k < k_end; ++k) {
                if (side == Side::Left) {
                    i = rank_ind;
                    j = k;
                }
                else {
                    i = k;
                    j = rank_ind;
                }
                if (C.tileIsLocal(i, j)) {
                    if (index % (2*step) == 0) {
                        if (index + step < nranks) {
                            // Send tile to dst.
                            int64_t k_dst = rank_indices[ index + step ].second;
                            if (side == Side::Left) {
                                i_dst = k_dst;
                                j_dst = k;
                            }
                            else {
                                i_dst = k;
                                j_dst = k_dst;
                            }
                            int dst = C.tileRank(i_dst, j_dst);
                            C.tileSend(i, j, dst, tag);
                        }
                    }
                    else {
                        // Receive tile from src.
                        int64_t k_src = rank_indices[ index - step ].second;
                        if (side == Side::Left) {
                            i1 = k_src;
                            j1 = k;
                        }
                        else {
                            i1 = k;
                            j1 = k_src;
                        }

                        int     src   = C.tileRank(i1, j1);
                        C.tileRecv(i1, j1, src, layout, tag);
                    }
                }
            }

            for (int64_t k = 0; k < k_end; ++k) {
                if (side == Side::Left) {
                    i = rank_ind;
                    j = k;
                }
                else {
                    i = k;
                    j = rank_ind;
                }
                if (C.tileIsLocal(i, j)) {
                    if (!(index % (2*step) == 0)) {
                        int64_t k_src = rank_indices[ index - step ].second;
                        if (side == Side::Left) {
                            i1 = k_src;
                            j1 = k;
                        }
                        else {
                            i1 = k;
                            j1 = k_src;
                        }

                        #pragma omp task shared(A, T, C)
                        {
                        A.tileGetForReading(0, rank_ind, LayoutConvert(layout));
                        T.tileGetForReading(0, rank_ind, LayoutConvert(layout));
                        C.tileGetForWriting(i, j, LayoutConvert(layout));

                        // Apply Q.
                        tpmlqt(side, op, std::min(A.tileMb(0), A.tileNb(rank_ind)),
                               A(0, rank_ind), T(0, rank_ind),
                               C(i1, j1), C(i, j));

                        // todo: should tileRelease()?
                        A.tileTick(0, rank_ind);
                        T.tileTick(0, rank_ind);
                        }
                    }
                }
            }
            #pragma omp taskwait

            for (int64_t k = 0; k < k_end; ++k) {
                if (side == Side::Left) {
                    i = rank_ind;
                    j = k;
                }
                else {
                    i = k;
                    j = rank_ind;
                }
                if (C.tileIsLocal(i, j)) {
                    if (index % (2*step) == 0) {
                        if (index + step < nranks) {
                            // Receive updated tile back.
                            int64_t k_dst = rank_indices[ index + step ].second;
                            if (side == Side::Left) {
                                i_dst = k_dst;
                                j_dst = k;
                            }
                            else {
                                i_dst = k;
                                j_dst = k_dst;
                            }
                            int dst = C.tileRank(i_dst, j_dst);
                            C.tileRecv(i, j, dst, layout, tag);
                        }
                    }
                    else {
                        int64_t k_src = rank_indices[ index - step ].second;
                        if (side == Side::Left) {
                            i1 = k_src;
                            j1 = k;
                        }
                        else {
                            i1 = k;
                            j1 = k_src;
                        }
                        int     src   = C.tileRank(i1, j1);
                        // Send updated tile back.
                        C.tileSend(i1, j1, src, tag);
                        C.tileTick(i1, j1);
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
