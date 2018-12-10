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

#include "slate_Matrix.hh"
#include "slate_types.hh"
#include "slate_Tile_tpmqrt.hh"
#include "slate_internal.hh"
#include "slate_internal_util.hh"

namespace slate {
namespace internal {

///-----------------------------------------------------------------------------
/// Distributed QR triangle-triangle factorization of column of tiles.
/// Each rank has one triangular tile, the result of local geqrf panel.
/// Dispatches to target implementations.
template <Target target, typename scalar_t>
void ttmqr(Side side, Op op,
           Matrix<scalar_t>&& A,
           Matrix<scalar_t>&& T,
           Matrix<scalar_t>&& C)
{
    ttmqr(internal::TargetType<target>(),
          side, op, A, T, C);
}

///-----------------------------------------------------------------------------
/// Distributed QR triangle-triangle factorization, host implementation.
template <typename scalar_t>
void ttmqr(internal::TargetType<Target::HostTask>,
           Side side, Op op,
           Matrix<scalar_t>& A,
           Matrix<scalar_t>& T,
           Matrix<scalar_t>& C)
{
    int64_t A_mt = A.mt();

    // Find ranks in this column.
    std::set<int> ranks_set;
    A.sub(0, A_mt-1, 0, 0).getRanks(&ranks_set);

    // Find each rank's top-most row in this column,
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
    std::sort(rank_rows.begin(), rank_rows.end(), compare_rank_rows);

    int nranks = rank_rows.size();
    int nlevels = int( ceil( log2( nranks ) ) );

    // Apply reduction tree from leaves down (NoTrans) or from root up (Trans).
    // if NoTrans, levels go from nlevels-1 down to 0 (inclusive)
    // if Trans,   levels go from 0 up to nlevels-1 (inclusive)
    int level_begin, level_end, level_step, step;
    if (op == Op::NoTrans) {
        level_begin = nlevels - 1;
        level_end   = -1;
        level_step  = -1;
        step        = pow(2, nlevels - 1);
    }
    else {
        level_begin = 0;
        level_end   = nlevels;
        level_step  = 1;
        step        = 1;
    }
    for (int level = level_begin; level != level_end; level += level_step) {
        for (int index = 0; index < nranks; index += step) {
            int64_t i = rank_rows[ index ].second;
            // Send V and T across row of C.
            // if (index % (2*step) != 0) {
            //     A.tileBcast(i, 0, C.sub(i, i, 0, C.nt()-1));
            //     T.tileBcast(i, 0, C.sub(i, i, 0, C.nt()-1));
            // }
            // At each level, scan rows of C for local tiles.
            // TODO: the j loop can be parallelized, but care needs to be
            // taken so that MPI makes progress.
            for (int64_t j = 0; j < C.nt(); ++j) {
                if (C.tileIsLocal(i, j)) {
                    if (index % (2*step) == 0) {
                        if (index + step < nranks) {
                            // Send tile to dst, then receive updated tile back.
                            int64_t i_dst = rank_rows[ index + step ].second;
                            int dst = C.tileRank(i_dst, j);
                            C.tileSend(i, j, dst);
                            C.tileRecv(i, j, dst);
                        }
                    }
                    else {
                        // Receive tile from src.
                        int64_t i_src = rank_rows[ index - step ].second;
                        int     src   = C.tileRank(i_src, j);
                        C.tileRecv(i_src, j, src);

                        // Apply Q
                        tpmqrt(side, op, A.tileNb(0), A(i, 0),
                               T(i, 0),
                               C(i_src, j), C(i, j));

                        // Send updated tile back.
                        C.tileSend(i_src, j, src);
                        C.tileTick(i_src, j);
                    }
                }
            }
        }
        if (op == Op::NoTrans)
            step /= 2;
        else
            step *= 2;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void ttmqr<Target::HostTask, float>(
    Side side, Op op,
    Matrix<float>&& A,
    Matrix<float>&& T,
    Matrix<float>&& C);

// ----------------------------------------
template
void ttmqr<Target::HostTask, double>(
    Side side, Op op,
    Matrix<double>&& A,
    Matrix<double>&& T,
    Matrix<double>&& C);

// ----------------------------------------
template
void ttmqr< Target::HostTask, std::complex<float> >(
    Side side, Op op,
    Matrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& T,
    Matrix< std::complex<float> >&& C);

// ----------------------------------------
template
void ttmqr< Target::HostTask, std::complex<double> >(
    Side side, Op op,
    Matrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& T,
    Matrix< std::complex<double> >&& C);

} // namespace internal
} // namespace slate
