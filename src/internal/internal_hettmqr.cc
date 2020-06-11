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

#include <array>

namespace slate {

//------------------------------------------------------------------------------
template <typename scalar_t>
void makeHermitian(Tile<scalar_t>&& T)
{
    using blas::conj;
    assert(T.mb() == T.nb());
    int64_t nb = T.nb();
    if (T.uplo() == Uplo::Lower) {
        for (int64_t j = 0; j < nb; ++j)
            for (int64_t i = j+1; i < nb; ++i) // lower
                T.at(j, i) = conj(T(i, j));
    }
    else { // upper
        for (int64_t j = 0; j < nb; ++j)
            for (int64_t i = j+1; i < nb; ++i) // lower
                T.at(i, j) = conj(T(j, i));
    }
}

namespace internal {

//------------------------------------------------------------------------------
/// Distributed multiply Hermitian matrix on left and right by Q from
/// QR triangle-triangle factorization of column of tiles.
/// Dispatches to target implementations.
/// todo: This assumes A and T have already been communicated as needed.
/// However, it necesarily handles communication for C.
/// Tag is used in geqrf to differentiate communication for look-ahead panel
/// from rest of trailing matrix.
/// @ingroup geqrf_internal
///
template <Target target, typename scalar_t>
void hettmqr(Op op,
             Matrix<scalar_t>&& V,
             Matrix<scalar_t>&& T,
             HermitianMatrix<scalar_t>&& C,
             int tag)
{
    hettmqr(internal::TargetType<target>(),
            op, V, T, C, tag);
}

//------------------------------------------------------------------------------
/// Distributed multiply Hermitian matrix on left and right by Q from
/// QR triangle-triangle factorization of column of tiles.
/// Host implementation.
/// @ingroup geqrf_internal
///
template <typename scalar_t>
void hettmqr(internal::TargetType<Target::HostTask>,
             Op op,
             Matrix<scalar_t>& V,
             Matrix<scalar_t>& T,
             HermitianMatrix<scalar_t>& C,
             int tag)
{
    using blas::conj;

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    int64_t V_mt = V.mt();
    assert(V.nt() == 1);
    assert(V_mt == C.mt());

    // Find ranks in this column of V.
    std::set<int> ranks_set;
    V.getRanks(&ranks_set);

    // Find each rank's first (top-most) row in this column of V,
    // which is the triangular tile resulting from local geqrf panel.
    std::vector< std::pair<int, int64_t> > rank_indices;
    rank_indices.reserve(ranks_set.size());
    for (int r: ranks_set) {
        for (int64_t i = 0; i < V_mt; ++i) {
            if (V.tileRank(i, 0) == r) {
                rank_indices.push_back({r, i});
                break;
            }
        }
    }
    // Sort rank_indices by index.
    std::sort(rank_indices.begin(), rank_indices.end(),
              compareSecond<int, int64_t>);

    int nranks = rank_indices.size();
    int nlevels = int( ceil( log2( nranks ) ) );

    // Applies op(Q) on left. Apply opposite operation on right, opR(Q).
    Op opR = (op == Op::NoTrans ? Op::ConjTrans : Op::NoTrans);

    // Apply reduction tree.
    // If NoTrans, multiply C = Q C Q^H, apply descending from root to leaves,
    // i.e., in reverse order of how they were created.
    // If ConjTrans, multiply C = Q^H C Q, apply ascending from leaves to root,
    // i.e., in same order as they were created.
    bool descend = (op == Op::NoTrans);
    int step;
    if (descend)
        step = pow(2, nlevels - 1);
    else
        step = 1;

    // Example with i1 = 2, i2 = 5 (which doesn't actually occur):
    //       [ .                             ]
    //       [ .   .                         ]
    // i1 => [ 2   2   1  2^H 2^H 1^H        ]
    //       [ .   .  (2)  .                 ]
    //       [ .   .  (2)  .   .             ]
    // i2 => [ 2   2  (1)  2   2   1         ]
    //       [ .   .   3   .   .   3   .     ]
    //       [ .   .   3   .   .   3   .   . ]
    // where numbers 1, 2, 3 indicate what step below updates tiles,
    // ^H are temporary conj-transposed copies, and
    // (*) shows where conj-transposed tiles come from.

    for (int level = 0; level < nlevels; ++level) {
        // index is first node of each pair.
        for (int index = 0; index + step < nranks; index += 2*step) {
            int64_t i1 = rank_indices[ index ].second;
            int64_t i2 = rank_indices[ index + step ].second;

            //--------------------
            // 1: Multiply Q^H * [ C(i1, i1)  C(i2, i1)^H ] * Q
            //                   [ C(i2, i1)  C(i2, i2)   ]
            // where C(i1, i1) and C(i2, i2) are Hermitian.
            // Send C(i1, i1) and C(i2, i2) to dst,
            // then receive updated tiles back.
            std::array<int64_t, 2> i1_i2 = { i1, i2 };
            for (int64_t i: i1_i2) {
                if (C.tileIsLocal(i, i)) {
                    C.tileGetForWriting(i, i, LayoutConvert(layout));
                    makeHermitian( C(i, i) );
                    int dst = C.tileRank(i2, i1);
                    C.tileSend(i, i, dst, tag);
                    C.tileRecv(i, i, dst, layout, tag);
                }
            }
            if (C.tileIsLocal(i2, i1)) {
                // Receive tiles from sources, apply Q on both sides,
                // then send updated tiles back.
                int src11 = C.tileRank(i1, i1);
                C.tileRecv(i1, i1, src11, layout, tag);

                int src22 = C.tileRank(i2, i2);
                C.tileRecv(i2, i2, src22, layout, tag);

                V.tileGetForReading(i2, 0,  LayoutConvert(layout));
                T.tileGetForReading(i2, 0,  LayoutConvert(layout));
                C.tileGetForWriting(i2, i1, LayoutConvert(layout));

                // Workspace C(i1, i2) = C(i2, i1)^H.
                C.tileInsert(i1, i2);
                deepConjTranspose(C(i2, i1), C(i1, i2));

                int64_t nb = std::min(V.tileMb(i2), V.tileNb(0));
                tpmqrt(Side::Left, op, nb,
                       V(i2, 0), T(i2, 0), C(i1, i1), C(i2, i1));  // 1st col
                tpmqrt(Side::Left, op, nb,
                       V(i2, 0), T(i2, 0), C(i1, i2), C(i2, i2));  // 2nd col
                tpmqrt(Side::Right, opR, nb,
                       V(i2, 0), T(i2, 0), C(i1, i1), C(i1, i2));  // 1st row
                tpmqrt(Side::Right, opR, nb,
                       V(i2, 0), T(i2, 0), C(i2, i1), C(i2, i2));  // 2nd row

                C.tileSend(i1, i1, src11, tag);
                C.tileSend(i2, i2, src22, tag);
                C.tileTick(i1, i1);
                C.tileTick(i2, i2);
                C.tileErase(i1, i2);  // Discard results; equals C(i2, i1)^H.
            }

            //--------------------
            // 2: Multiply Q^H * [ C(i1, j) ] for j = 0, ..., i1-1,
            //                   [ C(i2, j) ]         i1+1, ..., i2-1.
            // Note j = i1 is skipped because it is handled in step 1 above.
            // If  C(i1, j) is in upper triangle (i1 < j),
            // use C(i1, j) = C(j, i1)^H.
            // TODO: the j loop can be parallelized, with care for MPI.
            for (int64_t j = 0; j < i2; ++j) {
                if (j == i1)
                    continue;

                if ((i1 >= j && C.tileIsLocal(i1, j)) ||
                    (i1 <  j && C.tileIsLocal(j, i1)))
                {
                    // First node of each pair sends tile to dst,
                    // then receives updated tile back ().
                    int dst = C.tileRank(i2, j);
                    if (i1 >= j) {
                        C.tileSend(i1, j, dst, tag);
                        C.tileRecv(i1, j, dst, layout, tag);
                    }
                    else {
                        // Send transposed tile.
                        deepConjTranspose(C(j, i1));
                        C.tileSend(j, i1, dst, tag);
                        C.tileRecv(j, i1, dst, layout, tag);
                        deepConjTranspose(C(j, i1));
                    }
                }
                else if (C.tileIsLocal(i2, j)) {
                    // Second node of each pair receives tile from src,
                    // applies Q, then sends updated tile back.
                    int src = (i1 >= j ? C.tileRank(i1, j)
                                       : C.tileRank(j, i1));
                    C.tileRecv(i1, j, src, layout, tag);

                    V.tileGetForReading(i2, 0, LayoutConvert(layout));
                    T.tileGetForReading(i2, 0, LayoutConvert(layout));
                    C.tileGetForWriting(i2, j, LayoutConvert(layout));

                    // Multiply op(Q) * [ C(i1, j) ].
                    //                  [ C(i2, j) ]
                    tpmqrt(Side::Left, op, std::min(V.tileMb(i2), V.tileNb(0)),
                           V(i2, 0), T(i2, 0), C(i1, j), C(i2, j));

                    // todo: should tileRelease()?
                    V.tileTick(i2, 0);
                    T.tileTick(i2, 0);

                    // Send updated tile back.
                    C.tileSend(i1, j, src, tag);
                    C.tileTick(i1, j);
                }
            } // for j
        } // for index

        //--------------------
        // Finish updating all rows before updating columns.
        slate_mpi_call(
            MPI_Barrier(C.mpiComm()));

        for (int index = 0; index + step < nranks; index += 2*step) {
            int64_t j1 = rank_indices[ index ].second;
            int64_t j2 = rank_indices[ index + step ].second;

            //--------------------
            // 3: Multiply [ C(i, j1)  C(i, j2) ] * Q for i = j2+1, ..., mt-1.
            for (int64_t i = j2+1; i < C.mt(); ++i) {
                if (C.tileIsLocal(i, j1)) {
                    // First node of each pair sends tile to dst,
                    // then receives updated tile back.
                    int dst = C.tileRank(i, j2);
                    C.tileSend(i, j1, dst, tag);
                    C.tileRecv(i, j1, dst, layout, tag);
                }
                else if (C.tileIsLocal(i, j2)) {
                    // Second node of each pair receives tile from src,
                    // applies Q, then sends updated tile back.
                    int src = C.tileRank(i, j1);
                    C.tileRecv(i, j1, src, layout, tag);

                    V.tileGetForReading(j2, 0, LayoutConvert(layout));
                    T.tileGetForReading(j2, 0, LayoutConvert(layout));
                    C.tileGetForWriting(i, j2, LayoutConvert(layout));

                    // Multiply [ C(i, j1) C(i, j2) ] * opR(Q).
                    tpmqrt(Side::Right, opR, std::min(V.tileMb(i), V.tileNb(0)),
                           V(j2, 0), T(j2, 0), C(i, j1), C(i, j2));

                    // todo: should tileRelease()?
                    V.tileTick(j2, 0);
                    T.tileTick(j2, 0);

                    // Send updated tile back.
                    C.tileSend(i, j1, src, tag);
                    C.tileTick(i, j1);
                }
            } // for i
        } // for index

        //--------------------
        // Next level.
        if (descend)
            step /= 2;
        else
            step *= 2;
    } // for level
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void hettmqr<Target::HostTask, float>(
    Op op,
    Matrix<float>&& V,
    Matrix<float>&& T,
    HermitianMatrix<float>&& C,
    int tag);

// ----------------------------------------
template
void hettmqr<Target::HostTask, double>(
    Op op,
    Matrix<double>&& V,
    Matrix<double>&& T,
    HermitianMatrix<double>&& C,
    int tag);

// ----------------------------------------
template
void hettmqr< Target::HostTask, std::complex<float> >(
    Op op,
    Matrix< std::complex<float> >&& V,
    Matrix< std::complex<float> >&& T,
    HermitianMatrix< std::complex<float> >&& C,
    int tag);

// ----------------------------------------
template
void hettmqr< Target::HostTask, std::complex<double> >(
    Op op,
    Matrix< std::complex<double> >&& V,
    Matrix< std::complex<double> >&& T,
    HermitianMatrix< std::complex<double> >&& C,
    int tag);

} // namespace internal
} // namespace slate
