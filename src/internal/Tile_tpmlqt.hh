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

#ifndef SLATE_TILE_TPMLQT_HH
#define SLATE_TILE_TPMLQT_HH

#include "slate/Tile.hh"
#include "slate/types.hh"

#include <list>
#include <vector>

#include <blas.hh>
#include <lapack.hh>

namespace slate {

//------------------------------------------------------------------------------
/// Multiply the matrix C by the unitary matrix Q obtained from a
/// "triangular-pentagonal" block reflector H.
/// C consists of two tiles, C1 and C2.
///
/// If side == Left:
///
///     C = [ C1 ]  <- k-by-n
///         [ C2 ]  <- m-by-n
///
/// and on exit, $C = op(Q) C$.
/// C is (k+m)-by-n, C1 is k-by-n, C2 is m-by-n, and V2 is k-by-m.
/// l is the same in tplqt; m = tplqt's n; k = tplqt's m; n here is different.
///
/// If side == Right:
///
///     C = [ C1  C2 ]
///       m-by-k  m-by-n
///
/// and on exit, $C = C op(Q)$.
/// C is m-by-(k+n), C1 is m-by-k, C2 is m-by-n, and V2 is k-by-n.
/// n, l are the same in tplqt; k = tplqt's m; m here is different.
///
/// Q is a product of block reflectors,
///
///     Q = \prod_{j = r, ..., 1} I - Vj^H Tj^H Vj
///
/// where r is the number of blocks, Tj is the j-th block of T,
/// and Vj is the j-th block row of V, with internal blocking size ib.
///
/// See Further Details in tplqt.
///
/// @param[in] side
///     - Side::Left:  Multiply from the left:  $C = op(Q) C$.
///     - Side::Right: Multiply from the right: $C = C op(Q)$.
///
/// @param[in] op
///     - Op::NoTrans:   Multiply by $op(Q) = Q$.
///     - Op::Trans:     Multiply by $op(Q) = Q^T$ (only in real case).
///     - Op::ConjTrans: Multiply by $op(Q) = Q^H$.
///
/// @param[in] l
///     The number of columns of the lower trapezoidal part of V2.
///     - If side = left,  min(m, k) >= l >= 0.
///     - If side = right, min(n, k) >= l >= 0.
///
/// @param[in] V2
///     - If side == Left,  the k-by-m lower pentagonal tile V2.
///     - If side == Right, the k-by-n lower pentagonal tile V2.
///     The i-th row must contain the vector which defines the
///     elementary reflector H(i), for i = 1, 2, ..., k, as returned by
///     tplqt in A2. The left k-by-(m-l) or k-by-(n-l) portion is rectangular,
///     the right k-by-l portion is lower trapezoidal.
///     See Further Details in tplqt.
///
/// @param[in] T
///     The upper triangular factors of the block reflectors
///     as returned by tplqt, stored as an ib-by-k tile.
///
/// @param[in,out] C1
///     - If side == Left,  the k-by-n tile C1.
///       C1 can be k2-by-n for k2 >= k; only the upper k-by-n portion is used.
///     - If side == Right, the m-by-k tile C1.
///       C1 can be m-by-k2 for k2 >= k; only the left m-by-k portion is used.
///     On exit, C1 is overwritten by the corresponding block of
///     $op(Q) C$ or $C op(Q)$.
///
/// @param[in,out] C2
///     The m-by-n tile C2.
///     On exit, C2 is overwritten by the corresponding block of
///     $op(Q) C$ or $C op(Q)$.
///
/// Note in LAPACK, A = C1, B = C2, V = V2.
///
/// @ingroup gelqf_tile
///
template <typename scalar_t>
void tpmlqt(
    Side side, Op op, int64_t l,
    Tile<scalar_t> V2,
    Tile<scalar_t> T,
    Tile<scalar_t> C1,
    Tile<scalar_t> C2)
{
    trace::Block trace_block("lapack::tpmlqt");

    int64_t k = V2.mb();
    int64_t m = C2.mb();
    int64_t n = C2.nb();
    if (side == Side::Left) {
        assert(C1.mb() >= k);
        assert(C1.nb() == n);
        assert(V2.nb() == m);
        assert(std::min(m, k) >= l);
    }
    else {
        assert(C1.mb() == m);
        assert(C1.nb() >= k);
        assert(V2.nb() == n);
        assert(std::min(n, k) >= l);
    }

    int64_t ib = T.mb();
    assert(k >= ib);
    assert(T.nb() == k);

    lapack::tpmlqt(side, op, m, n, k, l, ib,
                   V2.data(), V2.stride(),
                   T.data(), T.stride(),
                   C1.data(), C1.stride(),
                   C2.data(), C2.stride());
}

} // namespace slate

#endif // SLATE_TILE_TPMLQT_HH
