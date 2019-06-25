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

#ifndef SLATE_TILE_TPMQRT_HH
#define SLATE_TILE_TPMQRT_HH

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
/// C consists of two tiles, C0 and C1.
///
/// If side == Left:
///
///     C = [ C0 ]
///         [ C1 ]
///
/// and on exit, $C = op(Q) C$.
/// C is (k+m)-by-n, C0 is k-by-n, C1 is m-by-n, and V1 is m-by-k.
///
/// If side == Right:
///
///     C = [ C0  C1 ]
///
/// and on exit, $C = C op(Q)$.
/// C is m-by-(k+n), C0 is m-by-k, C1 is m-by-n, and V1 is n-by-k.
///
/// See Further Details in tpqrt.
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
///     The number of rows of the upper trapezoidal part of V1.
///     min(m, k) >= l >= 0.
///     If l = 0, V1 is rectangular.
///     If l = k, V1 is triangular.
///     (Note n in tpqrt is k here.)
///
/// @param[in] V1
///     - If side == Left,  the m-by-k pentagonal tile V1.
///     - If side == Right, the n-by-k pentagonal tile V1.
///     The i-th column must contain the vector which defines the
///     elementary reflector H(i), for i = 1, 2, ..., k, as returned by
///     tpqrt in A1.  See Further Details in tpqrt.
///     The top (m-l)-by-k or (n-l)-by-k portion is rectangular,
///     the bottom l-by-k portion is upper trapezoidal.
///
/// @param[in] T
///     The upper triangular factors of the block reflectors
///     as returned by tpqrt, stored as an ib-by-k tile.
///
/// @param[in,out] C0
///     - If side == Left,  the k-by-n tile C0.
///     - If side == Right, the m-by-k tile C0.
///     On exit, C0 is overwritten by the corresponding block of
///     $op(Q) C$ or $C op(Q)$.
///
/// @param[in,out] C1
///     The m-by-n tile C1.
///     On exit, C1 is overwritten by the corresponding block of
///     $op(Q) C$ or $C op(Q)$.
///
/// Note in LAPACK, A = C0, B = C1, V = V1.
///
/// @ingroup geqrf_tile
///
template <typename scalar_t>
void tpmqrt(
    Side side, Op op, int64_t l,
    Tile<scalar_t> V1,
    Tile<scalar_t> T,
    Tile<scalar_t> C0,
    Tile<scalar_t> C1)
{
    trace::Block trace_block("lapack::tpmqrt");

    int64_t k = V1.nb();
    int64_t m = C1.mb();
    int64_t n = C1.nb();
    if (side == Side::Left) {
        assert(C0.mb() == k);
        assert(C0.nb() == n);
        assert(V1.mb() == m);
    }
    else {
        assert(C0.mb() == m);
        assert(C0.nb() == k);
        assert(V1.mb() == n);
    }

    int64_t ib = T.mb();
    assert(k >= ib);
    assert(T.nb() == k);
    assert(m >= l);

    lapack::tpmqrt(side, op, m, n, k, l, ib,
                   V1.data(), V1.stride(),
                   T.data(), T.stride(),
                   C0.data(), C0.stride(),
                   C1.data(), C1.stride());
}

} // namespace slate

#endif // SLATE_TILE_TPQRT_HH
