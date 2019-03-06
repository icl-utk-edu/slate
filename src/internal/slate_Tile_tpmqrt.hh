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

#include "slate_Tile.hh"
#include "slate_types.hh"

#include <list>
#include <vector>

#include <blas.hh>
#include <lapack.hh>

namespace slate {

///-----------------------------------------------------------------------------
/// Multiply the matrix C by the unitary matrix Q obtained from a
/// "triangular-pentagonal" block reflector H.
/// C consists of two tiles, A and B.
///
/// If side == Left:
///
///     C = [ A ]
///         [ B ]
///
/// and on exit, $C = op(Q) C$.
/// C is (k+m)-by-n, A is k-by-n, B is m-by-n, and V is m-by-k.
///
/// If side == Right:
///
///     C = [ A  B ]
///
/// and on exit, $C = C op(Q)$.
/// C is m-by-(k+n), A is m-by-k, B is m-by-n, and V is n-by-k.
///
/// See Further Details in tpqrt.
///
/// \param[in] side
///     - Side::Left:  Multiply from the left:  $C = op(Q) C$.
///     - Side::Right: Multiply from the right: $C = C op(Q)$.
///
/// \param[in] op
///     - Op::NoTrans:   Multiply by $op(Q) = Q$.
///     - Op::Trans:     Multiply by $op(Q) = Q^T$ (only in real case).
///     - Op::ConjTrans: Multiply by $op(Q) = Q^H$.
///
/// \param[in] l
///     The number of rows of the upper trapezoidal part of V.
///     k >= l >= 0.
///     If l = 0, V is rectangular.
///     If l = k, V is triangular.
///     (Note n in tpqrt is k here.)
///
/// \param[in] V
///     - If side == Left,  the m-by-k pentagonal tile V.
///     - If side == Right, the n-by-k pentagonal tile V.
///     The i-th column must contain the vector which defines the
///     elementary reflector H(i), for i = 1, 2, ..., k, as returned by
///     tpqrt in B.  See Further Details in tpqrt.
///     The top (m-l)-by-k or (n-l)-by-k portion is rectangular,
///     the bottom l-by-k portion is upper trapezoidal.
///
/// \param[in] T
///     The upper triangular factors of the block reflectors
///     as returned by tpqrt, stored as an ib-by-k tile.
///
/// \param[in,out] A
///     - If side == Left,  the k-by-n tile A.
///     - If side == Right, the m-by-k tile A.
///     On exit, A is overwritten by the corresponding block of
///     $op(Q) C$ or $C op(Q)$.
///
/// \param[in,out] B
///     The m-by-n tile B.
///     On exit, B is overwritten by the corresponding block of
///     $op(Q) C$ or $C op(Q)$.
///
template <typename scalar_t>
void tpmqrt(
    Side side, Op op, int64_t l,
    Tile<scalar_t> V,
    Tile<scalar_t> T,
    Tile<scalar_t> A,
    Tile<scalar_t> B)
{
    trace::Block trace_block("lapack::tpmqrt");

    int64_t k = V.nb();
    int64_t m = B.mb();
    int64_t n = B.nb();
    if (side == Side::Left) {
        assert(A.mb() == k);
        assert(A.nb() == n);
        assert(V.mb() == m);
    }
    else {
        assert(A.mb() == m);
        assert(A.nb() == k);
        assert(V.mb() == n);
    }

    int64_t ib = T.mb();
    assert(k >= ib);
    assert(T.nb() == k);

    lapack::tpmqrt(side, op, m, n, k, l, ib,
                   V.data(), V.stride(),
                   T.data(), T.stride(),
                   A.data(), A.stride(),
                   B.data(), B.stride());
}

} // namespace slate

#endif // SLATE_TILE_TPQRT_HH
