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

#ifndef SLATE_TILE_TPQRT_HH
#define SLATE_TILE_TPQRT_HH

#include "internal/internal.hh"
#include "slate/Tile.hh"
#include "slate/Tile_blas.hh"
#include "internal/Tile_lapack.hh"
#include "slate/types.hh"
#include "slate/internal/util.hh"

#include <list>
#include <vector>

#include <blas.hh>
#include <lapack.hh>

namespace slate {

//------------------------------------------------------------------------------
/// Compute the triangular-pentagonal QR factorization of 2 tiles, A1 and A2.
/// On exit, the pentagonal tile A2 has been eliminated.
///
/// @param[in] l
///     The number of rows of the upper trapezoidal part of A2.
///     min(m, n) >= l >= 0. See Further Details.
///
/// @param[in,out] A1
///     On entry, the n-by-n upper triangular tile A1.
///     A1 can be k-by-n for k >= n; only the upper n-by-n portion is used.
///
/// @param[in,out] A2
///     On entry, the m-by-n pentagonal tile A2.
///     On exit, the columns represent the Householder reflectors.
///     The top (m-l)-by-n portion is rectangular,
///     the bottom l-by-n portion is upper trapezoidal.
///
/// @param[out] T
///     Tile of size ib-by-n, where ib is the internal blocking to use.
///     n >= ib >= 1.
///     On exit, stores a sequence of ib-by-ib upper triangular T matrices
///     representing the block Householder reflectors. See Further Details.
///
/// Further Details
/// ---------------
///
/// Let A be the (n + m)-by-n matrix
///
///     A = [ A1 ]  <- n-by-n upper triangular
///         [ A2 ]  <- m-by-n upper pentagonal
///
/// For example, with m = 5, n = 4, l = 3, the non-zeros of A1 and A2 are
///
///     A1 = [ . . . . ]  <- n-by-n upper triangular
///          [   . . . ]
///          [     . . ]
///          [       . ]
///
///     A2 = [ . . . . ]  <- (m - l)-by-n rectangular
///          [ . . . . ]
///          [---------]
///          [ . . . . ]  <- l-by-n upper trapezoidal
///          [   . . . ]
///          [     . . ]
///
/// Depending on m, n, l, there are several cases.
/// If l < min(m, n), A2 is pentagonal, as shown above.
/// If l = 0, it becomes just the rectangular portion:
///
///     A2 = [ . . . . ]  <- m-by-n rectangular
///          [ . . . . ]
///
/// If m > n and l = min(m, n) = n, it becomes upper trapezoidal (tall):
///
///     A2 = [ . . . . ]  <- (m - l)-by-n rectangular
///          [---------]
///          [ . . . . ]  <- l-by-n upper trapezoidal (triangular)
///          [   . . . ]
///          [     . . ]
///          [       . ]
///
/// If m < n and l = min(m, n) = m, it becomes upper trapezoidal (wide):
///
///     A2 = [ . . . . . ]  <- l-by-n upper trapezoidal
///          [   . . . . ]
///          [     . . . ]
///          [       . . ]
///
/// If m = n = l, it becomes upper triangular:
///
///     A2 = [ . . . . ]  <- l-by-n upper trapezoidal (triangular)
///          [   . . . ]
///          [     . . ]
///          [       . ]
///
/// After factoring, the vector representing the elementary reflector H(i) is in
/// the i-th column of the (m + n)-by-n matrix V:
///
///     V = [ I  ] <- n-by-n identity
///         [ V2 ] <- m-by-n pentagonal, same form as A2.
///
/// Thus, all of the information needed for V is contained in V2, which
/// has the same form as A2 and overwrites A2 on exit.
///
/// The number of blocks is r = ceiling(n/ib), where each
/// block is of order ib except for the last block, which is of order
/// rb = n - (r-1)*ib. For each of the r blocks, an upper triangular block
/// reflector factor is computed: T1, T2, ..., Tr. The ib-by-ib (and rb-by-rb
/// for the last block) T's are stored in the ib-by-n matrix T as
///
///     T = [ T1 T2 ... Tr ]
///
/// Note in LAPACK, A = A1, B = A2, W = V, V = V2.
///
/// @ingroup geqrf_tile
///
template <typename scalar_t>
void tpqrt(
    int64_t l,
    Tile<scalar_t> A1,
    Tile<scalar_t> A2,
    Tile<scalar_t> T)
{
    trace::Block trace_block("lapack::tpqrt");

    int64_t n = A2.nb();
    int64_t m = std::min( A2.mb(), A2.nb() ); //A2.mb();

    assert(A1.mb() >= n);  // k >= n
    assert(A1.nb() == n);
    assert(std::min(m, n) >= l);

    int64_t ib = T.mb();
    assert(n >= ib);
    assert(T.nb() == n);

    lapack::tpqrt(m, n, l, ib,
                  A1.data(), A1.stride(),
                  A2.data(), A2.stride(),
                  T.data(), T.stride());
}

} // namespace slate

#endif // SLATE_TILE_TPQRT_HH
