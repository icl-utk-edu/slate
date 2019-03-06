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

#include "internal/slate_internal.hh"
#include "slate_Tile.hh"
#include "slate_Tile_blas.hh"
#include "internal/slate_Tile_lapack.hh"
#include "slate_types.hh"
#include "slate/slate_util.hh"

#include <list>
#include <vector>

#include <blas.hh>
#include <lapack.hh>

namespace slate {

///-----------------------------------------------------------------------------
/// Compute the triangle-pentagonal factorization of 2 tiles.
///
/// On exit, the pentagonal portion of B has been eliminated.
///
/// \param[in] l
///     The number of rows of the upper trapezoidal part of B.
///     min(m, n) >= l >= 0.  See Further Details.
///     If l = 0, B is rectangular.
///     If l = n, B is triangular.
///
/// \param[in,out] A
///     On entry, the k-by-n upper triangular tile A.
///     Only the upper n-by-n portion is accessed. k >= n; otherwise k is unused.
///
/// \param[in,out] B
///     On entry, the m-by-n pentagonal tile B.
///     On exit, the columns represent the Householder reflectors.
///     The top (m-l)-by-n portion is rectangular,
///     the bottom l-by-n portion is upper trapezoidal.
///
/// \param[out] T
///     Tile of size ib-by-n, where ib is the internal blocking to use. ib >= n.
///     On exit, stores a sequence of ib-by-ib upper triangular T matrices
///     representing the block Householder reflectors. See Further Details.
///
/// Further Details
/// ---------------
///
/// Let C be the (n + m)-by-n matrix
///
///     C = [ A ]
///         [ B ]
///
/// For example, with m = 5, n = 4, l = 3, the non-zeros of A and B are
///
///     A = [ . . . . ]  <- n-by-n upper triangular
///         [   . . . ]
///         [     . . ]
///         [       . ]
///
///     B = [ . . . . ]  <- (m - l)-by-n rectangular
///         [ . . . . ]
///         [---------]
///         [ . . . . ]  <- l-by-n upper trapezoidal
///         [   . . . ]
///         [     . . ]
///
/// After factoring, the vector representing the elementary reflector H(i) is in
/// the i-th column of the (m + n)-by-n matrix W:
///
///     W = [ I ] <- n-by-n identity
///         [ V ] <- m-by-n pentagonal, same form as B.
///
/// Thus, all of the information needed for W is contained in V, which
/// overwrites B on exit.
///
/// The number of blocks is r = ceiling(n/ib), where each
/// block is of order ib except for the last block, which is of order
/// rb = n - (r-1)*rb.  For each of the r blocks, a upper triangular block
/// reflector factor is computed: T1, T2, ..., Tr.  The ib-by-ib (and rb-by-rb
/// for the last block) T's are stored in the ib-by-n matrix T as
///
///     T = [ T1 T2 ... Tr ]
///
template <typename scalar_t>
void tpqrt(
    int64_t l,
    Tile<scalar_t> A,
    Tile<scalar_t> B,
    Tile<scalar_t> T)
{
    trace::Block trace_block("lapack::tpqrt");

    int64_t m = B.mb();
    int64_t n = B.nb();
    assert(A.mb() >= n);
    assert(A.nb() == n);
    assert(std::min(m, n) >= l);

    int64_t ib = T.mb();
    assert(n >= ib);
    assert(T.nb() == n);

    lapack::tpqrt(m, n, l, ib,
                  A.data(), A.stride(),
                  B.data(), B.stride(),
                  T.data(), T.stride());
}

} // namespace slate

#endif // SLATE_TILE_TPQRT_HH
