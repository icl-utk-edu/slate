// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

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
///     min( m, k ) >= l >= 0. See Further Details.
///
/// @param[in,out] A1
///     On entry, the k-by-k upper triangular matrix A1,
///     in a k1-by-k tile, where k1 >= k.
///
/// @param[in,out] A2
///     On entry, the m-by-k upper pentagonal matrix A2,
///     in an m2-by-k tile, where m2 >= m.
///     On exit, the columns represent the Householder reflectors.
///     The top (m-l)-by-k portion is rectangular,
///     the bottom l-by-k portion is upper trapezoidal.
///
/// @param[out] T
///     Array of size ib-by-k, where ib is the internal blocking to use,
///     1 <= ib <= k, in an ib3-by-k3 tile, ib3 >= ib and k3 >= k.
///     On exit, stores a sequence of ib-by-ib upper triangular T matrices
///     representing the block Householder reflectors. See Further Details.
///
/// Further Details
/// ---------------
///
/// Let A be the (k + m)-by-k matrix
///
///     A = [ A1 ]  <== k-by-k upper triangular
///         [ A2 ]  <== m-by-k upper pentagonal
///
/// For all cases, A1 is upper triangular.
/// Example with k = 4, with . representing non-zeros.
///
///     A1 = [ . . . . ]  <== k-by-k upper triangular
///          [   . . . ]
///          [     . . ]
///          [       . ]
///          [- - - - -]
///          [         ]  <== if k1 > k, all-zero rows are ignored
///
/// Depending on m, k, l, there are several cases for A2.
/// Currently, SLATE supports only cases 1, 2, and 3.
/// It assumes m = min( A2.mb, A2.nb ), and l = m or l = 0.
///
/// Case 1: m = k = l, A2 is upper triangular.
/// Example with m = k = l = 4.
///
///     A2 = [ . . . . ]  <== l-by-k upper triangular
///          [   . . . ]
///          [     . . ]
///          [       . ]
///          [- - - - -]
///          [         ]  <== if m2 > m, all-zero rows are ignored
///
/// Case 2: m < k and l = min( m, k ) = m, A2 is upper trapezoidal (wide).
/// Example with m = l = 3, k = 4.
///
///     A2 = [ . . . . ]  <== l-by-k upper trapezoidal
///          [   . . . ]
///          [     . . ]
///
/// Case 3: l = 0, A2 is just the rectangular portion.
/// Currently unused in SLATE, but should work.
/// Example with m = 3, l = 0, k = 4.
///
///     A2 = [ . . . . ]  <== m-by-k rectangular
///          [ . . . . ]
///          [ . . . . ]
///
/// Case 4: m > k and l = k, A2 is upper trapezoidal (tall).
/// Currently unsupported in SLATE; would require explicitly passing m.
/// Example with m = 6, l = k = 4.
///
///     A2 = [ . . . . ]  <== (m - l)-by-k rectangular
///          [ . . . . ]
///          [---------]
///          [ . . . . ]  <== l-by-k upper triangular
///          [   . . . ]
///          [     . . ]
///          [       . ]
///
/// Case 5: 0 < l < min( m, k ), A2 is upper pentagonal.
/// Currently unsupported in SLATE; would require explicitly passing m.
/// Example with m = 5, l = 3, k = 4.
///
///     A2 = [ . . . . ]  <== (m - l)-by-k rectangular
///          [ . . . . ]
///          [---------]
///          [ . . . . ]  <== l-by-k upper trapezoidal
///          [   . . . ]
///          [     . . ]
///
/// After factoring, the vector representing the elementary reflector H(i) is in
/// the i-th column of the (m + k)-by-k matrix V:
///
///     V = [ I  ] <== k-by-k identity
///         [ V2 ] <== m-by-k pentagonal, same form as A2.
///
/// Thus, all of the information needed for V is contained in V2, which
/// has the same form as A2 and overwrites A2 on exit.
///
/// The number of blocks is r = ceiling( k/ib ), where each
/// block is of order ib except for the last block, which is of order
/// rb = k - (r-1)*ib. For each of the r blocks, an upper triangular block
/// reflector factor is computed: T1, T2, ..., Tr. The ib-by-ib (and rb-by-rb
/// for the last block) T's are stored in the ib-by-k matrix T as
///
///     T = [ T1 T2 ... Tr ]
///
/// Note: compared to LAPACK, A is renamed here => A1, B => A2, W => V,
/// V => V2, and n => k. This makes k match k in tpmqrt.
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
#if LAPACK_VERSION >= 30400
    trace::Block trace_block("lapack::tpqrt");

    // Upper trapezoid of A2 is m-by-k with m <= k.
    int64_t k = A2.nb();
    int64_t m = std::min( A2.mb(), k );
    assert( l == m || l == 0 );

    // Upper triangle of A1 is k-by-k.
    assert( A1.mb() >= k );  // k1 >= k
    assert( A1.nb() == k );

    // T is ib-by-k, with ib <= k.
    int64_t ib = std::min( T.mb(), k );
    assert( T.nb() >= k );

    lapack::tpqrt( m, k, l, ib,
                   A1.data(), A1.stride(),
                   A2.data(), A2.stride(),
                   T.data(), T.stride() );
#else
    slate_not_implemented( "In geqrf: tpqrt requires LAPACK >= 3.4" );
#endif
}

} // namespace slate

#endif // SLATE_TILE_TPQRT_HH
