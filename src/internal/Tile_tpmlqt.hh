// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

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
///     V = [ I  V2 ]      C = [ C1 ]  <== k-by-n
///      k-by-k  k-by-m        [ C2 ]  <== m-by-n
///
/// and on exit, $C = op(Q) C$.
/// l, k are the same in tplqt; m = tplqt's n; n here is different.
///
/// If side == Right:
///
///     C = [ C1  C2 ]      V = [ I  V2 ]
///       m-by-k  m-by-n     k-by-k  k-by-n
///
/// and on exit, $C = C op(Q)$.
/// l, k, n are the same in tplqt; m here is different.
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
///     - If side = left,  min( m, k ) >= l >= 0.
///     - If side = right, min( n, k ) >= l >= 0.
///
/// @param[in] V2
///     The k-by-N, lower pentagonal matrix V2,
///     in an k-by-N2 tile, where N2 >= N.
///     - If side == Left,  N = m.
///     - If side == Right, N = n.
///     The i-th row must contain the vector which defines the
///     elementary reflector H(i), for i = 1, 2, ..., k, as returned by
///     tplqt in A2. The left k-by-(N-l) portion is rectangular,
///     the right k-by-l portion is lower trapezoidal.
///     See Further Details in tplqt.
///
/// @param[in] T
///     The upper triangular factors of the block reflectors
///     as returned by tplqt, stored as an ib-by-k tile.
///
/// @param[in,out] C1
///     - If side == Left,  the k-by-n matrix C1, in a  k1-by-n tile, k1 >= k.
///     - If side == Right, the m-by-k matrix C1, in an m-by-k1 tile, k1 >= k.
///     On exit, C1 is overwritten by the corresponding block of
///     $op(Q) C$ or $C op(Q)$.
///
/// @param[in,out] C2
///     The m-by-n matrix C2.
///     - If side == Left,  C2 is in an m2-by-n tile, m2 >= m.
///     - If side == Right, C2 is in an m-by-n2 tile, n2 >= n.
///     On exit, C2 is overwritten by the corresponding block of
///     $op(Q) C$ or $C op(Q)$.
///
/// Note: compared to LAPACK, A is renamed here => C1, B => C2, V => V2.
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
#if LAPACK_VERSION >= 30700
    trace::Block trace_block("lapack::tpmlqt");

    int64_t m, n, k;
    if (side == Side::Left) {
        // Lower trapezoid of V2 is k-by-m, with m <= k. Compare tplqt.
        k = V2.mb();
        m = std::min( V2.nb(), k );
        assert( l == m || l == 0 );

        // C1 is k-by-n.
        assert( C1.mb() >= k );  // k1 >= k
        n = C1.nb();

        // C2 is m-by-n.
        assert( C2.mb() >= m );  // m2 >= m
        assert( C2.nb() == n );
    }
    else { // Right
        // Lower trapezoid of V2 is k-by-n, with n <= k. Compare tplqt.
        k = V2.mb();
        n = std::min( V2.nb(), k );
        assert( l == n || l == 0 );

        // C1 is m-by-k.
        m = C1.mb();
        assert( C1.nb() >= k );  // k1 >= k

        // C2 is m-by-n.
        assert( C2.mb() == m );
        assert( C2.nb() >= n );  // n2 >= n
    }

    // T is ib-by-k with ib <= k.
    int64_t ib = std::min( T.mb(), k );
    assert( T.nb() >= k );

    lapack::tpmlqt(side, op, m, n, k, l, ib,
                   V2.data(), V2.stride(),
                   T.data(), T.stride(),
                   C1.data(), C1.stride(),
                   C2.data(), C2.stride());
#else
    slate_not_implemented( "In gelqf: tpmlqt requires LAPACK >= 3.7" );
#endif
}

} // namespace slate

#endif // SLATE_TILE_TPMLQT_HH
