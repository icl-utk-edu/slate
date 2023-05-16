// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_TILE_TPLQT_HH
#define SLATE_TILE_TPLQT_HH

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
/// Compute the triangular-pentagonal LQ factorization of 2 tiles, A1 and A2.
/// On exit, the pentagonal tile A2 has been eliminated.
///
/// @param[in] l
///     The number of columns of the lower trapezoidal part of A2.
///     min( k, n ) >= l >= 0. See Further Details.
///
/// @param[in,out] A1
///     On entry, the k-by-k lower triangular matrix A1,
///     in an k-by-k1 tile, where k1 >= k.
///
/// @param[in,out] A2
///     On entry, the k-by-n lower pentagonal matrix A2,
///     in an k-by-n2 tile, where n2 >= n.
///     On exit, the rows represent the Householder reflectors.
///     The left k-by-(n-l) portion is rectangular,
///     the right k-by-l portion is lower trapezoidal.
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
/// Let A be the k-by-(k + n) matrix
///
///     A = [ A1  A2 ]
///
/// where A1 is k-by-k lower triangular, and A2 is k-by-n lower pentagonal.
///
/// For all cases, A1 is lower triangular.
/// Example with k = 4, where "." represent non-zeros.
///
///     A1 = [ .        |  0 ]
///          [ . .      |  0 ]
///          [ . . .    |  0 ]
///          [ . . . .  |  0 ]
///       k-by-k lower    if k1 > k,
///         triangular    all-zero columns are ignored
///
/// Depending on n, k, l, there are several cases for A2.
/// Currently, SLATE supports only cases 1, 2, and 3.
/// It assumes n = min( A2.mb, A2.nb ), and l = n or l = 0.
///
/// Case 1: n = k = l, A2 is lower triangular.
/// Example with n = k = l = 4.
///
///     A1 = [ .       ]      A2 = [ .        |  0 ]
///          [ . .     ]           [ . .      |  0 ]
///          [ . . .   ]           [ . . .    |  0 ]
///          [ . . . . ]           [ . . . .  |  0 ]
///                             k-by-l lower     if n2 > n,
//                                triangular     all-zero columns are ignored
///
/// Case 2: n < k and l = min( n, k ) = n, A2 is lower trapezoidal (tall).
/// Example with n = l = 3, k = 4.
///
///     A1 = [ .       ]      A2 = [ .     ]  <== k-by-l lower trapezoidal
///          [ . .     ]           [ . .   ]
///          [ . . .   ]           [ . . . ]
///          [ . . . . ]           [ . . . ]
///
/// Case 3: l = 0, A2 is just the rectangular portion.
/// Currently unused in SLATE, but should work.
/// Example with n = 3, l = 0, k = 4.
///
///     A1 = [ .       ]      A2 = [ . . . ]  <== k-by-n rectangular
///          [ . .     ]           [ . . . ]
///          [ . . .   ]           [ . . . ]
///          [ . . . . ]           [ . . . ]
///
/// Case 4: n > k and l = k, A2 is upper trapezoidal (wide).
/// Currently unsupported in SLATE; would require explicitly passing n.
/// Example with n = 6, l = k = 4.
///
///     A1 = [ .       ]      A2 = [ . . | .       ]
///          [ . .     ]           [ . . | . .     ]
///          [ . . .   ]           [ . . | . . .   ]
///          [ . . . . ]           [ . . | . . . . ]
///                         k-by-(n - l)   k-by-l lower
//                           rectangular   triangular
///
/// Case 5: 0 < l < min( n, k ), A2 is lower pentagonal.
/// Currently unsupported in SLATE; would require explicitly passing n.
/// Example with n = 5, l = 3, k = 4.
///
///     A1 = [ .       ]      A2 = [ . . | .     ]
///          [ . .     ]           [ . . | . .   ]
///          [ . . .   ]           [ . . | . . . ]
///          [ . . . . ]           [ . . | . . . ]
///         k-by-k lower    k-by-(n - l)   k-by-l lower
///           triangular     rectangular   trapezoidal
///
/// After factoring, the vector representing the elementary reflector H(i) is in
/// the i-th row of the k-by-(k+n) matrix V:
///
///     V = [ I  V2 ]
///
/// where I is the k-by-k identity and V2 is k-by-n pentagonal, same form as A2.
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
/// V => V2, and m => k. This makes k match k in tpmlqt.
///
/// @ingroup gelqf_tile
///
template <typename scalar_t>
void tplqt(
    int64_t l,
    Tile<scalar_t> A1,
    Tile<scalar_t> A2,
    Tile<scalar_t> T)
{
#if LAPACK_VERSION >= 30700
    trace::Block trace_block("lapack::tplqt");

    // Lower trapezoid of A2 is k-by-n, with n <= k.
    int64_t k = A2.mb();
    int64_t n = std::min( A2.nb(), k );
    assert( l == n || l == 0 );

    // Lower triangle of A1 is k-by-k.
    assert( A1.mb() == k );
    assert( A1.nb() >= k );  // k1 >= k

    // T is ib-by-k, with ib <= k.
    int64_t ib = std::min( T.mb(), k );
    assert( T.nb() >= k );

    lapack::tplqt( k, n, l, ib,
                   A1.data(), A1.stride(),
                   A2.data(), A2.stride(),
                   T.data(), T.stride() );
#else
    slate_not_implemented( "In gelqf: tplqt requires LAPACK >= 3.7" );
#endif
}

} // namespace slate

#endif // SLATE_TILE_TPLQT_HH
