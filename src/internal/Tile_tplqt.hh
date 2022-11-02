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
///     min(m, n) >= l >= 0. See Further Details.
///
/// @param[in,out] A1
///     On entry, the m-by-m lower triangular tile A1.
///     A1 can be m-by-k for k >= m; only the lower m-by-m portion is used.
///
/// @param[in,out] A2
///     On entry, the m-by-n pentagonal tile A2.
///     On exit, the rows represent the Householder reflectors.
///     The left m-by-(n-l) portion is rectangular,
///     the right m-by-l portion is lower trapezoidal.
///
/// @param[out] T
///     Tile of size ib-by-m, where ib is the internal blocking to use.
///     m >= ib >= 1.
///     On exit, stores a sequence of ib-by-ib upper triangular T matrices
///     representing the block Householder reflectors. See Further Details.
///
/// Further Details
/// ---------------
///
/// Let A be the m-by-(m + n) matrix
///
///     A = [ A1  A2 ]
///
/// where A1 is m-by-m lower triangular, and A2 is m-by-n lower pentagonal.
///
/// For example, with m = 4, n = 5, l = 3, the non-zeros of A1 and A2 are
///
///     A1 = [ .       ]       A2 = [ . . | .     ]
///          [ . .     ]            [ . . | . .   ]
///          [ . . .   ]            [ . . | . . . ]
///          [ . . . . ]            [ . . | . . . ]
///          m-by-m lower    m-by-(n - l) | m-by-l
///          triangular       rectangular | trapezoidal
///
/// Depending on m, n, l, there are several cases.
/// If l < min(m, n), A2 is pentagonal, as shown above.
/// If l = 0, it becomes just the rectangular portion:
///
///     A2 = [ . . ]
///          [ . . ]
///          [ . . ]
///          [ . . ]
///          m-by-n rectangular
///
/// If m < n and l = min(m, n) = m, it becomes lower trapezoidal (wide):
///
///         A2 = [ . | .       ]
///              [ . | . .     ]
///              [ . | . . .   ]
///              [ . | . . . . ]
///     m-by-(n - l) | m-by-l
///      rectangular | trapezoidal (triangular)
///
/// If m > n and l = min(m, n) = n, it becomes lower trapezoidal (tall):
///
///     A2 = [ .     ]
///          [ . .   ]
///          [ . . . ]
///          [ . . . ]
///            m-by-l trapezoidal
///
/// If m = n = l, it becomes lower triangular:
///
///     A2 = [ .       ]
///          [ . .     ]
///          [ . . .   ]
///          [ . . . . ]
///            m-by-l trapezoidal (triangular)
///
/// After factoring, the vector representing the elementary reflector H(i) is in
/// the i-th row of the m-by-(m+n) matrix V:
///
///     V = [ I  V2 ]
///
/// where I is the m-by-m identity and V2 is m-by-n pentagonal, same form as A2.
///
/// Thus, all of the information needed for V is contained in V2, which
/// has the same form as A2 and overwrites A2 on exit.
///
/// The number of blocks is r = ceiling(m/ib), where each
/// block is of order ib except for the last block, which is of order
/// rb = m - (r-1)*ib. For each of the r blocks, an upper triangular block
/// reflector factor is computed: T1, T2, ..., Tr. The ib-by-ib (and rb-by-rb
/// for the last block) T's are stored in the ib-by-m matrix T as
///
///     T = [ T1 T2 ... Tr ]
///
/// Note in LAPACK, A = A1, B = A2, W = V, V = V2.
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

    int64_t m = A2.mb();
    int64_t n = A2.nb();

    assert(A1.mb() == m);
    assert(A1.nb() >= m);  // k >= m
    assert(std::min(m, n) >= l);
    assert(T.nb() == m);

    int64_t ib = std::min( T.mb(), m );
    lapack::tplqt(m, n, l, ib,
                  A1.data(), A1.stride(),
                  A2.data(), A2.stride(),
                  T.data(), T.stride());
#else
    slate_not_implemented( "In gelqf: tplqt requires LAPACK >= 3.7" );
#endif
}

} // namespace slate

#endif // SLATE_TILE_TPLQT_HH
