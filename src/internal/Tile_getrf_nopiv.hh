// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_TILE_GETRF_NOPIV_HH
#define SLATE_TILE_GETRF_NOPIV_HH

#include "internal/internal.hh"
#include "slate/Tile.hh"
#include "slate/types.hh"

#include <blas.hh>
#include <lapack.hh>

namespace slate {
namespace internal {
//------------------------------------------------------------------------------
/// Compute the LU factorization of a tile without pivoting.
///
/// @param[in] ib
///     internal blocking in the panel
///
/// @param[in,out] tile
///     tile to factor
///
/// @ingroup gesv_tile
///
template <typename scalar_t>
void getrf_nopiv(Tile<scalar_t> tile, int64_t ib)
{
    const scalar_t one = 1.0;
    int64_t nb = tile.nb();
    int64_t mb = tile.mb();
    int64_t diag_len = std::min(nb, mb);

    // Loop over ib-wide stripes.
    for (int64_t k = 0; k < diag_len; k += ib) {

        //=======================
        // ib panel factorization
        int64_t kb = std::min(diag_len-k, ib);

        // Loop over ib columns of a stripe.
        for (int64_t j = k; j < k+kb; ++j) {

            if (j+1 < mb) {
                // Update column
                blas::scal(mb-j-1, one/tile(j, j), &tile.at(j+1, j), 1);
            }

            // trailing update within ib block
            if (j+1 < k+kb) {
                // todo: make it a tile operation
                blas::geru(Layout::ColMajor,
                           mb-j-1, k+kb-j-1,
                           -one, &tile.at(j+1, j), 1,
                                 &tile.at(j, j+1), tile.stride(),
                                 &tile.at(j+1, j+1), tile.stride());
            }
        }

        // If there is a trailing submatrix.
        if (k+kb < nb) {

            blas::trsm(Layout::ColMajor,
                       Side::Left, Uplo::Lower,
                       Op::NoTrans, Diag::Unit,
                       kb, nb-k-kb,
                       one, &tile.at(k, k), tile.stride(),
                            &tile.at(k, k+kb), tile.stride());

            blas::gemm(blas::Layout::ColMajor,
                       Op::NoTrans, Op::NoTrans,
                       tile.mb()-k-kb, nb-k-kb, kb,
                       -one, &tile.at(k+kb,k   ), tile.stride(),
                             &tile.at(k,   k+kb), tile.stride(),
                       one,  &tile.at(k+kb,k+kb), tile.stride());
        }
    }
}

} // namespace internal
} // namespace slate

#endif // SLATE_TILE_GETRF_NOPIV_HH
