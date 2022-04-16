// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_TILE_GEQRF_REFLECTION_ONLY
#define SLATE_TILE_GEQRF_REFLECTION_ONLY

#include "internal/internal_util.hh"
#include "slate/Tile.hh"
#include "slate/Tile_blas.hh"
#include "internal/Tile_lapack.hh"
#include "slate/types.hh"
#include "slate/internal/util.hh"

#include <cmath>
#include <list>
#include <vector>

#include <blas.hh>
#include <lapack.hh>

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Compute only the Householder reflections of a panel, including tau scalars.
///     We care for two reasons:
///     1) Performance comparision, evaluate panel performance without 
///         trailing matrix update and constructing T
///     2) As a random orthogonal matrix generator code
///
/// @param[in] ib
///     internal blocking in the panel
///
/// @param[in,out] tiles
///     local tiles in the panel
///
/// @param[in] tile_indices
///     i indices of the tiles in the panel
///
/// @param[out] T
///     upper triangular factor of the block reflector
///
/// @param[in] thread_rank
///     rank of this thread
///
/// @param[in] thread_size
///     number of local threads
///
/// @param[in] thread_barrier
///     barrier for synchronizing local threads
///
/// todo: add missing params
///
/// @ingroup geqrf_tile
///
template <typename scalar_t>
void geqrf_reflection_only(
    int64_t ib,
    std::vector< Tile<scalar_t> >& tiles,
    std::vector<int64_t>& tile_indices,
    Tile<scalar_t>& T,
    int thread_rank, int thread_size,
    ThreadBarrier& thread_barrier,
    std::vector<blas::real_type<scalar_t>>& scale,
    std::vector<blas::real_type<scalar_t>>& sumsq,
    blas::real_type<scalar_t>& xnorm,
    std::vector< std::vector<scalar_t> >& W)
{
    trace::Block trace_block("lapack::reflection_only");

    using blas::conj;
    using real_t = blas::real_type<scalar_t>;

    const scalar_t zero = 0.0;
    const scalar_t one  = 1.0;

    Tile<scalar_t>& diag_tile = tiles.at(0);
    int64_t diag_len = std::min( diag_tile.mb(), diag_tile.nb() );

    std::vector<scalar_t> taus(diag_len);
    std::vector<real_t> betas(diag_len);

    std::vector<scalar_t> tempR(ib*ib);

    // Loop over ib-wide stripes within panel
    for (int64_t k = 0; k < diag_len; k += ib) {

        //=======================
        //
        // Compute Householder reflectors V(1:m,1:min(mb,nb))

        scalar_t alpha = diag_tile.at(k, k);
        real_t alphr = real(alpha);
        real_t alphi = imag(alpha);

        //------------------
        // thread local norm
        scale[thread_rank] = 0.0;
        sumsq[thread_rank] = 1.0;
        for (int64_t idx = thread_rank;
             idx < int64_t(tiles.size());
             idx += thread_size)
        {
            auto tile = tiles.at(idx);
            auto i_index = tile_indices.at(idx);

            // if diagonal tile
            if (i_index == tile_indices.at(0)) {
                if (k+1 < tile.mb())
                    lapack::lassq(tile.mb()-k-1, &tile.at(k+1, k), 1,
                                  &scale[thread_rank], &sumsq[thread_rank]);
            }
            // off diagonal tile
            else {
                lapack::lassq(tile.mb(), &tile.at(0, k), 1,
                              &scale[thread_rank], &sumsq[thread_rank]);
            }
        }
        thread_barrier.wait(thread_size);
        //----------------------
        // global norm reduction
        // setting diagonal to 1
        if (thread_rank == 0) {
            for (int rank = 1; rank < thread_size; ++rank) {
                combine_sumsq(scale[0], sumsq[0], scale[rank], sumsq[rank]);
            }
            xnorm = scale[0]*std::sqrt(sumsq[0]);
            diag_tile.at(k, k) = scalar_t(1.0);
            // Setting upper-diagonal to zero
            for (int64_t i = 0; i < k; ++i) 
                diag_tile.at(i,k) = scalar_t(zero);
        }
        thread_barrier.wait(thread_size);

        //----------------------
        //
        real_t beta =
            -std::copysign(lapack::lapy3(alphr, alphi, xnorm), alphr);

        scalar_t scal_alpha = scalar_t(1.0) / (alpha-beta);
        scalar_t tau = make<scalar_t>((beta-alphr)/beta, -alphi/beta);

        betas.at(k) = beta;
        taus.at(k) = tau;

        // Scale Householder reflection
        for (int64_t idx = thread_rank;
             idx < int64_t(tiles.size());
             idx += thread_size)
        {
            auto tile = tiles.at(idx);
            auto i_index = tile_indices.at(idx);

            if (i_index == tile_indices.at(0)) {
                if (k+1 < tile.mb())
                    blas::scal(tile.mb()-k-1,
                               scal_alpha, &tile.at(k+1, k), 1);
            }
            else {
                blas::scal(tile.mb(), scal_alpha, &tile.at(0, k), 1);
            }
        }
    }
}
} // namespace internal
} // namespace slate

#endif // SLATE_TILE_GEQRF_REFLECTION_ONLY
