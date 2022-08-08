// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_TILE_HOUSEHOLDER_REFLECTION_GENERATOR
#define SLATE_TILE_HOUSEHOLDER_REFLECTION_GENERATOR

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

//--------------------------------------------------------------------------------
/// Compute at the qr2 level, Householder reflections of a panel, with tau scalars.
/// Additionally, one can:
///    Compute blocking factor T column-by-column
///    Compute trailing matrix update at qr2 level (for a full factorization)
/// by manually changing #ifdef specific to desired computation
///     We care for two reasons:
///     1) Performance comparision, evaluate panel performance without
///         trailing matrix update and/or constructing T
///     2) As a random orthogonal matrix generator code (no need to update)
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
void householder_reflection_generator(
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
    trace::Block trace_block("householder_reflection_generator");

    using blas::conj;
    using real_t = blas::real_type<scalar_t>;

    Tile<scalar_t>& diag_tile = tiles.at(0);
    int64_t diag_len = std::min( diag_tile.mb(), diag_tile.nb() );

    std::vector<scalar_t> taus(diag_len);
    std::vector<real_t> betas(diag_len);

    bool construct_blocking_factor = true;
    bool update_trailing_submatrix = false;

    for (int64_t k = 0; k < diag_len; ++k) {

        // Compute Householder reflections
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

            if (i_index == tile_indices.at(0)) {
                if (k+1 < tile.mb())
                    lapack::lassq(tile.mb()-k-1, &tile.at(k+1, k), 1,
                                  &scale[thread_rank], &sumsq[thread_rank]);
            }
            else {
                lapack::lassq(tile.mb(), &tile.at(0, k), 1,
                              &scale[thread_rank], &sumsq[thread_rank]);
            }
        }
        thread_barrier.wait(thread_size);

        //----------------------
        // global norm reduction
        // setting diagonal to 1
        // setting upper-diag to 0
        //   only if used for random
        //   orthogonal matrix generator
        if (thread_rank == 0) {
            for (int rank = 1; rank < thread_size; ++rank) {
                combine_sumsq(scale[0], sumsq[0], scale[rank], sumsq[rank]);
            }
            xnorm = scale[0]*std::sqrt(sumsq[0]);
            diag_tile.at(k, k) = one;
            // Setting upper-diagonal to zero
            //for (int64_t i = 0; i < k; ++i)
            //    diag_tile.at(i,k) = zero;
        }
        thread_barrier.wait(thread_size);

        real_t beta =
            -std::copysign(lapack::lapy3(alphr, alphi, xnorm), alphr);

        scalar_t scal_alpha = one / (alpha-beta);
        scalar_t tau = make<scalar_t>((beta-alphr)/beta, -alphi/beta);
        betas.at(k) = beta; // only need beta for correct QR-factorization
        taus.at(k) = tau;

        for (int64_t idx = thread_rank;
             idx < int64_t(tiles.size());
             idx += thread_size)
        {
            auto tile = tiles.at(idx);
            auto i_index = tile_indices.at(idx);

            // Finish Householder reflection by scaling
            if (i_index == tile_indices.at(0)) {
                if (k+1 < tile.mb())
                    blas::scal(tile.mb()-k-1,
                               scal_alpha, &tile.at(k+1, k), 1);
            }
            else {
                blas::scal(tile.mb(), scal_alpha, &tile.at(0, k), 1);
            }
        }
        thread_barrier.wait(thread_size);

        // Constructing blocking factor T
        if (construct_blocking_factor) {
            for (int64_t idx = thread_rank;
                 idx < int64_t(tiles.size());
                 idx += thread_size)
            {
                auto tile = tiles.at(idx);
                auto i_index = tile_indices.at(idx);
                scalar_t gemv_beta = idx == thread_rank ? 0.0 : 1.0;
                scalar_t* gemv_c;

                if (thread_rank == 0) {
                    gemv_c = &T.at(0, k);
                }
                else {
                    gemv_c = W.at(thread_rank).data();
                }

                if (k > 0) {
                    // Compute V(k:m,1:k-1)^T V(k:m,k) for:
                    // T(1:k-1,k) = - T(1:k-1,1:k-1) * V(k:m,1:k-1)^T
                    //                               * V(k:m,k) * T(k,k)
                    if (i_index == tile_indices.at(0)) {
                        blas::gemv(
                            Layout::ColMajor, Op::ConjTrans,
                            tile.mb()-k, k,
                            one,       &tile.at(k, 0), tile.stride(),
                                       &tile.at(k, k), 1,
                            gemv_beta, gemv_c, 1 );
                    }
                    else {
                        blas::gemv(
                            Layout::ColMajor, Op::ConjTrans,
                            tile.mb(), k,
                            one,       &tile.at(0, 0), tile.stride(),
                                       &tile.at(0, k), 1,
                            gemv_beta, gemv_c, 1 );
                    }
                }
            }
            thread_barrier.wait(thread_size);

            // put tau on diagonal of T
            if (thread_rank == 0) {
                T.at(k,k) = taus.at(k);
            }
            thread_barrier.wait(thread_size);

            if (k > 0) {
                if (thread_rank == 0) {
                    for (int rank = 1; rank < thread_size; ++rank)
                        for (int64_t j = 0; j < k; ++j)
                            T.at(j, k) += W.at(rank).data()[j];

                    // finishing the column T(1:k-1,k)
                    for (int64_t j = 0; j < k; ++j)
                        T.at(j,k) = - T.at(k,k) * T.at(j,k);

                    blas::trmv(Layout::ColMajor,
                               Uplo::Upper, Op::NoTrans, Diag::NonUnit,
                               k,
                               &T.at(0, 0), T.stride(),
                               &T.at(0, k), 1);
                }
                thread_barrier.wait(thread_size);
            }
        }

        // Trailing matrix update
        if (update_trailing_submatrix) {
            int64_t nb = diag_tile.nb();
            for (int64_t j = 0; j < nb; ++j)
                W.at(thread_rank).data()[j] = zero;

            scalar_t ger_alpha = -conj(tau);
            for (int64_t idx = thread_rank;
                 idx < int64_t(tiles.size());
                 idx += thread_size)
            {
                auto tile = tiles.at(idx);
                auto i_index = tile_indices.at(idx);
                scalar_t gemv_beta = idx == thread_rank ? 0.0 : 1.0;

                if (k+1 < diag_len) {
                    if (i_index == tile_indices.at(0)) {
                        blas::gemv(
                            Layout::ColMajor, Op::ConjTrans,
                            tile.mb()-k, nb-k-1,
                            one,       &tile.at(k, k+1), tile.stride(),
                                       &tile.at(k, k), 1,
                            gemv_beta, W.at(thread_rank).data(), 1 );
                    }
                    else {
                        blas::gemv(
                            Layout::ColMajor, Op::ConjTrans,
                            tile.mb(), nb-k-1,
                            one,       &tile.at(0, k+1), tile.stride(),
                                       &tile.at(0, k), 1,
                            gemv_beta, W.at(thread_rank).data(), 1 );
                    }
                }
            }
            thread_barrier.wait(thread_size);

            if (thread_rank == 0) {
                for (int rank = 1; rank < thread_size; ++rank)
                    blas::axpy( nb-k-1,
                                one, W.at(rank).data(), 1,
                                     W.at(0).data(), 1 );
            }
            thread_barrier.wait(thread_size);

            if (k+1 < diag_len) {
                for (int64_t idx = thread_rank;
                     idx < int64_t(tiles.size());
                     idx += thread_size)
                {
                    auto tile = tiles.at(idx);
                    auto i_index = tile_indices.at(idx);

                    if (i_index == tile_indices.at(0)) {
                        blas::ger(Layout::ColMajor,
                                  tile.mb()-k, nb-k-1,
                                  ger_alpha, &tile.at(k, k), 1,
                                             W.at(0).data(), 1,
                                             &tile.at(k, k+1), tile.stride());
                    }
                    else {
                        blas::ger(Layout::ColMajor,
                                  tile.mb(), nb-k-1,
                                  ger_alpha, &tile.at(0, k), 1,
                                             W.at(0).data(), 1,
                                             &tile.at(0, k+1), tile.stride());
                    }
                }
            }
            thread_barrier.wait(thread_size);

            // this is needed for the current SLATE implementation
            if (k+1 >= diag_len) {
                if (thread_rank == 0) {
                    auto& tile = diag_tile;
                    for (int64_t j = 0; j < diag_len; ++j)
                        tile.at(j, j) = betas.at(j);
                }
                thread_barrier.wait(thread_size);
            }
        }
    }
}
} // namespace internal
} // namespace slate

#endif // SLATE_TILE_HOUSEHOLDER_REFLECTION_GENERATOR
