// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_TILE_GEQRF_HH
#define SLATE_TILE_GEQRF_HH

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
/// Compute the QR factorization of a panel.
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
/// @param[out] scale
///     component of lassq to compute Householder norm
///     in a stable manner
///
/// @param[out] sumsq
///     component of lassq to compute Householder norm
///     in a stable manner
///
/// @param[out] xnorm
///     Householder norm
///
/// @param[out] W
//      Workspace for the algorithm
///
/// @ingroup geqrf_tile
///
template <typename scalar_t>
void geqrf(
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
    trace::Block trace_block("lapack::geqrf");

    using blas::conj;
    using real_t = blas::real_type<scalar_t>;

    const scalar_t zero = 0.0;
    const scalar_t one  = 1.0;
    const real_t r_one  = 1.0;
    int knt;

    Tile<scalar_t>& diag_tile = tiles.at(0);
    int64_t diag_len = std::min( diag_tile.mb(), diag_tile.nb() );

    std::vector<scalar_t> taus(diag_len);
    std::vector<scalar_t> betas(diag_len);
    int64_t nb = diag_tile.nb();

    // Loop over ib-wide stripes within panel
    for (int64_t k = 0; k < diag_len; k += ib) {

        int64_t kb = std::min(diag_len-k, ib);

        //=======================
        //
        // ib-stripe factorization:
        // Compute Householder reflectors V(1:m,k:k+kb)
        // Update Householder reflectors (within ib-stripe of panel)
        // -- (I - V(1:m,j) * tau(j) * V(1:m,j)^T) * V(1:m,j+1:k+kb)
        for (int64_t j = k; j < k+kb; ++j) {

            scalar_t alpha = diag_tile.at(j, j);
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
                    if (j+1 < tile.mb())
                        lapack::lassq(tile.mb()-j-1, &tile.at(j+1, j), 1,
                                      &scale[thread_rank], &sumsq[thread_rank]);
                }
                // off diagonal tile
                else {
                    lapack::lassq(tile.mb(), &tile.at(0, j), 1,
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
                diag_tile.at(j, j) = one;
            }
            thread_barrier.wait(thread_size);

            real_t safemin = std::numeric_limits<real_t>::epsilon();
            real_t rsafemin = r_one / safemin;
            // if Householder norm is numerically "zero", set to identity and exit
            if (xnorm < safemin && j < diag_len) {
                betas.at(j) = alpha;
                taus.at(j) = zero;
                diag_tile.at(j,j) = betas.at(j);
            }
            else {
                real_t beta =
                    -std::copysign(lapack::lapy3(alphr, alphi, xnorm), alphr);
                knt = 0;
                if (std::abs(beta) < safemin) {
                    if (knt < 20 && std::abs(beta) < safemin) {
                        knt += 1;
                        // scale input vector accordingly
                        for (int64_t idx = thread_rank;
                             idx < int64_t(tiles.size());
                             idx += thread_size)
                        {
                            auto tile = tiles.at(idx);
                            auto i_index = tile_indices.at(idx);

                            if (i_index == tile_indices.at(0)) {
                                if (j+1 < tile.mb())
                                    blas::scal(tile.mb()-j-1,
                                               rsafemin, &tile.at(j+1, j), 1);
                            }
                            else {
                                blas::scal(tile.mb(), rsafemin, &tile.at(0, j), 1);
                            }
                        }
                        thread_barrier.wait(thread_size);

                        beta = beta*rsafemin;
                        alpha = alpha*rsafemin;
                    }
                    //-------------------
                    // thread local norm
                    // with scaled vector
                    scale[thread_rank] = 0.0;
                    sumsq[thread_rank] = 1.0;
                    for (int64_t idx = thread_rank;
                         idx < int64_t(tiles.size());
                         idx += thread_size)
                    {
                        auto tile = tiles.at(idx);
                        auto i_index = tile_indices.at(idx);

                        if (i_index == tile_indices.at(0)) {
                            if (j+1 < tile.mb())
                                lapack::lassq(tile.mb()-j-1, &tile.at(j+1, j), 1,
                                              &scale[thread_rank], &sumsq[thread_rank]);
                        }
                        else {
                            lapack::lassq(tile.mb(), &tile.at(0, j), 1,
                                          &scale[thread_rank], &sumsq[thread_rank]);
                        }
                    }
                    thread_barrier.wait(thread_size);
                    if (thread_rank == 0) {
                        for (int rank = 1; rank < thread_size; ++rank) {
                            combine_sumsq(scale[0], sumsq[0], scale[rank], sumsq[rank]);
                        }
                        xnorm = scale[0]*std::sqrt(sumsq[0]);
                    }
                    thread_barrier.wait(thread_size);
                    alphr = real(alpha);
                    alphi = imag(alpha);
                    beta = -std::copysign(lapack::lapy3(alphr, alphi, xnorm), alphr);
                }

                // todo: Use overflow-safe division (see CLADIV/ZLADIV)
                scalar_t tau = make<scalar_t>((beta-alphr)/beta, -alphi/beta);
                scalar_t scal_alpha = one / (alpha-beta);
                scalar_t ger_alpha = -conj(tau);
                betas.at(j) = beta;
                taus.at(j) = tau;

                for (int64_t i = 0; i < knt; ++i) {
                    beta = beta*safemin;
                }
                //----------------------------------
                // column scaling and thread local W
                for (int64_t idx = thread_rank;
                     idx < int64_t(tiles.size());
                     idx += thread_size)
                {
                    auto tile = tiles.at(idx);
                    auto i_index = tile_indices.at(idx);
                    scalar_t gemv_beta = idx == thread_rank ? 0.0 : 1.0;

                    // column scaling
                    if (i_index == tile_indices.at(0)) {
                        // diagonal tile
                        if (j+1 < tile.mb())
                            blas::scal(tile.mb()-j-1,
                                       scal_alpha, &tile.at(j+1, j), 1);
                    }
                    else {
                        // off diagonal tiles
                        blas::scal(tile.mb(), scal_alpha, &tile.at(0, j), 1);
                    }
                    // thread local W
                    if (j+1 < diag_len) {
                        if (i_index == tile_indices.at(0)) {
                            // diagonal tile
                            blas::gemv(Layout::ColMajor, Op::ConjTrans,
                                       tile.mb()-j, k+kb-j-1,
                                       one,       &tile.at(j, j+1), tile.stride(),
                                                  &tile.at(j, j), 1,
                                       gemv_beta, W.at(thread_rank).data(), 1);
                        }
                        else {
                            // off diagonal tile
                            blas::gemv(Layout::ColMajor, Op::ConjTrans,
                                       tile.mb(), k+kb-j-1,
                                       one,       &tile.at(0, j+1), tile.stride(),
                                                  &tile.at(0, j), 1,
                                       gemv_beta, W.at(thread_rank).data(), 1);
                        }
                    }
                }
                thread_barrier.wait(thread_size);

                //-------------------
                // global W reduction
                if (thread_rank == 0) {
                    for (int rank = 1; rank < thread_size; ++rank)
                        blas::axpy( k+kb-j-1,
                                    one, W.at(rank).data(), 1,
                                         W.at(0).data(), 1 );
                }
                thread_barrier.wait(thread_size);

                //-----------
                // ger update
                if (j+1 < diag_len) {
                    for (int64_t idx = thread_rank;
                         idx < int64_t(tiles.size());
                         idx += thread_size)
                    {
                        auto tile = tiles.at(idx);
                        auto i_index = tile_indices.at(idx);

                        if (i_index == tile_indices.at(0)) {
                            // diagonal tile
                            blas::ger(Layout::ColMajor,
                                      tile.mb()-j, k+kb-j-1,
                                      ger_alpha, &tile.at(j, j), 1,
                                                 W.at(0).data(), 1,
                                                 &tile.at(j, j+1), tile.stride());
                        }
                        else {
                            // off diagonal tile
                            blas::ger(Layout::ColMajor,
                                      tile.mb(), k+kb-j-1,
                                      ger_alpha, &tile.at(0, j), 1,
                                                 W.at(0).data(), 1,
                                                 &tile.at(0, j+1), tile.stride());
                        }
                    }
                }
                thread_barrier.wait(thread_size);
                diag_tile.at(j,j) = betas.at(j);
            }
        } // end of ib-factorization of Householder reflectors

        // Compute V(k:m,k:k+kb)^H * [V(k:m,1:k+kb), A(k:m,k+kb+1:nb)]
        for (int64_t idx = thread_rank;
             idx < int64_t(tiles.size());
             idx += thread_size)
        {
            auto tile = tiles.at(idx);
            auto i_index = tile_indices.at(idx);
            scalar_t gemm_beta = idx == thread_rank ? 0.0 : 1.0;
            scalar_t* gemm_c = W.at(thread_rank).data();
            int64_t c_stride = kb;

            // Break computation for triangular block
            // and lower-rectangular block, as this
            // implementation is CPU, on GPU it may
            // be more advantageous computing extra
            // flops at once
            if (i_index == tile_indices.at(0)) {
                for (int64_t j = 0; j < nb; ++j) {
                    for (int64_t i = 0; i < kb; ++i) {
                        W.at(thread_rank).data()[i+j*kb] = tile.at(k+i,j);
                    }
                }

                blas::trmm(Layout::ColMajor,
                           Side::Left, Uplo::Lower,
                           Op::ConjTrans, Diag::Unit,
                           kb, nb,
                           one, &tile.at(k, k), tile.stride(),
                                W.at(thread_rank).data(), kb);

                if (k+kb < tile.mb()) {
                    blas::gemm(Layout::ColMajor,
                               Op::ConjTrans, Op::NoTrans,
                               kb, nb, tile.mb()-k-kb,
                               one, &tile.at(k+kb, k), tile.stride(),
                                    &tile.at(k+kb, 0), tile.stride(),
                               one, gemm_c, c_stride);
                }
            }
            else {
                blas::gemm(Layout::ColMajor,
                           Op::ConjTrans, Op::NoTrans,
                           kb, nb, tile.mb(),
                           one,       &tile.at(0, k), tile.stride(),
                                      &tile.at(0, 0), tile.stride(),
                           gemm_beta, gemm_c, c_stride);
            }
        }
        thread_barrier.wait(thread_size);

        // All reduce for inner-product
        // better way to do this?
        // change inner-loop to axpy?
        if (thread_rank == 0) {
            for (int rank = 1; rank < thread_size; ++rank)
                for (int64_t j = 0; j < nb; ++j)
                    for (int64_t i = 0; i < kb; ++i)
                        W.at(0).data()[i+j*(kb)] += W.at(rank).data()[i+j*(kb)];

            // copy needed data for constructing rectangular block of T
            for (int64_t j = k; j < k+kb; ++j)
                for (int64_t i = 0; i < j; ++i)
                    T.at(i,j) = conj(W.at(0).data()[j-k+(i)*(kb)]);

            // Construct diagonal ib-block of T
            for (int64_t j = k; j < k+kb; ++j) {
                T.at(j,j) = taus.at(j);
                if (j > k) {
                    blas::scal(j-k, -T.at(j, j), &T.at(k, j), 1);

                    blas::trmv(Layout::ColMajor,
                               Uplo::Upper, Op::NoTrans, Diag::NonUnit,
                               j-k,
                               &T.at(k, k), T.stride(),
                               &T.at(k, j), 1);
                }
            }

            // Finish construction of rectangular block of T
            // T(1:k-1,k:k+kb) = - T(1:k-1,1:k-1) * W(1:k-1,1:kb)^H * T(k:k+kb,k:k+kb)
            // where W(1:k-1,1:kb) = V(k:m,k:k+kb)^H * V(k:m,1:k-1)
            if (k > 0) {
                blas::trmm(Layout::ColMajor,
                           Side::Right, Uplo::Upper,
                           Op::NoTrans, Diag::NonUnit,
                           k, kb,
                           -one, &T.at(k, k), T.stride(),
                                 &T.at(0, k), T.stride());

                blas::trmm(Layout::ColMajor,
                           Side::Left, Uplo::Upper,
                           Op::NoTrans, Diag::NonUnit,
                           k, kb,
                           one, &T.at(0, 0), T.stride(),
                                &T.at(0, k), T.stride());
            }
        }
        thread_barrier.wait(thread_size);

        // Update trailing sub-matrix using
        //       the GEMM performed earlier
        if (k+kb < nb) {

            // Apply blocking factor T
            if (thread_rank == 0) {
                // W(1:kb,1:nb-k-kb) = T(k:k+kb,k:k+kb)^H * W(1:kb,1:nb-k-kb)
                // where originally W(1:kb,1:nb-k-kb) = V(k:m,k:k+kb)^H A(k:m,k+kb+1:nb)
                blas::trmm(Layout::ColMajor,
                           Side::Left, Uplo::Upper,
                           Op::ConjTrans, Diag::NonUnit,
                           kb, nb-k-kb,
                           one, &T.at(k, k), T.stride(),
                                &W.at(0).data()[(k+kb)*(kb)], kb);
            }
            thread_barrier.wait(thread_size);

            // Finish projection:
            for (int64_t idx = thread_rank;
                 idx < int64_t(tiles.size());
                 idx += thread_size)
            {
                auto tile = tiles.at(idx);
                auto i_index = tile_indices.at(idx);

                // A(k:m,k+kb:nb) = A(k:m,k+kb:nb) - V(k:m,k:k+kb) * W(1:kb,1:nb-k+kb)
                // where W(1:kb,1:nb-k-kb) = (T(k:k+kb,k:k+kb)^H * V(k:m,k:k+kb)^H * A(k:m,k+kb+1:nb))
                if (i_index == tile_indices.at(0)) {
                    if (k+kb < tile.mb()) {
                        blas::gemm(Layout::ColMajor,
                                   Op::NoTrans, Op::NoTrans,
                                   tile.mb()-k-kb, nb-k-kb, kb,
                                   -one, &tile.at(k+kb, k), tile.stride(),
                                         &W.at(0).data()[(k+kb)*(kb)], kb,
                                    one, &tile.at(k+kb, k+kb), tile.stride());
                    }
                }
                else {
                    blas::gemm(Layout::ColMajor,
                               Op::NoTrans, Op::NoTrans,
                               tile.mb(), nb-k-kb, kb,
                               -one, &tile.at(0, k), tile.stride(),
                                     &W.at(0).data()[(k+kb)*(kb)], kb,
                                one, &tile.at(0, k+kb), tile.stride());
                }
            }
            thread_barrier.wait(thread_size);

            if (thread_rank == 0) {
                // Needed due to breaking up the projection
                // by triangular block and lower-rectangular
                // block. This finishes the projection.
                auto& tile = diag_tile;
                blas::trmm(Layout::ColMajor,
                           Side::Left, Uplo::Lower,
                           Op::NoTrans, Diag::Unit,
                           kb, nb-k-kb,
                           one, &tile.at(k, k), tile.stride(),
                                &W.at(thread_rank).data()[(k+kb)*kb], kb);

                for (int64_t j = 0; j < nb-k-kb; ++j)
                    for (int64_t i = 0; i < kb; ++i)
                        tile.at(k+i, k+kb+j) -=
                            W.at(thread_rank).data()[i+(j+k+kb)*(kb)];
            }
        }
    }
}

} // namespace internal
} // namespace slate

#endif // SLATE_TILE_GEQRF_HH
