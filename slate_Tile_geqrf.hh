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

#ifndef SLATE_TILE_GEQRF_HH
#define SLATE_TILE_GEQRF_HH

#include "slate_internal.hh"
#include "slate_Tile.hh"
#include "slate_Tile_blas.hh"
#include "slate_Tile_lapack.hh"
#include "slate_types.hh"
#include "slate_util.hh"

#include <cmath>
#include <list>
#include <vector>

#include <blas.hh>
#include <lapack.hh>

namespace slate {
namespace internal {
// todo: Perhaps we should put all Tile routines in "internal".

//-----------------------------------
float real(float val) { return val; }
double real(double val) { return val; }
float real(std::complex<float> val) { return val.real(); }
double real(std::complex<double> val) { return val.real(); }

//-----------------------------------
float imag(float val) { return 0.0; }
double imag(double val) { return 0.0; }
float imag(std::complex<float> val) { return val.imag(); }
double imag(std::complex<double> val) { return val.imag(); }

//--------------------------
template <typename scalar_t>
scalar_t make(blas::real_type<scalar_t> real, blas::real_type<scalar_t> imag);

template <>
float make<float>(float real, float imag) { return real; }

template <>
double make<double>(double real, double imag) { return real; }

template <>
std::complex<float> make<std::complex<float>>(float real, float imag)
{
    return std::complex<float>(real, imag);
}

template <>
std::complex<double> make<std::complex<double>>(double real, double imag)
{
    return std::complex<double>(real, imag);
}

///-----------------------------------------------------------------------------
/// \brief
/// Compute the QR factorization of a panel.
///
/// \param[in] diag_len
///     length of the panel diagonal
///
/// \param[in] ib
///     internal blocking in the panel
///
/// \param[inout] tiles
///     local tiles in the panel
///
/// \param[in] tile_indices
///     i indices of the tiles in the panel
///
/// \param[out] T
///     uppert triangular factor of the block reflector
///
/// \param[in] thread_rank
///     rank of this thread
///
/// \param[in] thread_size
///     number of local threads
///
/// \param[in] thread_barrier
///     barrier for synchronizing local threads
///
/// todo: add missing params
///
template <typename scalar_t>
void geqrf(
    int64_t diag_len, int64_t ib,
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

    using namespace blas;
    using namespace lapack;
    using real_t = real_type<scalar_t>;

    Tile<scalar_t>& diag_tile = tiles.at(0);
    const int64_t nb = diag_tile.nb();

    // Loop over ib-wide stripes.
    for (int64_t k = 0; k < diag_len; k += ib) {

        // ib panel factorization
        int64_t kb = std::min(diag_len-k, ib);

        // Loop over ib columns of a stripe.
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
                if (i_index == 0) {
                    if (j+1 < diag_len)
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
                    add_sumsq(scale[0], sumsq[0], scale[rank], sumsq[rank]);
                }
                xnorm = scale[0]*std::sqrt(sumsq[0]);
                diag_tile.at(j, j) = scalar_t(1.0);
            }
            thread_barrier.wait(thread_size);

            real_t beta =
                -std::copysign(lapack::lapy3(alphr, alphi, xnorm), alphr);
            // todo: IF( ABS( BETA ).LT.SAFMIN ) THEN

            // todo: Use overflow-safe division (see CLADIV/ZLADIV)
            scalar_t scal_alpha = scalar_t(1.0) / (alpha-beta);
            scalar_t tau = make<scalar_t>((beta-alphr)/beta, -alphi/beta);
            scalar_t ger_alpha = -conj(tau);

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
                if (i_index == 0) {
                    // diagonal tile
                    if (j+1 < diag_len)
                        blas::scal(tile.mb()-j-1,
                                   scal_alpha, &tile.at(j+1, j), 1);
                }
                else {
                    // off diagonal tiles
                    blas::scal(tile.mb(), scal_alpha, &tile.at(0, j), 1);
                }

                // thread local W
                if (j+1 < diag_len) {
                    if (i_index == 0) {
                        // diagonal tile
                        blas::gemv(Layout::ColMajor, Op::ConjTrans,
                                   tile.mb()-j, nb-j-1,
                                   scalar_t(1.0), &tile.at(j, j+1), tile.stride(),
                                                  &tile.at(j, j), 1,
                                   gemv_beta,     W.at(thread_rank).data(), 1);
                    }
                    else {
                        // off diagonal tile
                        blas::gemv(Layout::ColMajor, Op::ConjTrans,
                                   tile.mb(), nb-j-1,
                                   scalar_t(1.0), &tile.at(0, j+1), tile.stride(),
                                                  &tile.at(0, j), 1,
                                   gemv_beta,     W.at(thread_rank).data(), 1);
                    }
                }
            }
            thread_barrier.wait(thread_size);

            //-------------------
            // global W reduction
            if (thread_rank == 0) {
                for (int rank = 1; rank < thread_size; ++rank)
                    blas::axpy(nb-j-1, scalar_t(1.0),
                               W.at(rank).data(), 1,
                               W.at(0).data(), 1);
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

                    if (i_index == 0) {
                        // diagonal tile
                        blas::ger(Layout::ColMajor,
                                  tile.mb()-j, nb-j-1,
                                  ger_alpha, &tile.at(j, j), 1,
                                             W.at(0).data(), 1,
                                             &tile.at(j, j+1), tile.stride());
                    }
                    else {
                        // off diagonal tile
                        blas::ger(Layout::ColMajor,
                                  tile.mb(), nb-j-1,
                                  ger_alpha, &tile.at(0, j), 1,
                                             W.at(0).data(), 1,
                                             &tile.at(0, j+1), tile.stride());
                    }
                }
            }
            thread_barrier.wait(thread_size);

            //-----------------------
            // column of T local gemm
            if (j > 0) {
                for (int64_t idx = thread_rank;
                     idx < int64_t(tiles.size());
                     idx += thread_size)
                {
                    auto tile = tiles.at(idx);
                    auto i_index = tile_indices.at(idx);
                    scalar_t gemv_beta = idx == thread_rank ? 0.0 : 1.0;

                    if (i_index == 0) {
                        // diagonal tile
                        blas::gemv(Layout::ColMajor, Op::ConjTrans,
                                   tile.mb()-j, j,
                                   -tau,       &tile.at(j, 0), tile.stride(),
                                               &tile.at(j, j), 1,
                                    gemv_beta, W.at(thread_rank).data(), 1);
                    }
                    else {
                        // off diagonal tile
                        blas::gemv(Layout::ColMajor, Op::ConjTrans,
                                   tile.mb(), j,
                                   -tau,       &tile.at(0, 0), tile.stride(),
                                               &tile.at(0, j), 1,
                                    gemv_beta, W.at(thread_rank).data(), 1);
                    }
                }
            }
            thread_barrier.wait(thread_size);

            //-------------------------
            // gemv reducttion and trmv
            if (thread_rank == 0) {
                if (j > 0) {
                    for (int rank = 1; rank < thread_size; ++rank)
                        blas::axpy(j, scalar_t(1.0),
                                   W.at(rank).data(), 1,
                                   W.at(0).data(), 1);

                    memcpy(&T.at(0, j), W.at(0).data(), sizeof(scalar_t)*j);

                    blas::trmv(Layout::ColMajor,
                               Uplo::Upper, Op::NoTrans, Diag::NonUnit,
                               j,
                               &T.at(0, 0), T.stride(),
                               &T.at(0, j), 1);
                }
            }

            //---------------------
            // set tau and diagonal
            if (thread_rank == 0) {
                T.at(j, j) = tau;
                diag_tile.at(j, j) = beta;
            }
        }

        // If there is a trailing submatrix.
        if (k+kb < nb) {

        }
    }
}

} // namespace internal
} // namespace slate

#endif // SLATE_TILE_GEQRF_HH
