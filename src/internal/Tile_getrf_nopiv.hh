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
/// Compute the LU factorization of a panel.
///
/// @param[in] ib
///     internal blocking in the panel
///
/// @param[in,out] tile
///     tile to factor
///
/// @ingroup getrf_nopiv_tile
///
template <typename scalar_t>
void getrf_nopiv(Tile<scalar_t> tile, int64_t ib)
{
    const scalar_t one = 1.0;
    const int64_t nb = tile.nb();
    const int64_t mb = tile.mb();
    const int64_t diag_len = std::min(nb, mb);

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
