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

#include <list>
#include <vector>

#include <blas.hh>
#include <lapack.hh>

namespace slate {
namespace internal {
// todo: Perhaps we should put all Tile routines in "internal".

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
/// \param[in] thread_rank
///     rank of this thread
///
/// \param[in] thread_size
///     number of local threads
///
/// \param[in] thread_barrier
///     barrier for synchronizing local threads
///
template <typename scalar_t>
void geqrf(
    int64_t diag_len, int64_t ib,
    std::vector< Tile<scalar_t> >& tiles,
    std::vector<int64_t>& tile_indices,
    int thread_rank, int thread_size,
    ThreadBarrier& thread_barrier)
{
    trace::Block trace_block("lapack::geqrf");

    using namespace blas;
    using namespace lapack;
    using real_t = real_type<scalar_t>;

    const int64_t nb = tiles.at(0).nb();

    std::vector<scalar_t> tau(std::min(tiles.at(0).nb(), tiles.at(0).mb()));
    lapack::geqrf(tiles.at(0).mb(), tiles.at(0).nb(),
                  tiles.at(0).data(), tiles.at(0).stride(), tau.data());
/*
    // Loop over ib-wide stripes.
    for (int64_t k = 0; k < diag_len; k += ib) {

        // ib panel factorization
        int64_t kb = std::min(diag_len-k, ib);

        // Loop over ib columns of a stripe.
        for (int64_t j = k; j < k+kb; ++j) {

        }

        // If there is a trailing submatrix.
        if (k+kb < nb) {

        }
    }
*/
}

} // namespace internal
} // namespace slate

#endif // SLATE_TILE_GEQRF_HH
