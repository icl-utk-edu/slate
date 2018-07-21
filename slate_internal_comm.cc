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

#include "slate_internal_comm.hh"
#include "slate_internal_util.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// [internal]
/// Implements a hypercube broadcast pattern. For a given rank, finds the rank
/// to receive from and the list of ranks to forward to. Assumes rank 0 as the
/// root of the broadcast.
///
/// @param[in] size
///     Number of ranks participating in the broadcast.
///
/// @param[in] rank
///     Rank of the local process.
///
/// @param[in] radix
///     Dimension of the cube.
///
/// @param[out] recv_rank
///     List containing the the rank to receive from.
///     Empty list for rank 0.
///
/// @param[out] send_to
///     List of ranks to forward to.
///
void cubeBcastPattern(int size, int rank, int radix,
                      std::list<int>& recv_from, std::list<int>& send_to)
{
    //-------------------------------------------
    // Find the cube's and the rank's attributes:

    // Find the number of cube's dimensions.
    int num_dimensions = 1;
    int max_rank = size-1;
    while ((max_rank /= radix) > 0)
        ++num_dimensions;

    int dimension; // how deep the rank is in the cube
    int position;  // position in the last dimension
    int stride;    // stride of the last dimension

    // Find the rank's dimension, position, and stride.
    int radix_pow = pow(radix, num_dimensions-1);
    dimension = 0;
    while (rank%radix_pow != 0) {
        ++dimension;
        radix_pow /= radix;
    }
    stride = pow(radix, num_dimensions-dimension-1);
    position = rank%pow(radix, num_dimensions-dimension)/stride;

    //--------------------------------------
    // Find the origin and the destinations.

    // Unless root, receive from the predecessor.
    if (rank != 0)
        recv_from.push_back(rank-stride);

    // If not on the edge and successor exists, send to the successor.
    if (position < radix-1 && rank+stride < size)
        send_to.push_back(rank+stride);

    // Forward to all lower dimensions.
    for (int dim = dimension+1; dim < num_dimensions; ++dim) {
        stride /= radix;
        if (rank+stride < size)
            send_to.push_back(rank+stride);
    }
}

} // namespace internal
} // namespace slate
