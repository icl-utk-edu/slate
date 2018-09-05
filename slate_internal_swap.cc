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

#include "slate_Matrix.hh"
#include "slate_types.hh"
#include "slate_Tile_blas.hh"
#include "slate_internal.hh"

namespace slate {
namespace internal {

///-----------------------------------------------------------------------------
/// \brief
/// Swaps rows according to the pivot vector.
/// Dispatches to target implementations.
template <Target target, typename scalar_t>
void swap(Matrix<scalar_t>&& A, std::vector<Pivot>& pivot,
          int priority, int tag)
{
    swap(internal::TargetType<target>(), A, pivot, priority, tag);
}

///-----------------------------------------------------------------------------
/// \brief
/// Swaps rows according to the pivot vector, host implementation.
template <typename scalar_t>
void swap(internal::TargetType<Target::HostTask>,
          Matrix<scalar_t>& A, std::vector<Pivot>& pivot,
          int priority, int tag)
{
    for (int64_t i = 0; i < A.mt(); ++i) {
        for (int64_t j = 0; j < A.nt(); ++j) {
            if (A.tileIsLocal(i, j)) {
                #pragma omp task shared(A) priority(priority)
                {
                    A.tileMoveToHost(i, j, A.tileDevice(i, j));
                }
            }
        }
    }
    #pragma omp taskwait

    {
        trace::Block trace_block("internal::swap");

        for (int64_t j = 0; j < A.nt(); ++j) {
            bool root = A.mpiRank() == A.tileRank(0, j);

            for (int64_t i = 0; i < pivot.size(); ++i) {
                int pivot_rank = A.tileRank(pivot[i].tileIndex(), j);

                // If I own the pivot.
                if (pivot_rank == A.mpiRank()) {
                    // If I am the root.
                    if (root) {
                        // If pivot not on the diagonal.
                        if (pivot[i].tileIndex() > 0 ||
                            pivot[i].elementOffset() > i)
                        {
                            swap(0, A.tileNb(j),
                                 A(0, j), i,
                                 A(pivot[i].tileIndex(), j),
                                 pivot[i].elementOffset());
                        }
                    }
                    // I am not the root.
                    else {
                        // MPI swap with the root
                        swap(0, A.tileNb(j),
                             A(pivot[i].tileIndex(), j),
                             pivot[i].elementOffset(),
                             A.tileRank(0, j), A.mpiComm(),
                             tag);
                    }
                }
                // I don't own the pivot.
                else {
                    // If I am the root.
                    if (root) {
                        // MPI swap with the pivot owner
                        swap(0,  A.tileNb(j),
                             A(0, j), i,
                             pivot_rank, A.mpiComm(),
                             tag);
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void swap<Target::HostTask, float>(
    Matrix<float>&& A, std::vector<Pivot>& pivot,
    int priority, int tag);

// ----------------------------------------
template
void swap<Target::HostTask, double>(
    Matrix<double>&& A, std::vector<Pivot>& pivot,
    int priority, int tag);

// ----------------------------------------
template
void swap< Target::HostTask, std::complex<float> >(
    Matrix< std::complex<float> >&& A,
    std::vector<Pivot>& pivot,
    int priority, int tag);

// ----------------------------------------
template
void swap< Target::HostTask, std::complex<double> >(
    Matrix< std::complex<double> >&& A,
    std::vector<Pivot>& pivot,
    int priority, int tag);

} // namespace internal
} // namespace slate
