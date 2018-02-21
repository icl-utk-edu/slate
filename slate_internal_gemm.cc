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
// Need assistance with the SLATE software? Join the "SLATE User" Google group
// by going to https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user
// and clicking "Apply to join group". Upon acceptance, email your questions and
// comments to <slate-user@icl.utk.edu>.
//------------------------------------------------------------------------------

#include "slate_Matrix.hh"
#include "slate_types.hh"
#include "slate_Tile_blas.hh"

namespace slate {
namespace internal {

///-----------------------------------------------------------------------------
/// \brief
/// General matrix multiply to update trailing matrix,
/// where A is a single block column and B is a single block row.
/// Dispatches to target implementations.
template <Target target, typename scalar_t>
void gemm(scalar_t alpha, Matrix< scalar_t >&& A,
                          Matrix< scalar_t >&& B,
          scalar_t beta,  Matrix< scalar_t >&& C,
          int priority)
{
    gemm(internal::TargetType<target>(),
         alpha, A,
                B,
         beta,  C,
         priority);
}

///-----------------------------------------------------------------------------
/// \brief
/// General matrix multiply to update trailing matrix,
/// where A is a single block column and B is a single block row.
/// Host OpenMP task implementation.
template <typename scalar_t>
void gemm(internal::TargetType<Target::HostTask>,
          scalar_t alpha, Matrix< scalar_t >& A,
                          Matrix< scalar_t >& B,
          scalar_t beta,  Matrix< scalar_t >& C,
          int priority)
{
    // check dimensions
    assert(A.nt() == 1);
    assert(B.mt() == 1);
    assert(A.mt() == C.mt());
    assert(B.nt() == C.nt());

    for (int64_t i = 0; i < C.mt(); ++i)
        for (int64_t j = 0; j < C.nt(); ++j)
            if (C.tileIsLocal(i, j))
                #pragma omp task shared(A, B, C) priority(priority)
                {
                    A.tileCopyToHost(i, 0, A.tileDevice(i, 0));
                    B.tileCopyToHost(j, 0, B.tileDevice(j, 0));
                    C.tileMoveToHost(i, j, C.tileDevice(i, j));
                    gemm(alpha, A(i, 0),
                                B(0, j),
                         beta,  C(i, j));
                    A.tileTick(i, 0);
                    B.tileTick(j, 0);
                }

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gemm< Target::HostTask, double >(
    double alpha, Matrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  Matrix<double>&& C,
    int priority);

} // namespace internal
} // namespace slate
