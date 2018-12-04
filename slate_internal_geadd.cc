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
#include "slate_TriangularMatrix.hh"
#include "slate_types.hh"
#include "slate_Tile_blas.hh"
#include "slate_internal.hh"

namespace slate {
namespace internal {

///-----------------------------------------------------------------------------
/// \brief
/// General matrix add.
/// Dispatches to target implementations.
template <Target target, typename scalar_t>
void geadd(scalar_t alpha, Matrix<scalar_t>&& A,
           scalar_t beta, Matrix<scalar_t>&& B,
           int priority)
{
    geadd(internal::TargetType<target>(),
          alpha, A,
          beta,  B,
          priority);
}

///-----------------------------------------------------------------------------
/// \brief
/// General matrix add.
/// assumes A & B have same tile layout and dimensions, and have same distribution
/// TODO handle transpose A case
/// Host OpenMP task implementation.
template <typename scalar_t>
void geadd(internal::TargetType<Target::HostTask>,
           scalar_t alpha, Matrix<scalar_t>& A,
           scalar_t beta, Matrix<scalar_t>& B,
           int priority)
{
    // trace::Block trace_block("geadd");

    int64_t A_mt = A.mt();
    int64_t A_nt = A.nt();
    assert(A_mt == B.mt());
    assert(A_nt == B.nt());

    for (int64_t i = 0; i < A_mt; ++i) {
        for (int64_t j = 0; j < A_nt; ++j) {
            if (B.tileIsLocal(i, j)) {
                #pragma omp task shared(A, B) priority(priority)
                {
                    A.tileCopyToHost(i, j, A.tileDevice(i, j));
                    B.tileMoveToHost(i, j, B.tileDevice(i, j));
                    axby(alpha, A(i, j),
                         beta,  B(i, j));
                    A.tileTick(i, j);// TODO is this correct here?
                }
            }
        }
    }

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void geadd<Target::HostTask, float>(
    float alpha, Matrix<float>&& A,
    float beta, Matrix<float>&& B,
    int priority);

// ----------------------------------------
template
void geadd<Target::HostTask, double>(
    double alpha, Matrix<double>&& A,
    double beta, Matrix<double>&& B,
    int priority);

// ----------------------------------------
template
void geadd< Target::HostTask, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
    std::complex<float>  beta, Matrix< std::complex<float> >&& B,
    int priority);

// ----------------------------------------
template
void geadd< Target::HostTask, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
    std::complex<double> beta, Matrix< std::complex<double> >&& B,
    int priority);

} // namespace internal
} // namespace slate
