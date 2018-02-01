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

namespace slate {

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
template <Target target>
void Matrix<scalar_t>::gemm(blas::Op opa, blas::Op opb,
                             scalar_t alpha, Matrix &&a,
                                              Matrix &&b,
                             scalar_t beta,  Matrix &&c,
                             int priority)
{
    gemm(internal::TargetType<target>(),
         opa, opb,
         alpha, a,
                b,
         beta,  c);
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
void Matrix<scalar_t>::gemm(internal::TargetType<Target::HostTask>,
                             blas::Op opa, blas::Op opb,
                             scalar_t alpha, Matrix &a,
                                              Matrix &b,
                             scalar_t beta,  Matrix &c,
                             int priority)
{
    // NoTrans, Trans
    for (int m = 0; m < c.mt_; ++m)
        for (int n = 0; n < c.nt_; ++n)
            if (c.tileIsLocal(m, n))
                #pragma omp task shared(a, b, c) priority(priority)
                {
                    a.tileCopyToHost(m, 0, a.tileDevice(m, 0));
                    b.tileCopyToHost(n, 0, b.tileDevice(n, 0));
                    c.tileMoveToHost(m, n, c.tileDevice(m, n));
                    Tile<scalar_t>::gemm(opa, opb,
                                          alpha, a(m, 0),
                                                 b(n, 0),
                                          beta,  c(m, n));
                    a.tileTick(m, 0);
                    b.tileTick(n, 0);
                }

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
template
void Matrix<double>::gemm<Target::HostTask>(
    blas::Op opa, blas::Op opb,
    double alpha, Matrix &&a,
                  Matrix &&b,
    double beta,  Matrix &&c,
    int priority);

} // namespace slate
