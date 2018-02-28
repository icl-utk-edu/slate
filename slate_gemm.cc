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

#include "slate.hh"
#include "slate_Debug.hh"
#include "slate_Matrix.hh"
#include "slate_internal.hh"

namespace slate {

namespace internal_specialization {

///-----------------------------------------------------------------------------
/// \brief
/// Distributed parallel matrix multiplication.
/// Generic implementation for any target.
/// Dependencies enforce the following behavior:
/// - bcast communications are serialized,
/// - gemm operations are serialized,
/// - bcasts can get ahead of gemms by the value of lookahead.
template <Target target, typename scalar_t>
void gemm(slate::internal::TargetType<target>,
          blas::Op transA, blas::Op transB,
          scalar_t alpha, Matrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
          int64_t lookahead)
{
    using namespace blas;

    uint8_t *bcast = new uint8_t[A.nt()];
    uint8_t *gemm  = new uint8_t[A.nt()];

    #pragma omp parallel
    #pragma omp master
    {
        #pragma omp task depend(out:bcast[0])
        {
            for (int64_t i = 0; i < A.mt(); ++i)
                A.tileBcast(i, 0, C.sub(i, i, 0, C.nt()-1));

            for (int64_t j = 0; j < B.nt(); ++j)
                B.tileBcast(0, j, C.sub(0, C.mt()-1, j, j));
        }

        for (int64_t k = 1; k < lookahead+1 && k < A.nt(); ++k)
            #pragma omp task depend(in:bcast[k-1]) \
                             depend(out:bcast[k])
            {
                for (int64_t i = 0; i < A.mt(); ++i)
                    A.tileBcast(i, k, C.sub(i, i, 0, C.nt()-1));

                for (int64_t j = 0; j < B.nt(); ++j)
                    B.tileBcast(k, j, C.sub(0, C.mt()-1, j, j));
            }

        #pragma omp task depend(in:bcast[0]) \
                         depend(out:gemm[0])
        internal::gemm<Target::HostTask>(
            alpha, A.sub(0, A.mt()-1, 0, 0),
                   B.sub(0, 0, 0, B.nt()-1),
            beta,  C.sub(0, C.mt()-1, 0, C.nt()-1));

        for (int64_t k = 1; k < A.nt(); ++k) {

            if (k+lookahead < A.nt())
                #pragma omp task depend(in:gemm[k-1]) \
                                 depend(in:bcast[k+lookahead-1]) \
                                 depend(out:bcast[k+lookahead])
                {
                    for (int64_t i = 0; i < A.mt(); ++i)
                        A.tileBcast(i, k+lookahead, C.sub(i, i, 0, C.nt()-1));

                    for (int64_t j = 0; j < B.nt(); ++j)
                        B.tileBcast(k+lookahead, j, C.sub(0, C.mt()-1, j, j));
                }

            #pragma omp task depend(in:bcast[k]) \
                             depend(in:gemm[k-1]) \
                             depend(out:gemm[k])
            internal::gemm<Target::HostTask>(
                alpha, A.sub(0, A.mt()-1, k, k),
                       B.sub(k, k, 0, B.nt()-1),
                1.0,   C.sub(0, C.mt()-1, 0, C.nt()-1));
        }
    }

    A.clearWorkspace();
    B.clearWorkspace();

    delete[] bcast;
    delete[] gemm;
}

///-----------------------------------------------------------------------------
/// \brief
/// Distributed parallel matrix multiplication.
/// Implementation for GPU device target.
template <typename scalar_t>
void gemm(slate::internal::TargetType<Target::Devices>,
          blas::Op transA, blas::Op transB,
          scalar_t alpha, Matrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
          int64_t lookahead)
{

}

} // namespace internal_specialization

///-----------------------------------------------------------------------------
/// \brief
///
/// Precision and target templated function.
template <Target target, typename scalar_t>
void gemm(blas::Op transA, blas::Op transB,
          scalar_t alpha, Matrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
          int64_t lookahead)
{
    internal_specialization::gemm(internal::TargetType<target>(),
                                  transA, transB,
                                  alpha, A,
                                         B,
                                  beta,  C,
                                  lookahead);
}

//------------------------------------------------------------------------------
// Explicit instantiations for double precision and various targets.
template
void gemm< Target::HostTask, double >(
    blas::Op transA, blas::Op transB,
    double alpha, Matrix<double>& A,
                  Matrix<double>& B,
    double beta,  Matrix<double>& C,
    int64_t lookahead);

template
void gemm< Target::HostNest, double >(
    blas::Op transA, blas::Op transB,
    double alpha, Matrix<double>& A,
                  Matrix<double>& B,
    double beta,  Matrix<double>& C,
    int64_t lookahead);

template
void gemm< Target::HostBatch, double >(
    blas::Op transA, blas::Op transB,
    double alpha, Matrix<double>& A,
                  Matrix<double>& B,
    double beta,  Matrix<double>& C,
    int64_t lookahead);

template
void gemm< Target::Devices, double >(
    blas::Op transA, blas::Op transB,
    double alpha, Matrix<double>& A,
                  Matrix<double>& B,
    double beta,  Matrix<double>& C,
    int64_t lookahead);

} // namespace slate
