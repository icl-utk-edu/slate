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

// specialization namespace differentiates, e.g.,
// internal::syrk from internal::specialization::syrk
namespace internal {
namespace specialization {

///-----------------------------------------------------------------------------
/// \brief
/// Distributed parallel matrix multiplication.
/// Generic implementation for any target.
/// Dependencies enforce the following behavior:
/// - bcast communications are serialized,
/// - syrk operations are serialized,
/// - bcasts can get ahead of syrks by the value of lookahead.
// Note A and C are passed by value, so we can transpose if needed
// (for uplo = Upper) without affecting caller.
template <Target target, typename scalar_t>
void syrk(slate::internal::TargetType<target>,
          scalar_t alpha, Matrix<scalar_t> A,
          scalar_t beta,  SymmetricMatrix<scalar_t> C,
          int64_t lookahead)
{
    using namespace blas;

    // if upper, change to lower
    if ((C.uplo() == Uplo::Upper && C.op() == Op::NoTrans) ||
        (C.uplo() == Uplo::Lower && C.op() != Op::NoTrans)) {
        C = transpose(C);
    }

    // A is mt-by-nt, C is mt-by-mt
    assert(A.mt() == C.mt());

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > bcast_vector( A.nt() );
    std::vector< uint8_t >  gemm_vector( A.nt() );
    uint8_t *bcast = bcast_vector.data();
    uint8_t *gemm  =  gemm_vector.data();

    if (target == Target::Devices) {
        C.allocateBatchArrays();
        C.reserveDeviceWorkspace();
    }

    #pragma omp parallel
    #pragma omp master
    {
        // Lower/NoTrans or Upper/Trans case
        // send 1st block col of A
        #pragma omp task depend(out:bcast[0])
        {
            // broadcast A(i, 0) to ranks owning block row C(i, 0:i) and block col C(i:n, i)
            for (int64_t i = 0; i < A.mt(); ++i) {
                A.template tileBcast<target>(
                    i, 0, C.sub(i, i, 0, i),
                          C.sub(i, C.mt()-1, i, i));
            }
        }

        // send next lookahead block cols of A
        for (int64_t k = 1; k < lookahead+1 && k < A.nt(); ++k) {
            #pragma omp task depend(in:bcast[k-1]) \
                             depend(out:bcast[k])
            {
                // broadcast A(i, k) to ranks owning block row C(i, 0:i) and block col C(i:n, i)
                for (int64_t i = 0; i < A.mt(); ++i) {
                    A.template tileBcast<target>(
                        i, k, C.sub(i, i, 0, i),
                              C.sub(i, C.mt()-1, i, i));
                }
            }
        }

        // multiply alpha A(:, 0) A(0, :)^T + beta C
        #pragma omp task depend(in:bcast[0]) \
                         depend(out:gemm[0])
        {
            internal::syrk<target>(
                alpha, A.sub(0, A.mt()-1, 0, 0),
                beta,  std::move(C));
        }

        for (int64_t k = 1; k < A.nt(); ++k) {

            // send next block col of A and block row of B
            if (k+lookahead < A.nt()) {
                #pragma omp task depend(in:gemm[k-1]) \
                                 depend(in:bcast[k+lookahead-1]) \
                                 depend(out:bcast[k+lookahead])
                {
                    // broadcast A(k+la, i) to ranks owning block row C(i, 0:i) and block col C(i:n, i)
                    for (int64_t i = 0; i < A.mt(); ++i) {
                        A.template tileBcast<target>(
                            i, k+lookahead, C.sub(i, i, 0, i),
                                            C.sub(i, C.mt()-1, i, i));
                    }
                }
            }

            // multiply alpha A(:, k) A(k, :) + C, no beta
            #pragma omp task depend(in:bcast[k]) \
                             depend(in:gemm[k-1]) \
                             depend(out:gemm[k])
            {
                internal::syrk<target>(
                    alpha,         A.sub(0, A.mt()-1, k, k),
                    scalar_t(1.0), std::move(C));
            }
        }
    }

    // todo: we need a function that updates origins that are not valid
    for (int64_t j = 0; j < C.nt(); ++j)
        for (int64_t i = j; i < C.mt(); ++i)  // lower
            if (C.tileIsLocal(i, j))
                C.tileMoveToHost(i, j, C.tileDevice(i, j));

    C.clearWorkspace();
}

} // namespace specialization
} // namespace internal

///-----------------------------------------------------------------------------
/// \brief
///
/// Precision and target templated function.
template <Target target, typename scalar_t>
void syrk(scalar_t alpha, Matrix<scalar_t>& A,
          scalar_t beta,  SymmetricMatrix<scalar_t>& C,
          const std::map<Option, Value>& opts)
{
    int64_t lookahead;
    try {
        lookahead = opts.at(Option::Lookahead).i_;
    }
    catch (std::out_of_range) {
        lookahead = 1;
    }

    internal::specialization::syrk(internal::TargetType<target>(),
                                   alpha, A,
                                   beta,  C,
                                   lookahead);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void syrk< Target::HostTask, float >(
    float alpha, Matrix<float>& A,
    float beta,  SymmetricMatrix<float>& C,
    const std::map<Option, Value>& opts);

template
void syrk< Target::HostNest, float >(
    float alpha, Matrix<float>& A,
    float beta,  SymmetricMatrix<float>& C,
    const std::map<Option, Value>& opts);

template
void syrk< Target::HostBatch, float >(
    float alpha, Matrix<float>& A,
    float beta,  SymmetricMatrix<float>& C,
    const std::map<Option, Value>& opts);

template
void syrk< Target::Devices, float >(
    float alpha, Matrix<float>& A,
    float beta,  SymmetricMatrix<float>& C,
    const std::map<Option, Value>& opts);

// ----------------------------------------
template
void syrk< Target::HostTask, double >(
    double alpha, Matrix<double>& A,
    double beta,  SymmetricMatrix<double>& C,
    const std::map<Option, Value>& opts);

template
void syrk< Target::HostNest, double >(
    double alpha, Matrix<double>& A,
    double beta,  SymmetricMatrix<double>& C,
    const std::map<Option, Value>& opts);

template
void syrk< Target::HostBatch, double >(
    double alpha, Matrix<double>& A,
    double beta,  SymmetricMatrix<double>& C,
    const std::map<Option, Value>& opts);

template
void syrk< Target::Devices, double >(
    double alpha, Matrix<double>& A,
    double beta,  SymmetricMatrix<double>& C,
    const std::map<Option, Value>& opts);

// ----------------------------------------
template
void syrk< Target::HostTask,  std::complex<float>  >(
    std::complex<float> alpha, Matrix< std::complex<float> >& A,
    std::complex<float> beta,  SymmetricMatrix< std::complex<float> >& C,
    const std::map<Option, Value>& opts);

template
void syrk< Target::HostNest, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >& A,
    std::complex<float> beta,  SymmetricMatrix< std::complex<float> >& C,
    const std::map<Option, Value>& opts);

template
void syrk< Target::HostBatch, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >& A,
    std::complex<float> beta,  SymmetricMatrix< std::complex<float> >& C,
    const std::map<Option, Value>& opts);

template
void syrk< Target::Devices, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >& A,
    std::complex<float> beta,  SymmetricMatrix< std::complex<float> >& C,
    const std::map<Option, Value>& opts);

// ----------------------------------------
template
void syrk< Target::HostTask, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >& A,
    std::complex<double> beta,  SymmetricMatrix< std::complex<double> >& C,
    const std::map<Option, Value>& opts);

template
void syrk< Target::HostNest, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >& A,
    std::complex<double> beta,  SymmetricMatrix< std::complex<double> >& C,
    const std::map<Option, Value>& opts);

template
void syrk< Target::HostBatch, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >& A,
    std::complex<double> beta,  SymmetricMatrix< std::complex<double> >& C,
    const std::map<Option, Value>& opts);

template
void syrk< Target::Devices, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >& A,
    std::complex<double> beta,  SymmetricMatrix< std::complex<double> >& C,
    const std::map<Option, Value>& opts);

} // namespace slate
