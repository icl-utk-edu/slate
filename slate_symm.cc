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
// internal::symm from internal::specialization::symm
namespace internal {
namespace specialization {

///-----------------------------------------------------------------------------
/// \brief
/// Distributed parallel matrix multiplication.
/// Generic implementation for any target.
/// Dependencies enforce the following behavior:
/// - bcast communications are serialized,
/// - symm operations are serialized,
/// - bcasts can get ahead of symms by the value of lookahead.
// Note A, B, and C are passed by value, so we can transpose if needed
// (for side = right) without affecting caller.
template <Target target, typename scalar_t>
void symm(slate::internal::TargetType<target>,
          Side side,
          scalar_t alpha, SymmetricMatrix<scalar_t> A,
                          Matrix<scalar_t> B,
          scalar_t beta,  Matrix<scalar_t> C,
          int64_t lookahead)
{
    using namespace blas;

    // if on right, change to left by transposing A, B, C to get op(C) = op(A)*op(B)
    if (side == Side::Right) {
        A = transpose(A);
        B = transpose(B);
        C = transpose(C);
    }

    // B and C are mt-by-nt, A is mt-by-mt (assuming side = left)
    assert(A.mt() == B.mt());
    assert(A.nt() == B.mt());
    assert(B.mt() == C.mt());
    assert(B.nt() == C.nt());

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
        if ((A.uplo() == Uplo::Lower && A.op() == Op::NoTrans) ||
            (A.uplo() == Uplo::Upper && A.op() != Op::NoTrans)) {
            // ----------------------------------------
            // Left, Lower/NoTrans or Upper/Trans case

            // send 1st block col of A and block row of B
            #pragma omp task depend(out:bcast[0])
            {
                // broadcast A(i, 0) to ranks owning block row C(i, :)
                for (int64_t i = 0; i < A.mt(); ++i)
                    A.template tileBcast<target>(
                        i, 0, C.sub(i, i, 0, C.nt()-1));

                // broadcast B(0, j) to ranks owning block col C(:, j)
                for (int64_t j = 0; j < B.nt(); ++j)
                    B.template tileBcast<target>(
                        0, j, C.sub(0, C.mt()-1, j, j));
            }

            // send next lookahead block cols of A and block rows of B
            for (int64_t k = 1; k < lookahead+1 && k < A.nt(); ++k) {
                #pragma omp task depend(in:bcast[k-1]) \
                                 depend(out:bcast[k])
                {
                    // broadcast A(k, i) or A(i, k) to ranks owning block row C(i, :)
                    for (int64_t i = 0; i < k && i < A.mt(); ++i)
                        A.template tileBcast<target>(
                            k, i, C.sub(i, i, 0, C.nt()-1));

                    for (int64_t i = k; i < A.mt(); ++i)
                        A.template tileBcast<target>(
                            i, k, C.sub(i, i, 0, C.nt()-1));

                    // broadcast B(k, j) to ranks owning block col C(0:k, j)
                    for (int64_t j = 0; j < B.nt(); ++j)
                        B.template tileBcast<target>(
                            k, j, C.sub(0, C.mt()-1, j, j));
                }
            }

            // multiply alpha A(:, 0) B(0, :), which is:
            // C(0, :)      = alpha [ A(0, 0)      B(0, :) ] + beta C(0, :)       symm
            // C(1:mt-1, :) = alpha [ A(1:mt-1, 0) B(0, :) ] + beta C(1:mt-1, :)  gemm
            #pragma omp task depend(in:bcast[0]) \
                             depend(out:gemm[0])
            {
                internal::symm<Target::HostTask>(  // todo: target
                    Side::Left,
                    alpha, A.sub(0, 0),
                           B.sub(0, 0, 0, B.nt()-1),
                    beta,  C.sub(0, 0, 0, C.nt()-1));

                if (A.mt()-1 > 0) {
                    internal::gemm<target>(
                        alpha, A.sub(1, A.mt()-1, 0, 0),
                               B.sub(0, 0, 0, B.nt()-1),
                        beta,  C.sub(1, C.mt()-1, 0, C.nt()-1));
                }
            }

            for (int64_t k = 1; k < A.nt(); ++k) {

                // send next block col of A and block row of B
                if (k+lookahead < A.nt()) {
                    #pragma omp task depend(in:gemm[k-1]) \
                                     depend(in:bcast[k+lookahead-1]) \
                                     depend(out:bcast[k+lookahead])
                    {
                        // broadcast A(k+la, i) or A(i, k+la) to ranks owning block row C(i, :)
                        for (int64_t i = 0; i < k+lookahead; ++i)
                            A.template tileBcast<target>(
                                k+lookahead, i, C.sub(i, i, 0, C.nt()-1));

                        for (int64_t i = k+lookahead; i < A.mt(); ++i)
                            A.template tileBcast<target>(
                                i, k+lookahead, C.sub(i, i, 0, C.nt()-1));

                        // broadcast B(k+la, j) to ranks owning block col C(0:k+la, j)
                        for (int64_t j = 0; j < B.nt(); ++j)
                            B.template tileBcast<target>(
                                k+lookahead, j, C.sub(0, C.mt()-1, j, j));
                    }
                }

                // multiply alpha A(:, k) B(k, :), which is:
                // C(0:k-1, :)    += alpha [ A(k, 0:k-1)^T  B(k, :) ]  gemm
                // C(k, :)        += alpha [ A(k, k)        B(k, :) ]  symm
                // C(k+1:mt-1, :) += alpha [ A(k+1:mt-1, k) B(k, :) ]  gemm
                #pragma omp task depend(in:bcast[k]) \
                                 depend(in:gemm[k-1]) \
                                 depend(out:gemm[k])
                {
                    auto Arow_k = A.sub(k, k, 0, k-1);
                    internal::gemm<target>(
                        alpha,         transpose( Arow_k ),
                                       B.sub(k, k, 0, B.nt()-1),
                        scalar_t(1.0), C.sub(0, k-1, 0, C.nt()-1));

                    internal::symm<Target::HostTask>(  // todo: target
                        Side::Left,
                        alpha,          A.sub(k, k),
                                        B.sub(k, k, 0, B.nt()-1),
                        scalar_t(1.0),  C.sub(k, k, 0, C.nt()-1));

                    if (A.mt()-1 > k) {
                        internal::gemm<target>(
                            alpha,         A.sub(k+1, A.mt()-1, k, k),
                                           B.sub(k, k, 0, B.nt()-1),
                            scalar_t(1.0), C.sub(k+1, C.mt()-1, 0, C.nt()-1));
                    }
                }
            }
        }
        else {
            // ----------------------------------------
            // Left, Upper/NoTrans or Lower/Trans case

            // send 1st block col (row) of A and block row of B
            #pragma omp task depend(out:bcast[0])
            {
                // broadcast A(i, 0) to ranks owning block row C(i, :)
                for (int64_t i = 0; i < A.mt(); ++i)
                    A.template tileBcast<target>(
                        0, i, C.sub(i, i, 0, C.nt()-1));

                // broadcast B(0, j) to ranks owning block col C(:, j)
                for (int64_t j = 0; j < B.nt(); ++j)
                    B.template tileBcast<target>(
                        0, j, C.sub(0, C.mt()-1, j, j));
            }

            // send next lookahead block cols of A and block rows of B
            for (int64_t k = 1; k < lookahead+1 && k < A.nt(); ++k) {
                #pragma omp task depend(in:bcast[k-1]) \
                                 depend(out:bcast[k])
                {
                    // broadcast A(k, i) or A(i, k) to ranks owning block row C(i, :)
                    for (int64_t i = 0; i < k && i < A.mt(); ++i)
                        A.template tileBcast<target>(
                            i, k, C.sub(i, i, 0, C.nt()-1));

                    for (int64_t i = k; i < A.mt(); ++i)
                        A.template tileBcast<target>(
                            k, i, C.sub(i, i, 0, C.nt()-1));

                    // broadcast B(k, j) to ranks owning block col C(0:k, j)
                    for (int64_t j = 0; j < B.nt(); ++j)
                        B.template tileBcast<target>(
                            k, j, C.sub(0, C.mt()-1, j, j));
                }
            }

            // multiply alpha A(:, 0) B(0, :), which is:
            // C(0, :)      = alpha [ A(0, 0)      B(0, :) ] + beta C(0, :)       symm
            // C(1:mt-1, :) = alpha [ A(1:mt-1, 0) B(0, :) ] + beta C(1:mt-1, :)  gemm
            #pragma omp task depend(in:bcast[0]) \
                             depend(out:gemm[0])
            {
                internal::symm<Target::HostTask>(  // todo: target
                    Side::Left,
                    alpha, A.sub(0, 0),
                           B.sub(0, 0, 0, B.nt()-1),
                    beta,  C.sub(0, 0, 0, C.nt()-1));

                if (A.mt()-1 > 0) {
                    auto Arow_k = A.sub(0, 0, 1, A.mt()-1);
                    internal::gemm<target>(
                        alpha, transpose( Arow_k ),
                               B.sub(0, 0, 0, B.nt()-1),
                        beta,  C.sub(1, C.mt()-1, 0, C.nt()-1));
                }
            }

            for (int64_t k = 1; k < A.nt(); ++k) {

                // send next block col of A and block row of B
                if (k+lookahead < A.nt()) {
                    #pragma omp task depend(in:gemm[k-1]) \
                                     depend(in:bcast[k+lookahead-1]) \
                                     depend(out:bcast[k+lookahead])
                    {
                        // broadcast A(k+la, i) or A(i, k+la) to ranks owning block row C(i, :)
                        for (int64_t i = 0; i < k+lookahead; ++i)
                            A.template tileBcast<target>(
                                i, k+lookahead, C.sub(i, i, 0, C.nt()-1));

                        for (int64_t i = k+lookahead; i < A.mt(); ++i)
                            A.template tileBcast<target>(
                                k+lookahead, i, C.sub(i, i, 0, C.nt()-1));

                        // broadcast B(k+la, j) to ranks owning block col C(0:k+la, j)
                        for (int64_t j = 0; j < B.nt(); ++j)
                            B.template tileBcast<target>(
                                k+lookahead, j, C.sub(0, C.mt()-1, j, j));
                    }
                }

                // multiply alpha A(:, k) B(k, :), which is:
                // C(0:k-1, :)    += alpha [ A(k, 0:k-1)^T  B(k, :) ]  gemm
                // C(k, :)        += alpha [ A(k, k)        B(k, :) ]  symm
                // C(k+1:mt-1, :) += alpha [ A(k+1:mt-1, k) B(k, :) ]  gemm
                #pragma omp task depend(in:bcast[k]) \
                                 depend(in:gemm[k-1]) \
                                 depend(out:gemm[k])
                {
                    internal::gemm<target>(
                        alpha,         A.sub(0, k-1, k, k),
                                       B.sub(k, k, 0, B.nt()-1),
                        scalar_t(1.0), C.sub(0, k-1, 0, C.nt()-1));

                    internal::symm<Target::HostTask>(  // todo: target
                        Side::Left,
                        alpha,          A.sub(k, k),
                                        B.sub(k, k, 0, B.nt()-1),
                        scalar_t(1.0),  C.sub(k, k, 0, C.nt()-1));

                    if (A.mt()-1 > k) {
                        auto Arow_k = A.sub(k, k, k+1, A.mt()-1);
                        internal::gemm<target>(
                            alpha,         transpose( Arow_k ),
                                           B.sub(k, k, 0, B.nt()-1),
                            scalar_t(1.0), C.sub(k+1, C.mt()-1, 0, C.nt()-1));
                    }
                }
            }
        }
    }

    // todo: we need a function that updates origins that are not valid
    for (int64_t i = 0; i < C.mt(); ++i)
        for (int64_t j = 0; j < C.nt(); ++j)
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
void symm(Side side,
          scalar_t alpha, SymmetricMatrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
          const std::map<Option, Value>& opts)
{
    int64_t lookahead;
    try {
        lookahead = opts.at(Option::Lookahead).i_;
    }
    catch (std::out_of_range) {
        lookahead = 1;
    }

    internal::specialization::symm(internal::TargetType<target>(),
                                   side,
                                   alpha, A,
                                          B,
                                   beta,  C,
                                   lookahead);
}

//------------------------------------------------------------------------------
// Explicit instantiations for double precision and various targets.
template
void symm< Target::HostTask, float >(
    Side side,
    float alpha, SymmetricMatrix<float>& A,
                 Matrix<float>& B,
    float beta,  Matrix<float>& C,
    const std::map<Option, Value>& opts);

template
void symm< Target::HostNest, float >(
    Side side,
    float alpha, SymmetricMatrix<float>& A,
                 Matrix<float>& B,
    float beta,  Matrix<float>& C,
    const std::map<Option, Value>& opts);

template
void symm< Target::HostBatch, float >(
    Side side,
    float alpha, SymmetricMatrix<float>& A,
                 Matrix<float>& B,
    float beta,  Matrix<float>& C,
    const std::map<Option, Value>& opts);

template
void symm< Target::Devices, float >(
    Side side,
    float alpha, SymmetricMatrix<float>& A,
                 Matrix<float>& B,
    float beta,  Matrix<float>& C,
    const std::map<Option, Value>& opts);

// ----------------------------------------
template
void symm< Target::HostTask, double >(
    Side side,
    double alpha, SymmetricMatrix<double>& A,
                  Matrix<double>& B,
    double beta,  Matrix<double>& C,
    const std::map<Option, Value>& opts);

template
void symm< Target::HostNest, double >(
    Side side,
    double alpha, SymmetricMatrix<double>& A,
                  Matrix<double>& B,
    double beta,  Matrix<double>& C,
    const std::map<Option, Value>& opts);

template
void symm< Target::HostBatch, double >(
    Side side,
    double alpha, SymmetricMatrix<double>& A,
                  Matrix<double>& B,
    double beta,  Matrix<double>& C,
    const std::map<Option, Value>& opts);

template
void symm< Target::Devices, double >(
    Side side,
    double alpha, SymmetricMatrix<double>& A,
                  Matrix<double>& B,
    double beta,  Matrix<double>& C,
    const std::map<Option, Value>& opts);

// ----------------------------------------
template
void symm< Target::HostTask,  std::complex<float>  >(
    Side side,
    std::complex<float> alpha, SymmetricMatrix< std::complex<float> >& A,
                               Matrix< std::complex<float> >& B,
    std::complex<float> beta,  Matrix< std::complex<float> >& C,
    const std::map<Option, Value>& opts);

template
void symm< Target::HostNest, std::complex<float> >(
    Side side,
    std::complex<float> alpha, SymmetricMatrix< std::complex<float> >& A,
                               Matrix< std::complex<float> >& B,
    std::complex<float> beta,  Matrix< std::complex<float> >& C,
    const std::map<Option, Value>& opts);

template
void symm< Target::HostBatch, std::complex<float> >(
    Side side,
    std::complex<float> alpha, SymmetricMatrix< std::complex<float> >& A,
                               Matrix< std::complex<float> >& B,
    std::complex<float> beta,  Matrix< std::complex<float> >& C,
    const std::map<Option, Value>& opts);

template
void symm< Target::Devices, std::complex<float> >(
    Side side,
    std::complex<float> alpha, SymmetricMatrix< std::complex<float> >& A,
                               Matrix< std::complex<float> >& B,
    std::complex<float> beta,  Matrix< std::complex<float> >& C,
    const std::map<Option, Value>& opts);

// ----------------------------------------
template
void symm< Target::HostTask, std::complex<double> >(
    Side side,
    std::complex<double> alpha, SymmetricMatrix< std::complex<double> >& A,
                                Matrix< std::complex<double> >& B,
    std::complex<double> beta,  Matrix< std::complex<double> >& C,
    const std::map<Option, Value>& opts);

template
void symm< Target::HostNest, std::complex<double> >(
    Side side,
    std::complex<double> alpha, SymmetricMatrix< std::complex<double> >& A,
                                Matrix< std::complex<double> >& B,
    std::complex<double> beta,  Matrix< std::complex<double> >& C,
    const std::map<Option, Value>& opts);

template
void symm< Target::HostBatch, std::complex<double> >(
    Side side,
    std::complex<double> alpha, SymmetricMatrix< std::complex<double> >& A,
                                Matrix< std::complex<double> >& B,
    std::complex<double> beta,  Matrix< std::complex<double> >& C,
    const std::map<Option, Value>& opts);

template
void symm< Target::Devices, std::complex<double> >(
    Side side,
    std::complex<double> alpha, SymmetricMatrix< std::complex<double> >& A,
                                Matrix< std::complex<double> >& B,
    std::complex<double> beta,  Matrix< std::complex<double> >& C,
    const std::map<Option, Value>& opts);

} // namespace slate
