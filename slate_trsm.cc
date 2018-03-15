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
// internal::trsm from internal::specialization::trsm
namespace internal {
namespace specialization {

///-----------------------------------------------------------------------------
/// \brief
/// Distributed parallel triangular matrix multiplication.
/// Generic implementation for any target.
// Note A and B are passed by value, so we can transpose if needed
// (for side = right) without affecting caller.
template <Target target, typename scalar_t>
void trsm(slate::internal::TargetType<target>,
          Side side, Diag diag,
          scalar_t alpha, TriangularMatrix<scalar_t> A,
                                    Matrix<scalar_t> B,
          int64_t lookahead)
{
    using namespace blas;

    // if on right, change to left by (conj)-transposing A and B to get
    // op(B) = op(A)^{-1} * op(B)
    if (side == Side::Right) {
        if (A.op() == Op::ConjTrans || B.op() == Op::ConjTrans) {
            A = conj_transpose(A);
            B = conj_transpose(B);
            alpha = conj(alpha);
        }
        else {
            A = transpose(A);
            B = transpose(B);
        }
    }

    // B is mt-by-nt, A is mt-by-mt (assuming side = left)
    assert(A.mt() == B.mt());
    assert(A.nt() == B.mt());

    int64_t mt = B.mt();
    int64_t nt = B.nt();

    if (target == Target::Devices) {
        B.allocateBatchArrays();
        B.reserveDeviceWorkspace();
    }

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > bcast_vector(mt);
    std::vector< uint8_t >  trsm_vector(mt);
    std::vector< uint8_t >  gemm_vector(mt);
    uint8_t *bcast = bcast_vector.data();
    uint8_t *trsm  =  trsm_vector.data();
    uint8_t *gemm  =  gemm_vector.data();

    #pragma omp parallel
    #pragma omp master
    {
        if ((A.uplo() == Uplo::Upper && A.op() == Op::NoTrans) ||
            (A.uplo() == Uplo::Lower && A.op() != Op::NoTrans)) {
            // ----------------------------------------
            // Left, Upper/NoTrans or Lower/Trans case
            // Backward sweep

            // send 1st block col of A
            #pragma omp task depend(out:bcast[mt-1])
            {
                // broadcast A(i, 0) to ranks owning block row B(i, :)
                for (int64_t i = 0; i < mt; ++i)
                    A.template tileBcast<target>(i, mt-1, B.sub(i, i, 0, nt-1));
            }

            // send next lookahead block cols of A
            for (int64_t k = mt-2; k >= mt-1-lookahead && k >= 0; --k) {
                #pragma omp task depend(in:bcast[k+1]) \
                                 depend(out:bcast[k])
                {
                    // broadcast A(i, k) to ranks owning block row B(i, :)
                    for (int64_t i = 0; i <= k; ++i)  // upper
                        A.template tileBcast<target>(i, k, B.sub(i, i, 0, nt-1));
                }
            }

            #pragma omp task depend(in:bcast[mt-1]) \
                             depend(out:trsm[mt-1])
            {
                // solve B(mt-1, :) = alpha A(mt-1, :) B(mt-1, :)
                internal::trsm<Target::HostTask>(
                    Side::Left, diag,
                    alpha, A.sub(mt-1, mt-1),
                           B.sub(mt-1, mt-1, 0, nt-1));

                // broadcast B(0, j) to ranks owning block col B(0:0, j)
                for (int64_t j = 0; j < nt; ++j)
                    B.template tileBcast<target>(mt-1, j, B.sub(0, mt-2, j, j));
            }

            // update B(0:mt-2, :) = -A(0:mt-2, mt-1) B(mt-1, :) + alpha B(0:mt-2, :)
            #pragma omp task depend(in:bcast[mt-1]) \
                             depend(in:trsm[mt-1]) \
                             depend(out:gemm[mt-1])
            {
                internal::gemm<target>(
                    scalar_t(-1.0), A.sub(0, mt-2, mt-1, mt-1),
                                    B.sub(mt-1, mt-1, 0, nt-1),
                    alpha,          B.sub(0, mt-2, 0, nt-1));
            }

            for (int64_t k = mt-2; k >= 0; --k) {

                // send next block col of A
                if (k-lookahead >= 0) {
                    #pragma omp task depend(in:gemm[k+1]) \
                                     depend(in:bcast[k-lookahead+1]) \
                                     depend(out:bcast[k-lookahead])
                    {
                        // broadcast A(i, k-la) to ranks owning block row B(i, :)
                        for (int64_t i = 0; i <= k-lookahead; ++i)  // upper
                            A.template tileBcast<target>(i, k-lookahead, B.sub(i, i, 0, nt-1));
                    }
                }

                #pragma omp task depend(in:bcast[k]) \
                                 depend(in:gemm[k+1]) \
                                 depend(out:trsm[k])
                {
                    // solve A(k, k) X(k, :) = B(k, :)
                    // note no alpha; dealt with above
                    internal::trsm<Target::HostTask>(  // todo: target? needs batch trsm
                        Side::Left, diag,
                        scalar_t(1.0), A.sub(k, k),
                                       B.sub(k, k, 0, nt-1));

                    // broadcast B(k, j) to ranks owning block col B(0:k-1, j)
                    if (k > 0) {
                        for (int64_t j = 0; j < nt; ++j)
                            B.template tileBcast<target>(k, j, B.sub(0, k-1, j, j));
                    }
                }

                // update B(0:k-1, :) -= A(0:k-1, k) B(k, :)
                // note no alpha; dealt with above
                if (k > 0) {
                    #pragma omp task depend(in:bcast[k]) \
                                     depend(in:trsm[k]) \
                                     depend(out:gemm[k])
                    {
                        internal::gemm<target>(
                            scalar_t(-1.0), A.sub(0, k-1, k, k),
                                            B.sub(k, k, 0, nt-1),
                            scalar_t(1.0),  B.sub(0, k-1, 0, nt-1));
                    }
                }
            }
        }
        else {
            // ----------------------------------------
            // Left, Lower/NoTrans or Upper/Trans case
            // Forward sweep

            // send 1st block col of A
            #pragma omp task depend(out:bcast[0])
            {
                // broadcast A(i, 0) to ranks owning block row B(i, :)
                for (int64_t i = 0; i < mt; ++i)
                    A.template tileBcast<target>(i, 0, B.sub(i, i, 0, nt-1));
            }

            // send next lookahead block cols of A
            for (int64_t k = 1; k < lookahead && k < mt; ++k) {
                #pragma omp task depend(in:bcast[k-1]) \
                                 depend(out:bcast[k])
                {
                    // broadcast A(i, k) to ranks owning block row B(i, :)
                    for (int64_t i = k; i < mt; ++i)  // lower
                        A.template tileBcast<target>(i, k, B.sub(i, i, 0, nt-1));
                }
            }

            #pragma omp task depend(in:bcast[0]) \
                             depend(out:trsm[0])
            {
                // solve B(0, :) = alpha A(0, :) B(0, :)
                internal::trsm<Target::HostTask>(
                    Side::Left, diag,
                    alpha, A.sub(0, 0),
                           B.sub(0, 0, 0, nt-1));

                // broadcast B(0, j) to ranks owning block col B(1:mt-1, j)
                for (int64_t j = 0; j < nt; ++j)
                    B.template tileBcast<target>(0, j, B.sub(1, mt-1, j, j));
            }

            // update B(1:mt-1, :) = -A(1:mt-1, 0) B(0, :) + alpha B(1:mt-1, :)
            #pragma omp task depend(in:bcast[0]) \
                             depend(in:trsm[0]) \
                             depend(out:gemm[0])
            {
                internal::gemm<target>(
                    scalar_t(-1.0), A.sub(1, mt-1, 0, 0),
                                    B.sub(0, 0, 0, nt-1),
                    alpha,          B.sub(1, mt-1, 0, nt-1));
            }

            for (int64_t k = 1; k < mt; ++k) {

                // send next block col of A
                if (k+lookahead < mt) {
                    #pragma omp task depend(in:gemm[k-1]) \
                                     depend(in:bcast[k+lookahead-1]) \
                                     depend(out:bcast[k+lookahead])
                    {
                        // broadcast A(i, k-la) to ranks owning block row B(i, :)
                        for (int64_t i = k-lookahead; i < mt; ++i)  // lower
                            A.template tileBcast<target>(i, k+lookahead, B.sub(i, i, 0, nt-1));
                    }
                }

                #pragma omp task depend(in:bcast[k]) \
                                 depend(in:gemm[k-1]) \
                                 depend(out:trsm[k])
                {
                    // solve A(k, k) X(k, :) = B(k, :)
                    // note no alpha; dealt with above
                    internal::trsm<Target::HostTask>(  // todo: target? needs batch trsm
                        Side::Left, diag,
                        scalar_t(1.0), A.sub(k, k),
                                       B.sub(k, k, 0, nt-1));

                    // broadcast B(k, j) to ranks owning block col B(k+1:mt-1, j)
                    if (k < mt-1) {
                        for (int64_t j = 0; j < nt; ++j)
                            B.template tileBcast<target>(k, j, B.sub(k+1, mt-1, j, j));
                    }
                }

                // update B(k+1:mt-1, :) -= A(k+1:mt-1, k) B(k, :)
                // note no alpha; dealt with above
                if (k < mt-1) {
                    #pragma omp task depend(in:bcast[k]) \
                                     depend(in:trsm[k]) \
                                     depend(out:gemm[k])
                    {
                        internal::gemm<target>(
                            scalar_t(-1.0), A.sub(k+1, mt-1, k, k),
                                            B.sub(k, k, 0, nt-1),
                            scalar_t(1.0),  B.sub(k+1, mt-1, 0, nt-1));
                    }
                }
            }
        } // end Lower/NoTrans
    } // end omp master

    // todo: restoreToOrigin, moveToOrigin

    B.clearWorkspace();
}

} // namespace specialization
} // namespace internal

///-----------------------------------------------------------------------------
/// \brief
///
/// Precision and target templated function.
template <Target target, typename scalar_t>
void trsm(blas::Side side, blas::Diag diag,
          scalar_t alpha, TriangularMatrix<scalar_t>& A,
                                    Matrix<scalar_t>& B,
          const std::map<Option, Value>& opts)
{
    int64_t lookahead;
    try {
        lookahead = opts.at(Option::Lookahead).i_;
    }
    catch (std::out_of_range) {
        lookahead = 1;
    }

    internal::specialization::trsm(internal::TargetType<target>(),
                                   side, diag,
                                   alpha, A,
                                          B,
                                   lookahead);
}

//------------------------------------------------------------------------------
// Explicit instantiations for double precision and various targets.
template
void trsm< Target::HostTask, float >(
    blas::Side side, blas::Diag diag,
    float alpha, TriangularMatrix<float>& A,
                           Matrix<float>& B,
    const std::map<Option, Value>& opts);

template
void trsm< Target::HostNest, float >(
    blas::Side side, blas::Diag diag,
    float alpha, TriangularMatrix<float>& A,
                           Matrix<float>& B,
    const std::map<Option, Value>& opts);

template
void trsm< Target::HostBatch, float >(
    blas::Side side, blas::Diag diag,
    float alpha, TriangularMatrix<float>& A,
                           Matrix<float>& B,
    const std::map<Option, Value>& opts);

template
void trsm< Target::Devices, float >(
    blas::Side side, blas::Diag diag,
    float alpha, TriangularMatrix<float>& A,
                           Matrix<float>& B,
    const std::map<Option, Value>& opts);

// ----------------------------------------
template
void trsm< Target::HostTask, double >(
    blas::Side side, blas::Diag diag,
    double alpha, TriangularMatrix<double>& A,
                            Matrix<double>& B,
    const std::map<Option, Value>& opts);

template
void trsm< Target::HostNest, double >(
    blas::Side side, blas::Diag diag,
    double alpha, TriangularMatrix<double>& A,
                            Matrix<double>& B,
    const std::map<Option, Value>& opts);

template
void trsm< Target::HostBatch, double >(
    blas::Side side, blas::Diag diag,
    double alpha, TriangularMatrix<double>& A,
                            Matrix<double>& B,
    const std::map<Option, Value>& opts);

template
void trsm< Target::Devices, double >(
    blas::Side side, blas::Diag diag,
    double alpha, TriangularMatrix<double>& A,
                            Matrix<double>& B,
    const std::map<Option, Value>& opts);

// ----------------------------------------
template
void trsm< Target::HostTask, std::complex<float> >(
    blas::Side side, blas::Diag diag,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>>& A,
                                         Matrix<std::complex<float>>& B,
    const std::map<Option, Value>& opts);

template
void trsm< Target::HostNest, std::complex<float> >(
    blas::Side side, blas::Diag diag,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>>& A,
                                         Matrix<std::complex<float>>& B,
    const std::map<Option, Value>& opts);

template
void trsm< Target::HostBatch, std::complex<float> >(
    blas::Side side, blas::Diag diag,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>>& A,
                                         Matrix<std::complex<float>>& B,
    const std::map<Option, Value>& opts);

template
void trsm< Target::Devices, std::complex<float> >(
    blas::Side side, blas::Diag diag,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>>& A,
                                         Matrix<std::complex<float>>& B,
    const std::map<Option, Value>& opts);

// ----------------------------------------
template
void trsm< Target::HostTask, std::complex<double> >(
    blas::Side side, blas::Diag diag,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>>& A,
                                          Matrix<std::complex<double>>& B,
    const std::map<Option, Value>& opts);

template
void trsm< Target::HostNest, std::complex<double> >(
    blas::Side side, blas::Diag diag,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>>& A,
                                          Matrix<std::complex<double>>& B,
    const std::map<Option, Value>& opts);

template
void trsm< Target::HostBatch, std::complex<double> >(
    blas::Side side, blas::Diag diag,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>>& A,
                                          Matrix<std::complex<double>>& B,
    const std::map<Option, Value>& opts);

template
void trsm< Target::Devices, std::complex<double> >(
    blas::Side side, blas::Diag diag,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>>& A,
                                          Matrix<std::complex<double>>& B,
    const std::map<Option, Value>& opts);

} // namespace slate
