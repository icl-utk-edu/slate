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
// internal::trmm from internal::specialization::trmm
namespace internal {
namespace specialization {

///-----------------------------------------------------------------------------
/// \brief
/// Distributed parallel triangular matrix multiplication.
/// Generic implementation for any target.
// Note A and B are passed by value, so we can transpose if needed
// (for side = right) without affecting caller.
template <Target target, typename scalar_t>
void trmm(slate::internal::TargetType<target>,
          Side side, Diag diag,
          scalar_t alpha, TriangularMatrix<scalar_t> A,
                                    Matrix<scalar_t> B,
          int64_t lookahead)
{
    using namespace blas;

    // if on right, change to left by (conj)-transposing A and B to get op(B) = op(A)*op(B)
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
    std::vector< uint8_t >  gemm_vector(mt);
    uint8_t *bcast = bcast_vector.data();
    uint8_t *gemm  =  gemm_vector.data();

    #pragma omp parallel
    #pragma omp master
    {
        if ((A.uplo() == Uplo::Upper && A.op() == Op::NoTrans) ||
            (A.uplo() == Uplo::Lower && A.op() != Op::NoTrans)) {
            // ----------------------------------------
            // Left, Upper/NoTrans or Lower/Trans case
            // Forward sweep

            // send 1st block col of A and block row of B
            #pragma omp task depend(out:bcast[0])
            {
                // broadcast A(i, 0) to ranks owning block row B(i, :), for i = 0
                A.template tileBcast<target>(
                    0, 0, B.sub(0, 0, 0, nt-1));

                // broadcast B(0, j) to ranks owning block col B(0:0, j)
                // todo: nowhere to send?
                for (int64_t j = 0; j < nt; ++j)
                    B.template tileBcast<target>(
                        0, j, B.sub(0, 0, j, j));
            }

            // send next lookahead block cols of A and block rows of B
            for (int64_t k = 1; k < lookahead+1 && k < mt; ++k) {
                #pragma omp task depend(in:bcast[k-1]) \
                                 depend(out:bcast[k])
                {
                    // broadcast A(i, k) to ranks owning block row B(i, :)
                    for (int64_t i = 0; i <= k; ++i)  // upper
                        A.template tileBcast<target>(
                            i, k, B.sub(i, i, 0, nt-1));

                    // broadcast B(k, j) to ranks owning block col B(0:k, j)
                    for (int64_t j = 0; j < nt; ++j)
                        B.template tileBcast<target>(
                            k, j, B.sub(0, k, j, j));
                }
            }

            // multiply alpha A(:, 0) B(0, :), which is:
            // B(0, :) = alpha [ A(0, 0) B(0, :) ]  trmm
            #pragma omp task depend(in:bcast[0]) \
                             depend(out:gemm[0])
            internal::trmm<Target::HostTask>(
                Side::Left, diag,
                alpha, A.sub(0, 0),
                       B.sub(0, 0, 0, nt-1));

            for (int64_t k = 1; k < mt; ++k) {

                // send next block col of A and block row of B
                if (k+lookahead < mt) {
                    #pragma omp task depend(in:gemm[k-1]) \
                                     depend(in:bcast[k+lookahead-1]) \
                                     depend(out:bcast[k+lookahead])
                    {
                        // broadcast A(i, k+la) to ranks owning block row B(i, :)
                        for (int64_t i = 0; i <= k+lookahead; ++i)  // upper
                            A.template tileBcast<target>(
                                i, k+lookahead, B.sub(i, i, 0, nt-1));

                        // broadcast B(k+la, j) to ranks owning block col B(0:k+la, j)
                        for (int64_t j = 0; j < nt; ++j)
                            B.template tileBcast<target>(
                                k+lookahead, j, B.sub(0, k+lookahead, j, j));
                    }
                }

                // multiply alpha A(:, k) B(k, :), which is:
                // B(0:k-1, :) += alpha [ A(0:k-1, k) B(k, :) ]  gemm
                // B(k, :)      = alpha [ A(k, k)     B(k, :) ]  trmm
                #pragma omp task depend(in:bcast[k]) \
                                 depend(in:gemm[k-1]) \
                                 depend(out:gemm[k])
                {
                    internal::gemm<target>(
                        alpha,         A.sub(0, k-1, k, k),
                                       B.sub(k, k, 0, nt-1),
                        scalar_t(1.0), B.sub(0, k-1, 0, nt-1));

                    internal::trmm<Target::HostTask>(  // todo: target? needs batch trmm
                        Side::Left, diag,
                        alpha, A.sub(k, k),
                               B.sub(k, k, 0, nt-1));
                }
            }
        }
        else {
            // ----------------------------------------
            // Left, Lower/NoTrans or Upper/Trans case
            // Backward sweep

            // send 1st block col of A and block row of B
            #pragma omp task depend(out:bcast[mt-1])
            {
                // broadcast A(i, 0) to ranks owning block row B(i, :), for i = m-1
                A.template tileBcast<target>(
                    mt-1, mt-1, B.sub(mt-1, mt-1, 0, nt-1));

                // broadcast B(m-1, j) to ranks owning block col B(m-1:m-1, j)
                // todo: nowhere to send?
                for (int64_t j = 0; j < nt; ++j)
                    B.template tileBcast<target>(
                        mt-1, j, B.sub(mt-1, mt-1, j, j));
            }

            // send next lookahead block cols of A and block rows of B
            for (int64_t k = mt-2; k >= mt-1-lookahead && k >= 0; --k) {
                #pragma omp task depend(in:bcast[k+1]) \
                                 depend(out:bcast[k])
                {
                    // broadcast A(i, k) to ranks owning block row B(i, :)
                    for (int64_t i = k; i < mt; ++i)  // lower
                        A.template tileBcast<target>(
                            i, k, B.sub(i, i, 0, nt-1));

                    // broadcast B(k, j) to ranks owning block col B(k:m-1, j)
                    for (int64_t j = 0; j < nt; ++j)
                        B.template tileBcast<target>(
                            k, j, B.sub(k, mt-1, j, j));
                }
            }

            // multiply B = alpha A(:, mt-1) B(mt-1, :), which is:
            // B(mt-1, :) = alpha [ A(mt-1, mt-1) B(mt-1, :) ]  trmm
            #pragma omp task depend(in:bcast[mt-1]) \
                             depend(out:gemm[mt-1])
            internal::trmm<Target::HostTask>(
                Side::Left, diag,
                alpha, A.sub(mt-1, mt-1),
                       B.sub(mt-1, mt-1, 0, nt-1));

            for (int64_t k = mt-2; k >= 0; --k) {

                // send next block col of A and block row of B
                if (k-lookahead >= 0) {
                    #pragma omp task depend(in:gemm[k+1]) \
                                     depend(in:bcast[k-lookahead+1]) \
                                     depend(out:bcast[k-lookahead])
                    {
                        // broadcast A(i, k-la) to ranks owning block row B(i, :)
                        for (int64_t i = k-lookahead; i < mt; ++i)  // lower
                            A.template tileBcast<target>(
                                i, k-lookahead, B.sub(i, i, 0, nt-1));

                        // broadcast B(k-la, j) to ranks owning block col B(k-la:m-1, j)
                        for (int64_t j = 0; j < nt; ++j)
                            B.template tileBcast<target>(
                                k-lookahead, j, B.sub(k-lookahead, mt-1, j, j));
                    }
                }

                // multiply alpha A(:, k) B(k, :), which is:
                // B(k+1:m-1, :) += alpha [ A(k+1:m-1, k) B(k, :) ]  gemm
                // B(k, :)        = alpha [ A(k, k)       B(k, :) ]  trmm
                #pragma omp task depend(in:bcast[k]) \
                                 depend(in:gemm[k+1]) \
                                 depend(out:gemm[k])
                {
                    internal::gemm<target>(
                        alpha,         A.sub(k+1, mt-1, k, k),
                                       B.sub(k, k, 0, nt-1),
                        scalar_t(1.0), B.sub(k+1, mt-1, 0, nt-1));

                    internal::trmm<Target::HostTask>(  // todo: target? needs batch trmm
                        Side::Left, diag,
                        alpha, A.sub(k, k),
                               B.sub(k, k, 0, nt-1));
                }
            }
        } // end Lower/NoTrans
    } // end omp master

    // todo: we need a function that updates origins that are not valid
    for (int64_t i = 0; i < B.mt(); ++i)
        for (int64_t j = 0; j < B.nt(); ++j)
            if (B.tileIsLocal(i, j))
                B.tileMoveToHost(i, j, B.tileDevice(i, j));

    B.clearWorkspace();
}

} // namespace specialization
} // namespace internal

///-----------------------------------------------------------------------------
/// \brief
///
/// Precision and target templated function.
template <Target target, typename scalar_t>
void trmm(blas::Side side, blas::Diag diag,
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

    internal::specialization::trmm(internal::TargetType<target>(),
                                   side, diag,
                                   alpha, A,
                                          B,
                                   lookahead);
}

//------------------------------------------------------------------------------
// Explicit instantiations for double precision and various targets.
template
void trmm< Target::HostTask, float >(
    blas::Side side, blas::Diag diag,
    float alpha, TriangularMatrix<float>& A,
                           Matrix<float>& B,
    const std::map<Option, Value>& opts);

template
void trmm< Target::HostNest, float >(
    blas::Side side, blas::Diag diag,
    float alpha, TriangularMatrix<float>& A,
                           Matrix<float>& B,
    const std::map<Option, Value>& opts);

template
void trmm< Target::HostBatch, float >(
    blas::Side side, blas::Diag diag,
    float alpha, TriangularMatrix<float>& A,
                           Matrix<float>& B,
    const std::map<Option, Value>& opts);

template
void trmm< Target::Devices, float >(
    blas::Side side, blas::Diag diag,
    float alpha, TriangularMatrix<float>& A,
                           Matrix<float>& B,
    const std::map<Option, Value>& opts);

// ----------------------------------------
template
void trmm< Target::HostTask, double >(
    blas::Side side, blas::Diag diag,
    double alpha, TriangularMatrix<double>& A,
                            Matrix<double>& B,
    const std::map<Option, Value>& opts);

template
void trmm< Target::HostNest, double >(
    blas::Side side, blas::Diag diag,
    double alpha, TriangularMatrix<double>& A,
                            Matrix<double>& B,
    const std::map<Option, Value>& opts);

template
void trmm< Target::HostBatch, double >(
    blas::Side side, blas::Diag diag,
    double alpha, TriangularMatrix<double>& A,
                            Matrix<double>& B,
    const std::map<Option, Value>& opts);

template
void trmm< Target::Devices, double >(
    blas::Side side, blas::Diag diag,
    double alpha, TriangularMatrix<double>& A,
                            Matrix<double>& B,
    const std::map<Option, Value>& opts);

// ----------------------------------------
template
void trmm< Target::HostTask, std::complex<float> >(
    blas::Side side, blas::Diag diag,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>>& A,
                                         Matrix<std::complex<float>>& B,
    const std::map<Option, Value>& opts);

template
void trmm< Target::HostNest, std::complex<float> >(
    blas::Side side, blas::Diag diag,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>>& A,
                                         Matrix<std::complex<float>>& B,
    const std::map<Option, Value>& opts);

template
void trmm< Target::HostBatch, std::complex<float> >(
    blas::Side side, blas::Diag diag,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>>& A,
                                         Matrix<std::complex<float>>& B,
    const std::map<Option, Value>& opts);

template
void trmm< Target::Devices, std::complex<float> >(
    blas::Side side, blas::Diag diag,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>>& A,
                                         Matrix<std::complex<float>>& B,
    const std::map<Option, Value>& opts);

// ----------------------------------------
template
void trmm< Target::HostTask, std::complex<double> >(
    blas::Side side, blas::Diag diag,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>>& A,
                                          Matrix<std::complex<double>>& B,
    const std::map<Option, Value>& opts);

template
void trmm< Target::HostNest, std::complex<double> >(
    blas::Side side, blas::Diag diag,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>>& A,
                                          Matrix<std::complex<double>>& B,
    const std::map<Option, Value>& opts);

template
void trmm< Target::HostBatch, std::complex<double> >(
    blas::Side side, blas::Diag diag,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>>& A,
                                          Matrix<std::complex<double>>& B,
    const std::map<Option, Value>& opts);

template
void trmm< Target::Devices, std::complex<double> >(
    blas::Side side, blas::Diag diag,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>>& A,
                                          Matrix<std::complex<double>>& B,
    const std::map<Option, Value>& opts);

} // namespace slate
