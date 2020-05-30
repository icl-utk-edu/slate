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

#include "slate/slate.hh"
#include "aux/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/SymmetricMatrix.hh"
#include "internal/internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::symm from internal::specialization::symm
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// @internal
/// Distributed parallel symmetric matrix-matrix multiplication.
/// Generic implementation for any target.
/// Dependencies enforce the following behavior:
/// - bcast communications are serialized,
/// - symm operations are serialized,
/// - bcasts can get ahead of symms by the value of lookahead.
/// Note A, B, and C are passed by value, so we can transpose if needed
/// (for side = right) without affecting caller.
/// @ingroup symm_specialization
///
template <Target target, typename scalar_t>
void symm(slate::internal::TargetType<target>,
          Side side,
          scalar_t alpha, SymmetricMatrix<scalar_t> A,
                          Matrix<scalar_t> B,
          scalar_t beta,  Matrix<scalar_t> C,
          int64_t lookahead)
{
    // Due to the symmetry, each off diagonal tile is sent twice, once as part
    // of A and once as part of A^T. In principle, this could be avoided by
    // sending each tile only once and retaining it until it is used twice.
    // This would, however, violate the upper bound on the size of communication
    // buffers.
    // The same happens in the hemm routine.
    // See also the implementation remarks in the BaseMatrix::listBcast routine.

    using BcastList = typename Matrix<scalar_t>::BcastList;

    // Assumes column major
    const Layout layout = Layout::ColMajor;
    // if on right, change to left by transposing A, B, C to get
    // op(C) = op(A)*op(B)
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
    std::vector<uint8_t> bcast_vector(A.nt());
    std::vector<uint8_t>  gemm_vector(A.nt());
    uint8_t* bcast = bcast_vector.data();
    uint8_t* gemm  =  gemm_vector.data();

    if (target == Target::Devices) {
        C.allocateBatchArrays();
        C.reserveDeviceWorkspace();
    }

    #pragma omp parallel
    #pragma omp master
    {
        omp_set_nested(1);
        if (A.uplo() == Uplo::Lower) {
            // ----------------------------------------
            // Left, Lower/NoTrans or Upper/Trans case

            // send 1st block col of A and block row of B
            #pragma omp task depend(out:bcast[0])
            {
                // broadcast A(i, 0) to ranks owning block row C(i, :)
                BcastList bcast_list_A;
                for (int64_t i = 0; i < A.mt(); ++i)
                    bcast_list_A.push_back({i, 0, {C.sub(i, i, 0, C.nt()-1)}});
                A.template listBcast<target>(bcast_list_A, layout);

                // broadcast B(0, j) to ranks owning block col C(:, j)
                BcastList bcast_list_B;
                for (int64_t j = 0; j < B.nt(); ++j)
                    bcast_list_B.push_back({0, j, {C.sub(0, C.mt()-1, j, j)}});
                B.template listBcast<target>(bcast_list_B, layout);
            }

            // send next lookahead block cols of A and block rows of B
            for (int64_t k = 1; k < lookahead+1 && k < A.nt(); ++k) {
                #pragma omp task depend(in:bcast[k-1]) \
                                 depend(out:bcast[k])
                {
                    // broadcast A(k, i) or A(i, k)
                    // to ranks owning block row C(i, :)
                    BcastList bcast_list_A;
                    for (int64_t i = 0; i < k && i < A.mt(); ++i) {
                        bcast_list_A.push_back(
                            {k, i, {C.sub(i, i, 0, C.nt()-1)}});
                    }
                    for (int64_t i = k; i < A.mt(); ++i) {
                        bcast_list_A.push_back(
                            {i, k, {C.sub(i, i, 0, C.nt()-1)}});
                    }
                    A.template listBcast<target>(bcast_list_A, layout);

                    // broadcast B(k, j) to ranks owning block col C(0:k, j)
                    BcastList bcast_list_B;
                    for (int64_t j = 0; j < B.nt(); ++j) {
                        bcast_list_B.push_back(
                            {k, j, {C.sub(0, C.mt()-1, j, j)}});
                    }
                    B.template listBcast<target>(bcast_list_B, layout);
                }
            }

            // multiply alpha A(:, 0) B(0, :), which is: (symm / gemm):
            // C(0, :)      = alpha [ A(0, 0)      B(0, :) ] + beta C(0, :)
            // C(1:mt-1, :) = alpha [ A(1:mt-1, 0) B(0, :) ] + beta C(1:mt-1, :)
            #pragma omp task depend(in:bcast[0]) \
                             depend(out:gemm[0])
            {
                internal::symm<Target::HostTask>(
                    Side::Left,
                    alpha, A.sub(0, 0),
                           B.sub(0, 0, 0, B.nt()-1),
                    beta,  C.sub(0, 0, 0, C.nt()-1));

                if (A.mt()-1 > 0) {
                    internal::gemm<target>(
                        alpha, A.sub(1, A.mt()-1, 0, 0),
                               B.sub(0, 0, 0, B.nt()-1),
                        beta,  C.sub(1, C.mt()-1, 0, C.nt()-1),
                        layout);
                }
            }

            for (int64_t k = 1; k < A.nt(); ++k) {

                // send next block col of A and block row of B
                if (k+lookahead < A.nt()) {
                    #pragma omp task depend(in:gemm[k-1]) \
                                     depend(in:bcast[k+lookahead-1]) \
                                     depend(out:bcast[k+lookahead])
                    {
                        // broadcast A(k+la, i) or A(i, k+la)
                        // to ranks owning block row C(i, :)
                        BcastList bcast_list_A;
                        for (int64_t i = 0; i < k+lookahead; ++i) {
                            bcast_list_A.push_back(
                                {k+lookahead, i, {C.sub(i, i, 0, C.nt()-1)}});
                        }
                        for (int64_t i = k+lookahead; i < A.mt(); ++i) {
                            bcast_list_A.push_back(
                                {i, k+lookahead, {C.sub(i, i, 0, C.nt()-1)}});
                        }
                        A.template listBcast<target>(bcast_list_A, layout);

                        // broadcast B(k+la, j) to ranks
                        // owning block col C(0:k+la, j)
                        BcastList bcast_list_B;
                        for (int64_t j = 0; j < B.nt(); ++j) {
                            bcast_list_B.push_back(
                                {k+lookahead, j, {C.sub(0, C.mt()-1, j, j)}});
                        }
                        B.template listBcast<target>(bcast_list_B, layout);
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
                        alpha,         transpose(Arow_k),
                                       B.sub(k, k, 0, B.nt()-1),
                        scalar_t(1.0), C.sub(0, k-1, 0, C.nt()-1),
                        layout);

                    internal::symm<Target::HostTask>(
                        Side::Left,
                        alpha,         A.sub(k, k),
                                       B.sub(k, k, 0, B.nt()-1),
                        scalar_t(1.0), C.sub(k, k, 0, C.nt()-1));

                    if (A.mt()-1 > k) {
                        internal::gemm<target>(
                            alpha,         A.sub(k+1, A.mt()-1, k, k),
                                           B.sub(k, k, 0, B.nt()-1),
                            scalar_t(1.0), C.sub(k+1, C.mt()-1, 0, C.nt()-1),
                            layout);
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
                BcastList bcast_list_A;
                for (int64_t i = 0; i < A.mt(); ++i)
                    bcast_list_A.push_back({0, i, {C.sub(i, i, 0, C.nt()-1)}});
                A.template listBcast<target>(bcast_list_A, layout);

                BcastList bcast_list_B;
                // broadcast B(0, j) to ranks owning block col C(:, j)
                for (int64_t j = 0; j < B.nt(); ++j)
                    bcast_list_B.push_back({0, j, {C.sub(0, C.mt()-1, j, j)}});
                B.template listBcast<target>(bcast_list_B, layout);
            }

            // send next lookahead block cols of A and block rows of B
            for (int64_t k = 1; k < lookahead+1 && k < A.nt(); ++k) {
                #pragma omp task depend(in:bcast[k-1]) \
                                 depend(out:bcast[k])
                {
                    // broadcast A(k, i) or A(i, k)
                    // to ranks owning block row C(i, :)
                    BcastList bcast_list_A;
                    for (int64_t i = 0; i < k && i < A.mt(); ++i) {
                        bcast_list_A.push_back(
                            {i, k, {C.sub(i, i, 0, C.nt()-1)}});
                    }
                    for (int64_t i = k; i < A.mt(); ++i) {
                        bcast_list_A.push_back(
                            {k, i, {C.sub(i, i, 0, C.nt()-1)}});
                    }
                    A.template listBcast<target>(bcast_list_A, layout);

                    // broadcast B(k, j) to ranks owning block col C(0:k, j)
                    BcastList bcast_list_B;
                    for (int64_t j = 0; j < B.nt(); ++j) {
                        bcast_list_B.push_back(
                            {k, j, {C.sub(0, C.mt()-1, j, j)}});
                    }
                    B.template listBcast<target>(bcast_list_B, layout);
                }
            }

            // multiply alpha A(:, 0) B(0, :), which is (symm / gemm):
            // C(0, :)      = alpha [ A(0, 0)      B(0, :) ] + beta C(0, :)
            // C(1:mt-1, :) = alpha [ A(1:mt-1, 0) B(0, :) ] + beta C(1:mt-1, :)
            #pragma omp task depend(in:bcast[0]) \
                             depend(out:gemm[0])
            {
                internal::symm<Target::HostTask>(
                    Side::Left,
                    alpha, A.sub(0, 0),
                           B.sub(0, 0, 0, B.nt()-1),
                    beta,  C.sub(0, 0, 0, C.nt()-1));

                if (A.mt()-1 > 0) {
                    auto Arow_k = A.sub(0, 0, 1, A.mt()-1);
                    internal::gemm<target>(
                        alpha, transpose(Arow_k),
                               B.sub(0, 0, 0, B.nt()-1),
                        beta,  C.sub(1, C.mt()-1, 0, C.nt()-1),
                        layout);
                }
            }

            for (int64_t k = 1; k < A.nt(); ++k) {

                // send next block col of A and block row of B
                if (k+lookahead < A.nt()) {
                    #pragma omp task depend(in:gemm[k-1]) \
                                     depend(in:bcast[k+lookahead-1]) \
                                     depend(out:bcast[k+lookahead])
                    {
                        // broadcast A(k+la, i) or A(i, k+la)
                        // to ranks owning block row C(i, :)
                        BcastList bcast_list_A;
                        for (int64_t i = 0; i < k+lookahead; ++i) {
                            bcast_list_A.push_back(
                                {i, k+lookahead, {C.sub(i, i, 0, C.nt()-1)}});
                        }
                        for (int64_t i = k+lookahead; i < A.mt(); ++i) {
                            bcast_list_A.push_back(
                                {k+lookahead, i, {C.sub(i, i, 0, C.nt()-1)}});
                        }
                        A.template listBcast<target>(bcast_list_A, layout);

                        // broadcast B(k+la, j) to ranks
                        // owning block col C(0:k+la, j)
                        BcastList bcast_list_B;
                        for (int64_t j = 0; j < B.nt(); ++j) {
                            bcast_list_B.push_back(
                                {k+lookahead, j, {C.sub(0, C.mt()-1, j, j)}});
                        }
                        B.template listBcast<target>(bcast_list_B, layout);
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
                        scalar_t(1.0), C.sub(0, k-1, 0, C.nt()-1),
                        layout);

                    internal::symm<Target::HostTask>(
                        Side::Left,
                        alpha,         A.sub(k, k),
                                       B.sub(k, k, 0, B.nt()-1),
                        scalar_t(1.0), C.sub(k, k, 0, C.nt()-1));

                    if (A.mt()-1 > k) {
                        auto Arow_k = A.sub(k, k, k+1, A.mt()-1);
                        internal::gemm<target>(
                            alpha,         transpose(Arow_k),
                                           B.sub(k, k, 0, B.nt()-1),
                            scalar_t(1.0), C.sub(k+1, C.mt()-1, 0, C.nt()-1),
                            layout);
                    }
                }
            }
        }
        #pragma omp taskwait
        C.tileUpdateAllOrigin();
    }

    C.clearWorkspace();
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup symm_specialization
///
template <Target target, typename scalar_t>
void symm(Side side,
          scalar_t alpha, SymmetricMatrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
          Options const& opts)
{
    int64_t lookahead;
    try {
        lookahead = opts.at(Option::Lookahead).i_;
    }
    catch (std::out_of_range&) {
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
/// Distributed parallel symmetric matrix-matrix multiplication.
/// Performs one of the matrix-matrix operations
/// \[
///     C = \alpha A B + \beta C
/// \]
/// or
/// \[
///     C = \alpha B A + \beta C
/// \]
/// where alpha and beta are scalars, A is a symmetric matrix and B and
/// C are m-by-n matrices.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///         One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] side
///         Whether the symmetric matrix A appears on the left or right:
///         - Side::Left:  $C = \alpha A B + \beta C$
///         - Side::Right: $C = \alpha B A + \beta C$
///
/// @param[in] alpha
///         The scalar alpha.
///
/// @param[in] A
///         - If side = left,  the m-by-m symmetric matrix A;
///         - if side = right, the n-by-n symmetric matrix A.
///
/// @param[in] B
///         The m-by-n matrix B.
///
/// @param[in] beta
///         The scalar beta.
///
/// @param[in,out] C
///         On entry, the m-by-n matrix C.
///         On exit, overwritten by the result
///         $\alpha A B + \beta C$ or $\alpha B A + \beta C$.
///
/// @param[in] opts
///         Additional options, as map of name = value pairs. Possible options:
///         - Option::Lookahead:
///           Number of blocks to overlap communication and computation.
///           lookahead >= 0. Default 1.
///         - Option::Target:
///           Implementation to target. Possible values:
///           - HostTask:  OpenMP tasks on CPU host [default].
///           - HostNest:  nested OpenMP parallel for loop on CPU host.
///           - HostBatch: batched BLAS on CPU host.
///           - Devices:   batched BLAS on GPU device.
///
/// @ingroup symm
///
template <typename scalar_t>
void symm(Side side,
          scalar_t alpha, SymmetricMatrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
          Options const& opts)
{
    Target target;
    try {
        target = Target(opts.at(Option::Target).i_);
    }
    catch (std::out_of_range&) {
        target = Target::HostTask;
    }

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            symm<Target::HostTask>(side, alpha, A, B, beta, C, opts);
            break;
        case Target::HostNest:
            symm<Target::HostNest>(side, alpha, A, B, beta, C, opts);
            break;
        case Target::HostBatch:
            symm<Target::HostBatch>(side, alpha, A, B, beta, C, opts);
            break;
        case Target::Devices:
            symm<Target::Devices>(side, alpha, A, B, beta, C, opts);
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void symm< float >(
    Side side,
    float alpha, SymmetricMatrix<float>& A,
                 Matrix<float>& B,
    float beta,  Matrix<float>& C,
    Options const& opts);

template
void symm< double >(
    Side side,
    double alpha, SymmetricMatrix<double>& A,
                  Matrix<double>& B,
    double beta,  Matrix<double>& C,
    Options const& opts);

template
void symm< std::complex<float> >(
    Side side,
    std::complex<float> alpha, SymmetricMatrix< std::complex<float> >& A,
                               Matrix< std::complex<float> >& B,
    std::complex<float> beta,  Matrix< std::complex<float> >& C,
    Options const& opts);

template
void symm< std::complex<double> >(
    Side side,
    std::complex<double> alpha, SymmetricMatrix< std::complex<double> >& A,
                                Matrix< std::complex<double> >& B,
    std::complex<double> beta,  Matrix< std::complex<double> >& C,
    Options const& opts);

} // namespace slate
