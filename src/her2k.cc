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
#include "internal/internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::her2k from internal::specialization::her2k
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// @internal
/// Distributed parallel Hermitian rank 2k update.
/// Generic implementation for any target.
/// Dependencies enforce the following behavior:
/// - bcast communications are serialized,
/// - her2k operations are serialized,
/// - bcasts can get ahead of her2ks by the value of lookahead.
/// Note A, B, and C are passed by value, so we can transpose if needed
/// (for uplo = Upper) without affecting caller.
/// @ingroup her2k_specialization
///
template <Target target, typename scalar_t>
void her2k(slate::internal::TargetType<target>,
           scalar_t alpha,                  Matrix<scalar_t> A,
                                            Matrix<scalar_t> B,
           blas::real_type<scalar_t> beta,  HermitianMatrix<scalar_t> C,
           int64_t lookahead)
{
    using real_t = blas::real_type<scalar_t>;
    using BcastList = typename Matrix<scalar_t>::BcastList;

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    // if upper, change to lower
    if (C.uplo() == Uplo::Upper)
        C = conjTranspose(C);

    // A is mt-by-nt, C is mt-by-mt
    assert(A.mt() == C.mt());
    assert(B.mt() == C.mt());
    assert(A.nt() == B.nt());

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
        // Lower/NoTrans or Upper/ConjTrans case
        // send 1st block col of A
        #pragma omp task depend(out:bcast[0])
        {
            // broadcast A(i, 0) and B(i, 0) to ranks owning
            // block row C(i, 0:i) and block col C(i:n, i)
            BcastList bcast_list_A;
            BcastList bcast_list_B;
            for (int64_t i = 0; i < A.mt(); ++i) {
                bcast_list_A.push_back({i, 0, {C.sub(i, i, 0, i),
                                               C.sub(i, C.mt()-1, i, i)
                                              }});
                bcast_list_B.push_back({i, 0, {C.sub(i, i, 0, i),
                                               C.sub(i, C.mt()-1, i, i)
                                              }});
            }
            A.template listBcast<target>(bcast_list_A, layout);
            B.template listBcast<target>(bcast_list_B, layout);
        }

        // send next lookahead block cols of A
        for (int64_t k = 1; k < lookahead+1 && k < A.nt(); ++k) {
            #pragma omp task depend(in:bcast[k-1]) \
                             depend(out:bcast[k])
            {
                // broadcast A(i, k) and B(i, k) to ranks owning
                // block row C(i, 0:i) and block col C(i:n, i)
                BcastList bcast_list_A;
                BcastList bcast_list_B;
                for (int64_t i = 0; i < A.mt(); ++i) {
                    bcast_list_A.push_back({i, k, {C.sub(i, i, 0, i),
                                                   C.sub(i, C.mt()-1, i, i)}});
                    bcast_list_B.push_back({i, k, {C.sub(i, i, 0, i),
                                                   C.sub(i, C.mt()-1, i, i)}});
                }
                A.template listBcast<target>(bcast_list_A, layout);
                B.template listBcast<target>(bcast_list_B, layout);
            }
        }

        // multiply alpha A(:, 0) B(0, :)^H + beta C
        #pragma omp task depend(in:bcast[0]) \
                         depend(out:gemm[0])
        {
            internal::her2k<target>(
                alpha, A.sub(0, A.mt()-1, 0, 0),
                       B.sub(0, B.mt()-1, 0, 0),
                beta,  std::move(C));
        }

        for (int64_t k = 1; k < A.nt(); ++k) {

            // send next block col of A and block row of B
            if (k+lookahead < A.nt()) {
                #pragma omp task depend(in:gemm[k-1]) \
                                 depend(in:bcast[k+lookahead-1]) \
                                 depend(out:bcast[k+lookahead])
                {
                    // broadcast A(k+la, i) to ranks owning
                    // block row C(i, 0:i) and block col C(i:n, i)
                    BcastList bcast_list_A;
                    BcastList bcast_list_B;
                    for (int64_t i = 0; i < A.mt(); ++i) {
                        bcast_list_A.push_back(
                            {i, k+lookahead, {C.sub(i, i, 0, i),
                                              C.sub(i, C.mt()-1, i, i)}});
                        bcast_list_B.push_back(
                            {i, k+lookahead, {C.sub(i, i, 0, i),
                                              C.sub(i, C.mt()-1, i, i)}});
                    }
                    A.template listBcast<target>(bcast_list_A, layout);
                    B.template listBcast<target>(bcast_list_B, layout);
                }
            }

            // multiply alpha A(:, k) A(k, :) + C, no beta
            #pragma omp task depend(in:bcast[k]) \
                             depend(in:gemm[k-1]) \
                             depend(out:gemm[k])
            {
                internal::her2k<target>(
                    alpha,       A.sub(0, A.mt()-1, k, k),
                                 B.sub(0, B.mt()-1, k, k),
                    real_t(1.0), std::move(C));
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
/// @ingroup her2k_specialization
///
template <Target target, typename scalar_t>
void her2k(scalar_t alpha,                  Matrix<scalar_t>& A,
                                            Matrix<scalar_t>& B,
           blas::real_type<scalar_t> beta,  HermitianMatrix<scalar_t>& C,
           Options const& opts)
{
    int64_t lookahead;
    try {
        lookahead = opts.at(Option::Lookahead).i_;
        assert(lookahead >= 0);
    }
    catch (std::out_of_range&) {
        lookahead = 1;
    }

    internal::specialization::her2k(internal::TargetType<target>(),
                                    alpha, A,
                                           B,
                                    beta,  C,
                                    lookahead);
}

//------------------------------------------------------------------------------
/// Distributed parallel Hermitian rank 2k update.
/// Performs the Hermitian rank 2k operation
/// \[
///     C = \alpha A B^H + \alpha B A^H + \beta C,
/// \]
/// where alpha and beta are scalars, C is an n-by-n Hermitian
/// matrix, and A and B are an n-by-k matrices.
/// The matrices can be conjugate-transposed beforehand, e.g.,
///
///     auto AT = slate::conjTranspose( A );
///     auto BT = slate::conjTranspose( B );
///     slate::her2k( alpha, AT, BT, beta, C );
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///         One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] alpha
///         The scalar alpha.
///
/// @param[in] A
///         The n-by-k matrix A.
///
/// @param[in] B
///         The n-by-k matrix B.
///
/// @param[in] beta
///         The real scalar beta.
///
/// @param[in,out] C
///         On entry, the n-by-n Hermitian matrix C.
///         On exit, overwritten by the result
///         $C = \alpha A B^H + \alpha B A^H + \beta C$.
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
/// @ingroup her2k
///
template <typename scalar_t>
void her2k(scalar_t alpha,                 Matrix<scalar_t>& A,
                                           Matrix<scalar_t>& B,
           blas::real_type<scalar_t> beta, HermitianMatrix<scalar_t>& C,
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
            her2k<Target::HostTask>(alpha, A, B, beta, C, opts);
            break;
        case Target::HostNest:
            her2k<Target::HostNest>(alpha, A, B, beta, C, opts);
            break;
        case Target::HostBatch:
            her2k<Target::HostBatch>(alpha, A, B, beta, C, opts);
            break;
        case Target::Devices:
            her2k<Target::Devices>(alpha, A, B, beta, C, opts);
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void her2k<float>(
    float alpha, Matrix<float>& A,
                 Matrix<float>& B,
    float beta,  HermitianMatrix<float>& C,
    Options const& opts);

template
void her2k<double>(
    double alpha, Matrix<double>& A,
                  Matrix<double>& B,
    double beta,  HermitianMatrix<double>& C,
    Options const& opts);

template
void her2k< std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >& A,
                               Matrix< std::complex<float> >& B,
    float beta,                HermitianMatrix< std::complex<float> >& C,
    Options const& opts);

template
void her2k< std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >& A,
                                Matrix< std::complex<double> >& B,
    double beta,                HermitianMatrix< std::complex<double> >& C,
    Options const& opts);

} // namespace slate
