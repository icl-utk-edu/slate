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

#include <list>
#include <tuple>

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::gbmm from internal::specialization::gbmm
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// @internal
/// Distributed parallel general matrix-matrix multiplication.
/// Generic implementation for any target.
/// Dependencies enforce the following behavior:
/// - bcast communications are serialized,
/// - gemm operations are serialized,
/// - bcasts can get ahead of gemms by the value of lookahead.
/// ColMajor layout is assumed
///
/// @ingroup gbmm_specialization
///
template <Target target, typename scalar_t>
void gbmm(slate::internal::TargetType<target>,
          scalar_t alpha, BandMatrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
          int64_t lookahead)
{
    using blas::min;
    using blas::max;
    using BcastList = typename Matrix<scalar_t>::BcastList;

    // Assumes column major
    // todo: relax this assumption, by ?
    //       or pass as parameter
    const Layout layout = Layout::ColMajor;

    const scalar_t one = 1.0;

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector<uint8_t> bcast_vector(A.nt());
    std::vector<uint8_t>  gemm_vector(A.nt());
    uint8_t* bcast = bcast_vector.data();
    uint8_t* gemm  =  gemm_vector.data();

    int64_t kl = A.lowerBandwidth();
    int64_t ku = A.upperBandwidth();

    // todo: initially, assume fixed size, square tiles for simplicity
    int64_t klt = ceildiv(kl, A.tileNb(0));
    int64_t kut = ceildiv(ku, A.tileNb(0));

    if (target == Target::Devices) {
        C.allocateBatchArrays();
        C.reserveDeviceWorkspace();
    }

    #pragma omp parallel
    #pragma omp master
    {
        omp_set_nested(1);
        // send first block col of A and block row of B
        #pragma omp task depend(out:bcast[0])
        {
            int64_t i_begin = 0;
            int64_t i_end   = min(0 + klt + 1, A.mt());

            // broadcast A(i, 0) to ranks owning block row C(i, :)
            BcastList bcast_list_A;
            for (int64_t i = i_begin; i < i_end; ++i)
                bcast_list_A.push_back({i, 0, {C.sub(i, i, 0, C.nt()-1)}});
            A.template listBcast<target>(bcast_list_A, layout);

            // broadcast B(0, j) to ranks owning block col C(:, j)
            BcastList bcast_list_B;
            for (int64_t j = 0; j < B.nt(); ++j)
                bcast_list_B.push_back({0, j, {C.sub(i_begin, i_end-1, j, j)}});
            B.template listBcast<target>(bcast_list_B, layout);
        }

        // send next lookahead block cols of A and block rows of B
        for (int64_t k = 1; k < lookahead+1 && k < A.nt(); ++k) {
            #pragma omp task depend(in:bcast[k-1]) \
                             depend(out:bcast[k])
            {
                int64_t i_begin = max(k - kut, 0);
                int64_t i_end   = min(k + klt + 1, A.mt());

                // broadcast A(i, k) to ranks owning block row C(i, :)
                BcastList bcast_list_A;
                for (int64_t i = i_begin; i < i_end; ++i)
                    bcast_list_A.push_back({i, k, {C.sub(i, i, 0, C.nt()-1)}});
                A.template listBcast<target>(bcast_list_A, layout);

                // broadcast B(k, j) to ranks owning block col C(:, j)
                BcastList bcast_list_B;
                for (int64_t j = 0; j < B.nt(); ++j)
                    bcast_list_B.push_back({k, j, {C.sub(i_begin, i_end-1, j, j)}});
                B.template listBcast<target>(bcast_list_B, layout);
            }
        }

        // multiply alpha A(:, 0) B(0, :) + beta C
        #pragma omp task depend(in:bcast[0]) \
                         depend(out:gemm[0])
        {
            int64_t i_begin = 0;
            int64_t i_end   = min(0 + klt + 1, A.mt());

            internal::gemm<target>(
                    alpha, A.sub(i_begin, i_end-1, 0, 0),
                           B.sub(0, 0, 0, B.nt()-1),
                    beta,  C.sub(i_begin, i_end-1, 0, C.nt()-1),
                    layout);

            if (beta != one) {
                // Scale block rows of C below the bandwidth of A:
                // C(i_end : mt-1, :) = beta * C(i_end : mt-1, :)
                // todo: make internal::scale routine. This is HostTask.
                for (int64_t i = i_end; i < C.mt(); ++i) {
                    for (int64_t j = 0; j < C.nt(); ++j) {
                        if (C.tileIsLocal(i, j)) {
                            #pragma omp task shared(C)
                            {
                                C.tileGetForWriting(i, j, LayoutConvert(layout));
                                scale(beta, C(i, j));
                            }
                        }
                    }
                }
                #pragma omp taskwait
            }
        }

        for (int64_t k = 1; k < A.nt(); ++k) {

            // send next block col of A and block row of B
            if (k+lookahead < A.nt()) {
                #pragma omp task depend(in:gemm[k-1]) \
                                 depend(in:bcast[k+lookahead-1]) \
                                 depend(out:bcast[k+lookahead])
                {
                    int64_t i_begin = max(k + lookahead - kut, 0);
                    int64_t i_end   = min(k + lookahead + klt + 1, A.mt());

                    // broadcast A(i, k+la) to ranks owning block row C(i, :)
                    BcastList bcast_list_A;
                    for (int64_t i = i_begin; i < i_end; ++i) {
                        bcast_list_A.push_back(
                            {i, k+lookahead, {C.sub(i, i, 0, C.nt()-1)}});
                    }
                    A.template listBcast<target>(bcast_list_A, layout);

                    // broadcast B(k+la, j) to ranks owning block col C(:, j)
                    BcastList bcast_list_B;
                    for (int64_t j = 0; j < B.nt(); ++j) {
                        bcast_list_B.push_back(
                            {k+lookahead, j, {C.sub(i_begin, i_end-1, j, j)}});
                    }
                    B.template listBcast<target>(bcast_list_B, layout);
                }
            }

            int64_t i_begin = max(k - kut, 0);
            int64_t i_end   = min(k + klt + 1, A.mt());

            if (i_begin <= i_end-1) {
                // multiply alpha A(:, k) B(k, :) + C, no beta
                #pragma omp task depend(in:bcast[k]) \
                                 depend(in:gemm[k-1]) \
                                 depend(out:gemm[k])
                {
                    internal::gemm<target>(
                        alpha, A.sub(i_begin, i_end-1, k, k),
                               B.sub(k, k, 0, B.nt()-1),
                        one,   C.sub(i_begin, i_end-1, 0, C.nt()-1),
                        layout);
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
/// @ingroup gbmm_specialization
///
template <Target target, typename scalar_t>
void gbmm(scalar_t alpha, BandMatrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
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

    internal::specialization::gbmm(internal::TargetType<target>(),
                                   alpha, A,
                                          B,
                                   beta,  C,
                                   lookahead);
}

//------------------------------------------------------------------------------
/// Distributed parallel general matrix-matrix multiplication.
/// Performs the matrix-matrix operation
/// \[
///     C = \alpha A B + \beta C,
/// \]
/// where alpha and beta are scalars, and $A$, $B$, and $C$ are matrices, with
/// $A$ an m-by-k band matrix, $B$ a k-by-n matrix, and $C$ an m-by-n matrix.
/// The matrices can be transposed or conjugate-transposed beforehand, e.g.,
///
///     auto AT = slate::transpose( A );
///     auto BT = slate::conjTranspose( B );
///     slate::gbmm( alpha, AT, BT, beta, C );
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///         One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] alpha
///         The scalar alpha.
///
/// @param[in] A
///         The m-by-k band matrix A.
///
/// @param[in] B
///         The k-by-n matrix B.
///
/// @param[in] beta
///         The scalar beta.
///
/// @param[in,out] C
///         On entry, the m-by-n matrix C.
///         On exit, overwritten by the result $\alpha A B + \beta C$.
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
/// @ingroup gbmm
///
template <typename scalar_t>
void gbmm(scalar_t alpha, BandMatrix<scalar_t>& A,
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
            gbmm<Target::HostTask>(alpha, A, B, beta, C, opts);
            break;
        case Target::HostNest:
            gbmm<Target::HostNest>(alpha, A, B, beta, C, opts);
            break;
        case Target::HostBatch:
            gbmm<Target::HostBatch>(alpha, A, B, beta, C, opts);
            break;
        case Target::Devices:
            gbmm<Target::Devices>(alpha, A, B, beta, C, opts);
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gbmm<float>(
    float alpha, BandMatrix<float>& A,
                 Matrix<float>& B,
    float beta,  Matrix<float>& C,
    Options const& opts);

template
void gbmm<double>(
    double alpha, BandMatrix<double>& A,
                  Matrix<double>& B,
    double beta,  Matrix<double>& C,
    Options const& opts);

template
void gbmm< std::complex<float> >(
    std::complex<float> alpha, BandMatrix< std::complex<float> >& A,
                               Matrix< std::complex<float> >& B,
    std::complex<float> beta,  Matrix< std::complex<float> >& C,
    Options const& opts);

template
void gbmm< std::complex<double> >(
    std::complex<double> alpha, BandMatrix< std::complex<double> >& A,
                                Matrix< std::complex<double> >& B,
    std::complex<double> beta,  Matrix< std::complex<double> >& C,
    Options const& opts);

} // namespace slate
