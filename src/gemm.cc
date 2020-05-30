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
#include "internal/internal_batch.hh"
#include "aux/Debug.hh"

#include <list>
#include <tuple>

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::gemm from internal::specialization::gemm
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
/// @ingroup gemm_specialization
///
template <Target target, typename scalar_t>
void gemm(slate::internal::TargetType<target>,
          scalar_t alpha, Matrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
          int64_t lookahead)
{
    using BcastList = typename Matrix<scalar_t>::BcastList;

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector<uint8_t> bcast_vector(A.nt());
    std::vector<uint8_t> gemm_vector(A.nt());
    std::vector<uint8_t> c_vector(1);
    uint8_t* bcast = bcast_vector.data();
    uint8_t* gemm  =  gemm_vector.data();
    uint8_t* c     =     c_vector.data();

    if (target == Target::Devices) {
        C.allocateBatchArrays();
        C.reserveDeviceWorkspace();
    }

    #pragma omp parallel
    #pragma omp master
    {
        omp_set_nested(1);

        if (target == Target::Devices) {
            // fetch C matrix tiles into devices in parallel with first MPI broadcast
            #pragma omp task depend(out:c[0])
            {
                trace::Block trace_block("fetch_C");
                C.tileGetAllForWritingOnDevices(LayoutConvert(layout));
            }
        }

        // send first block col of A and block row of B
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
                // broadcast A(i, k) to ranks owning block row C(i, :)
                BcastList bcast_list_A;
                for (int64_t i = 0; i < A.mt(); ++i)
                    bcast_list_A.push_back({i, k, {C.sub(i, i, 0, C.nt()-1)}});
                A.template listBcast<target>(bcast_list_A, layout);

                // broadcast B(k, j) to ranks owning block col C(:, j)
                BcastList bcast_list_B;
                for (int64_t j = 0; j < B.nt(); ++j)
                    bcast_list_B.push_back({k, j, {C.sub(0, C.mt()-1, j, j)}});
                B.template listBcast<target>(bcast_list_B, layout);
            }
        }

        // multiply alpha A(:, 0) B(0, :) + beta C
        #pragma omp task depend(in:bcast[0]) \
                         depend(in:c[0]) \
                         depend(out:gemm[0])
        {
            internal::gemm<target>(
                    alpha, A.sub(0, A.mt()-1, 0, 0),
                           B.sub(0, 0, 0, B.nt()-1),
                    beta,  std::move(C),
                    layout);
        }

        for (int64_t k = 1; k < A.nt(); ++k) {

            // send next block col of A and block row of B
            if (k+lookahead < A.nt()) {
                #pragma omp task depend(in:gemm[k-1]) \
                                 depend(in:bcast[k+lookahead-1]) \
                                 depend(out:bcast[k+lookahead])
                {
                    // broadcast A(i, k+la) to ranks owning block row C(i, :)
                    BcastList bcast_list_A;
                    for (int64_t i = 0; i < A.mt(); ++i) {
                        bcast_list_A.push_back(
                            {i, k+lookahead, {C.sub(i, i, 0, C.nt()-1)}});
                    }
                    A.template listBcast<target>(bcast_list_A, layout);

                    // broadcast B(k+la, j) to ranks owning block col C(:, j)
                    BcastList bcast_list_B;
                    for (int64_t j = 0; j < B.nt(); ++j) {
                        bcast_list_B.push_back(
                            {k+lookahead, j, {C.sub(0, C.mt()-1, j, j)}});
                    }
                    B.template listBcast<target>(bcast_list_B, layout);
                }
            }

            // multiply alpha A(:, k) B(k, :) + C, no beta
            #pragma omp task depend(in:bcast[k]) \
                             depend(in:gemm[k-1]) \
                             depend(out:gemm[k])
            {
                internal::gemm<target>(
                    alpha,         A.sub(0, A.mt()-1, k, k),
                                   B.sub(k, k, 0, B.nt()-1),
                    scalar_t(1.0), std::move(C),
                    layout);
            }
        }
        #pragma omp taskwait
        C.tileUpdateAllOrigin();
    }

    C.releaseWorkspace();
}

//------------------------------------------------------------------------------
/// @internal
/// Distributed parallel general matrix-matrix multiplication.
/// Device target implementation.
/// Dependencies enforce the following behavior:
/// - bcast communications are serialized,
/// - gemm operations are serialized,
/// - bcasts can get ahead of gemms by the value of lookahead.
/// ColMajor layout is assumed
///
/// @ingroup gemm_specialization
///
template <typename scalar_t>
void gemm(slate::internal::TargetType<Target::Devices>,
          scalar_t alpha, Matrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
          int64_t lookahead)
{
    using BcastList = typename Matrix<scalar_t>::BcastList;

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector<uint8_t> bcast_vector(A.nt());
    std::vector<uint8_t>  gemm_vector(A.nt());
    uint8_t* bcast = bcast_vector.data();
    uint8_t* gemm  =  gemm_vector.data();

    std::vector< GemmBatchArrays<scalar_t> > gemmBatch(
                    lookahead+1,
                    GemmBatchArrays<scalar_t>(A.num_devices()));

    #pragma omp parallel
    #pragma omp master
    {
        omp_set_nested(1);

        C.reserveDeviceWorkspace();

        // send first block col of A and block row of B
        #pragma omp task depend(out:bcast[0])
        {
            // broadcast A(i, 0) to ranks owning block row C(i, :)
            {
                BcastList bcast_list_A;
                for (int64_t i = 0; i < A.mt(); ++i)
                    bcast_list_A.push_back({i, 0, {C.sub(i, i, 0, C.nt()-1)}});
                A.template listBcast<Target::Devices>(bcast_list_A, layout);
            }

            // broadcast B(0, j) to ranks owning block col C(:, j)
            {
                BcastList bcast_list_B;
                for (int64_t j = 0; j < B.nt(); ++j)
                    bcast_list_B.push_back({0, j, {C.sub(0, C.mt()-1, j, j)}});
                B.template listBcast<Target::Devices>(bcast_list_B, layout);
            }

            // prepare first internal::gemm pointer's arrays and prefetch its data
            internal::gemmPrep<Target::Devices>(
                    alpha, A.sub(0, A.mt()-1, 0, 0),
                           B.sub(0, 0, 0, B.nt()-1),
                    beta,  std::move(C),
                    std::move(&gemmBatch[0]),
                    layout);
        }

        // send next lookahead block cols of A and block rows of B
        for (int64_t k = 1; k < lookahead+1 && k < A.nt(); ++k) {
            #pragma omp task depend(in:bcast[k-1]) \
                             depend(out:bcast[k])
            {
                // broadcast A(i, k) to ranks owning block row C(i, :)
                {

                    BcastList bcast_list_A;
                    for (int64_t i = 0; i < A.mt(); ++i)
                        bcast_list_A.push_back({i, k, {C.sub(i, i, 0, C.nt()-1)}});
                    A.template listBcast<Target::Devices>(bcast_list_A, layout);
                }

                // broadcast B(k, j) to ranks owning block col C(:, j)
                {
                    BcastList bcast_list_B;
                    for (int64_t j = 0; j < B.nt(); ++j)
                        bcast_list_B.push_back({k, j, {C.sub(0, C.mt()-1, j, j)}});
                    B.template listBcast<Target::Devices>(bcast_list_B, layout);
                }

                // prepare lookahead internal::gemm pointer's arrays
                // data is already prefetched
                internal::gemmPrep<Target::Devices>(
                        alpha,         A.sub(0, A.mt()-1, k, k),
                                       B.sub(k, k, 0, B.nt()-1),
                        scalar_t(1.0), std::move(C),
                        std::move(&gemmBatch[k % (lookahead+1)]),
                        layout, true);
            }
        }

        // multiply alpha A(:, 0) B(0, :) + beta C
        #pragma omp task depend(in:bcast[0]) \
                         depend(out:gemm[0])
        {
            internal::gemmExec<Target::Devices>(
                    alpha, A.sub(0, A.mt()-1, 0, 0),
                           B.sub(0, 0, 0, B.nt()-1),
                    beta,  std::move(C),
                    std::move(&gemmBatch[0]),
                    layout);
        }

        for (int64_t k = 1; k < A.nt(); ++k) {

            // send next block col of A and block row of B
            if (k+lookahead < A.nt()) {
                #pragma omp task depend(in:gemm[k-1]) \
                                 depend(in:bcast[k+lookahead-1]) \
                                 depend(out:bcast[k+lookahead])
                {
                    // broadcast A(i, k+la) to ranks owning block row C(i, :)
                    {
                        BcastList bcast_list_A;
                        for (int64_t i = 0; i < A.mt(); ++i) {
                            bcast_list_A.push_back(
                                {i, k+lookahead, {C.sub(i, i, 0, C.nt()-1)}});
                        }
                        A.template listBcast<Target::Devices>(bcast_list_A, layout);
                    }

                    // broadcast B(k+la, j) to ranks owning block col C(:, j)
                    {
                        BcastList bcast_list_B;
                        for (int64_t j = 0; j < B.nt(); ++j) {
                            bcast_list_B.push_back(
                                {k+lookahead, j, {C.sub(0, C.mt()-1, j, j)}});
                        }
                        B.template listBcast<Target::Devices>(bcast_list_B, layout);
                    }

                    // prepare lookahead internal::gemm pointer's arrays
                    // data is already prefetched
                    internal::gemmPrep<Target::Devices>(
                        alpha,         A.sub(0, A.mt()-1, k+lookahead, k+lookahead),
                                       B.sub(k+lookahead, k+lookahead, 0, B.nt()-1),
                        scalar_t(1.0), std::move(C),
                        std::move(&gemmBatch[(k+lookahead) % (lookahead+1)]),
                        layout, true);
                }
            }

            // multiply alpha A(:, k) B(k, :) + C, no beta
            #pragma omp task depend(in:bcast[k]) \
                             depend(in:gemm[k-1]) \
                             depend(out:gemm[k])
            {
                internal::gemmExec<Target::Devices>(
                    alpha,         A.sub(0, A.mt()-1, k, k),
                                   B.sub(k, k, 0, B.nt()-1),
                    scalar_t(1.0), std::move(C),
                    std::move(&gemmBatch[k % (lookahead+1)]),
                    layout);
            }
        }

        #pragma omp taskwait
        C.tileUpdateAllOrigin();
    }

    C.releaseWorkspace();
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup gemm_specialization
///
template <Target target, typename scalar_t>
void gemm(scalar_t alpha, Matrix<scalar_t>& A,
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

    internal::specialization::gemm(internal::TargetType<target>(),
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
/// $A$ an m-by-k matrix, $B$ a k-by-n matrix, and $C$ an m-by-n matrix.
/// The matrices can be transposed or conjugate-transposed beforehand, e.g.,
///
///     auto AT = slate::transpose( A );
///     auto BT = slate::conjTranspose( B );
///     slate::gemm( alpha, AT, BT, beta, C );
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///         One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] alpha
///         The scalar alpha.
///
/// @param[in] A
///         The m-by-k matrix A.
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
/// @ingroup gemm
///
template <typename scalar_t>
void gemm(scalar_t alpha, Matrix<scalar_t>& A,
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
            gemm<Target::HostTask>(alpha, A, B, beta, C, opts);
            break;
        case Target::HostNest:
            gemm<Target::HostNest>(alpha, A, B, beta, C, opts);
            break;
        case Target::HostBatch:
            gemm<Target::HostBatch>(alpha, A, B, beta, C, opts);
            break;
        case Target::Devices:
            gemm<Target::Devices>(alpha, A, B, beta, C, opts);
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gemm<float>(
    float alpha, Matrix<float>& A,
                 Matrix<float>& B,
    float beta,  Matrix<float>& C,
    Options const& opts);

template
void gemm<double>(
    double alpha, Matrix<double>& A,
                  Matrix<double>& B,
    double beta,  Matrix<double>& C,
    Options const& opts);

template
void gemm< std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >& A,
                               Matrix< std::complex<float> >& B,
    std::complex<float> beta,  Matrix< std::complex<float> >& C,
    Options const& opts);

template
void gemm< std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >& A,
                                Matrix< std::complex<double> >& B,
    std::complex<double> beta,  Matrix< std::complex<double> >& C,
    Options const& opts);

} // namespace slate
