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
#include "work/work.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::trmm from internal::specialization::trmm
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// @internal
/// Distributed parallel triangular matrix-matrix multiplication.
/// Generic implementation for any target.
/// Note A and B are passed by value, so we can transpose if needed
/// (for side = right) without affecting caller.
/// @ingroup trmm_specialization
///
template <Target target, typename scalar_t>
void trmm(slate::internal::TargetType<target>,
          Side side,
          scalar_t alpha, TriangularMatrix<scalar_t> A,
                                    Matrix<scalar_t> B,
          int64_t lookahead)
{
    if (target == Target::Devices) {
        B.allocateBatchArrays();
        B.reserveDeviceWorkspace();
    }

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector<uint8_t> bcast_vector(B.mt());
    std::vector<uint8_t>  gemm_vector(B.nt());
    uint8_t* bcast = bcast_vector.data();
    uint8_t* gemm  =  gemm_vector.data();

    #pragma omp parallel
    #pragma omp master
    {
        omp_set_nested(1);
        #pragma omp task
        {
            work::trmm<target, scalar_t>(side, alpha, A, B, bcast, gemm, lookahead);
            B.tileUpdateAllOrigin();
        }
    }
    B.clearWorkspace();
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup trmm_specialization
///
template <Target target, typename scalar_t>
void trmm(blas::Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>& A,
                                    Matrix<scalar_t>& B,
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

    internal::specialization::trmm(internal::TargetType<target>(),
                                   side,
                                   alpha, A,
                                          B,
                                   lookahead);
}

//------------------------------------------------------------------------------
/// Distributed parallel triangular matrix-matrix multiplication.
/// Performs one of the triangular matrix-matrix operations
/// \[
///     B = \alpha A B,
/// \]
/// or
/// \[
///     B = \alpha B A,
/// \]
/// where alpha is a scalar, B is an m-by-n matrix and A is a unit or non-unit,
/// upper or lower triangular matrix.
/// The matrices can be transposed or conjugate-transposed beforehand, e.g.,
///
///     auto AT = slate::transpose( A );
///     slate::trmm( Side::Left, alpha, AT, B );
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///         One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] side
///         Whether A appears on the left or on the right of B:
///         - Side::Left:  $B = \alpha A B$
///         - Side::Right: $B = \alpha B A$
///
/// @param[in] alpha
///         The scalar alpha.
///
/// @param[in] A
///         - If side = left,  the m-by-m triangular matrix A;
///         - if side = right, the n-by-n triangular matrix A.
///
/// @param[in,out] B
///         On entry, the m-by-n matrix B.
///         On exit, overwritten by the result $\alpha A B$ or $\alpha B A$.
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
/// @ingroup trmm
///
template <typename scalar_t>
void trmm(blas::Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>& A,
                                    Matrix<scalar_t>& B,
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
            trmm<Target::HostTask>(side, alpha, A, B, opts);
            break;
        case Target::HostNest:
            trmm<Target::HostNest>(side, alpha, A, B, opts);
            break;
        case Target::HostBatch:
            trmm<Target::HostBatch>(side, alpha, A, B, opts);
            break;
        case Target::Devices:
            trmm<Target::Devices>(side, alpha, A, B, opts);
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void trmm<float>(
    blas::Side side,
    float alpha, TriangularMatrix<float>& A,
                           Matrix<float>& B,
    Options const& opts);

template
void trmm<double>(
    blas::Side side,
    double alpha, TriangularMatrix<double>& A,
                            Matrix<double>& B,
    Options const& opts);

template
void trmm< std::complex<float> >(
    blas::Side side,
    std::complex<float> alpha, TriangularMatrix< std::complex<float> >& A,
                                         Matrix< std::complex<float> >& B,
    Options const& opts);

template
void trmm< std::complex<double> >(
    blas::Side side,
    std::complex<double> alpha, TriangularMatrix< std::complex<double> >& A,
                                          Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
