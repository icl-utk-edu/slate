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
// internal::geadd from internal::specialization::geadd
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// @internal
/// Distributed parallel general matrix-matrix addition.
/// Generic implementation for any target.
/// @ingroup geadd_specialization
///
template <Target target, typename scalar_t>
void geadd(slate::internal::TargetType<target>,
           scalar_t alpha, Matrix<scalar_t>& A,
           scalar_t beta,  Matrix<scalar_t>& B,
           int64_t lookahead)
{
    if (target == Target::Devices) {
        B.allocateBatchArrays();
        B.reserveDeviceWorkspace();
    }

    #pragma omp parallel
    #pragma omp master
    {
        internal::geadd<target>(alpha, std::move(A),
                                beta, std::move(B));
        #pragma omp taskwait
        B.tileUpdateAllOrigin();
    }

    B.releaseWorkspace();
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup geadd_specialization
///
template <Target target, typename scalar_t>
void geadd(scalar_t alpha, Matrix<scalar_t>& A,
           scalar_t beta,  Matrix<scalar_t>& B,
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

    internal::specialization::geadd(internal::TargetType<target>(),
                                    alpha, A,
                                    beta,  B,
                                    lookahead);
}

//------------------------------------------------------------------------------
/// Distributed parallel general matrix-matrix addition.
/// Performs the matrix-matrix operation
/// \[
///     B = \alpha A + \beta B,
/// \]
/// where alpha and beta are scalars, and $A$ and $B$ are matrices, with
/// $A$ an m-by-n matrix and $B$ a m-by-n matrix.
/// Transposition is currently not supported.
/// TODO: Support transposition.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///         One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] alpha
///         The scalar alpha.
///
/// @param[in] A
///         The m-by-n matrix A.
///
/// @param[in] beta
///         The scalar beta.
///
/// @param[in,out] B
///         On entry, the m-by-n matrix B.
///         On exit, overwritten by the result $\alpha A + \beta B$.
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
/// @ingroup geadd
///
template <typename scalar_t>
void geadd(scalar_t alpha, Matrix<scalar_t>& A,
           scalar_t beta,  Matrix<scalar_t>& B,
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
            geadd<Target::HostTask>(alpha, A, beta, B, opts);
            break;
        case Target::HostNest:
            geadd<Target::HostNest>(alpha, A, beta, B, opts);
            break;
        case Target::HostBatch:
            geadd<Target::HostBatch>(alpha, A, beta, B, opts);
            break;
        case Target::Devices:
            geadd<Target::Devices>(alpha, A, beta, B, opts);
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void geadd<float>(
    float alpha, Matrix<float>& A,
    float beta,  Matrix<float>& B,
    Options const& opts);

template
void geadd<double>(
    double alpha, Matrix<double>& A,
    double beta,  Matrix<double>& B,
    Options const& opts);

template
void geadd< std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >& A,
    std::complex<float> beta,  Matrix< std::complex<float> >& B,
    Options const& opts);

template
void geadd< std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >& A,
    std::complex<double> beta,  Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
