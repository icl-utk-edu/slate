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
// internal::set from internal::specialization::set
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// @internal
/// Set matrix entries.
/// Generic implementation for any target.
/// @ingroup set_specialization
///
template <Target target, typename scalar_t>
void set(slate::internal::TargetType<target>,
         scalar_t alpha, scalar_t beta, Matrix<scalar_t>& A)
{
    if (target == Target::Devices) {
        A.allocateBatchArrays();
        // todo: is this needed here when the matrix is already on devices?
        A.reserveDeviceWorkspace();
    }

    #pragma omp parallel
    #pragma omp master
    {
        internal::set<target>(alpha, beta, std::move(A));
        #pragma omp taskwait
        A.tileUpdateAllOrigin();
    }

    A.releaseWorkspace();
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup set_specialization
///
template <Target target, typename scalar_t>
void set(scalar_t alpha, scalar_t beta, Matrix<scalar_t>& A,
         Options const& opts)
{
    internal::specialization::set(internal::TargetType<target>(),
                                  alpha, beta, A);
}

//------------------------------------------------------------------------------
/// Set matrix entries.
/// Transposition is currently ignored.
/// TODO: Inspect transposition?
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] A
///         The m-by-n matrix A.
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
/// @ingroup set
///
template <typename scalar_t>
void set(scalar_t alpha, scalar_t beta, Matrix<scalar_t>& A,
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
        default: // todo: this is to silence a warning, should err otherwise
            set<Target::HostTask>(alpha, beta, A, opts);
            break;
//      case Target::HostNest:
//          set<Target::HostNest>(alpha, beta, A, opts);
//          break;
//      case Target::HostBatch:
//          set<Target::HostBatch>(alpha, beta, A, opts);
//          break;
        case Target::Devices:
            set<Target::Devices>(alpha, beta, A, opts);
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void set(
    float alpha, float beta, Matrix<float>& A,
    Options const& opts);

template
void set(
    double alpha, double beta, Matrix<double>& A,
    Options const& opts);

template
void set(
    std::complex<float> alpha, std::complex<float> beta,
    Matrix<std::complex<float> >& A,
    Options const& opts);

template
void set(
    std::complex<double> alpha, std::complex<double> beta,
    Matrix<std::complex<double> >& A,
    Options const& opts);

} // namespace slate
