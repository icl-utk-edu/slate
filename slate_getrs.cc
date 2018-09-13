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

#include "slate.hh"
#include "slate_Debug.hh"
#include "slate_Matrix.hh"
#include "slate_HermitianMatrix.hh"
#include "slate_Tile_blas.hh"
#include "slate_TriangularMatrix.hh"
#include "slate_internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::getrs from internal::specialization::getrs
namespace internal {
namespace specialization {

///-----------------------------------------------------------------------------
/// \brief
/// Distributed parallel LU solve.
/// Generic implementation for any target.
template <Target target, typename scalar_t>
void getrs(slate::internal::TargetType<target>,
           Matrix<scalar_t>& A, Pivots& pivots,
           Matrix<scalar_t>& B, int64_t lookahead)
{
    assert(A.mt() == A.nt());
    assert(B.mt() == A.mt());

    // TODO: handle the transpose case
    
    // Pivot the right hand side matrix.
    for (int64_t k = 0; k < B.mt(); ++k) {
        // swap rows in B(k:mt-1, 0:nt-1)
        internal::swap<Target::HostTask>(
            B.sub(k, B.mt()-1, 0, B.nt()-1), pivots.at(k));
    }

    auto L = TriangularMatrix<scalar_t>(Uplo::Lower, Diag::Unit, A);
    auto U = TriangularMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, A);

    // forward substitution
    trsm(Side::Left, scalar_t(1.0), L, B,
         {{Option::Lookahead, lookahead},
          {Option::Target, target}});

    // backward substitution
    trsm(Side::Left, scalar_t(1.0), U, B,
         {{Option::Lookahead, lookahead},
          {Option::Target, target}});
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup gesv_comp
template <Target target, typename scalar_t>
void getrs(Matrix<scalar_t>& A, Pivots& pivots,
           Matrix<scalar_t>& B,
           const std::map<Option, Value>& opts)
{
    int64_t lookahead;
    try {
        lookahead = opts.at(Option::Lookahead).i_;
        assert(lookahead >= 0);
    }
    catch (std::out_of_range) {
        lookahead = 1;
    }

    internal::specialization::getrs(internal::TargetType<target>(),
                                    A, pivots, B, lookahead);
}

//------------------------------------------------------------------------------
/// Distributed parallel LU factorization.
///
template <typename scalar_t>
void getrs(Matrix<scalar_t>& A, Pivots& pivots,
           Matrix<scalar_t>& B,
           const std::map<Option, Value>& opts)
{
    Target target;
    try {
        target = Target(opts.at(Option::Target).i_);
    }
    catch (std::out_of_range) {
        target = Target::HostTask;
    }

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            getrs<Target::HostTask>(A, pivots, B, opts);
            break;
        case Target::HostNest:
            getrs<Target::HostNest>(A, pivots, B, opts);
            break;
        case Target::HostBatch:
            getrs<Target::HostBatch>(A, pivots, B, opts);
            break;
        case Target::Devices:
            getrs<Target::Devices>(A, pivots, B, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void getrs<float>(
    Matrix<float>& A, Pivots& pivots,
    Matrix<float>& B,
    const std::map<Option, Value>& opts);

template
void getrs<double>(
    Matrix<double>& A, Pivots& pivots,
    Matrix<double>& B,
    const std::map<Option, Value>& opts);

template
void getrs< std::complex<float> >(
    Matrix< std::complex<float> >& A, Pivots& pivots,
    Matrix< std::complex<float> >& B,
    const std::map<Option, Value>& opts);

template
void getrs< std::complex<double> >(
    Matrix< std::complex<double> >& A, Pivots& pivots,
    Matrix< std::complex<double> >& B,
    const std::map<Option, Value>& opts);

} // namespace slate
