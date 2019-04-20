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
#include "slate/HermitianMatrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::hesv from internal::specialization::hesv
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// Distributed parallel symmetric indefinite $LTL^T$ factorization and solve.
/// Generic implementation for any target.
/// @ingroup hesv_specialization
///
template <Target target, typename scalar_t>
void hesv(slate::internal::TargetType<target>,
          HermitianMatrix<scalar_t>& A, Pivots& pivots,
               BandMatrix<scalar_t>& T, Pivots& pivots2,
                   Matrix<scalar_t>& H,
          Matrix<scalar_t>& B,
          int64_t ib, int64_t max_panel_threads, int64_t lookahead)
{
    assert(B.mt() == A.mt());

    // if upper, change to lower
    if (A.uplo() == Uplo::Upper)
        A = conj_transpose(A);

    // factorization
    hetrf(A, pivots, T, pivots2, H,
          {{Option::InnerBlocking,   ib},
           {Option::MaxPanelThreads, max_panel_threads},
           {Option::Lookahead,       lookahead},
           {Option::Target, target}});

    // solve
    hetrs(A, pivots, T, pivots2, B,
          {{Option::Lookahead, lookahead},
           {Option::Target, target}});
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup hesv_specialization
///
template <Target target, typename scalar_t>
void hesv(HermitianMatrix<scalar_t>& A, Pivots& pivots,
               BandMatrix<scalar_t>& T, Pivots& pivots2,
                   Matrix<scalar_t>& H,
          Matrix<scalar_t>& B,
          const std::map<Option, Value>& opts)
{
    int64_t ib;
    try {
        ib = opts.at(Option::InnerBlocking).i_;
        assert(ib >= 0);
    }
    catch (std::out_of_range) {
        ib = 16;
    }

    int64_t max_panel_threads;
    try {
        max_panel_threads = opts.at(Option::MaxPanelThreads).i_;
        assert(max_panel_threads >= 0);
    }
    catch (std::out_of_range) {
        max_panel_threads = std::max(omp_get_max_threads()/2, 1);
    }

    int64_t lookahead;
    try {
        lookahead = opts.at(Option::Lookahead).i_;
        assert(lookahead >= 0);
    }
    catch (std::out_of_range) {
        lookahead = 1;
    }

    internal::specialization::hesv(internal::TargetType<target>(),
                                   A, pivots, T, pivots2, H, B,
                                   ib, max_panel_threads, lookahead);
}

//------------------------------------------------------------------------------
/// Distributed parallel symmetric indefinite $LTL^T$ factorization and solve.
/// @ingroup hesv
///
template <typename scalar_t>
void hesv(HermitianMatrix<scalar_t>& A, Pivots& pivots,
               BandMatrix<scalar_t>& T, Pivots& pivots2,
                   Matrix<scalar_t>& H,
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
            hesv<Target::HostTask>(A, pivots, T, pivots2, H, B, opts);
            break;
        case Target::HostNest:
            //hesv<Target::HostNest>(A, B, opts);
            break;
        case Target::HostBatch:
             //hesv<Target::HostBatch>(A, B, opts);
            break;
        case Target::Devices:
            //posv<Target::Devices>(A, B, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void hesv<float>(
    HermitianMatrix<float>& A, Pivots& pivots,
         BandMatrix<float>& T, Pivots& pivots2,
             Matrix<float>& H,
    Matrix<float>& B,
    const std::map<Option, Value>& opts);

template
void hesv<double>(
    HermitianMatrix<double>& A, Pivots& pivots,
         BandMatrix<double>& T, Pivots& pivots2,
             Matrix<double>& H,
    Matrix<double>& B,
    const std::map<Option, Value>& opts);

template
void hesv< std::complex<float> >(
    HermitianMatrix< std::complex<float> >& A, Pivots& pivots,
         BandMatrix< std::complex<float> >& T, Pivots& pivots2,
             Matrix< std::complex<float> >& H,
    Matrix< std::complex<float> >& B,
    const std::map<Option, Value>& opts);

template
void hesv< std::complex<double> >(
    HermitianMatrix< std::complex<double> >& A, Pivots& pivots,
         BandMatrix< std::complex<double> >& T, Pivots& pivots2,
             Matrix< std::complex<double> >& H,
    Matrix< std::complex<double> >& B,
    const std::map<Option, Value>& opts);

} // namespace slate
