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
// #include "aux/Debug.hh"
#include "slate/Tile_blas.hh"
#include "slate/TriangularBandMatrix.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"

#include <atomic>

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::bdsqr from internal::specialization::bdsqr
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// computes the singular values and, optionally, the right and/or
/// left singular vectors from the singular value decomposition (SVD) of
/// a real (upper or lower) bidiagonal matrix.
/// Generic implementation for any target.
/// @ingroup bdsqr_specialization
///
// ATTENTION: only singular values computed for now, no singular vectors.
// only host computation supported for now
//
template <Target target, typename scalar_t>
void bdsqr(slate::internal::TargetType<target>,
           TriangularBandMatrix<scalar_t> A,
           std::vector< blas::real_type<scalar_t> >& D)
{
    trace::Block trace_block("slate::bdsqr");
    using real_t = blas::real_type<scalar_t>;

    // if lower, change to upper
    if (A.uplo() == Uplo::Lower) {
        A = conj_transpose(A);
    }

    // make sure it is a bi-diagobal matrix
    slate_assert(A.bandwidth() == 1);

    // Find the set of participating ranks.
    std::set<int> rank_set;
    A.getRanks(&rank_set);// todo: is this needed? aren't all ranks needed?

    // gather A on each rank
    // todo: this is over-communicating, try gathering the vectors only
    A.gatherAll(rank_set);

    slate_assert(A.m() == A.n()); // Triangular matrix has square dimensions
    slate_assert(A.mt() == A.nt());

    D.resize(A.n());
    std::vector<real_t> E(A.n() - 1);  // super-diagonal
    scalar_t dummy[1];  // U, VT, C not needed for NoVec

    // Copy diagonal & super-diagonal.
    internal::copytb2bd(A, D, E);

    {
        trace::Block trace_block("lapack::bdsqr");

        lapack::bdsqr(A.uplo(), A.n(), 0, 0, 0,
                      &D[0], &E[0], dummy, 1, dummy, 1, dummy, 1);
    }
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup bdsqr_specialization
///
template <Target target, typename scalar_t>
void bdsqr(TriangularBandMatrix<scalar_t>& A,
           std::vector< blas::real_type<scalar_t> >& D,
           const std::map<Option, Value>& opts)
{
    internal::specialization::bdsqr(internal::TargetType<target>(),
                                    A, D);
}

//------------------------------------------------------------------------------
///
template <typename scalar_t>
void bdsqr(TriangularBandMatrix<scalar_t>& A,
           std::vector< blas::real_type<scalar_t> >& D,
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
            bdsqr<Target::HostTask>(A, D, opts);
            break;
        case Target::HostNest:
            bdsqr<Target::HostNest>(A, D, opts);
            break;
        case Target::HostBatch:
            bdsqr<Target::HostBatch>(A, D, opts);
            break;
        case Target::Devices:
            bdsqr<Target::Devices>(A, D, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void bdsqr<float>(
    TriangularBandMatrix<float>& A,
    std::vector<float>& D,
    const std::map<Option, Value>& opts);

template
void bdsqr<double>(
    TriangularBandMatrix<double>& A,
    std::vector<double>& D,
    const std::map<Option, Value>& opts);

template
void bdsqr< std::complex<float> >(
    TriangularBandMatrix< std::complex<float> >& A,
    std::vector<float>& D,
    const std::map<Option, Value>& opts);

template
void bdsqr< std::complex<double> >(
    TriangularBandMatrix< std::complex<double> >& A,
    std::vector<double>& D,
    const std::map<Option, Value>& opts);

} // namespace slate
