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
#include "slate/Tile_blas.hh"
#include "slate/TriangularBandMatrix.hh"
#include "internal/internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::gbtrs from internal::specialization::gbtrs
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// Distributed parallel band LU solve.
/// Generic implementation for any target.
/// @ingroup gbsv_specialization
///
template <Target target, typename scalar_t>
void gbtrs(slate::internal::TargetType<target>,
           BandMatrix<scalar_t>& A, Pivots& pivots,
           Matrix<scalar_t>& B, int64_t lookahead)
{
    assert(A.mt() == A.nt());
    assert(B.mt() == A.mt());

    auto L = TriangularBandMatrix<scalar_t>(Uplo::Lower, Diag::Unit, A);
    auto U = TriangularBandMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, A);
//printf( "L kd %lld\n", L.bandwidth() );
//printf( "U kd %lld\n", U.bandwidth() );

    if (A.op() == Op::NoTrans) {
        // forward substitution, Y = L^{-1} P B
        tbsm(Side::Left, scalar_t(1.0), L, pivots, B,
             {{Option::Lookahead, lookahead},
              {Option::Target, target}});

        // backward substitution, X = U^{-1} Y
        tbsm(Side::Left, scalar_t(1.0), U, B,
             {{Option::Lookahead, lookahead},
              {Option::Target, target}});
    }
    else {
        // forward substitution, Y = U^{-T} B
        tbsm(Side::Left, scalar_t(1.0), U, B,
             {{Option::Lookahead, lookahead},
              {Option::Target, target}});

        // backward substitution, X = P^T L^{-T} Y
        tbsm(Side::Left, scalar_t(1.0), L, pivots, B,
             {{Option::Lookahead, lookahead},
              {Option::Target, target}});
    }
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup gbsv_specialization
///
template <Target target, typename scalar_t>
void gbtrs(BandMatrix<scalar_t>& A, Pivots& pivots,
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

    internal::specialization::gbtrs(internal::TargetType<target>(),
                                    A, pivots, B, lookahead);
}

//------------------------------------------------------------------------------
/// Distributed parallel band LU solve.
///
/// Solves a system of linear equations
/// \[
///     A X = B
/// \]
/// with a general n-by-n band matrix $A$ using the LU factorization computed
/// by gbtrf. $A$ can be transposed or conjugate-transposed.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] A
///     The factors $L$ and $U$ from the factorization $A = L U$
///     as computed by gbtrf.
///
/// @param[in] pivots
///     The pivot indices that define the permutations
///     as computed by gbtrf.
///
/// @param[in,out] B
///     On entry, the n-by-nrhs right hand side matrix $B$.
///     On exit, the n-by-nrhs solution matrix $X$.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::Lookahead:
///       Number of panels to overlap with matrix updates.
///       lookahead >= 0. Default 1.
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  nested OpenMP parallel for loop on CPU host.
///       - HostBatch: batched BLAS on CPU host.
///       - Devices:   batched BLAS on GPU device.
///
/// @ingroup gbsv_computational
///
template <typename scalar_t>
void gbtrs(BandMatrix<scalar_t>& A, Pivots& pivots,
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
            gbtrs<Target::HostTask>(A, pivots, B, opts);
            break;
        case Target::HostNest:
            gbtrs<Target::HostNest>(A, pivots, B, opts);
            break;
        case Target::HostBatch:
            gbtrs<Target::HostBatch>(A, pivots, B, opts);
            break;
        case Target::Devices:
            gbtrs<Target::Devices>(A, pivots, B, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gbtrs<float>(
    BandMatrix<float>& A, Pivots& pivots,
    Matrix<float>& B,
    Options const& opts);

template
void gbtrs<double>(
    BandMatrix<double>& A, Pivots& pivots,
    Matrix<double>& B,
    Options const& opts);

template
void gbtrs< std::complex<float> >(
    BandMatrix< std::complex<float> >& A, Pivots& pivots,
    Matrix< std::complex<float> >& B,
    Options const& opts);

template
void gbtrs< std::complex<double> >(
    BandMatrix< std::complex<double> >& A, Pivots& pivots,
    Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
