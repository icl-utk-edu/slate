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
// internal::hetrs from internal::specialization::hetrs
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// Distributed parallel Hermitian indefinite $LTL^T$ solve.
/// Generic implementation for any target.
/// @ingroup hesv_specialization
///
template <Target target, typename scalar_t>
void hetrs(slate::internal::TargetType<target>,
           HermitianMatrix<scalar_t>& A, Pivots& pivots,
                BandMatrix<scalar_t>& T, Pivots& pivots2,
                    Matrix<scalar_t>& B, int64_t lookahead)
{
    // assert(A.mt() == A.nt());
    assert(B.mt() == A.mt());

    // if upper, change to lower
    if (A.uplo() == Uplo::Upper)
        A = conjTranspose(A);

    const int64_t A_nt = A.nt();
    const int64_t A_mt = A.mt();
    const int64_t B_nt = B.nt();
    const int64_t B_mt = B.mt();

    if (A_nt > 1) {
        // pivot right-hand-sides
        for (int64_t k = 1; k < B.mt(); ++k) {
            // swap rows in B(k:mt-1, 0:nt-1)
            internal::permuteRows<Target::HostTask>(
                Direction::Forward, B.sub(k, B.mt()-1, 0, B.nt()-1),
                pivots.at(k), Layout::ColMajor);
        }

        // forward substitution with L from Aasen's
        auto Lkk = TriangularMatrix<scalar_t>( Diag::NonUnit, A, 1, A_mt-1, 0, A_nt-2 );
        auto Bkk = B.sub(1, B_mt-1, 0, B_nt-1);
        trsm(Side::Left, scalar_t(1.0), Lkk, Bkk,
             {{Option::Lookahead, lookahead},
              {Option::Target, target}});
    }

    // band solve
    gbtrs(T, pivots2, B,
          {{Option::Lookahead, lookahead}});

    if (A_nt > 1) {
        // backward substitution with L^T from Aasen's
        auto Lkk = TriangularMatrix<scalar_t>( Diag::NonUnit, A, 1, A_mt-1, 0, A_nt-2 );
        auto Bkk = B.sub(1, B_mt-1, 0, B_nt-1);
        Lkk = conjTranspose(Lkk);
        trsm(Side::Left, scalar_t(1.0), Lkk, Bkk,
             {{Option::Lookahead, lookahead},
              {Option::Target, target}});

        // pivot right-hand-sides
        for (int64_t k = B.mt()-1; k > 0; --k) {
            // swap rows in B(k:mt-1, 0:nt-1)
            internal::permuteRows<Target::HostTask>(
                Direction::Backward, B.sub(k, B.mt()-1, 0, B.nt()-1),
                pivots.at(k), Layout::ColMajor);
        }
    }
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup hesv_specialization
///
template <Target target, typename scalar_t>
void hetrs(HermitianMatrix<scalar_t>& A, Pivots& pivots,
                BandMatrix<scalar_t>& T, Pivots& pivots2,
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

    internal::specialization::hetrs(internal::TargetType<target>(),
                                    A, pivots, T, pivots2, B, lookahead);
}

//------------------------------------------------------------------------------
/// Distributed parallel Hermitian indefinite $LTL^T$ solve.
///
/// Solves a system of linear equations $A X = B$ with a
/// Hermitian matrix $A$ using the factorization $A = U^H T U$ or
/// $A = L T L^H$ computed by hetrf.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     Details of the factors $U$ or $L$ as computed by hetrf.
///     If scalar_t is real, $A$ can be a SymmetricMatrix object.
///
/// @param[out] pivots
///     Details of the interchanges applied to $A$ as computed by hetrf.
///
/// @param[out] T
///     Details of the LU factorization of the band matrix as computed by hetrf.
///
/// @param[out] pivots2
///     Details of the interchanges applied to $T$ as computed by hetrf.
///
/// @param[in,out] B
///     On entry, the n-by-nrhs right hand side matrix $B$.
///     On exit, if return value = 0, the n-by-nrhs solution matrix $X$.
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
/// @ingroup hesv_computational
///
template <typename scalar_t>
void hetrs(HermitianMatrix<scalar_t>& A, Pivots& pivots,
                BandMatrix<scalar_t>& T, Pivots& pivots2,
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
            hetrs<Target::HostTask>(A, pivots, T, pivots2, B, opts);
            break;
        case Target::HostNest:
            hetrs<Target::HostNest>(A, pivots, T, pivots2, B, opts);
            break;
        case Target::HostBatch:
            hetrs<Target::HostBatch>(A, pivots, T, pivots2, B, opts);
            break;
        case Target::Devices:
            hetrs<Target::Devices>(A, pivots, T, pivots2, B, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void hetrs<float>(
    HermitianMatrix<float>& A, Pivots& pivots,
         BandMatrix<float>& T, Pivots& pivots2,
             Matrix<float>& B,
    Options const& opts);

template
void hetrs<double>(
    HermitianMatrix<double>& A, Pivots& pivots,
         BandMatrix<double>& T, Pivots& pivots2,
             Matrix<double>& B,
    Options const& opts);

template
void hetrs< std::complex<float> >(
    HermitianMatrix< std::complex<float> >& A, Pivots& pivots,
         BandMatrix< std::complex<float> >& T, Pivots& pivots2,
             Matrix< std::complex<float> >& B,
    Options const& opts);

template
void hetrs< std::complex<double> >(
    HermitianMatrix< std::complex<double> >& A, Pivots& pivots,
         BandMatrix< std::complex<double> >& T, Pivots& pivots2,
             Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
