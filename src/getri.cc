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
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::getri from internal::specialization::getri
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// Distributed parallel inverse of a general matrix.
/// Generic implementation for any target.
/// @ingroup getri_specialization
///
/// todo: This routine is in-place and does not support GPUs.
///       There is another one (out-of-place) that does.
///       What if this one is called with Target::Devices?
///       a) execute on CPUs,
///       b) error out (not supported)?
///
template <Target target, typename scalar_t>
void getri(slate::internal::TargetType<target>,
           Matrix<scalar_t>& A, Pivots& pivots,
           int64_t lookahead)
{
    using BcastList = typename Matrix<scalar_t>::BcastList;
    using ReduceList = typename Matrix<scalar_t>::ReduceList;

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    // auto U = TriangularMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, A);
    auto L = TriangularMatrix<scalar_t>(Uplo::Lower, Diag::Unit, A);

    #pragma omp parallel
    #pragma omp master
    {
        int64_t k = A.nt()-1;
        {
            auto Akk = A.sub(k, k, k, k);
            auto W = Akk.template emptyLike<scalar_t>();
            W.insertLocalTiles(Target::HostTask);

            // Copy A(k, k) to W.
            // todo: Copy L(k, k) to W.
            internal::copy<Target::HostTask>(std::move(Akk), std::move(W));

            // Zero L(k, k).
            if (L.tileIsLocal(k, k)) {
                auto Lkk = L(k, k);
                tzset(scalar_t(0.0), Lkk);
            }

            // send W down col A(0:nt-1, k)
            W.template tileBcast(
                0, 0, A.sub(0, A.nt()-1, k, k), layout);

            auto Wkk = TriangularMatrix<scalar_t>(Uplo::Lower, Diag::Unit, W);
            internal::trsm<Target::HostTask>(
                Side::Right,
                scalar_t(1.0), std::move(Wkk), A.sub(0, A.nt()-1, k, k));
        }
        --k;

        for (; k >= 0; --k) {

            auto Lk = A.sub(k, A.nt()-1, k, k);
            auto W = Lk.template emptyLike<scalar_t>();
            W.insertLocalTiles(Target::HostTask);

            // Copy L(:, k) to W.
            internal::copy<Target::HostTask>(std::move(Lk), std::move(W));

            // Zero L(k, k).
            if (L.tileIsLocal(k, k)) {
                auto Lkk = L(k, k);
                tzset(scalar_t(0.0), Lkk);
            }

            // Zero L(k+1:A_nt-1, k).
            for (int64_t i = k+1; i < A.nt(); ++i) {
                if (L.tileIsLocal(i, k)) {
                    L(i, k).set(0.0);
                }
            }

            // send W across A
            BcastList bcast_list_W;
            for (int64_t i = 1; i < W.mt(); ++i) {
                // send W(i) down column A(0:nt-1, k+i)
                bcast_list_W.push_back({i, 0, {A.sub(0, A.nt()-1, k+i, k+i)}});
            }
            W.template listBcast(bcast_list_W, layout);

            // A(:, k) -= A(:, k+1:nt-1) * W
            internal::gemmA<Target::HostTask>(
                scalar_t(-1.0), A.sub(0, A.nt()-1, k+1, A.nt()-1),
                                W.sub(1, W.mt()-1, 0, 0),
                scalar_t(1.0),  A.sub(0, A.nt()-1, k, k), layout);

            // reduce A(0:nt-1, k)
            ReduceList reduce_list_A;
            for (int64_t i = 0; i < A.nt(); ++i) {
                // recude A(i, k) across A(i, k+1:nt-1)
                reduce_list_A.push_back({i, k, {A.sub(i, i, k+1, A.nt()-1)}});
            }
            A.template listReduce(reduce_list_A, layout);

            // send W(0, 0) down col A(0:nt-1, k)
            W.tileBcast(0, 0, A.sub(0, A.nt()-1, k, k), layout);

            auto Wkk = W.sub(0, 0, 0, 0);
            auto Tkk = TriangularMatrix<scalar_t>(Uplo::Lower, Diag::Unit, Wkk);
            internal::trsm<Target::HostTask>(
                Side::Right,
                scalar_t(1.0), std::move(Tkk), A.sub(0, A.nt()-1, k, k));
        }

        // Apply column pivoting.
        for (int64_t k = A.nt()-1; k >= 0; --k) {
            internal::permuteRows<Target::HostTask>(
                Direction::Backward, transpose(A).sub(k, A.nt()-1, 0, A.nt()-1),
                pivots.at(k), Layout::ColMajor);
        }
    }
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup getri_specialization
///
template <Target target, typename scalar_t>
void getri(Matrix<scalar_t>& A, Pivots& pivots,
           Options const& opts)
{
    slate_assert(A.mt() == A.nt());  // square

    int64_t lookahead;
    try {
        lookahead = opts.at(Option::Lookahead).i_;
        assert(lookahead >= 0);
    }
    catch (std::out_of_range&) {
        lookahead = 1;
    }

    internal::specialization::getri(internal::TargetType<target>(),
                                    A, pivots, lookahead);
}

//------------------------------------------------------------------------------
/// Distributed parallel LU inversion.
///
/// Computes the inverse of a matrix $A$ using the LU factorization $A = L*U$
/// computed by `getrf`.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] A
///     On entry, the factors $L$ and $U$ from the factorization $A = P L U$
///     as computed by getrf.
///     On exit, the inverse of the original matrix $A$.
///
/// @param[in] pivots
///     The pivot indices that define the permutation matrix $P$
///     as computed by getrf.
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
/// @ingroup getri_computational
///
template <typename scalar_t>
void getri(Matrix<scalar_t>& A, Pivots& pivots,
           Options const& opts)
{
    // triangular inversion
    auto U = TriangularMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, A);
    trtri(U, opts);

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
            getri<Target::HostTask>(A, pivots, opts);
            break;
        case Target::HostNest:
            getri<Target::HostNest>(A, pivots, opts);
            break;
        case Target::HostBatch:
            getri<Target::HostBatch>(A, pivots, opts);
            break;
        case Target::Devices:
            getri<Target::Devices>(A, pivots, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void getri<float>(
    Matrix<float>& A, Pivots& pivots,
    Options const& opts);

template
void getri<double>(
    Matrix<double>& A, Pivots& pivots,
    Options const& opts);

template
void getri< std::complex<float> >(
    Matrix< std::complex<float> >& A, Pivots& pivots,
    Options const& opts);

template
void getri< std::complex<double> >(
    Matrix< std::complex<double> >& A, Pivots& pivots,
    Options const& opts);

} // namespace slate
