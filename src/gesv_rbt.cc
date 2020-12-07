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
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "internal/internal.hh"

namespace slate {

namespace internal {

template<Target target, typename scalar_t>
void gesv_rbt(slate::internal::TargetType<target>,
              Matrix<scalar_t>& A,
              Matrix<scalar_t>& B,
              const std::map<Option, Value>& opts,
              const int64_t depth,
              const int64_t iters)
{

    using real_t = blas::real_type<scalar_t>;

    slate_assert(A.mt() == A.nt());  // square
    slate_assert(B.mt() == A.mt());

    const scalar_t one = 1.0;

    std::pair<Matrix<scalar_t>, Matrix<scalar_t>> transforms = rbt_generate(A, depth, 42);
	Matrix<scalar_t> U = transforms.first;
	Matrix<scalar_t> V = transforms.second;

    // Make copies for IR
    Matrix<scalar_t> B_copy = B.emptyLike();
    B_copy.insertLocalTiles(Target::Host);
    copy<Matrix<scalar_t>, Matrix<scalar_t>>(B, B_copy, {{Option::Target, Target::HostTask}});

    Matrix<scalar_t> A_copy = A.emptyLike();
    A_copy.insertLocalTiles(Target::Host);
    copy<Matrix<scalar_t>, Matrix<scalar_t>>(A, A_copy, {{Option::Target, Target::HostTask}});

    // Factor
    gerbt(U, A, V);
    getrf_nopiv(A, opts);

    // Solve
    gerbt(U, B);
    getrs_nopiv(A, B, opts);
    gerbt(V, B);


    // refine
    Matrix<scalar_t> R = B.emptyLike();
    R.insertLocalTiles(Target::Host);

    copy<Matrix<scalar_t>, Matrix<scalar_t>>(B_copy, R, {{Option::Target, Target::HostTask}});
    gemmA(-one, A_copy, B, one, R, {{Option::Target, Target::HostTask}});

    real_t R_norm = slate::norm(slate::Norm::One, R);
    real_t X_norm = slate::norm(slate::Norm::One, B);
    const real_t A_norm = slate::norm(slate::Norm::One, A_copy);
    const real_t eps = std::numeric_limits<real_t>::epsilon();

    for (int64_t i = 0; i < iters && R_norm / (A_norm * X_norm) > eps; ++i) {
        gerbt(U, R);
        getrs_nopiv(A, R, opts);
        gerbt(V, R);
        add(one, R, one, B, {{Option::Target, Target::HostTask}});
        copy<Matrix<scalar_t>, Matrix<scalar_t>>(B_copy, R, {{Option::Target, Target::HostTask}});
        gemmA(-one, A_copy, B, one, R, {{Option::Target, Target::HostTask}});

        R_norm = slate::norm(slate::Norm::One, R);
        X_norm = slate::norm(slate::Norm::One, B);
    }

    if (R_norm / (A_norm * X_norm) > eps) {
        // Not accurate enough, fall back to pivoted LU

        copy<Matrix<scalar_t>, Matrix<scalar_t>>(B_copy, B, {{Option::Target, Target::HostTask}});
        copy<Matrix<scalar_t>, Matrix<scalar_t>>(A_copy, A, {{Option::Target, Target::HostTask}});
        Pivots pivots;
        gesv(A_copy, pivots, B, opts);
    }

    // todo: return value for errors?
}

} // namespace internal


//------------------------------------------------------------------------------
/// Distributed parallel LU factorization and solve.
///
/// Computes the solution to a system of linear equations
/// \[
///     A X = B,
/// \]
/// where $A$ is an n-by-n matrix and $X$ and $B$ are n-by-nrhs matrices.
///
/// The LU decomposition with partial pivoting and row interchanges is
/// used to factor $A$ as
/// \[
///     A = P L U,
/// \]
/// where $P$ is a permutation matrix, $L$ is unit lower triangular, and $U$ is
/// upper triangular.  The factored form of $A$ is then used to solve the
/// system of equations $A X = B$.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the n-by-n matrix $A$ to be factored.
///     On exit, the factors $L$ and $U$ from the factorization $A = P L U$;
///     the unit diagonal elements of $L$ are not stored.
///
/// @param[out] pivots
///     The pivot indices that define the permutation matrix $P$.
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
///     - Option::InnerBlocking:
///       Inner blocking to use for panel. Default 16.
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  nested OpenMP parallel for loop on CPU host.
///       - HostBatch: batched BLAS on CPU host.
///       - Devices:   batched BLAS on GPU device.
///
/// @param[in] refine
///     Whether an iteration of refinement should be applied
///
/// TODO: return value
/// @retval 0 successful exit
/// @retval >0 for return value = $i$, the computed $U(i,i)$ is exactly zero.
///         The factorization has been completed, but the factor U is exactly
///         singular, so the solution could not be computed.
///
/// @ingroup gesv
///
template <typename scalar_t>
void gesv_rbt(Matrix<scalar_t>& A,
              Matrix<scalar_t>& B,
              std::map<Option, Value> const& opts,
              const int64_t depth,
              const int64_t refine)
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
            internal::gesv_rbt(internal::TargetType<Target::HostTask>(), A, B, opts, depth, refine);
            break;
        case Target::HostNest:
            internal::gesv_rbt(internal::TargetType<Target::HostNest>(), A, B, opts, depth, refine);
            break;
        case Target::HostBatch:
            internal::gesv_rbt(internal::TargetType<Target::HostBatch>(), A, B, opts, depth, refine);
            break;
        case Target::Devices:
            internal::gesv_rbt(internal::TargetType<Target::Devices>(), A, B, opts, depth, refine);
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gesv_rbt<float>(
    Matrix<float>& A,
    Matrix<float>& B,
    const std::map<Option, Value>& opts,
    const int64_t depth,
    const int64_t refine);

template
void gesv_rbt<double>(
    Matrix<double>& A,
    Matrix<double>& B,
    const std::map<Option, Value>& opts,
    const int64_t depth,
    const int64_t refine);

template
void gesv_rbt< std::complex<float> >(
    Matrix< std::complex<float> >& A,
    Matrix< std::complex<float> >& B,
    const std::map<Option, Value>& opts,
    const int64_t depth,
    const int64_t refine);

template
void gesv_rbt< std::complex<double> >(
    Matrix< std::complex<double> >& A,
    Matrix< std::complex<double> >& B,
    const std::map<Option, Value>& opts,
    const int64_t depth,
    const int64_t refine);

} // namespace slate
