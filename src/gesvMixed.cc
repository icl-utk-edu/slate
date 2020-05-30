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
#include "internal/internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::gesvMixed from internal::specialization::gesvMixed
namespace internal {
namespace specialization {

template <typename scalar_t>
bool iterRefConverged(std::vector<scalar_t>& colnorms_R,
                      std::vector<scalar_t>& colnorms_X,
                      scalar_t cte)
{
    assert(colnorms_X.size() == colnorms_R.size());
    bool value = true;
    int64_t size = colnorms_X.size();

    for (int64_t i = 0; i < size; i++) {
        if (colnorms_R[i] > colnorms_X[i] * cte) {
            value = false;
            break;
        }
    }

    return value;
}

//------------------------------------------------------------------------------
/// Distributed parallel LU factorization and solve.
/// Generic implementation for any target.
/// @ingroup gesv_specialization
///
template <Target target, typename scalar_hi, typename scalar_lo>
void gesvMixed( slate::internal::TargetType<target>,
                Matrix<scalar_hi>& A, Pivots& pivots,
                Matrix<scalar_hi>& B,
                Matrix<scalar_hi>& X,
                int& iter,
                int64_t ib, int max_panel_threads, int64_t lookahead)
{
    // Assumes column major
    const Layout layout = Layout::ColMajor;

    bool converged = false;
    const int itermax = 30;
    using real_hi = blas::real_type<scalar_hi>;
    const real_hi eps = std::numeric_limits<real_hi>::epsilon();
    iter = 0;

    assert(B.mt() == A.mt());

    // workspace
    auto R    = B.emptyLike();
    auto A_lo = A.template emptyLike<scalar_lo>();
    auto X_lo = X.template emptyLike<scalar_lo>();

    std::vector<real_hi> colnorms_X(X.n());
    std::vector<real_hi> colnorms_R(R.n());

    // insert local tiles
    X_lo.insertLocalTiles(target);
    R.   insertLocalTiles(target);
    A_lo.insertLocalTiles(target);

    if (target == Target::Devices) {
        #pragma omp parallel
        #pragma omp master
        {
            #pragma omp task default(shared)
            {
                A.tileGetAndHoldAllOnDevices(LayoutConvert(layout));
            }
            #pragma omp task default(shared)
            {
                B.tileGetAndHoldAllOnDevices(LayoutConvert(layout));
            }
            #pragma omp task default(shared)
            {
                X.tileGetAndHoldAllOnDevices(LayoutConvert(layout));
            }
        }
    }

    // norm of A
    real_hi Anorm = norm(Norm::Inf, A,
                         {{Option::Target, target}});

    // stopping criteria
    real_hi cte = Anorm * eps * std::sqrt(A.n());

    // Convert B from high to low precision, store result in X_lo.
    copy(B, X_lo,
         {{Option::Target, target}});

    // Convert A from high to low precision, store result in A_lo.
    copy(A, A_lo,
         {{Option::Target, target}});

    // Compute the LU factorization of A_lo.
    getrf(A_lo, pivots,
          {{Option::InnerBlocking, ib},
           {Option::Lookahead, lookahead},
           {Option::MaxPanelThreads, int64_t(max_panel_threads)},
           {Option::Target, target}});


    // Solve the system A_lo * X_lo = B_lo.
    getrs(A_lo, pivots, X_lo,
          {{Option::Lookahead, lookahead},
           {Option::Target, target}});

    // Convert X_lo to high precision.
    copy(X_lo, X,
         {{Option::Target, target}});

    // Compute R = B - A * X.
    copy(B, R,
         {{Option::Target, target}});
    gemm<scalar_hi>(
        scalar_hi(-1.0), A,
                         X,
        scalar_hi(1.0),  R,
        {{Option::Lookahead, lookahead},
         {Option::Target, target}});

    // Check whether the nrhs normwise backward error satisfies the
    // stopping criterion. If yes, set iter=0 and return.
    colNorms( Norm::Max, X, colnorms_X.data(),
                {{Option::Target, target}});
    colNorms( Norm::Max, R, colnorms_R.data(),
                {{Option::Target, target}});

    if (iterRefConverged<real_hi>(colnorms_R, colnorms_X, cte)) {
        iter = 0;
        converged = true;
    }

    // iterative refinement
    for (int iiter = 0; iiter < itermax && ! converged; iiter++) {
        // Convert R from high to low precision, store result in X_lo.
        copy(R, X_lo,
             {{Option::Target, target}});

        // Solve the system A_lo * X_lo = R_lo.
        getrs(A_lo, pivots, X_lo,
              {{Option::Lookahead, lookahead},
               {Option::Target, target}});

        // Convert X_lo back to double precision and update the current iterate.
        copy(X_lo, R,
             {{Option::Target, target}});
        geadd<scalar_hi>(
              scalar_hi(1.0), R,
              scalar_hi(1.0), X,
              {{Option::Target, target}});

        // Compute R = B - A * X.
        copy(B, R,
             {{Option::Target, target}});
        gemmA<scalar_hi>(
            scalar_hi(-1.0), A,
                             X,
            scalar_hi(1.0),  R,
            {{Option::Lookahead, lookahead},
             {Option::Target, target}});


        // Check whether nrhs normwise backward error satisfies the
        // stopping criterion. If yes, set iter = iiter > 0 and return.
        colNorms( Norm::Max, X, colnorms_X.data(),
                    {{Option::Target, target}});
        colNorms( Norm::Max, R, colnorms_R.data(),
                    {{Option::Target, target}});

        if (iterRefConverged<real_hi>(colnorms_R, colnorms_X, cte)) {
            iter = iiter+1;
            converged = true;
        }
    }

    if (! converged) {
        // If we are at this place of the code, this is because we have performed
        // iter = itermax iterations and never satisfied the stopping criterion,
        // set up the iter flag accordingly and follow up with double precision
        // routine.
        iter = -itermax - 1;

        // Compute the LU factorization of A.
        getrf(A, pivots,
              {{Option::InnerBlocking, ib},
               {Option::Lookahead, lookahead},
               {Option::MaxPanelThreads, int64_t(max_panel_threads)},
               {Option::Target, target}});

        // Solve the system A * X = B.
        copy(B, X,
             {{Option::Target, target}});
        getrs(A, pivots, X,
              {{Option::Lookahead, lookahead},
               {Option::Target, target}});
    }

    if (target == Target::Devices) {
        // clear instead of release due to previous hold
        A.clearWorkspace();
        B.clearWorkspace();
        X.clearWorkspace();
    }
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup gesv_specialization
///
template <Target target, typename scalar_hi, typename scalar_lo>
void gesvMixed( Matrix<scalar_hi>& A, Pivots& pivots,
                Matrix<scalar_hi>& B,
                Matrix<scalar_hi>& X,
                int& iter,
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

    int64_t ib;
    try {
        ib = opts.at(Option::InnerBlocking).i_;
        assert(ib >= 0);
    }
    catch (std::out_of_range&) {
        ib = 16;
    }

    int64_t max_panel_threads;
    try {
        max_panel_threads = opts.at(Option::MaxPanelThreads).i_;
        assert(max_panel_threads >= 0);
    }
    catch (std::out_of_range&) {
        max_panel_threads = std::max(omp_get_max_threads()/2, 1);
    }

    internal::specialization::gesvMixed<target, scalar_hi, scalar_lo>(
                                        internal::TargetType<target>(),
                                        A, pivots, B, X, iter,
                                        ib, max_panel_threads, lookahead);
}

//------------------------------------------------------------------------------
/// Distributed parallel iterative-refinement LU factorization and solve.
///
/// Computes the solution to a system of linear equations
/// \[
///     A X = B,
/// \]
/// where $A$ is an n-by-n matrix and $X$ and $B$ are n-by-nrhs matrices.
///
/// gesvMixed first factorizes the matrix using getrf in low precision (single)
/// and uses this factorization within an iterative refinement procedure to
/// produce a solution with high precision (double) normwise backward error
/// quality (see below). If the approach fails, the method falls back to a
/// high precision (double) factorization and solve.
///
/// The iterative refinement is not going to be a winning strategy if
/// the ratio of low-precision performance over high-precision performance is
/// too small. A reasonable strategy should take the number of right-hand
/// sides and the size of the matrix into account. This might be automated
/// in the future. Up to now, we always try iterative refinement.
///
/// The iterative refinement process is stopped if iter > itermax or
/// for all the RHS, $1 \le j \le nrhs$, we have:
///     $\norm{r_j}_{inf} < \sqrt{n} \norm{x_j}_{inf} \norm{A}_{inf} \epsilon,$
/// where:
/// - iter is the number of the current iteration in the iterative refinement
///    process
/// - $\norm{r_j}_{inf}$ is the infinity-norm of the residual, $r_j = Ax_j - b_j$
/// - $\norm{x_j}_{inf}$ is the infinity-norm of the solution
/// - $\norm{A}_{inf}$ is the infinity-operator-norm of the matrix $A$
/// - $\epsilon$ is the machine epsilon.
///
/// The value itermax is fixed to 30.
///
//------------------------------------------------------------------------------
/// @tparam scalar_hi
///     One of double, std::complex<double>.
///
/// @tparam scalar_lo
///     One of float, std::complex<float>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the n-by-n matrix $A$.
///     On exit, if iterative refinement has been successfully used
///     (return value = 0 and iter >= 0, see description below), then $A$ is
///     unchanged. If high precision (double) factorization has been used
///     (return value = 0 and iter < 0, see description below), then the
///     array $A$ contains the factors $L$ and $U$ from the
///     factorization $A = P L U$.
///
/// @param[out] pivots
///     The pivot indices that define the permutation matrix $P$.
///
/// @param[in] B
///     On entry, the n-by-nrhs right hand side matrix $B$.
///
/// @param[out] X
///     On exit, if return value = 0, the n-by-nrhs solution matrix $X$.
///
/// @param[out] iter
///     The number of the iterations in the iterative refinement
///     process, needed for the convergence. If failed, it is set
///     to be -(1+itermax), where itermax = 30.
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
/// TODO: return value
/// @retval 0 successful exit
/// @retval >0 for return value = $i$, the computed $U(i,i)$ is exactly zero.
///         The factorization has been completed, but the factor U is exactly
///         singular, so the solution could not be computed.
///
/// @ingroup gesv
///
template <typename scalar_hi, typename scalar_lo>
void gesvMixed( Matrix<scalar_hi>& A, Pivots& pivots,
                Matrix<scalar_hi>& B,
                Matrix<scalar_hi>& X,
                int& iter,
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
            gesvMixed<Target::HostTask,  scalar_hi, scalar_lo>(
                     A, pivots, B, X, iter, opts);
            break;
        case Target::HostNest:
            gesvMixed<Target::HostNest,  scalar_hi, scalar_lo>(
                     A, pivots, B, X, iter, opts);
            break;
        case Target::HostBatch:
            gesvMixed<Target::HostBatch, scalar_hi, scalar_lo>(
                     A, pivots, B, X, iter, opts);
            break;
        case Target::Devices:
            gesvMixed<Target::Devices,   scalar_hi, scalar_lo>(
                     A, pivots, B, X, iter, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template <>
void gesvMixed<double>(
    Matrix<double>& A, Pivots& pivots,
    Matrix<double>& B,
    Matrix<double>& X,
    int& iter,
    Options const& opts)
{
    gesvMixed<double, float>(A, pivots, B, X, iter, opts);
}

template <>
void gesvMixed< std::complex<double> >(
    Matrix< std::complex<double> >& A, Pivots& pivots,
    Matrix< std::complex<double> >& B,
    Matrix< std::complex<double> >& X,
    int& iter,
    Options const& opts)
{
    gesvMixed<std::complex<double>, std::complex<float>>(A, pivots, B, X, iter, opts);
}

} // namespace slate
