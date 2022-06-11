// Copyright (c) 2020-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "internal/internal.hh"

namespace slate {

//------------------------------------------------------------------------------
/// todo: replicated in gesvMixed.cc; move to common header.
/// @ingroup posv_specialization
///
template <typename scalar_t>
bool iterRefConverged(std::vector<scalar_t>& colnorms_R,
                      std::vector<scalar_t>& colnorms_X,
                      scalar_t cte)
{
    assert(colnorms_X.size() == colnorms_R.size());
    bool value = true;
    int64_t size = colnorms_X.size();

    for (int64_t i = 0; i < size; i++) {
        // Use negated condition for correct NaN behavior
        if (! (colnorms_R[i] <= colnorms_X[i] * cte)) {
            value = false;
            break;
        }
    }

    return value;
}


//------------------------------------------------------------------------------
/// Distributed parallel LU factorization and solve.
///
/// Computes the solution to a system of linear equations
/// \[
///     A X = B,
/// \]
/// where $A$ is an n-by-n matrix and $X$ and $B$ are n-by-nrhs matrices.
///
/// gesv_rbt first transforms the matrix with random butterfly transforms,
/// factorizes the transformed matrix using getrf_nopiv, and uses this
/// factorization within an iterative refinement procedure to produce a
/// solution with full normwise backward error quality (see below).  If the
/// approach fails and the UseFallbackSolver is true, the problem is re-solved
/// with a partial pivoted factorization.
///
/// The iterative refinement process is stopped if iter > itermax or
/// for all the RHS, $1 \le j \le nrhs$, we have:
///     $\norm{r_j}_{inf} < tol \norm{x_j}_{inf} \norm{A}_{inf},$
/// where:
/// - iter is the number of the current iteration in the iterative refinement
///    process
/// - $\norm{r_j}_{inf}$ is the infinity-norm of the residual, $r_j = Ax_j - b_j$
/// - $\norm{x_j}_{inf}$ is the infinity-norm of the solution
/// - $\norm{A}_{inf}$ is the infinity-operator-norm of the matrix $A$
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
/// @param[in] B
///     On entry, the n-by-nrhs right hand side matrix $B$.
///
/// @param[out] X
///     On exit, if return value = 0, the n-by-nrhs solution matrix $X$.
///
/// @param[out] iter
///     The number of the iterations in the iterative refinement
///     process, needed for the convergence. If iterative refinement failed,
///     it is set to -(1+itermax), regardless of whether the fallback solver
///     was used.
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
///     - Option::Depth:
///       Depth for butterfly transform. Default 2
///     - Option::Tolerance:
///       Iterative refinement tolerance. Default epsilon * sqrt(m)
///     - Option::MaxIterations:
///       Maximum number of refinement iterations. Default 30
///     - Option::UseFallbackSolver:
///       If true and iterative refinement fails to convergene, the problem is
///       resolved with partial-pivoted LU. Default true
///
/// TODO: return value
/// @retval 0 successful exit
/// @retval >0 for return value = $i$, the computed $U(i,i)$ is exactly zero.
///         The factorization has been completed, but the factor U is exactly
///         singular, so the solution could not be computed.
///
/// @ingroup gesv
///
template<typename scalar_t>
void gesv_rbt(Matrix<scalar_t>& A,
              Matrix<scalar_t>& B,
              Matrix<scalar_t>& X,
              int& iter,
              Options const& opts)
{

    using real_t = blas::real_type<scalar_t>;

    // gemmA and rbt are currently only implemented on the host
    Options host_opts = opts;
    host_opts.insert_or_assign( Option::Target, Target::Host );

    const scalar_t one = 1.0;
    const real_t eps = std::numeric_limits<real_t>::epsilon();

    int64_t depth = get_option<int64_t>( opts, Option::Depth, 2 );
    int64_t itermax = get_option<int64_t>( opts, Option::MaxIterations, 30 );
    double tol = get_option<double>( opts, Option::Tolerance, eps*std::sqrt(A.m()) );
    bool use_fallback = get_option<int64_t>( opts, Option::UseFallbackSolver, true );

    slate_assert(A.mt() == A.nt());  // square
    slate_assert(B.mt() == A.mt());

    auto transforms = internal::rbt_generate( A, depth, 42 );
    Matrix<scalar_t> U = transforms.first;
    Matrix<scalar_t> V = transforms.second;

    // Workspace
    Matrix<scalar_t> A_copy = A.emptyLike();
    Matrix<scalar_t> R = B.emptyLike();

    A_copy.insertLocalTiles( Target::Host );
    R.insertLocalTiles( Target::Host );

    slate::copy( B, X, opts );
    slate::copy( A, A_copy, host_opts );

    std::vector<real_t> colnorms_X( X.n() );
    std::vector<real_t> colnorms_R( R.n() );

    // norm of A
    real_t Anorm = norm( Norm::Inf, A, opts );
    real_t cte = Anorm*tol;
    bool converged = false;

    // Factor
    gerbt( U, A, V );
    getrf_nopiv( A, opts );

    // Solve
    gerbt( U, X );
    getrs_nopiv( A, X, opts );
    gerbt( V, X );


    // refine
    slate::copy( B, R, host_opts );
    gemmA( -one, A_copy, X,
            one, R, host_opts );

    // Check whether the nrhs normwise backward error satisfies the
    // stopping criterion. If yes, set iter=0 and return.
    colNorms( Norm::Max, X, colnorms_X.data(), opts );
    colNorms( Norm::Max, R, colnorms_R.data(), opts );

    if (iterRefConverged<real_t>( colnorms_R, colnorms_X, cte )) {
        iter = 0;
        converged = true;
    }

    for (int64_t iiter = 0; iiter < itermax && ! converged; ++iiter) {
        gerbt( U, R );
        getrs_nopiv( A, R, opts );
        gerbt( V, R );
        add( one, R, one, X, host_opts );
        slate::copy( B, R, host_opts );
        gemmA( -one, A_copy, X,
                one, R, host_opts );

        // Check whether nrhs normwise backward error satisfies the
        // stopping criterion. If yes, set iter = iiter > 0 and return.
        colNorms( Norm::Max, X, colnorms_X.data(), opts );
        colNorms( Norm::Max, R, colnorms_R.data(), opts );

        if (iterRefConverged<real_t>( colnorms_R, colnorms_X, cte )) {
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

        if (use_fallback) {
            slate::copy( B, X, opts );
            slate::copy( A_copy, A, host_opts );
            Pivots pivots;
            gesv( A_copy, pivots, X, opts );
        }
    }

    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gesv_rbt<float>(
    Matrix<float>& A,
    Matrix<float>& B,
    Matrix<float>& X,
    int& iter,
    Options const& opts);

template
void gesv_rbt<double>(
    Matrix<double>& A,
    Matrix<double>& B,
    Matrix<double>& X,
    int& iter,
    Options const& opts);

template
void gesv_rbt< std::complex<float> >(
    Matrix< std::complex<float> >& A,
    Matrix< std::complex<float> >& B,
    Matrix< std::complex<float> >& X,
    int& iter,
    Options const& opts);

template
void gesv_rbt< std::complex<double> >(
    Matrix< std::complex<double> >& A,
    Matrix< std::complex<double> >& B,
    Matrix< std::complex<double> >& X,
    int& iter,
    Options const& opts);

} // namespace slate
