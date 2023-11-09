// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/Tile_blas.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Distributed parallel iterative-refinement Cholesky factorization and solve.
///
/// Computes the solution to a system of linear equations
/// \[
///     A X = B,
/// \]
/// where $A$ is an n-by-n Hermitian positive definite matrix and $X$ and $B$
/// are n-by-nrhs matrices.
///
/// posv_mixed first factorizes the matrix using potrf in low precision (single)
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
///     $\norm{r_j}_{inf} < tol \norm{x_j}_{inf} \norm{A}_{inf},$
/// where:
/// - iter is the number of the current iteration in the iterative refinement
///    process
/// - $\norm{r_j}_{inf}$ is the infinity-norm of the residual, $r_j = Ax_j - b_j$
/// - $\norm{x_j}_{inf}$ is the infinity-norm of the solution
/// - $\norm{A}_{inf}$ is the infinity-operator-norm of the matrix $A$
///
//------------------------------------------------------------------------------
/// @tparam scalar_hi
///     One of double, std::complex<double>.
///
/// @tparam scalar_lo
///     One of float, std::complex<float>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the n-by-n Hermitian positive definite matrix $A$.
///     On exit, if iterative refinement has been successfully used
///     (return value = 0 and iter >= 0, see description below), then $A$ is
///     unchanged. If high precision (double) factorization has been used
///     (return value = 0 and iter < 0, see description below), then the
///     array $A$ contains the factor $U$ or $L$ from the Cholesky
///     factorization $A = U^H U$ or $A = L L^H$.
///     If scalar_t is real, $A$ can be a SymmetricMatrix object.
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
/// @retval >0 for return value = $i$, the leading minor of order $i$ of $A$ is not
///         positive definite, so the factorization could not
///         be completed, and the solution has not been computed.
///
/// @ingroup posv
///
template <typename scalar_hi, typename scalar_lo>
void posv_mixed(
    HermitianMatrix<scalar_hi>& A,
    Matrix<scalar_hi>& B,
    Matrix<scalar_hi>& X,
    int& iter,
    Options const& opts)
{
    Timer t_posv_mixed;

    // XXX This is only used for the memory management and may be inconsistent
    // with the routines called in this routine.
    Target target = get_option( opts, Option::Target, Target::HostTask );

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    bool converged = false;
    using real_hi = blas::real_type<scalar_hi>;
    const real_hi eps = std::numeric_limits<real_hi>::epsilon();
    const scalar_hi one_hi      = 1.0;

    int64_t itermax = get_option<int64_t>( opts, Option::MaxIterations, 30 );
    double tol = get_option<double>( opts, Option::Tolerance, eps*std::sqrt(A.m()) );
    bool use_fallback = get_option<int64_t>( opts, Option::UseFallbackSolver, true );
    iter = 0;

    assert( B.mt() == A.mt() );

    // workspace
    auto R    = B.emptyLike();
    auto A_lo = A.template emptyLike<scalar_lo>();
    auto X_lo = X.template emptyLike<scalar_lo>();

    std::vector<real_hi> colnorms_X( X.n() );
    std::vector<real_hi> colnorms_R( R.n() );

    // insert local tiles
    X_lo.insertLocalTiles( target );
    R.   insertLocalTiles( target );
    A_lo.insertLocalTiles( target );

    if (target == Target::Devices) {
        #pragma omp parallel
        #pragma omp master
        {
            #pragma omp task slate_omp_default_none \
                shared( A ) firstprivate( layout )
            {
                A.tileGetAndHoldAllOnDevices( LayoutConvert( layout ) );
            }
            #pragma omp task slate_omp_default_none \
                shared( B ) firstprivate( layout )
            {
                B.tileGetAndHoldAllOnDevices( LayoutConvert( layout ) );
            }
            #pragma omp task slate_omp_default_none \
                shared( X ) firstprivate( layout )
            {
                X.tileGetAndHoldAllOnDevices( LayoutConvert( layout ) );
            }
        }
    }

    // norm of A
    real_hi Anorm = norm( Norm::Inf, A, opts );

    // stopping criteria
    real_hi cte = Anorm * tol;

    // Convert B from high to low precision, store result in X_lo.
    copy( B, X_lo, opts );

    // Convert A from high to low precision, store result in A_lo.
    copy( A, A_lo, opts );

    // Compute the Cholesky factorization of A_lo.
    Timer t_potrf_lo;
    potrf(  A_lo, opts );
    timers[ "posv_mixed::potrf_lo" ] = t_potrf_lo.stop();

    // Solve the system A_lo * X_lo = B_lo.
    Timer t_potrs_lo;
    potrs( A_lo, X_lo, opts );
    timers[ "posv_mixed::potrs_lo" ] = t_potrs_lo.stop();

    // Convert X_lo to high precision.
    copy( X_lo, X, opts );

    // Compute R = B - A * X.
    slate::copy( B, R, opts );
    Timer t_hemm_hi;
    hemm<scalar_hi>(
        Side::Left,
        -one_hi, A,
                 X,
        one_hi,  R, opts );
    timers[ "posv_mixed::hemm_hi" ] = t_hemm_hi.stop();

    // Check whether the nrhs normwise backward error satisfies the
    // stopping criterion. If yes, set iter=0 and return.
    colNorms( Norm::Max, X, colnorms_X.data(), opts );
    colNorms( Norm::Max, R, colnorms_R.data(), opts );

    if (internal::iterRefConverged<real_hi>( colnorms_R, colnorms_X, cte) ) {
        iter = 0;
        converged = true;
    }

    // iterative refinement
    timers[ "posv_mixed::add_hi" ] = 0;
    for (int iiter = 0; iiter < itermax && ! converged; ++iiter) {
        // Convert R from high to low precision, store result in X_lo.
        copy( R, X_lo, opts );

        // Solve the system A_lo * X_lo = R_lo.
        t_potrs_lo.start();
        potrs( A_lo, X_lo, opts );
        timers[ "posv_mixed::potrs_lo" ] += t_potrs_lo.stop();

        // Convert X_lo back to double precision and update the current iterate.
        copy( X_lo, R, opts );
        Timer t_add_hi;
        add<scalar_hi>(
              one_hi, R,
              one_hi, X, opts );
        timers[ "posv_mixed::add_hi" ] += t_add_hi.stop();

        // Compute R = B - A * X.
        slate::copy( B, R, opts );
        t_hemm_hi.start();
        hemm<scalar_hi>(
            Side::Left,
            -one_hi, A,
                     X,
            one_hi,  R, opts );
        timers[ "posv_mixed::hemm_hi" ] += t_hemm_hi.stop();

        // Check whether nrhs normwise backward error satisfies the
        // stopping criterion. If yes, set iter = iiter > 0 and return.
        colNorms( Norm::Max, X, colnorms_X.data(), opts );
        colNorms( Norm::Max, R, colnorms_R.data(), opts );

        if (internal::iterRefConverged<real_hi>( colnorms_R, colnorms_X, cte )) {
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
            // Compute the Cholesky factorization of A.
            Timer t_potrf_hi;
            potrf( A, opts );
            timers[ "posv_mixed::potrf_hi" ] = t_potrf_hi.stop();

            // Solve the system A * X = B.
            slate::copy( B, X, opts );
            Timer t_potrs_hi;
            potrs( A, X, opts );
            timers[ "posv_mixed::potrs_hi" ] = t_potrs_hi.stop();
        }
    }

    if (target == Target::Devices) {
        // clear instead of release due to previous hold
        A.clearWorkspace();
        B.clearWorkspace();
        X.clearWorkspace();
    }
    timers[ "posv_mixed" ] = t_posv_mixed.stop();

    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template <>
void posv_mixed<double>(
    HermitianMatrix<double>& A,
    Matrix<double>& B,
    Matrix<double>& X,
    int& iter,
    Options const& opts)
{
    posv_mixed<double, float>( A, B, X, iter, opts );
}

template <>
void posv_mixed< std::complex<double> >(
    HermitianMatrix< std::complex<double> >& A,
    Matrix< std::complex<double> >& B,
    Matrix< std::complex<double> >& X,
    int& iter,
    Options const& opts)
{
    posv_mixed<std::complex<double>, std::complex<float>>( A, B, X, iter, opts );
}

} // namespace slate
