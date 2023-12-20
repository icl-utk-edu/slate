// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/Tile_blas.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"

namespace slate {

template <typename scalar_t>
void gesv_addmod_ir( Matrix<scalar_t>& A, AddModFactors<scalar_t>& W,
                     Matrix<scalar_t>& B,
                     Matrix<scalar_t>& X,
                     int& iter,
                     Options const& opts)
{
    using real_t = blas::real_type<scalar_t>;

    Target target = get_option( opts, Option::Target, Target::HostTask );

    // Most routines prefer column major
    const Layout layout = Layout::ColMajor;
    const scalar_t one = 1.0;
    const real_t eps = std::numeric_limits<real_t>::epsilon();

    int64_t itermax = get_option<int64_t>( opts, Option::MaxIterations, 30 );
    double tol = get_option<double>( opts, Option::Tolerance, eps*std::sqrt(A.m()) );
    bool use_fallback = get_option<int64_t>( opts, Option::UseFallbackSolver, true );

    assert( B.mt() == A.mt() );

    // workspace
    auto R    = B.emptyLike();
    auto A_lu = A.template emptyLike();

    // insert local tiles
    R.   insertLocalTiles( target );
    A_lu.insertLocalTiles( target );

    if (target == Target::Devices && itermax != 0) {
        #pragma omp parallel
        #pragma omp master
        #pragma omp taskgroup
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

    std::vector<real_t> colnorms_X( X.n() );
    std::vector<real_t> colnorms_R( R.n() );

    // stopping criteria
    real_t Anorm = norm( Norm::Inf, A, opts );
    real_t cte = Anorm*tol;
    bool converged = false;

    // Compute the LU factorization of A_lo.
    slate::copy( A, A_lu, opts );
    getrf_addmod( A_lu, W, opts );


    // Solve the system A_lu * X = B.
    slate::copy( B, X, opts );
    getrs_addmod( W, X, opts );

    if (itermax == 0) {
        return;
    }

    // compute r = b - a * x.
    slate::copy( B, R, opts );
    gemm<scalar_t>(
        -one, A,
              X,
        one,  R, opts );

    // Check whether the nrhs normwise backward error satisfies the
    // stopping criterion. If yes, set iter=0 and return.
    colNorms( Norm::Max, X, colnorms_X.data(), opts );
    colNorms( Norm::Max, R, colnorms_R.data(), opts );

    if (internal::iterRefConverged<real_t>( colnorms_R, colnorms_X, cte )) {
        iter = 0;
        converged = true;
    }

    // iterative refinement
    for (int iiter = 0; iiter < itermax && ! converged; iiter++) {
        // Solve the system A_lu * X = R.
        getrs_addmod( W, R, opts );

        // Update the current iterate.
        add<scalar_t>(
              one, R,
              one, X, opts );

        // Compute R = B - A * X.
        slate::copy( B, R, opts );
        gemm<scalar_t>(
            -one, A,
                  X,
            one,  R, opts );

        // Check whether nrhs normwise backward error satisfies the
        // stopping criterion. If yes, set iter = iiter > 0 and return.
        colNorms( Norm::Max, X, colnorms_X.data(), opts );
        colNorms( Norm::Max, R, colnorms_R.data(), opts );

        if (internal::iterRefConverged<real_t>( colnorms_R, colnorms_X, cte )) {
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
            Pivots pivots;
            gesv( A, pivots, X, opts );
        }
    }

    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gesv_addmod_ir<float>(
    Matrix<float>& A, AddModFactors<float>& W,
    Matrix<float>& B,
    Matrix<float>& X,
    int& iter,
    Options const& opts);

template
void gesv_addmod_ir<double>(
    Matrix<double>& A, AddModFactors<double>& W,
    Matrix<double>& B,
    Matrix<double>& X,
    int& iter,
    Options const& opts);

template
void gesv_addmod_ir< std::complex<float> >(
    Matrix< std::complex<float> >& A, AddModFactors< std::complex<float> >& W,
    Matrix< std::complex<float> >& B,
    Matrix< std::complex<float> >& X,
    int& iter,
    Options const& opts);

template
void gesv_addmod_ir< std::complex<double> >(
    Matrix< std::complex<double> >& A, AddModFactors< std::complex<double> >& W,
    Matrix< std::complex<double> >& B,
    Matrix< std::complex<double> >& X,
    int& iter,
    Options const& opts);

} // namespace slate
