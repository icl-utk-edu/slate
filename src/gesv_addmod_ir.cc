// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/Tile_blas.hh"
#include "internal/internal.hh"

namespace slate {

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

template <typename scalar_t>
void gesv_addmod_ir( Matrix<scalar_t>& A, AddModFactors<scalar_t>& W,
                     Matrix<scalar_t>& B,
                     Matrix<scalar_t>& X,
                     int& iter,
                     Options const& opts)
{
    // gemmA and trsmA_addmod are both limited to the host (for # devices > 1)
    // use host_opts for those and other memory-bound ops
    Options host_opts = opts;
    host_opts[ Option::Target ] = Target::HostTask;

    bool converged = false;
    const int itermax = 30;
    using real_t = blas::real_type<scalar_t>;
    const scalar_t one = 1.0;
    const real_t eps = std::numeric_limits<real_t>::epsilon();
    iter = 0;

    assert( B.mt() == A.mt() );

    // workspace
    auto R    = B.emptyLike();
    auto A_lu = A.template emptyLike();

    // insert local tiles
    R.   insertLocalTiles( Target::HostTask );
    A_lu.insertLocalTiles( Target::HostTask );

    std::vector<real_t> colnorms_X( X.n() );
    std::vector<real_t> colnorms_R( R.n() );

    // norm of A and B
    real_t Anorm = norm( Norm::Inf, A, host_opts );

    // stopping criteria
    real_t cte = Anorm * eps * std::sqrt( A.n() );

    // Compute the LU factorization of A_lo.
    slate::copy( A, A_lu, opts );
    getrf_addmod( A_lu, W, opts );


    // Solve the system A_lu * X = B.
    slate::copy( B, X, opts );
    getrs_addmod( W, X, opts );

    // Compute R = B - A * X.
    slate::copy( B, R, host_opts );
    gemm<scalar_t>(
        -one, A,
              X,
        one,  R, host_opts );

    // Check whether the nrhs normwise backward error satisfies the
    // stopping criterion. If yes, set iter=0 and return.
    colNorms( Norm::Max, X, colnorms_X.data(), opts );
    colNorms( Norm::Max, R, colnorms_R.data(), opts );

    if (iterRefConverged<real_t>( colnorms_R, colnorms_X, cte )) {
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
              one, X, host_opts );

        // Compute R = B - A * X.
        slate::copy( B, R, host_opts );
        gemm<scalar_t>(
            -one, A,
                  X,
            one,  R, host_opts );

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
        // set up the iter flag accordingly and follow up with the pivoted routine.
        iter = -itermax - 1;

        // Compute the LU factorization of A.
        Pivots pivots;
        getrf( A, pivots, opts );

        // Solve the system A * X = B.
        slate::copy( B, X, opts );
        getrs( A, pivots, X, opts );
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
