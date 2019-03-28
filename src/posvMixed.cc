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
// #include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::posvMixed from internal::specialization::posvMixed
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

///-----------------------------------------------------------------------------
/// \brief
/// Distributed parallel iterative-refinement Cholesky factorization and solve.
///
/// Computes the solution to a system of linear equations A * X = B, where A is
/// an n-by-n Hermitian positive definite matrix and X and B are n-by-nrhs matrices.
///
/// posvMixed first factorizes the matrix using potrf on lower precision and uses
/// this factorization within an iterative refinement procedure to produce a
/// solution with ?? normwise backward error quality (see below). If
/// the approach fails the method falls back to a full precision factorization and
/// solve.
///
/// The iterative refinement is not going to be a winning strategy if
/// the ratio COMPLEX performance over COMPLEX*16 performance is too
/// small. A reasonable strategy should take the number of right-hand
/// sides and the size of the matrix into account. This might be done
/// with a call to ILAENV in the future. Up to now, we always try
/// iterative refinement.
///
/// The iterative refinement process is stopped if iter > itermax or
/// for all the RHS we have: Rnorm < sqrt(n)*Xnorm*Anorm*eps, where:
///
/// - iter is the number of the current iteration in the iterative refinement
///    process
/// - Rnorm is the Infinity-norm of the residual
/// - Xnorm is the Infinity-norm of the solution
/// - Anorm is the Infinity-operator-norm of the matrix A
/// - eps is the machine epsilon.
/// The values itermax is fixed to 30.
/// Generic implementation for any target.
template <Target target, typename scalar_hi, typename scalar_lo>
void posvMixed( slate::internal::TargetType<target>,
                HermitianMatrix<scalar_hi>& A,
                Matrix<scalar_hi>& B,
                Matrix<scalar_hi>& X,
                int& iter,
                int64_t lookahead)
{
    bool converged = false;
    const int itermax = 30;
    using real_hi = blas::real_type<scalar_hi>;
    const real_hi eps = std::numeric_limits<real_hi>::epsilon();
    iter = 0;

    assert(B.mt() == A.mt());

    // if upper, change to lower
    if (A.uplo_logical() == Uplo::Upper)
        A = conj_transpose(A);

    // workspace
    auto R    = B.emptyLike();
    auto A_lo = A.template emptyLike<scalar_lo>();
    auto X_lo = X.template emptyLike<scalar_lo>();

    std::vector<real_hi> colnorms_X(X.n());
    std::vector<real_hi> colnorms_R(R.n());

    // insert local tiles
    X_lo.insertLocalTiles(target == Target::Devices);
    R.insertLocalTiles(target == Target::Devices);
    A_lo.insertLocalTiles(target == Target::Devices);

    if (target == Target::Devices){
        A.tileGetAndHoldAllOnDevices();
        B.tileGetAndHoldAllOnDevices();
        X.tileGetAndHoldAllOnDevices();
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

    // Compute the Cholesky factorization of A_lo.
    potrf(A_lo,
          {{Option::Lookahead, lookahead},
           {Option::Target, target}});

    // Solve the system A_lo * X_lo = B_lo.
    potrs(A_lo, X_lo,
          {{Option::Lookahead, lookahead},
           {Option::Target, target}});

    // Convert X_lo to high precision.
    copy(X_lo, X,
         {{Option::Target, target}});

    // Compute R = B - A * X.
    copy(B, R,
         {{Option::Target, target}});
    hemm<scalar_hi>(
        Side::Left,
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
    for (int iiter = 0; iiter < itermax && !converged; iiter++) {
        // Convert R from high to low precision, store result in X_lo.
        copy(R, X_lo,
             {{Option::Target, target}});

        // Solve the system A_lo * X_lo = R_lo.
        potrs(A_lo, X_lo,
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
        hemm<scalar_hi>(
            Side::Left,
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

    if ( ! converged ) {
        // If we are at this place of the code, this is because we have performed
        // iter = itermax iterations and never satisfied the stopping criterion,
        // set up the iter flag accordingly and follow up with double precision
        // routine.
        iter = -itermax - 1;

        // Compute the Cholesky factorization of A.
        potrf(A,
              {{Option::Lookahead, lookahead},
               {Option::Target, target}});

        // Solve the system A * X = B.
        copy(B, X,
             {{Option::Target, target}});
        potrs(A, X,
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
/// @ingroup gesv_comp
template <Target target, typename scalar_hi, typename scalar_lo>
void posvMixed( HermitianMatrix<scalar_hi>& A,
                Matrix<scalar_hi>& B,
                Matrix<scalar_hi>& X,
                int& iter,
                const std::map<Option, Value>& opts)
{
    int64_t lookahead;
    try {
        lookahead = opts.at(Option::Lookahead).i_;
        assert(lookahead >= 0);
    }
    catch (std::out_of_range) {
        lookahead = 1;
    }

    internal::specialization::posvMixed<target, scalar_hi, scalar_lo>(
                                        internal::TargetType<target>(),
                                        A, B, X, iter,
                                        lookahead);
}

//------------------------------------------------------------------------------
/// Distributed parallel iterative-refinement Cholesky factorization and solve.
///
template <typename scalar_hi, typename scalar_lo>
void posvMixed( HermitianMatrix<scalar_hi>& A,
                Matrix<scalar_hi>& B,
                Matrix<scalar_hi>& X,
                int& iter,
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
            posvMixed<Target::HostTask,  scalar_hi, scalar_lo>(
                      A, B, X, iter, opts);
            break;
        case Target::HostNest:
            posvMixed<Target::HostNest,  scalar_hi, scalar_lo>(
                      A, B, X, iter, opts);
            break;
        case Target::HostBatch:
            posvMixed<Target::HostBatch, scalar_hi, scalar_lo>(
                      A, B, X, iter, opts);
            break;
        case Target::Devices:
            posvMixed<Target::Devices,   scalar_hi, scalar_lo>(
                      A, B, X, iter, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template <>
void posvMixed<double>(
                HermitianMatrix<double>& A,
                Matrix<double>& B,
                Matrix<double>& X,
                int& iter,
                const std::map<Option, Value>& opts)
{
    posvMixed<double, float>( A, B, X, iter, opts);
}

template <>
void posvMixed< std::complex<double> >(
                HermitianMatrix< std::complex<double> >& A,
                Matrix< std::complex<double> >& B,
                Matrix< std::complex<double> >& X,
                int& iter,
                const std::map<Option, Value>& opts)
{
    posvMixed<std::complex<double>, std::complex<float>>( A, B, X, iter, opts);
}

} // namespace slate
