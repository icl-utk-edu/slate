// Copyright (c) 2022, University of Tennessee. All rights reserved.
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
/// Distributed parallel GMRES-IR Cholesky factorization and solve.
///
/// Computes the solution to a system of linear equations
/// \[
///     A X = B,
/// \]
/// where $A$ is an n-by-n Hermitian positive definite matrix and $X$ and $B$
/// are n-by-nrhs matrices.
///
/// posv_mixed_gmres first factorizes the matrix using potrf in low precision
/// (single) and uses this factorization within a GMRES-IR procedure to
/// produce a solution with high precision (double) normwise backward error
/// quality (see below). If the approach fails, the method falls back to a
/// high precision (double) factorization and solve.
///
/// GMRES-IR is not going to be a winning strategy if the ratio of
/// low-precision performance over high-precision performance is too small.
/// A reasonable strategy should take the number of right-hand sides and the
/// the size of the matrix into account. This might be automated in the future.
/// Up to now, we always try iterative refinement.
///
/// GMRES-IR process is stopped if iter > itermax or for all the RHS,
/// $1 \le j \le nrhs$, we have:
///     $\norm{r_j}_{inf} < \sqrt{n} \norm{x_j}_{inf} \norm{A}_{inf} \epsilon_{\mathrm{hi}},$
/// where:
/// - iter is the number of the current iteration in the iterative refinement
///    process
/// - $\norm{r_j}_{inf}$ is the infinity-norm of the residual, $r_j = Ax_j - b_j$
/// - $\norm{x_j}_{inf}$ is the infinity-norm of the solution
/// - $\norm{A}_{inf}$ is the infinity-operator-norm of the matrix $A$
/// - $\epsilon_{\mathrm{hi}}$ is the machine epsilon of double precision.
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
///     On entry, the n-by-n Hermitian positive definite matrix $A$.
///     On exit, if iterative refinement has been successfully used
///     (return value = 0 and iter >= 0, see description below), then $A$ is
///     unchanged. If high precision (double) factorization has been used
///     (return value = 0 and iter < 0, see description below), then the
///     array $A$ contains the factors $L$ or $U$ from the Cholesky
///     factorization $A = U^H U$ or $A = L L^H$.
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
///     to be -(1+itermax), where itermax is controlled by the opts argument.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::Lookahead:
///       Number of panels to overlap with matrix updates.
///       lookahead >= 0. Default 1.
///     - Option::MaxIterations
///       The iteration limit for refinement. Default 30.
///     - Option::MaxPanelThreads:
///       Number of threads to use for panel. Default omp_get_max_threads()/2.
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
/// @ingroup posv
///
template <typename scalar_hi, typename scalar_lo>
void posv_mixed_gmres(
    HermitianMatrix<scalar_hi>& A,
    Matrix<scalar_hi>& B,
    Matrix<scalar_hi>& X,
    int& iter,
    Options const& opts)
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    bool converged = false;
    using real_hi = blas::real_type<scalar_hi>;
    const real_hi eps = std::numeric_limits<real_hi>::epsilon();
    iter = 0;
    const int64_t itermax = 30;
    const int64_t restart = std::min(std::min(int64_t(30), itermax), A.tileMb(0)-1);
    const int64_t mpi_rank = A.mpiRank();

    assert(B.mt() == A.mt());

    // TODO: implement block gmres
    if (B.n() != 1) {
        slate_not_implemented("block-GMRES is not yet supported");
    }

    // workspace
    auto R    = B.emptyLike();
    R.   insertLocalTiles(target);
    auto A_lo = A.template emptyLike<scalar_lo>();
    A_lo.insertLocalTiles(target);
    auto X_lo = X.template emptyLike<scalar_lo>();
    X_lo.insertLocalTiles(target);

    std::vector<real_hi> colnorms_X(X.n());
    std::vector<real_hi> colnorms_R(R.n());

    // test basis.  First column corresponds to the residual
    auto V = internal::alloc_basis(A, restart+1, target);
    // solution basis.  Columns correspond to those in V.  First column is unused
    auto W = internal::alloc_basis(A, restart+1, target);

    // workspace vector for the orthogonalization process
    auto z = X.template emptyLike<scalar_hi>();
    z.insertLocalTiles(target);

    // Hessenberg Matrix.  Allocate as a single tile
    slate::Matrix<scalar_hi> H(restart+1, restart+1, restart+1, 1, 1, A.mpiComm());
    H.insertLocalTiles(Target::Host);
    // least squares RHS.  Allocate as a single tile
    slate::Matrix<scalar_hi> S(restart+1, 1, restart+1, 1, 1, A.mpiComm());
    S.insertLocalTiles(Target::Host);
    // Rotations
    std::vector<real_hi>   givens_alpha(restart);
    std::vector<scalar_hi> givens_beta (restart);


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
    real_hi Anorm = norm(Norm::Inf, A, opts);

    // stopping criteria
    real_hi cte = Anorm * eps * std::sqrt(A.n());

    // Compute the Cholesky factorization of A in single-precision.
    slate::copy(A, A_lo, opts);
    potrf(A_lo, opts);


    // Solve the system A * X = B in low precision.
    slate::copy(B, X_lo, opts);
    potrs(A_lo, X_lo, opts);
    slate::copy(X_lo, X, opts);


    // IR
    int iiter = 0;
    while (iiter < itermax) {

        // Check for convergence
        slate::copy(B, R, opts);
        hemm<scalar_hi>(
            Side::Left,
            scalar_hi(-1.0), A,
                             X,
            scalar_hi(1.0),  R,
            opts);
        colNorms( Norm::Max, X, colnorms_X.data(), opts );
        colNorms( Norm::Max, R, colnorms_R.data(), opts );
        if (internal::iterRefConverged<real_hi>(colnorms_R, colnorms_X, cte)) {
            iter = iiter;
            converged = true;
            break;
        }

        // GMRES

        // Compute initial vector
        auto v0 = V.slice(0, V.m()-1, 0, 0);
        slate::copy(R, v0, opts);

        std::vector<real_hi> arnoldi_residual = {norm(Norm::Fro, v0, opts)};
        if (arnoldi_residual[0] == 0) {
            // Solver broke down, but residual is not small enough yet.
            iter = iiter;
            converged = false;
            break;
        }
        scale(1.0, arnoldi_residual[0], v0, opts);
        if (S.tileRank(0, 0) == mpi_rank) {
            S.tileGetForWriting(0, 0, LayoutConvert::ColMajor);
            auto S_00 = S(0, 0);
            S_00.at(0, 0) = arnoldi_residual[0];
            for (int i = 1; i < S_00.mb(); ++i) {
                S_00.at(i, 0) = 0.0;
            }
        }


        // N.B. convergence is detected using norm(X) at the beginning of the
        // outer iteration. Thus, changes in the magnitude of X may lead to
        // excessive restarting or delayed completion.
        int j = 0;
        for (; j < restart && iiter < itermax
                   && !internal::iterRefConverged(arnoldi_residual, colnorms_X, cte);
             j++, iiter++) {
            auto Vj1 = V.slice(0, V.m()-1, j+1, j+1);
            auto Wj1 = W.slice(0, W.m()-1, j+1, j+1);

            auto Vj = V.slice(0, V.m()-1, j, j);

            // Wj1 = M^-1 A Vj
            slate::copy(Vj, X_lo, opts);
            potrs(A_lo, X_lo, opts);
            slate::copy(X_lo, Wj1, opts);

            hemm<scalar_hi>(
                Side::Left,
                scalar_hi(1.0), A,
                                Wj1,
                scalar_hi(0.0), Vj1,
                opts);

            // orthogonalize w/ CGS2
            auto V0j = V.slice(0, V.m()-1, 0, j);
            auto V0jT = conj_transpose(V0j);
            auto Hj = H.slice(0, j, j, j);
            gemm<scalar_hi>(
                scalar_hi(1.0), V0jT,
                                Vj1,
                scalar_hi(0.0), Hj,
                opts);
            gemm<scalar_hi>(
                scalar_hi(-1.0), V0j,
                                 Hj,
                scalar_hi(1.0),  Vj1,
                opts);
            auto zj = z.slice(0, j, 0, 0);
            gemm<scalar_hi>(
                scalar_hi(1.0), V0jT,
                                Vj1,
                scalar_hi(0.0), zj,
                opts);
            gemm<scalar_hi>(
                scalar_hi(-1.0), V0j,
                                 zj,
                scalar_hi(1.0),  Vj1,
                opts);
            add(scalar_hi(1.0), zj, scalar_hi(1.0), Hj,
                opts);
            auto Vj1_norm = norm(Norm::Fro, Vj1, opts);
            scale(1.0, Vj1_norm, Vj1, opts);
            if (H.tileRank(0, 0) == mpi_rank) {
                H.tileGetForWriting(0, 0, LayoutConvert::ColMajor);
                auto H_00 = H(0, 0);
                H_00.at(j+1, j) = Vj1_norm;
            }

            // apply givens rotations
            if (H.tileRank(0, 0) == mpi_rank) {
                auto H_00 = H(0, 0);
                for (int64_t i = 0; i < j; ++i) {
                    blas::rot(1, &H_00.at(i, j), 1, &H_00.at(i+1, j), 1,
                              givens_alpha[i], givens_beta[i]);
                }
                scalar_hi H_jj = H_00.at(j, j), H_j1j = H_00.at(j+1, j);
                blas::rotg(&H_jj, & H_j1j, &givens_alpha[j], &givens_beta[j]);
                blas::rot(1, &H_00.at(j, j), 1, &H_00.at(j+1, j), 1,
                          givens_alpha[j], givens_beta[j]);
                auto S_00 = S(0, 0);
                blas::rot(1, &S_00.at(j, 0), 1, &S_00.at(j+1, 0), 1,
                          givens_alpha[j], givens_beta[j]);
                arnoldi_residual[0] = cabs1(S_00.at(j+1, 0));
            }
            MPI_Bcast(arnoldi_residual.data(), arnoldi_residual.size(),
                      mpi_type<scalar_hi>::value, S.tileRank(0, 0), A.mpiComm());
        }
        // update X
        auto H_j = H.slice(0, j-1, 0, j-1);
        auto S_j = S.slice(0, j-1, 0, 0);
        auto H_tri = TriangularMatrix<scalar_hi>(Uplo::Upper, Diag::NonUnit, H_j);
        trsm(Side::Left, scalar_hi(1.0), H_tri, S_j, opts);
        auto W_0j = W.slice(0, W.m()-1, 1, j); // first column of W is unused
        gemm<scalar_hi>(
            scalar_hi(1.0), W_0j,
                            S_j,
            scalar_hi(1.0), X,
            opts);
    }

    if (! converged) {
        // If we are at this place of the code, this is because we have performed
        // iter = itermax iterations and never satisfied the stopping criterion,
        // set up the iter flag accordingly and follow up with double precision
        // routine.
        iter = -iiter-1;

        // Compute the Cholesky factorization of A.
        potrf(A, opts);

        // Solve the system A * X = B.
        slate::copy(B, X, opts);
        potrs(A, X, opts);
    }

    if (target == Target::Devices) {
        // clear instead of release due to previous hold
        A.clearWorkspace();
        B.clearWorkspace();
        X.clearWorkspace();
    }
}


//------------------------------------------------------------------------------
// Explicit instantiations.
template <>
void posv_mixed_gmres<double>(
    HermitianMatrix<double>& A,
    Matrix<double>& B,
    Matrix<double>& X,
    int& iter,
    Options const& opts)
{
    posv_mixed_gmres<double, float>(A, B, X, iter, opts);
}

template <>
void posv_mixed_gmres< std::complex<double> >(
    HermitianMatrix< std::complex<double> >& A,
    Matrix< std::complex<double> >& B,
    Matrix< std::complex<double> >& X,
    int& iter,
    Options const& opts)
{
    posv_mixed_gmres<std::complex<double>, std::complex<float>>(A, B, X, iter, opts);
}

} // namespace slate
