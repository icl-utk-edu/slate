// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
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
static bool iterRefConverged(std::vector<scalar_t>& colnorms_R,
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

template<typename scalar_t>
static slate::Matrix<scalar_t> alloc_V(slate::Matrix<scalar_t> A, int64_t n)
{
    typedef typename Matrix<scalar_t>::ij_tuple ij_tuple;

    const int64_t m = A.m();
    const int64_t mt = A.mt();
    const int64_t nt = A.nt();
    const MPI_Comm mpi_comm = A.mpiComm();

    std::vector<int64_t> tileMb(mt);
    for (int64_t i = 0; i < mt; ++i) {
        tileMb[i] = A.tileMb(i);
    }
    std::function<int64_t(int64_t)> tileMb_lambda = [tileMb] (int64_t i) {
        return tileMb[i];
    };

    std::vector<int64_t> tileNb(nt);
    for (int64_t i = 0; i < nt; ++i) {
        tileNb[i] = A.tileNb(i);
    }
    std::function<int64_t(int64_t)> tileNb_lambda = [tileNb] (int64_t i) {
        return tileNb[i];
    };

    std::vector<int> tileRank(nt);
    for (int64_t i = 0; i < mt; ++i) {
        tileRank[i] = A.tileRank(i, 0);
    }
    std::function<int(ij_tuple)> tileRank_lambda = [tileRank] (ij_tuple ij) {
        return tileRank[std::get<0>(ij)];
    };

    std::function<int(ij_tuple)> tileDevice_lambda = [] (ij_tuple) {
        return HostNum;
    };


    Matrix<scalar_t> V(m, n, tileMb_lambda, tileNb_lambda, tileRank_lambda, tileDevice_lambda, mpi_comm);
    V.insertLocalTiles(Target::Host);
    return V;
}

//------------------------------------------------------------------------------
/// Distributed parallel LU factorization and solve.
/// Generic implementation for any target.
/// @ingroup gesv_specialization
///
template <Target target, typename scalar_hi, typename scalar_lo>
void gesv_mixed_gmres(Matrix<scalar_hi>& A, Pivots& pivots,
                      Matrix<scalar_hi>& B,
                      Matrix<scalar_hi>& X,
                      int& iter,
                      Options const& opts)
{
    // Assumes column major
    const Layout layout = Layout::ColMajor;

    bool converged = false;
    using real_hi = blas::real_type<scalar_hi>;
    const real_hi eps = std::numeric_limits<real_hi>::epsilon();
    iter = 0;
    const int64_t itermax = 30;
    const int64_t restart = std::min(int64_t(30), itermax);
    const int64_t mpi_rank = A.mpiRank();

    assert(B.mt() == A.mt());

    // Only a single rhs is currently supported; TODO: implement block gmres
    assert(B.n() == 1);

    // workspace
    auto R    = B.emptyLike();
    auto A_lo = A.template emptyLike<scalar_lo>();
    auto X_lo = X.template emptyLike<scalar_lo>();

    std::vector<real_hi> colnorms_X(X.n());
    std::vector<real_hi> colnorms_R(R.n());

    auto V = alloc_V(A, itermax+1);
    auto z = alloc_V(A, 1);
    // allocate H as a single tile
    slate::Matrix<scalar_hi> H(itermax+1, itermax+1, itermax+1, 1, 1, A.mpiComm());
    H.insertLocalTiles(Target::Host);
    // allocate S as a single tile
    slate::Matrix<scalar_hi> S(itermax+1, 1, itermax+1, 1, 1, A.mpiComm());
    S.insertLocalTiles(Target::Host);
    std::vector<real_hi>   givens_alpha(itermax);
    std::vector<scalar_hi> givens_beta (itermax);

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
    real_hi Anorm = norm(Norm::Inf, A, opts);

    // stopping criteria
    real_hi cte = Anorm * eps * std::sqrt(A.n());

    // Convert A from high to low precision, store result in A_lo.
    slate::copy(A, A_lo, opts);

    // Compute the LU factorization of A_lo.
    getrf(A_lo, pivots, opts);


    // Solve the system A * X = B in low precision.
    slate::copy(B, X_lo, opts);
    getrs(A_lo, pivots, X_lo, opts);
    slate::copy(X_lo, X, opts);



    // IR
    int iiter = 0;
    while (iiter < itermax) {
        std::cout << "IR: " << iiter << std::endl;


        // Check for convergence
        slate::copy(B, R, opts);
        gemm<scalar_hi>(
            scalar_hi(-1.0), A,
                             X,
            scalar_hi(1.0),  R,
            opts);
        colNorms( Norm::Max, X, colnorms_X.data(), opts );
        colNorms( Norm::Max, R, colnorms_R.data(), opts );
        std::cout << "error: " << (colnorms_R[0]/(colnorms_X[0]*cte)) << " = " << colnorms_R[0] << " / " << colnorms_X[0] << "*" << cte << std::endl;
        if (iterRefConverged<real_hi>(colnorms_R, colnorms_X, cte)) {
            iter = iiter;
            converged = true;
            break;
        }

        // GMRES

        // Compute initial vector
        auto v0 = V.slice(0, V.m()-1, 0, 0);
        slate::copy(R, v0, opts);

        real_hi arnoldi_residual = norm(Norm::Fro, v0, opts);
        std::cout << "  beta = " << arnoldi_residual << std::endl;
        if (arnoldi_residual == 0) {
            // Solver broke down, but residual is not small enough yet.
            converged = false;
            break;
        }
        scale(1.0, arnoldi_residual, v0, opts);
        if (S.tileRank(0, 0) == mpi_rank) {
            S.tileGetForWriting(0, 0, LayoutConvert::ColMajor);
            auto S_00 = S(0, 0);
            S_00.at(0, 0) = arnoldi_residual;
            for (int i = 1; i < S_00.mb(); ++i) {
                S_00.at(i, 0) = 0.0;
            }
        }

        real_hi restart_tol = arnoldi_residual*eps*std::sqrt(A.n());

        int j = 0;
        for (; j < restart && iiter < itermax && arnoldi_residual > restart_tol;
               j++, iiter++) {
            auto Vj1 = V.slice(0, V.m()-1, j+1, j+1);

            auto Vj = V.slice(0, V.m()-1, j, j);

            // Vj1 = M^-1 A Vj
            slate::copy(Vj, X_lo, opts);
            getrs(A_lo, pivots, X_lo, opts);
            slate::copy(X_lo, z, opts);

            gemm<scalar_hi>(
                scalar_hi(1.0), A,
                                z,
                scalar_hi(0.0), Vj1,
                opts);

            // orthogonalize w/ CGS2
            auto V0j = V.slice(0, V.m()-1, 0, j);
            auto V0jT = conjTranspose(V0j);
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

                // apply givens rotations
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
                arnoldi_residual = cabs1(S_00.at(j+1, 0));
            }
            MPI_Bcast(&arnoldi_residual, 1, mpi_type<scalar_hi>::value,
                      S.tileRank(0, 0), A.mpiComm());
            if (A.mpiRank() == 0) {
                std::cout << "0: arnoldi_residual = " << arnoldi_residual << std::endl;
            }
        }
        // update X
        auto H_j = H.slice(0, j-1, 0, j-1);
        auto S_j = S.slice(0, j-1, 0, 0);
        auto H_tri = TriangularMatrix<scalar_hi>(Uplo::Upper, Diag::NonUnit, H_j);
        trsm(Side::Left, scalar_hi(1.0), H_tri, S_j, opts);
        auto V_j = V.slice(0, V.m()-1, 0, j-1);
        gemm<scalar_hi>(
            scalar_hi(1.0), V_j,
                            S_j,
            scalar_hi(0.0), z,
            opts);
        slate::copy(z, X_lo, opts);
        getrs(A_lo, pivots, X_lo, opts);
        slate::copy(X_lo, z, opts);
        add(scalar_hi(1.0), z, scalar_hi(1.0), X, opts);

    }

    if (! converged) {
        // If we are at this place of the code, this is because we have performed
        // iter = itermax iterations and never satisfied the stopping criterion,
        // set up the iter flag accordingly and follow up with double precision
        // routine.
        iter = -iiter-1;

        // Compute the LU factorization of A.
        //getrf(A, pivots, opts);

        // Solve the system A * X = B.
        //slate::copy(B, X, opts);
        //getrs(A, pivots, X, opts);
    }

    if (target == Target::Devices) {
        // clear instead of release due to previous hold
        A.clearWorkspace();
        B.clearWorkspace();
        X.clearWorkspace();
    }
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
/// @ingroup gesv
///
template <typename scalar_hi, typename scalar_lo>
void gesv_mixed_gmres( Matrix<scalar_hi>& A, Pivots& pivots,
                    Matrix<scalar_hi>& B,
                    Matrix<scalar_hi>& X,
                    int& iter,
                    Options const& opts)
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            gesv_mixed_gmres<Target::HostTask,  scalar_hi, scalar_lo>(
                     A, pivots, B, X, iter, opts);
            break;
        case Target::HostNest:
            gesv_mixed_gmres<Target::HostNest,  scalar_hi, scalar_lo>(
                     A, pivots, B, X, iter, opts);
            break;
        case Target::HostBatch:
            gesv_mixed_gmres<Target::HostBatch, scalar_hi, scalar_lo>(
                     A, pivots, B, X, iter, opts);
            break;
        case Target::Devices:
            gesv_mixed_gmres<Target::Devices,   scalar_hi, scalar_lo>(
                     A, pivots, B, X, iter, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template <>
void gesv_mixed_gmres<double>(
    Matrix<double>& A, Pivots& pivots,
    Matrix<double>& B,
    Matrix<double>& X,
    int& iter,
    Options const& opts)
{
    gesv_mixed_gmres<double, float>(A, pivots, B, X, iter, opts);
}

template <>
void gesv_mixed_gmres< std::complex<double> >(
    Matrix< std::complex<double> >& A, Pivots& pivots,
    Matrix< std::complex<double> >& B,
    Matrix< std::complex<double> >& X,
    int& iter,
    Options const& opts)
{
    gesv_mixed_gmres<std::complex<double>, std::complex<float>>(A, pivots, B, X, iter, opts);
}

} // namespace slate
