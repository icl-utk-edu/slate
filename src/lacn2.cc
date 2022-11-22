// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "slate/Matrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/HermitianBandMatrix.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"
#include "slate_steqr2.hh"

namespace slate {

// todo:
// 1. documentation in lacn2
// 1. documentation in gecon
// 2. V and X as vectors or matrices
// 3. scale and sum (est) with no loop
// 4. quick return inn both
// 5. add tester
// 6. push all changes: lacn2, gecon, scalapck_wrapper
// 7. test multiple mpi, gpus
// 8. test complex: c and z
// 9. check isave[2] increment
// 10. check when inf norm
// re write code as slate structure
// specialization namespace differentiates, e.g.,
// internal::lacn2 from impl::lacn2
namespace impl {

//------------------------------------------------------------------------------
/// An auxiliary routine to alter the sign of a vector
template <typename scalar_t>
void lacn2_altsgn(Matrix<scalar_t>& A)
{
    using real_t = blas::real_type<scalar_t>;

    int64_t mt = A.mt();
    int64_t nt = A.nt();
    int64_t n  = A.n();

    const scalar_t one  = 1.0;
    real_t altsgn;

    //altsgn = 1.;
    // x( i ) = altsgn*( one+dble( i-1 ) / dble( n-1 ) )
    //for (int64_t i = 0; i < n; ++i) {
    //    X[i] = altsgn * ( one + (scalar_t)( i-1 ) / (scalar_t)( n-1 ) );
    //    altsgn = -altsgn;
    //}

    for (int64_t i = 0; i < mt; ++i) {
        for (int64_t j = 0; j < nt; ++j) {
            if (A.tileIsLocal(i, j)) {
                auto Aij = A(i, j);
                auto Aij_data = Aij.data();
                int64_t nb = A.tileMb(i);
                for (int64_t ii = 0; ii < nb; ++ii) {
                    altsgn = pow(-1., i*nb+ii) * altsgn;
                    Aij_data[ii] = altsgn * ( one + (scalar_t)( (i*nb+ii)-1 ) / (scalar_t)( n-1 ) );
                }
            }
        }
    }
}
//------------------------------------------------------------------------------
/// Distributed parallel Cholesky factorization.
/// Generic implementation for any target.
/// Panel and lookahead computed on host using Host OpenMP task.
/// @ingroup posv_specialization
///
template <Target target, typename scalar_t>
void lacn2(
           int64_t n,
           std::vector<scalar_t>& X,
           std::vector<scalar_t>& V,
           std::vector<int64_t>& isgn,
           blas::real_type<scalar_t>* est,
           int* kase,
           std::vector<int64_t>& isave,
           Options const& opts)
{
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    //printf("\n ====== kase %ld mpi_rank %d isave[0] %ld isave[1] %ld \n", *kase, mpi_rank, isave[0], isave[1]);
    using real_t = blas::real_type<scalar_t>;
    using blas::real;

    const scalar_t one  = 1.0;
    const scalar_t zero = 0.0;
    scalar_t xi = 1./n;
    int itmax = 5, jlast;

    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );

    // Assumes column major
    const Layout layout = Layout::ColMajor;
    const auto mpi_real_type = mpi_type< blas::real_type<scalar_t> >::value;

    // isave[0] = jump
    // isave[1] = j which is the index of the max element in X
    // isave[2] = iter

    real_t altsgn;
    real_t estold = 0., temp, absxi;
    real_t safemin = std::numeric_limits< real_t >::min();
    int64_t xs;
    int isavemax = 0;

    int64_t m = n, nb = 10, p = 1, q = 1;
    auto AX = slate::Matrix<scalar_t>::fromLAPACK(
            m, 1, &X[0], m, nb, 1, p, q, MPI_COMM_WORLD);
    auto AV = slate::Matrix<scalar_t>::fromLAPACK(
            m, 1, &V[0], m, nb, 1, p, q, MPI_COMM_WORLD);
    auto Aisgn = slate::Matrix<int64_t>::fromLAPACK(
            m, 1, &isgn[0], m, nb, 1, p, q, MPI_COMM_WORLD);
    int64_t nt = AX.nt();
    int64_t mt = AX.mt();

    // First iteration, kase = 0
    // Initialize X = 1./n
    if (*kase == 0) {
        slate::set(xi, xi, AX);
        // X to be overwritten by AX, so kase = 1.
        *kase = 1;
        isave[0] = 1;
        return;
    }
    // Iterations 2, 3, ..., itmax.
    // kase = 1 or 2
    // X has been overwitten by AX.
    else {
        if (isave[0] == 1) {
            // quick return
            if (n == 1) {
                if (AX.tileIsLocal(0, 0)) {
                    //*est = fabs( V0_data[0] );
                    //V[0] = X[0];
                    //*est = fabs( V[0] );
                    auto X0 = AX(0, 0);
                    auto V0 = AV(0, 0);
                    auto X0_data = X0.data();
                    auto V0_data = V0.data();
                    V0_data[0] = X0_data[0];
                    *est = std::abs( V0_data[0] );
                }
                MPI_Bcast( est, 1, mpi_real_type, 0, MPI_COMM_WORLD );
                // Converged, set kase back to zero
                *kase = 0;
                return;
            }

            // Initial value of est
            *est = slate::norm(slate::Norm::One, AX, opts);
            //printf("\n isave[0] %ld est %e \n", isave[0], *est);

            // Update X to 1. or -1.:
            // X[i] = {1   if X[i] > 0
            //        {-1  if X[i] < 0
            //
            // todo: do I need offset in isgn?
            for (int64_t i = 0; i < mt; ++i) {
                if (AX.tileIsLocal(i, 0)) {
                    auto X0 = AX(i, 0);
                    auto X0_data = X0.data();
                    auto Aisgn0 = Aisgn(i, 0);
                    auto Aisgn0_data = Aisgn0.data();
                    for (int64_t ii = 0; ii < AX.tileMb(i); ++ii) {
                        if (real( X0_data[ii] ) >= 0.0) {
                            X0_data[ii] = one;
                            Aisgn0_data[ii] = 1;
                        }
                        else {
                            X0_data[ii] = -one;
                            Aisgn0_data[ii] = -1;
                        }
                    }
                }
            }
            // Update kase ad isave
            // X be overwritten by A^TX, so kase = 2
            *kase = 2;
            isave[0] = 2;
            return;
        }
        else if (isave[0] == 2) {
            //printf("\n isave[0] %ld \n", isave[0]);
            isave[1] = 0;
            real_t A_max = 0.;
            for (int64_t i = 0; i < mt; ++i) {
                if (AX.tileIsLocal(i, 0)) {
                    auto X0 = AX(i, 0);
                    auto X0_data = X0.data();
                    int64_t nb = AX.tileMb(i);
                    for (int64_t ii = 0; ii < nb; ++ii) {
                            printf("\n i %ld ii %ld isave[1] %ld X0_data[ii] %e X0_data[ isave[1] ] %e \n", i, ii, isave[1], std::abs(X0_data[ii]), A_max);
                        if (std::abs( X0_data[ii] ) > A_max) {
                            isave[1] = i*AX.tileMb(0)+ii;
                            A_max = std::abs( X0_data[ ii ] );
                        }
                    }
                }
            }
            // MPI reduce for the max isave[1]
            MPI_Allreduce(&isave[1], &isavemax, 1, MPI_INT, MPI_MAX, AX.mpiComm());
            isave[2] = 2;
            isave[1] = isavemax;

            // main loop - iterations 2,3,..., itmax
            slate::set(zero, zero, AX);
            int64_t i_isave = std::floor(isave[1] / AX.tileMb(0) );
            //printf("\n isave[1] %ld i_isave %d iloc %ld \n", isave[1], i_isave, isave[1] - i_isave*nb);
            if (AX.tileIsLocal(i_isave, 0)) {
                auto X0 = AX(i_isave, 0);
                auto X0_data = X0.data();
                X0_data[ isave[1] - i_isave*AX.tileMb(0) ] = one;
            }
            *kase = 1;
            isave[0] = 3;
            return;
        }
        else if (isave[0] == 3) {
            // X HAS BEEN OVERWRITTEN BY A*X.
            copy(AX, AV);
            estold = *est;
            *est = slate::norm(slate::Norm::One, AV, opts);
            //printf("\n isave[0] %ld est %e estold %e \n", isave[0], *est, estold);
            for (int64_t i = 0; i < mt; ++i) {
                if (AX.tileIsLocal(i, 0)) {
                    auto X0 = AX(i, 0);
                    auto X0_data = X0.data();
                    auto Aisgn0 = Aisgn(i, 0);
                    auto Aisgn0_data = Aisgn0.data();
                    for (int64_t ii = 0; ii < AX.tileMb(i); ++ii) {
                        if (real( X0_data[ii] ) > 0.) {
                            xs = 1;
                        }
                        else {
                            xs = -1;
                        }
                        if (xs != Aisgn0_data[ii]) {
                            if (*est <= estold) {
                                lacn2_altsgn( AX );
                                *kase = 1;
                                //printf("\n isave[0] %ld go to 5 \n", isave[0]);
                                isave[0] = 5;
                                return;
                            }
                            //for (int64_t i = 0; i < n; ++i) {
                            if (real( X0_data[ii] ) >= 0) {
                                X0_data[ii] = 1.0;
                                Aisgn0_data[ii] = 1;
                            }
                            else {
                                X0_data[ii] = -1.0;
                                Aisgn0_data[ii] = -1;
                            }
                            //}
                            *kase = 2;
                                //printf("\n isave[0] %ld go to 4 \n", isave[0]);
                            isave[0] = 4;
                            return;
                        } // if (xs != isgn[i])
                    } // for ii = 0:nb
                } // if (AX.tileIsLocal(i, j))
            } // for i = 0:mt
            lacn2_altsgn( AX );
            *kase = 1;
            isave[0] = 5;
            return;
        }

        else if (isave[0] == 4) {
            //printf("\n isave[0] %ld \n", isave[0]);
            // X HAS BEEN OVERWRITTEN BY TRANSPOSE(A)*X.
            jlast = isave[1];
            isave[1] = 0;
            real_t A_max = 0.;
            for (int64_t i = 0; i < mt; ++i) {
                if (AX.tileIsLocal(i, 0)) {
                    auto X0 = AX(i, 0);
                    auto X0_data = X0.data();
                    int64_t nb = AX.tileMb(i);
                    for (int64_t ii = 0; ii < nb; ++ii) {
                        if (std::abs( X0_data[ii] ) > A_max) {
                            isave[1] = i*AX.tileMb(0)+ii;
                            A_max = std::abs( X0_data[ ii ] );
                        }
                    }
                }
            }
            // MPI reduce for the max isave[1]
            MPI_Allreduce(&isave[1], &isavemax, 1, MPI_INT, MPI_MAX, AX.mpiComm());
            isave[1] = isavemax;

            if (std::abs( X[jlast] ) != std::abs( X[isave[1]] ) && isave[2] < itmax) {
                //printf("\n jlast is diff than isave[1] %ld %ld \n", jlast, isave[1]);
                isave[2] = isave[2] + 1;
                #if 0
                for (int64_t i = 0; i < mt; ++i) {
                    if (AX.tileIsLocal(i, 0)) {
                        auto X0 = AX(i, 0);
                        auto X0_data = X0.data();
                        int64_t nb = AX.tileMb(i);
                        X0_data[i] = zero;
                        if (i*nb < isave[1] < (i+1)*nb)
                            X0_data[isave[1]] = one;
                    }
                }
                #endif
                //printf("\n X[jlast] %e X[isave[1]] %e \n", X[jlast], X[isave[1]]);
                slate::set(zero, zero, AX);
                int64_t i_isave = std::floor(isave[1] / AX.tileMb(0) );
                if (AX.tileIsLocal(i_isave, 0)) {
                    auto X0 = AX(i_isave, 0);
                    auto X0_data = X0.data();
                    X0_data[isave[1] - i_isave*AX.tileMb(0) ] = one;
                }
                *kase = 1;
                isave[0] = 3;
                return;
            }
            lacn2_altsgn( AX );
            *kase = 1;
            isave[0] = 5;
            return;
        }

        else if (isave[0] == 5) {
            //printf("\n isave[0] %ld \n", isave[0]);
            temp = slate::norm(slate::Norm::One, AV, opts);
            temp = 2.0 * temp / (3.0 * n);
            if (temp > *est) {
                copy( AX, AV);
                *est = temp;
            }
            *kase = 0;
            return;
        }
    }

}

} // namespace impl

//------------------------------------------------------------------------------
/// Distributed parallel Cholesky factorization.
///
/// Performs the Cholesky factorization of a Hermitian positive definite
/// matrix $A$.
///
/// The factorization has the form
/// \[
///     A = L L^H,
/// \]
/// if $A$ is stored lower, where $L$ is a lower triangular matrix, or
/// \[
///     A = U^H U,
/// \]
/// if $A$ is stored upper, where $U$ is an upper triangular matrix.
///
/// Complexity (in real): $\approx \frac{1}{3} n^{3}$ flops.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the n-by-n Hermitian positive definite matrix $A$.
///     On exit, if return value = 0, the factor $U$ or $L$ from the Cholesky
///     factorization $A = U^H U$ or $A = L L^H$.
///     If scalar_t is real, $A$ can be a SymmetricMatrix object.
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
/// @retval >0 for return value = $i$, the leading minor of order $i$ of $A$ is not
///         positive definite, so the factorization could not
///         be completed, and the solution has not been computed.
///
/// @ingroup posv_computational
///
template <typename scalar_t>
void lacn2(
           int64_t n,
           std::vector<scalar_t>& X,
           std::vector<scalar_t>& V,
           std::vector<int64_t>& isgn,
           blas::real_type<scalar_t>* est,
           int* kase,
           std::vector<int64_t>& isave,
           Options const& opts)
{
    using internal::TargetType;
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            impl::lacn2<Target::HostTask>( n, X, V,
                                           isgn, est,
                                           kase, isave, opts);
            break;
        case Target::HostNest:
            impl::lacn2<Target::HostNest>( n, X, V,
                                           isgn, est,
                                           kase, isave, opts);
            break;
        case Target::HostBatch:
            impl::lacn2<Target::HostBatch>( n, X, V,
                                           isgn, est,
                                           kase, isave, opts);
            break;
        case Target::Devices:
            impl::lacn2<Target::Devices>( n, X, V,
                                          isgn, est,
                                          kase, isave, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void lacn2<float>(
    int64_t n,
    std::vector<float>& X,
    std::vector<float>& V,
    std::vector<int64_t>& isgn,
    float* est,
    int* kase,
    std::vector<int64_t>& isave,
    Options const& opts);

template
void lacn2<double>(
    int64_t n,
    std::vector<double>& X,
    std::vector<double>& V,
    std::vector<int64_t>& isgn,
    double* est,
    int* kase,
    std::vector<int64_t>& isave,
    Options const& opts);

template
void lacn2< std::complex<float> >(
    int64_t n,
    std::vector< std::complex<float> >& X,
    std::vector< std::complex<float> >& V,
    std::vector<int64_t>& isgn,
    float* est,
    int* kase,
    std::vector<int64_t>& isave,
    Options const& opts);

template
void lacn2< std::complex<double> >(
    int64_t n,
    std::vector< std::complex<double> >& X,
    std::vector< std::complex<double> >& V,
    std::vector<int64_t>& isgn,
    double* est,
    int* kase,
    std::vector<int64_t>& isave,
    Options const& opts);

} // namespace slate
