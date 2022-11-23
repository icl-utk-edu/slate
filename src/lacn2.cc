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
// 2. test gpus
// 3. way to know tile number
// 4. quick return inn both
// 8. test complex: c and z
// 9. check isave[2] increment
// 10. check when inf norm
// re write code as slate structure
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
    real_t altsgn = 1.;

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
/// An auxiliary routine to set the sign of A and isgn vectors
template <typename scalar_t>
void lacn2_set(Matrix<int64_t>& isgn, Matrix<scalar_t>& A,
               int* kase, int64_t* isave0)
{
    using blas::real;

    int64_t mt = A.mt();

    for (int64_t i = 0; i < mt; ++i) {
        if (A.tileIsLocal(i, 0)) {
            auto X0 = A(i, 0);
            auto X0_data = X0.data();
            auto isgn0 = isgn(i, 0);
            auto isgn0_data = isgn0.data();
            for (int64_t ii = 0; ii < A.tileMb(i); ++ii) {
                if (real( X0_data[ii] ) >= 0) {
                    X0_data[ii] = 1.0;
                    isgn0_data[ii] = 1;
                }
                else {
                    X0_data[ii] = -1.0;
                    isgn0_data[ii] = -1;
                }
                *kase = 2;
                //printf("\n isave[0] %ld go to 4 \n", *isave0);
                *isave0 = 4;
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Distributed parallel estimates of the 1-norm of a square matrix A.
/// Generic implementation for any target.
/// @ingroup norm_specialization
///
template <Target target, typename scalar_t>
void lacn2(
           Matrix<scalar_t>& X,
           Matrix<scalar_t>& V,
           Matrix<int64_t>& isgn,
           blas::real_type<scalar_t>* est,
           int* kase,
           std::vector<int64_t>& isave,
           Options const& opts)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    const auto mpi_real_type = mpi_type< blas::real_type<scalar_t> >::value;

    const scalar_t one  = 1.0;
    const scalar_t zero = 0.0;

    int64_t n = X.m();
    scalar_t xi = 1./n;

    int itmax = 5, jlast;

    // isave[0] = jump
    // isave[1] = j which is the index of the max element in X
    // isave[2] = iter

    real_t estold = 0., temp;
    int64_t xs;

    int64_t mt = X.mt();

    // First iteration, kase = 0
    // Initialize X = 1./n
    if (*kase == 0) {
        slate::set(xi, xi, X);
        // X to be overwritten by A*X, so kase = 1.
        *kase = 1;
        isave[0] = 1;
        return;
    }

    // Iterations 2, 3, ..., itmax.
    // kase = 1 or 2
    // X has been overwitten by A*X.
    else {
        if (isave[0] == 1) {
            // quick return
            if (n == 1) {
                if (X.tileIsLocal(0, 0)) {
                    //*est = fabs( V0_data[0] );
                    //V[0] = X[0];
                    //*est = fabs( V[0] );
                    auto X0 = X(0, 0);
                    auto V0 = V(0, 0);
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
            *est = slate::norm(slate::Norm::One, X, opts);

            for (int64_t i = 0; i < mt; ++i) {
                if (X.tileIsLocal(i, 0)) {
                    auto X0 = X(i, 0);
                    auto X0_data = X0.data();
                    auto isgn0 = isgn(i, 0);
                    auto isgn0_data = isgn0.data();
                    for (int64_t ii = 0; ii < X.tileMb(i); ++ii) {
                        if (real( X0_data[ii] ) >= 0.0) {
                            X0_data[ii] = one;
                            isgn0_data[ii] = 1;
                        }
                        else {
                            X0_data[ii] = -one;
                            isgn0_data[ii] = -1;
                        }
                    }
                }
            }
            // Update kase and isave
            // X be overwritten by A^*X, so kase = 2
            *kase = 2;
            isave[0] = 2;
            return;
        }
        else if (isave[0] == 2) {
            isave[1] = 0;
            real_t A_max = 0.;

            struct { real_t max; int loc; } max_loc_in[1], max_loc[1];
            max_loc_in[0].max = 0.;
            max_loc_in[0].loc = 0;

            for (int64_t i = 0; i < mt; ++i) {
                if (X.tileIsLocal(i, 0)) {
                    auto X0 = X(i, 0);
                    auto X0_data = X0.data();
                    int64_t nb = X.tileMb(i);
                    for (int64_t ii = 0; ii < nb; ++ii) {
                        if (std::abs( X0_data[ii] ) > A_max) {
                            isave[1] = i*X.tileMb(0)+ii;
                            A_max = std::abs( X0_data[ ii ] );
                            max_loc_in[0].max = A_max;
                            max_loc_in[0].loc = X.tileRank(i, 0);
                        }
                    }
                }
            }
            slate_mpi_call(
                    MPI_Allreduce(max_loc_in, max_loc, 1,
                        mpi_type< max_loc_type<real_t> >::value,
                        MPI_MAXLOC, X.mpiComm()));

            int root_rank = max_loc[0].loc;
            MPI_Bcast( &isave[1], 1, MPI_INT, root_rank, X.mpiComm() );

            isave[2] = 2;

            // main loop - iterations 2,3,..., itmax
            slate::set(zero, zero, X);
            int64_t i_isave = std::floor(isave[1] / X.tileMb(0) );

            if (X.tileIsLocal(i_isave, 0)) {
                auto X0 = X(i_isave, 0);
                auto X0_data = X0.data();
                X0_data[ isave[1] - i_isave*X.tileMb(0) ] = one;
            }
            *kase = 1;
            isave[0] = 3;
            return;
        }
        else if (isave[0] == 3) {
            // X has been overwritten by A*X
            copy(X, V);
            estold = *est;
            *est = slate::norm(slate::Norm::One, V, opts);

            for (int64_t i = 0; i < mt; ++i) {
                if (X.tileIsLocal(i, 0)) {
                    auto X0 = X(i, 0);
                    auto X0_data = X0.data();
                    auto isgn0 = isgn(i, 0);
                    auto isgn0_data = isgn0.data();
                    for (int64_t ii = 0; ii < X.tileMb(i); ++ii) {
                        if (real( X0_data[ii] ) > 0.) {
                            xs = 1;
                        }
                        else {
                            xs = -1;
                        }
                        if (xs != isgn0_data[ii]) {
                            if (*est <= estold) {
                                lacn2_altsgn( X );
                                *kase = 1;
                                isave[0] = 5;
                                return;
                            }
                            lacn2_set( isgn, X, kase, &isave[0]);
                            return;
                        } // if (xs != isgn[i])
                    } // for ii = 0:nb
                } // if (X.tileIsLocal(i, j))
            } // for i = 0:mt

            lacn2_altsgn( X );
            *kase = 1;
            isave[0] = 5;
            return;
        }

        else if (isave[0] == 4) {
            // X has been overwritten by A^T*X
            jlast = isave[1];
            isave[1] = 0;
            real_t A_max = 0.;

            struct { real_t max; int loc; } max_loc_in[1], max_loc[1];
            max_loc_in[0].max = 0.;
            max_loc_in[0].loc = 0;

            for (int64_t i = 0; i < mt; ++i) {
                if (X.tileIsLocal(i, 0)) {
                    auto X0 = X(i, 0);
                    auto X0_data = X0.data();
                    int64_t nb = X.tileMb(i);
                    for (int64_t ii = 0; ii < nb; ++ii) {
                        if (std::abs( X0_data[ii] ) > A_max) {
                            isave[1] = i*X.tileMb(0)+ii;
                            A_max = std::abs( X0_data[ ii ] );
                            max_loc_in[0].max = A_max;
                            max_loc_in[0].loc = X.tileRank(i, 0);
                        }
                    }
                }
            }

            slate_mpi_call(
                    MPI_Allreduce(max_loc_in, max_loc, 1,
                        mpi_type< max_loc_type<real_t> >::value,
                        MPI_MAXLOC, X.mpiComm()));
            int root_rank = max_loc[0].loc;
            MPI_Bcast( &isave[1], 1, MPI_INT, root_rank, X.mpiComm() );

            int64_t i_jlast = std::floor(jlast / X.tileMb(0) );
            int64_t i_isave1 = std::floor(isave[1] / X.tileMb(0) );

            real_t A_jlast = 0., A_i_isave1 = 0.;
            if (X.tileIsLocal(i_jlast, 0)) {
                auto X0 = X(i_jlast, 0);
                auto X0_data = X0.data();
                A_jlast = std::abs( X0_data[jlast - i_jlast*X.tileMb(0) ] );
            }
            else if (X.tileIsLocal(i_isave1, 0)) {
                auto X0 = X(i_isave1, 0);
                auto X0_data = X0.data();
                A_i_isave1 = std::abs( X0_data[isave[1] - i_isave1*X.tileMb(0) ] );
            }

            MPI_Bcast( &A_jlast, 1, mpi_real_type, X.tileRank(i_jlast, 0), X.mpiComm() );
            MPI_Bcast( &A_i_isave1, 1, mpi_real_type, X.tileRank(i_isave1, 0), X.mpiComm() );

            if (A_jlast != A_i_isave1 && isave[2] < itmax) {
                isave[2] = isave[2] + 1;

                slate::set(zero, zero, X);
                int64_t i_isave = std::floor(isave[1] / X.tileMb(0) );
                if (X.tileIsLocal(i_isave, 0)) {
                    auto X0 = X(i_isave, 0);
                    auto X0_data = X0.data();
                    X0_data[isave[1] - i_isave*X.tileMb(0) ] = one;
                }
                *kase = 1;
                isave[0] = 3;
                return;
            }
            lacn2_altsgn( X );
            *kase = 1;
            isave[0] = 5;
            return;
        }

        else if (isave[0] == 5) {
            temp = slate::norm(slate::Norm::One, V, opts);
            temp = 2.0 * temp / (3.0 * n);
            if (temp > *est) {
                copy( X, V);
                *est = temp;
            }
            *kase = 0;
            return;
        }
    }

}

} // namespace impl

//------------------------------------------------------------------------------
/// Distributed parallel estimates of the 1-norm of a square matrix A.
///
/// Estimates the 1-norm of a square matrix, using reverse communication for
/// evaluating matrix-vector products.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] X
///     On entry, the n-by-1 matrix $X$.
///     On an intermediate return, X should be overwritten by
///       A * X,   if kase=1
///       A**T * X,   if kase=2
///
/// @param[in,out] V
///     On entry, the n-by-1 matrix $V$.
///     On exit, V = A*W, where est = norm(A) / norm(W)
///     (W is not retured).
///
/// @param[out] isgn
///     isgn is integer matrix with size n-by-1.
///
/// @param[in,out] est
///     On entry, with kase =1 or 2 and isave[0] = 3, est should be unchaged
///     from the previous call to lacn2.
///     On exit, est is an estimate for norm(A).
///
/// @param[in,out] kase
///     On the initial call to lacn2, kase should be 0.
///     On an intermediate return, kase will be 1 or 2, indicating whether
///     X should be overwritte by A * X or A**T * X.
///     On exit, kase will again be 0.
///
/// @param[in,out] isave
///     isave is an integer vector, of size (3).
///     isave is used to save variables between calls to lacn2.
///     isave[0]: the step to do in the next iteration
///     isave[1]: index of maximum element in X
///     isave[2]: number of iterations
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
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
/// @ingroup norm_specialization
///
template <typename scalar_t>
void lacn2(
           Matrix<scalar_t>& X,
           Matrix<scalar_t>& V,
           Matrix<int64_t>& isgn,
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
            impl::lacn2<Target::HostTask>( X, V, isgn,
                                           est,
                                           kase, isave, opts);
            break;
        case Target::HostNest:
            impl::lacn2<Target::HostNest>( X, V, isgn,
                                           est,
                                           kase, isave, opts);
            break;
        case Target::HostBatch:
            impl::lacn2<Target::HostBatch>( X, V, isgn,
                                            est,
                                           kase, isave, opts);
            break;
        case Target::Devices:
            impl::lacn2<Target::Devices>( X, V, isgn,
                                          est,
                                          kase, isave, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void lacn2<float>(
    Matrix<float>& X,
    Matrix<float>& V,
    Matrix<int64_t>& isgn,
    float* est,
    int* kase,
    std::vector<int64_t>& isave,
    Options const& opts);

template
void lacn2<double>(
    Matrix<double>& X,
    Matrix<double>& V,
    Matrix<int64_t>& isgn,
    double* est,
    int* kase,
    std::vector<int64_t>& isave,
    Options const& opts);

template
void lacn2< std::complex<float> >(
    Matrix< std::complex<float> >& X,
    Matrix< std::complex<float> >& V,
    Matrix<int64_t>& isgn,
    float* est,
    int* kase,
    std::vector<int64_t>& isave,
    Options const& opts);

template
void lacn2< std::complex<double> >(
    Matrix< std::complex<double> >& X,
    Matrix< std::complex<double> >& V,
    Matrix<int64_t>& isgn,
    double* est,
    int* kase,
    std::vector<int64_t>& isave,
    Options const& opts);

} // namespace slate
