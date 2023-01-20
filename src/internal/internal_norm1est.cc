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

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// An auxiliary routine to set the entries of a vector and alternating the
/// vector entries signs.
/// Vector is stored as a matrix
/// For each iteration in norm1est, if the new estimation is smaller than the
/// current one, then call this routine to set a new search direction.
template <typename scalar_t>
void norm1est_altsgn(Matrix<scalar_t>& A)
{
    using real_t = blas::real_type<scalar_t>;

    int64_t mt = A.mt();
    int64_t n  = A.n();

    const scalar_t one  = 1.0;
    real_t altsgn = 1.;

    for (int64_t i = 0; i < mt; ++i) {
        if (A.tileIsLocal(i, 0)) {
            A.tileGetForWriting( i, 0, LayoutConvert::ColMajor );
            auto Aij = A(i, 0);
            auto Aij_data = Aij.data();
            int64_t nb = A.tileMb(i);
            for (int64_t ii = 0; ii < nb; ++ii) {
                altsgn = pow(-1, int(i*nb+ii)) * altsgn;
                Aij_data[ii] = altsgn * ( one + scalar_t( (i*nb+ii)-1 ) / scalar_t( n-1 ) );
            }
        }
    }
}

//------------------------------------------------------------------------------
/// An auxiliary routine to replace each entry of a vector by its sign (+1 or -1),
/// for each entry a_i = {1.0,  if a_i >=0
///                      {-1.0, if a_i < 0
/// Store the sign of each entry in isgn vector as well.
template <typename scalar_t>
void norm1est_set(Matrix<int64_t>& isgn, Matrix<scalar_t>& A)
{
    using blas::real;

    int64_t mt = A.mt();

    for (int64_t i = 0; i < mt; ++i) {
        if (A.tileIsLocal(i, 0)) {
            auto Ai = A(i, 0);
            auto Ai_data = Ai.data();
            auto isgn0 = isgn(i, 0);
            auto isgn0_data = isgn0.data();
            for (int64_t ii = 0; ii < A.tileMb(i); ++ii) {
                if (real( Ai_data[ii] ) >= 0) {
                    Ai_data[ii] = 1.0;
                    isgn0_data[ii] = 1;
                }
                else {
                    Ai_data[ii] = -1.0;
                    isgn0_data[ii] = -1;
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Distributed parallel estimates of the 1-norm of a square matrix A.
/// Generic implementation for any target.
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
///       A^H * X,   if kase=2
///
/// @param[in,out] V
///     On entry, the n-by-1 matrix $V$.
///     On exit, V = A*W, where est = norm(A) / norm(W)
///     (W is not returned).
///
/// @param[out] isgn
///     isgn is integer matrix with size n-by-1.
///
/// @param[in,out] est
///     On entry, with kase = 1 or 2 and isave[0] = 3, est should be unchanged
///     from the previous call to norm1est.
///     On exit, est is an estimate for norm(A).
///
/// @param[in,out] kase
///     On the initial call to norm1est, kase should be 0.
///     On an intermediate return, kase will be 1 or 2, indicating whether
///     X should be overwritten by A * X or A^H * X.
///     On exit, kase will again be 0.
///
/// @param[in,out] isave
///     isave is an integer vector, of size (3).
///     isave is used to save variables between calls to norm1est.
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
/// Note in LAPACK, norm1est is lacn2
///
/// @ingroup cond_internal
///
template <typename scalar_t>
void norm1est(
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
    using blas::imag;
    const auto mpi_real_type = mpi_type< blas::real_type<scalar_t> >::value;
    real_t safmin = std::numeric_limits< real_t >::min();
    SLATE_UNUSED(safmin);

    const scalar_t one  = 1.0;
    const scalar_t zero = 0.0;

    int64_t n = X.m();
    scalar_t alpha = one /scalar_t( n );

    int itmax = 5, jlast;

    // isave[0] = jump
    // isave[1] = j which is the index of the max element in X
    // isave[2] = iter

    real_t estold = 0., temp;
    int64_t sign_x;

    int64_t mt = X.mt();

    // First iteration, kase = 0
    // Initialize X = 1./n
    if (*kase == 0) {
        slate::set(alpha, alpha, X);
        // X to be overwritten by A*X, so kase = 1.
        *kase = 1;
        isave[0] = 1;
        return;
    }
    else {
        // Iterations 2, 3, ..., itmax.
        // kase = 1 or 2
        // X has been overwitten by A*X.
        if (isave[0] == 1) {
            // quick return
            if (n == 1) {
                if (X.tileIsLocal(0, 0)) {
                    auto Xi = X(0, 0);
                    auto Vi = V(0, 0);
                    auto Xi_data = Xi.data();
                    auto Vi_data = Vi.data();
                    Vi_data[0] = Xi_data[0];
                    *est = std::abs( Vi_data[0] );
                }
                MPI_Bcast( est, 1, mpi_real_type, X.tileRank(0, 0), X.mpiComm() );
                // Converged, set kase back to zero
                *kase = 0;
                return;
            }

            // Initial value of est
            *est = slate::norm(slate::Norm::One, X);

            // Update the vector X
            // For real case, the vector X will be 1 or -1
            // For complex case, vector X will be X/abs(X) or 1
            // Loop over the tiles and over the elements of each tile
            for (int64_t i = 0; i < mt; ++i) {
                if (X.tileIsLocal(i, 0)) {
                    auto Xi = X(i, 0);
                    auto Xi_data = Xi.data();
                    auto isgn0 = isgn(i, 0);
                    auto isgn0_data = isgn0.data();
                    for (int64_t ii = 0; ii < X.tileMb(i); ++ii) {
                        // for complex number
                        if constexpr (blas::is_complex<scalar_t>::value) {
                            real_t absx1 = std::abs( Xi_data[ii] );
                            if (absx1 > safmin) {
                                scalar_t Xi_ii = blas::MakeScalarTraits<scalar_t>::make(
                                        real( Xi_data[ii] ) / absx1,
                                        imag( Xi_data[ii] ) / absx1 );
                                Xi_data[ii] = Xi_ii;
                            }
                            else {
                                Xi_data[ii] = one;
                            }
                        }
                        else {
                            if (Xi_data[ii] >= zero) {
                                Xi_data[ii] = one;
                                isgn0_data[ii] = 1;
                            }
                            else {
                                Xi_data[ii] = -one;
                                isgn0_data[ii] = -1;
                            }
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
            // X has been overwritten by A^T*X
            // Find the index of the largest entry of X,
            // The index of the largest element will be saved in isave[1]
            isave[1] = 0;
            real_t X_max = 0.;

            struct { real_t max; int loc; } max_loc_in[1], max_loc[1];
            max_loc_in[0].max = 0.;
            max_loc_in[0].loc = 0;

            for (int64_t i = 0; i < mt; ++i) {
                if (X.tileIsLocal(i, 0)) {
                    auto Xi = X(i, 0);
                    auto Xi_data = Xi.data();
                    int64_t nb = X.tileMb(i);
                    for (int64_t ii = 0; ii < nb; ++ii) {
                        if (std::abs( Xi_data[ii] ) > X_max) {
                            isave[1] = i*X.tileMb(0)+ii;
                            X_max = std::abs( Xi_data[ ii ] );
                            max_loc_in[0].max = X_max;
                            max_loc_in[0].loc = X.tileRank(i, 0);
                        }
                    }
                }
            }
            slate_mpi_call(
                    MPI_Allreduce(max_loc_in, max_loc, 1,
                        mpi_type< max_loc_type<real_t> >::value,
                        MPI_MAXLOC, X.mpiComm()) );

            int root_rank = max_loc[0].loc;
            MPI_Bcast( &isave[1], 1, MPI_INT64_T, root_rank, X.mpiComm() );

            isave[2] = 2;

            // main loop - iterations 2,3,..., itmax
            slate::set(zero, zero, X);
            int64_t i_isave = std::floor(isave[1] / X.tileMb(0) );

            if (X.tileIsLocal(i_isave, 0)) {
                auto Xi = X(i_isave, 0);
                auto Xi_data = Xi.data();
                Xi_data[ isave[1] - i_isave*X.tileMb(0) ] = one;
            }
            *kase = 1;
            isave[0] = 3;
            return;
        }
        else if (isave[0] == 3) {
            // X has been overwritten by A*X
            copy(X, V);
            estold = *est;
            *est = slate::norm(slate::Norm::One, V);

            if (*est <= estold) {
                norm1est_altsgn( X );
                *kase = 1;
                isave[0] = 5;
                return;
            }
            if constexpr (blas::is_complex<scalar_t>::value) {
                for (int64_t i = 0; i < mt; ++i) {
                    if (X.tileIsLocal(i, 0)) {
                        auto Xi = X(i, 0);
                        auto Xi_data = Xi.data();
                        for (int64_t ii = 0; ii < X.tileMb(i); ++ii) {
                            real_t absx1 = std::abs( Xi_data[ii] );
                            if (absx1 > safmin) {
                                scalar_t Xi_ii = blas::MakeScalarTraits<scalar_t>::make( real( Xi_data[ii] ) / absx1,
                                        imag( Xi_data[ii] ) / absx1);
                                Xi_data[ii] = Xi_ii;
                            }
                            else {
                                Xi_data[ii] = one;
                            }
                        }
                    }
                }
                *kase = 2;
                isave[0] = 4;
                return;
            }
            else {
                int break_outer = -1;
                for (int64_t i = 0; i < mt; ++i) {
                    if (X.tileIsLocal(i, 0)) {
                        auto Xi = X(i, 0);
                        auto Xi_data = Xi.data();
                        auto isgn0 = isgn(i, 0);
                        auto isgn0_data = isgn0.data();
                        for (int64_t ii = 0; ii < X.tileMb(i); ++ii) {
                            if (real( Xi_data[ii] ) > 0.) {
                                sign_x = 1;
                            }
                            else {
                                sign_x = -1;
                            }
                            if (sign_x != isgn0_data[ii]) {
                                if (*est <= estold) {
                                    *kase = 1;
                                    isave[0] = 5;
                                    break_outer = 1;
                                    break;
                                }
                                *kase = 2;
                                isave[0] = 4;
                                break_outer = 2;
                                break;
                            } // if (sign_x != isgn[i])
                        } // for ii = 0:nb
                        if (break_outer == 1 || break_outer == 2) break;
                    } // if (X.tileIsLocal(i, j))
                } // for i = 0:mt
                if (break_outer == 1) {
                    norm1est_altsgn( X );
                    return;
                }
                else if (break_outer == 2) {
                    norm1est_set( isgn, X );
                    return;
                }
            } // if constexpr

            norm1est_altsgn( X );
            *kase = 1;
            isave[0] = 5;
            return;
        }
        else if (isave[0] == 4) {
            // X has been overwritten by A^T*X
            jlast = isave[1];
            isave[1] = 0;
            real_t X_max = 0.;

            struct { real_t max; int loc; } max_loc_in[1], max_loc[1];
            max_loc_in[0].max = 0.;
            max_loc_in[0].loc = 0;

            // Find the max value/index on each rank
            for (int64_t i = 0; i < mt; ++i) {
                if (X.tileIsLocal(i, 0)) {
                    auto Xi = X(i, 0);
                    auto Xi_data = Xi.data();
                    int64_t nb = X.tileMb(i);
                    for (int64_t ii = 0; ii < nb; ++ii) {
                        if (std::abs( Xi_data[ii] ) > X_max) {
                            isave[1] = i*X.tileMb(0)+ii;
                            X_max = std::abs( Xi_data[ ii ] );
                            max_loc_in[0].max = X_max;
                            max_loc_in[0].loc = X.tileRank(i, 0);
                        }
                    }
                }
            }

            // Find the max value/index among all mpi ranks
            slate_mpi_call(
                    MPI_Allreduce(max_loc_in, max_loc, 1,
                        mpi_type< max_loc_type<real_t> >::value,
                        MPI_MAXLOC, X.mpiComm()) );

            // Bcast the index of the max value to all mpi ranks
            int root_rank = max_loc[0].loc;
            MPI_Bcast( &isave[1], 1, MPI_INT64_T, root_rank, X.mpiComm() );

            // Find the tile index which has the jlast entry
            int64_t i_jlast = std::floor(jlast / X.tileMb(0) );
            // Find the tile which has the isave[1] entry (max value)
            int64_t i_max = std::floor(isave[1] / X.tileMb(0) );

            // Find the value at i_jlast
            real_t X_jlast = 0., X_i_max = 0.;
            if (X.tileIsLocal(i_jlast, 0)) {
                auto Xi = X(i_jlast, 0);
                auto Xi_data = Xi.data();
                X_jlast = std::abs( Xi_data[jlast - i_jlast*X.tileMb(0) ] );
            }
            // Find the value at i_max
            else if (X.tileIsLocal(i_max, 0)) {
                auto Xi = X(i_max, 0);
                auto Xi_data = Xi.data();
                X_i_max = std::abs( Xi_data[isave[1] - i_max*X.tileMb(0) ] );
            }

            // Bcast previous max value (X_jlast) and the current max
            // value (X_i_max)
            MPI_Bcast( &X_jlast, 1, mpi_real_type, X.tileRank(i_jlast, 0), X.mpiComm() );
            MPI_Bcast( &X_i_max, 1, mpi_real_type, X.tileRank(i_max, 0), X.mpiComm() );

            // If the currennt max value is not equal to the previous and
            // still the number of iterations is less than itmax,
            // then do one more iteration
            if (X_jlast != X_i_max && isave[2] < itmax) {
                isave[2] = isave[2] + 1;
                slate::set(zero, zero, X);

                // Find the tile index which has the max entry,
                // then set X to a canonical form:
                // X_i = 0, all i, except X_i_max = 1
                int64_t i_isave = std::floor(isave[1] / X.tileMb(0) );
                if (X.tileIsLocal(i_isave, 0)) {
                    auto Xi = X(i_isave, 0);
                    auto Xi_data = Xi.data();
                    Xi_data[isave[1] - i_isave*X.tileMb(0) ] = one;
                }
                *kase = 1;
                isave[0] = 3;
                return;
            }
            norm1est_altsgn( X );
            *kase = 1;
            isave[0] = 5;
            return;
        }
        else if (isave[0] == 5) {
            // X has been overwritten by A*X.
            temp = slate::norm(slate::Norm::One, V);
            temp = real_t(2.0) * temp / ( real_t(3.0) * real_t(n) );
            if (temp > *est) {
                copy( X, V);
                *est = temp;
            }
            // Set kase to zero, norm1est converged
            *kase = 0;
            return;
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void norm1est<float>(
    Matrix<float>& X,
    Matrix<float>& V,
    Matrix<int64_t>& isgn,
    float* est,
    int* kase,
    std::vector<int64_t>& isave,
    Options const& opts);

template
void norm1est<double>(
    Matrix<double>& X,
    Matrix<double>& V,
    Matrix<int64_t>& isgn,
    double* est,
    int* kase,
    std::vector<int64_t>& isave,
    Options const& opts);

template
void norm1est< std::complex<float> >(
    Matrix< std::complex<float> >& X,
    Matrix< std::complex<float> >& V,
    Matrix<int64_t>& isgn,
    float* est,
    int* kase,
    std::vector<int64_t>& isave,
    Options const& opts);

template
void norm1est< std::complex<double> >(
    Matrix< std::complex<double> >& X,
    Matrix< std::complex<double> >& V,
    Matrix<int64_t>& isgn,
    double* est,
    int* kase,
    std::vector<int64_t>& isave,
    Options const& opts);

} // namespace internal
} // namespace slate
