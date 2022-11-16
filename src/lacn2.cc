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
// 1. documentation
// 2. V and X as vectors or matrices
// 3. scale and sum (est) with no loop
// re write code as slate structure
// specialization namespace differentiates, e.g.,
// internal::lacn2 from impl::lacn2
namespace impl {

//------------------------------------------------------------------------------
/// Distributed parallel Cholesky factorization.
/// Generic implementation for any target.
/// Panel and lookahead computed on host using Host OpenMP task.
/// @ingroup posv_specialization
///
template <Target target, typename scalar_t>
void lacn2(
           int64_t n,
           std::vector<scalar_t>& V,
           std::vector<scalar_t>& X,
           std::vector<int64_t>& isgn,
           blas::real_type<scalar_t>* est,
           int* kase,
           std::vector<int64_t>& isave,
           Options const& opts)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;

    const scalar_t one  = 1.0;
    const scalar_t zero = 0.0;
    int itmax = 5, jlast;

    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    // isave[0] = jump
    // isave[1] = j which is the index of the max element in X
    // isave[2] = iter

    scalar_t xs, altsgn;
    real_t estold, temp;

    // First iteration, kase = 0
    // Initialize X = 1./n
    if (*kase == 0) {
        for (int64_t i = 0; i < n; ++i) {
            X[i] = 1. / n;
        }
        // X to be overwritten by AX, so kase = 1.
        *kase = 1;
        // Next iteration will skip this.
        isave[0] = 1;
    }
    // Iterations 2, 3, ..., itmax.
    // kase = 1 or 2
    // X has been overwitten by AX.
    else {
        if (isave[0] == 1) {
            // quick return
            if (n == 1) {
                // Set V[0] = X[0];
                //auto X0 = X(0, 0);
                //auto V0 = X(0, 0);
                //auto X0_data = X0.data();
                //auto V0_data = V0.data();
                //V0_data[0] = X0_data[0];
                //*est = fabs( V0_data[0] );
                V[0] = X[0];
                *est = fabs( V[0] );
                // Converged, set kase back to zero
                kase = 0;
                return;
            }

            // Initial value of est
            for (int64_t i = 1; i < n; ++i) {
                *est = fabs( X[0] ) + fabs( X[i] );
            }

            // Update X to 1. or -1.:
            // X[i] = {1   if X[i] > 0
            //        {-1  if X[i] < 0
            for (int64_t i = 0; i < n; ++i) {
                if (real( X[i] ) >= 0.0) {
                    X[i] = one;
                    isgn[i] = 1;
                }
                else {
                    X[i] = -one;
                    isgn[i] = -1;
                }
            }
            // Update kase ad isave
            // X be overwritten by A^TX, so kase = 2
            *kase = 2;
            isave[0] = 2;
            return;
        }
        else if (isave[0] == 2) {
            // isave( 2 ) = idamax( n, x, 1 )
            isave[1] = 0;
            for (int64_t i = 1; i < n; ++i) {
                if (fabs( X[i] ) > fabs( X[ isave[1] ] )) {
                    isave[1] = i;
                }
            }
            isave[2] = 2;

            // main loop - iterations 2,3,..., itmax
            for (int64_t i = 0; i < n; ++i) {
                X[i] = zero;
            }
            X[isave[1]] = one;
            *kase = 1;
            isave[0] = 3;
            return;
        }
        else if (isave[0] == 3) {
            // X HAS BEEN OVERWRITTEN BY A*X.
            for (int64_t i = 0; i < n; ++i) {
                V[i] = X[i];
            }
            estold = *est;
            *est = fabs( V[0] );
            for (int64_t i = 1; i < n; ++i) {
                *est = *est + fabs( V[i] );
            }

            for (int64_t i = 0; i < n; ++i) {
                //if (X[i] >= 0) {
                //    xs = one;
                //}
                //else {
                //    xs = -one;
                //}
                //if ((int64_t) xs != isgn[i]) {
                    //if (real(est1) <= real(estold)) {
                    if (*est <= estold) {
                        altsgn = one;
                        // x( i ) = altsgn*( one+dble( i-1 ) / dble( n-1 ) )
                        for (int64_t i = 0; i < n; ++i) {
                            X[i] = altsgn * ( one + (scalar_t)( i-1 ) / (scalar_t)( n-1 ) );
                            altsgn = -altsgn;
                        }
                        *kase = 1;
                        isave[0] = 5;
                    }
                //}
            }
            return;
        }

        else if (isave[0] == 4) {
            // X HAS BEEN OVERWRITTEN BY TRANSPOSE(A)*X.
            jlast = isave[1];
            isave[1] = 0;
            for (int64_t i = 1; i < n; ++i) {
                if (fabs( X[0] ) > fabs( X[i] )) {
                    isave[1] = i;
                }
                if (X[jlast] != fabs( X[isave[1]] ) && isave[2] < itmax) {
                    isave[2] = isave[2] + 1;
                    for (int64_t i = 0; i < n; ++i) {
                        X[i] = zero;
                    }
                    X[isave[1]] = one;
                    *kase = 1;
                    isave[0] = 3;
                }
            }
            return;
        }

        else if (isave[0] == 5) {
            temp = real(X[0]);
            for (int64_t i = 1; i < n; ++i) {
                //temp = 2.0 * ( (temp + fabs( X[i] )) / ( 3.0 *  n ) );
            }
            if (temp > *est) {
                for (int64_t i = 0; i < n; ++i) {
                    V[i] = X[i];
                }
                *est = temp;
            }
            *kase = 0;
        }
    }

    // Debug::checkTilesLives(A);
    // Debug::printTilesLives(A);
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
           std::vector<scalar_t>& V,
           std::vector<scalar_t>& X,
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
            impl::lacn2<Target::HostTask>( n, V, X,
                                           isgn, est,
                                           kase, isave, opts);
            break;
        case Target::HostNest:
            impl::lacn2<Target::HostNest>( n, V, X,
                                           isgn, est,
                                           kase, isave, opts);
            break;
        case Target::HostBatch:
            impl::lacn2<Target::HostBatch>( n, V, X,
                                           isgn, est,
                                           kase, isave, opts);
            break;
        case Target::Devices:
            impl::lacn2<Target::Devices>( n, V, X,
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
    std::vector<float>& V,
    std::vector<float>& X,
    std::vector<int64_t>& isgn,
    float* est,
    int* kase,
    std::vector<int64_t>& isave,
    Options const& opts);

template
void lacn2<double>(
    int64_t n,
    std::vector<double>& V,
    std::vector<double>& X,
    std::vector<int64_t>& isgn,
    double* est,
    int* kase,
    std::vector<int64_t>& isave,
    Options const& opts);

template
void lacn2< std::complex<float> >(
    int64_t n,
    std::vector< std::complex<float> >& V,
    std::vector< std::complex<float> >& X,
    std::vector<int64_t>& isgn,
    float* est,
    int* kase,
    std::vector<int64_t>& isave,
    Options const& opts);

template
void lacn2< std::complex<double> >(
    int64_t n,
    std::vector< std::complex<double> >& V,
    std::vector< std::complex<double> >& X,
    std::vector<int64_t>& isgn,
    double* est,
    int* kase,
    std::vector<int64_t>& isave,
    Options const& opts);

} // namespace slate
