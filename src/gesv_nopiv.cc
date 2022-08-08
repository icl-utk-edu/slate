// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Distributed parallel LU factorization and solve.
///
/// Computes the solution to a system of linear equations
/// \[
///     A X = B,
/// \]
/// where $A$ is an n-by-n matrix and $X$ and $B$ are n-by-nrhs matrices.
///
/// The LU decomposition with no pivoting
/// \[
///     A = L U,
/// \]
/// where $L$ is unit lower triangular, and $U$ is
/// upper triangular.  The factored form of $A$ is then used to solve the
/// system of equations $A X = B$.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the n-by-n matrix $A$ to be factored.
///     On exit, the factors $L$ and $U$ from the factorization $A = L U$;
///     the unit diagonal elements of $L$ are not stored.
///
/// @param[in,out] B
///     On entry, the n-by-nrhs right hand side matrix $B$.
///     On exit, if return value = 0, the n-by-nrhs solution matrix $X$.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::Lookahead:
///       Number of panels to overlap with matrix updates.
///       lookahead >= 0. Default 1.
///     - Option::InnerBlocking:
///       Inner blocking to use for panel. Default 16.
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
template <typename scalar_t>
void gesv_nopiv(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& B,
    Options const& opts)
{
    slate_assert(A.mt() == A.nt());  // square
    slate_assert(B.mt() == A.mt());

    // factorization
    getrf_nopiv(A, opts);

    // solve
    getrs_nopiv(A, B, opts);

    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gesv_nopiv<float>(
    Matrix<float>& A,
    Matrix<float>& B,
    Options const& opts);

template
void gesv_nopiv<double>(
    Matrix<double>& A,
    Matrix<double>& B,
    Options const& opts);

template
void gesv_nopiv< std::complex<float> >(
    Matrix< std::complex<float> >& A,
    Matrix< std::complex<float> >& B,
    Options const& opts);

template
void gesv_nopiv< std::complex<double> >(
    Matrix< std::complex<double> >& A,
    Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
