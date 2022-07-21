// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Distributed parallel Cholesky factorization and solve.
///
/// Computes the solution to a system of linear equations
/// \[
///     A X = B,
/// \]
/// where $A$ is an n-by-n Hermitian positive definite matrix and $X$ and $B$ are
/// n-by-nrhs matrices. The Cholesky decomposition is used to factor $A$ as
/// \[
///     A = L L^H,
/// \]
/// if $A$ is stored lower, where $L$ is a lower triangular matrix, or
/// \[
///     A = U^H U,
/// \]
/// if $A$ is stored upper, where $U$ is an upper triangular matrix.
/// The factored form of $A$ is then used to solve the system of equations
/// $A X = B$.
///
/// Complexity (in real): $\approx \frac{1}{3} n^{3} + 2 n^{2} r$ flops.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the n-by-n Hermitian positive definite matrix $A$.
///     On exit, if return value = 0, overwritten by the factor $U$ or $L$ from
///     the Cholesky factorization $A = U^H U$ or $A = L L^H$.
///     If scalar_t is real, $A$ can be a SymmetricMatrix object.
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
/// @ingroup posv
///
template <typename scalar_t>
void posv(HermitianMatrix<scalar_t>& A,
          Matrix<scalar_t>& B,
          Options const& opts)
{
    slate_assert(B.mt() == A.mt());

    // factorization
    potrf(A, opts);

    // solve
    potrs(A, B, opts);

    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void posv<float>(
    HermitianMatrix<float>& A,
    Matrix<float>& B,
    Options const& opts);

template
void posv<double>(
    HermitianMatrix<double>& A,
    Matrix<double>& B,
    Options const& opts);

template
void posv< std::complex<float> >(
    HermitianMatrix< std::complex<float> >& A,
    Matrix< std::complex<float> >& B,
    Options const& opts);

template
void posv< std::complex<double> >(
    HermitianMatrix< std::complex<double> >& A,
    Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
