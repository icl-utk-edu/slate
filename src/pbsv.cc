// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/HermitianBandMatrix.hh"
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
/// where $A$ is an n-by-n Hermitian positive definite band matrix and $X$ and $B$ are
/// n-by-nrhs matrices. The Cholesky decomposition is used to factor $A$ as
/// \[
///     A = L L^H,
/// \]
/// if $A$ is stored lower, where $L$ is a lower triangular band matrix, or
/// \[
///     A = U^H U,
/// \]
/// if $A$ is stored upper, where $U$ is an upper triangular band matrix.
/// The factored form of $A$ is then used to solve the system of equations
/// $A X = B$.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the n-by-n Hermitian positive definite band matrix $A$.
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
/// @return 0: successful exit
/// @return i > 0: the leading minor of order $i$ of $A$ is not
///         positive definite, so the factorization could not
///         be completed, and the solution has not been computed.
///
/// @ingroup pbsv
///
template <typename scalar_t>
int64_t pbsv(
    HermitianBandMatrix<scalar_t>& A,
    Matrix<scalar_t>& B,
    Options const& opts )
{
    // factorization
    int64_t info = pbtrf(A, opts);

    // solve
    if (info == 0) {
        pbtrs(A, B, opts);
    }
    return info;
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
int64_t pbsv<float>(
    HermitianBandMatrix<float>& A,
    Matrix<float>& B,
    Options const& opts);

template
int64_t pbsv<double>(
    HermitianBandMatrix<double>& A,
    Matrix<double>& B,
    Options const& opts);

template
int64_t pbsv< std::complex<float> >(
    HermitianBandMatrix< std::complex<float> >& A,
    Matrix< std::complex<float> >& B,
    Options const& opts);

template
int64_t pbsv< std::complex<double> >(
    HermitianBandMatrix< std::complex<double> >& A,
    Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
