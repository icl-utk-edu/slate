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
/// Distributed parallel Cholesky solve.
///
/// Solves a system of linear equations
/// \[
///     A X = B
/// \]
/// with a Hermitian positive definite matrix $A$ using the Cholesky
/// factorization $A = U^H U$ or $A = L L^H$ computed by potrf.
///
/// Complexity (in real): $2 n^{2} r$ flops.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] A
///     The n-by-n triangular factor $U$ or $L$ from the Cholesky
///     factorization $A = U^H U$ or $A = L L^H$, computed by potrf.
///     If scalar_t is real, $A$ can be a SymmetricMatrix object.
///
/// @param[in,out] B
///     On entry, the n-by-nrhs right hand side matrix $B$.
///     On exit, the n-by-nrhs solution matrix $X$.
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
/// @ingroup posv_computational
///
template <typename scalar_t>
void potrs(HermitianMatrix<scalar_t>& A,
           Matrix<scalar_t>& B,
           Options const& opts)
{
    // Constants
    const scalar_t one  = 1;

    // assert(A.mt() == A.nt());
    assert(B.mt() == A.mt());

    auto A_ = A;  // local shallow copy to transpose

    // if upper, change to lower
    if (A_.uplo() == Uplo::Upper)
        A_ = conjTranspose(A_);

    auto L = TriangularMatrix<scalar_t>(Diag::NonUnit, A_);
    auto LT = conjTranspose(L);

    trsm(Side::Left, one, L, B, opts);

    trsm(Side::Left, one, LT, B, opts);
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void potrs<float>(
    HermitianMatrix<float>& A,
    Matrix<float>& B,
    Options const& opts);

template
void potrs<double>(
    HermitianMatrix<double>& A,
    Matrix<double>& B,
    Options const& opts);

template
void potrs< std::complex<float> >(
    HermitianMatrix< std::complex<float> >& A,
    Matrix< std::complex<float> >& B,
    Options const& opts);

template
void potrs< std::complex<double> >(
    HermitianMatrix< std::complex<double> >& A,
    Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
