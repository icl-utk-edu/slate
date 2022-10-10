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
/// @deprecated Use getrs( A, pivots, B, { Option::MethodLU, MethodLU::NoPiv } ).
///
/// Distributed parallel LU solve.
///
/// Solves a system of linear equations
/// \[
///     A X = B
/// \]
/// with a general n-by-n matrix $A$ using the LU factorization computed
/// by getrf. $A$ can be transposed or conjugate-transposed.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] A
///     The factors $L$ and $U$ from the factorization $A = P L U$
///     as computed by getrf.
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
/// @ingroup gesv_computational
///
template <typename scalar_t>
void getrs_nopiv(Matrix<scalar_t>& A,
           Matrix<scalar_t>& B,
           Options const& opts)
{
    // Constants
    const scalar_t one  = 1;

    assert(A.mt() == A.nt());
    assert(B.mt() == A.mt());

    auto L = TriangularMatrix<scalar_t>(Uplo::Lower, Diag::Unit, A);
    auto U = TriangularMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, A);

    if (A.op() == Op::NoTrans) {
        // Forward substitution, Y = L^{-1} P B.
        trsm(Side::Left, one, L, B, opts);

        // Backward substitution, X = U^{-1} Y.
        trsm(Side::Left, one, U, B, opts);
    }
    else {
        // Forward substitution, Y = U^{-T} B.
        trsm(Side::Left, one, U, B, opts);

        // Backward substitution, Xhat = L^{-T} Y.
        trsm(Side::Left, one, L, B, opts);
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void getrs_nopiv<float>(
    Matrix<float>& A,
    Matrix<float>& B,
    Options const& opts);

template
void getrs_nopiv<double>(
    Matrix<double>& A,
    Matrix<double>& B,
    Options const& opts);

template
void getrs_nopiv< std::complex<float> >(
    Matrix< std::complex<float> >& A,
    Matrix< std::complex<float> >& B,
    Options const& opts);

template
void getrs_nopiv< std::complex<double> >(
    Matrix< std::complex<double> >& A,
    Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
