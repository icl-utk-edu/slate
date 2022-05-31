// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/TriangularBandMatrix.hh"
#include "internal/internal.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Distributed parallel band LU solve.
///
/// Solves a system of linear equations
/// \[
///     A X = B
/// \]
/// with a general n-by-n band matrix $A$ using the LU factorization computed
/// by gbtrf. $A$ can be transposed or conjugate-transposed.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] A
///     The factors $L$ and $U$ from the factorization $A = L U$
///     as computed by gbtrf.
///
/// @param[in] pivots
///     The pivot indices that define the permutations
///     as computed by gbtrf.
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
/// @ingroup gbsv_computational
///
template <typename scalar_t>
void gbtrs(BandMatrix<scalar_t>& A, Pivots& pivots,
           Matrix<scalar_t>& B,
           Options const& opts)
{
    // Constants
    const scalar_t one  = 1;

    assert(A.mt() == A.nt());
    assert(B.mt() == A.mt());

    auto L = TriangularBandMatrix<scalar_t>(Uplo::Lower, Diag::Unit, A);
    auto U = TriangularBandMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, A);
//printf( "L kd %lld\n", L.bandwidth() );
//printf( "U kd %lld\n", U.bandwidth() );

    if (A.op() == Op::NoTrans) {
        // forward substitution, Y = L^{-1} P B
        tbsm(Side::Left, one, L, pivots, B, opts);

        // backward substitution, X = U^{-1} Y
        tbsm(Side::Left, one, U, B, opts);
    }
    else {
        // forward substitution, Y = U^{-T} B
        tbsm(Side::Left, one, U, B, opts);

        // backward substitution, X = P^T L^{-T} Y
        tbsm(Side::Left, one, L, pivots, B, opts);
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gbtrs<float>(
    BandMatrix<float>& A, Pivots& pivots,
    Matrix<float>& B,
    Options const& opts);

template
void gbtrs<double>(
    BandMatrix<double>& A, Pivots& pivots,
    Matrix<double>& B,
    Options const& opts);

template
void gbtrs< std::complex<float> >(
    BandMatrix< std::complex<float> >& A, Pivots& pivots,
    Matrix< std::complex<float> >& B,
    Options const& opts);

template
void gbtrs< std::complex<double> >(
    BandMatrix< std::complex<double> >& A, Pivots& pivots,
    Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
