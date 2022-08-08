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
/// Distributed parallel least squares solve via QR or LQ factorization.
///
/// Solves overdetermined or underdetermined complex linear systems
/// involving an m-by-n matrix $A$, using a QR
/// or LQ factorization of $A$.  It is assumed that $A$ has full rank.
/// $X$ is n-by-nrhs, $B$ is m-by-nrhs, $BX$ is max(m, n)-by-nrhs.
///
/// If m >= n, solves over-determined $A X = B$
/// with least squares solution $X$ that minimizes $\norm{ A X - B }_2$.
/// $BX$ is m-by-nrhs.
/// On input, $B$ is all m rows of $BX$.
/// On output, $X$ is first n rows of $BX$.
/// Currently in this case $A$ must be not transposed.
///
/// If m < n, solves under-determined $A X = B$
/// with minimum norm solution $X$ that minimizes $\norm{ X }_2$.
/// $BX$ is n-by-nrhs.
/// On input, $B$ is first m rows of $BX$.
/// On output, $X$ is all n rows of $BX$.
/// Currently in this case $A$ must be transposed (only if real) or
/// conjugate-transposed.
///
/// Several right hand side vectors $b$ and solution vectors $x$ can be
/// handled in a single call; they are stored as the columns of the
/// m-by-nrhs right hand side matrix $B$ and the n-by-nrhs solution
/// matrix $X$.
///
/// Note these (m, n) differ from (M, N) in (Sca)LAPACK, where the original
/// $A$ is M-by-N, _before_ appyling any transpose,
/// while here $A$ is m-by-n, _after_ applying any transpose,.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the m-by-n matrix $A$.
///     On exit:
///     <br>
///     If (m >= n and $A$ is not transposed) or
///        (m <  n and $A$ is (conjugate) transposed),
///     $A$ is overwritten by details of its QR factorization
///     as returned by geqrf.
///     <br>
///     If (m >= n and $A$ is (conjugate) transposed) or
///        (m <  n and $A$ is not transposed),
///     $A$ is overwritten by details of its LQ factorization
///     as returned by gelqf (todo: not currently supported).
///
/// @param[out] T
///     The triangular matrices of the block reflectors from the
///     QR or LQ factorization, as returned by geqrf or gelqf.
///
/// @param[in,out] BX
///     Matrix of size max(m,n)-by-nrhs.
///     On entry, the m-by-nrhs right hand side matrix $B$.
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
/// @retval >0 for return value = $i$, in the computed triangular factor,
///         $R(i,i)$ is exactly zero, so that $A$ does not have full rank;
///         the least squares solution could not be computed.
///
/// @ingroup gels
///
template <typename scalar_t>
void gels_qr(
    Matrix<scalar_t>& A,
    TriangularFactors<scalar_t>& T,
    Matrix<scalar_t>& BX,
    Options const& opts)
{
    // m, n of op(A) as in docs above.
    int64_t m = A.m();
    int64_t n = A.n();
    int64_t nrhs = BX.n();

    const scalar_t one  = 1.0;
    const scalar_t zero = 0.0;

    // Get original, un-transposed matrix A0.
    slate::Matrix<scalar_t> A0;
    if (A.op() == Op::NoTrans)
        A0 = A;
    else if (A.op() == Op::ConjTrans)
        A0 = conj_transpose( A );
    else if (A.op() == Op::Trans && A.is_real)
        A0 = transpose( A );
    else
        slate_error( "Unsupported op(A)" );

    int64_t A0_M = (A.op() == Op::NoTrans ? m : n);
    int64_t A0_N = (A.op() == Op::NoTrans ? n : m);
    if (A0_M >= A0_N) {
        assert( A0.m() >= A0.n() );

        // A0 itself is tall: QR factorization
        geqrf( A0, T, opts );

        int64_t min_mn = std::min( m, n );
        auto R_ = A0.slice( 0, min_mn-1, 0, min_mn-1 );
        auto R = TriangularMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, R_);

        if (A.op() == Op::NoTrans) {
            // Solve A0 X = (QR) X = B.
            // Least squares solution X = R^{-1} Y = R^{-1} (Q^H B).

            // Y = Q^H B
            // B is all m rows of BX.
            unmqr( Side::Left, Op::ConjTrans, A0, T, BX, opts );

            // X is first n rows of BX.
            auto X = BX.slice( 0, n-1, 0, nrhs-1 );

            // X = R^{-1} Y
            trsm( Side::Left, one, R, X, opts );
        }
        else {
            // Solve A X = A0^H X = (QR)^H X = B.
            // Minimum norm solution X = Q Y = Q (R^{-H} B).

            // B is first m rows of BX.
            auto B = BX.slice( 0, m-1, 0, nrhs-1 );

            // Y = R^{-H} B
            auto RH = conj_transpose( R );
            trsm( Side::Left, one, RH, B, opts );

            // X is all n rows of BX.
            // Zero out rows m:n-1 of BX.
            if (m < n) {
                auto Z = BX.slice( m, n-1, 0, nrhs-1 );
                set( zero, Z );
            }

            // X = Q Y
            unmqr( Side::Left, Op::NoTrans, A0, T, BX, opts );
        }
    }
    else {
        // todo: LQ factorization
        slate_not_implemented( "least squares using LQ" );
    }
    // todo: return value for errors?
    // R or L is singular => A is not full rank
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gels_qr<float>(
    Matrix<float>& A,
    TriangularFactors<float>& T,
    Matrix<float>& B,
    Options const& opts);

template
void gels_qr<double>(
    Matrix<double>& A,
    TriangularFactors<double>& T,
    Matrix<double>& B,
    Options const& opts);

template
void gels_qr< std::complex<float> >(
    Matrix< std::complex<float> >& A,
    TriangularFactors< std::complex<float> >& T,
    Matrix< std::complex<float> >& B,
    Options const& opts);

template
void gels_qr< std::complex<double> >(
    Matrix< std::complex<double> >& A,
    TriangularFactors< std::complex<double> >& T,
    Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
