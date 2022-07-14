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
/// Distributed parallel least squares solve via CholeskyQR factorization.
///
/// Solves overdetermined or underdetermined complex linear systems
/// involving an m-by-n matrix $A$, using a CholeskyQR
/// factorization of $A$.  It is assumed that $A$ has full rank.
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
/// @param[out] R
///     The triangular matrix from the CholeskyQR factorization,
///     as returned by cholqr.
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
void gels_cholqr(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& R,
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
        R = A0.emptyLike();
        R = R.slice( 0, A0_N-1, 0, A0_N-1 );
        R.insertLocalTiles();

        cholqr( A0, R, opts );

        auto R_U = TriangularMatrix( Uplo::Upper, Diag::NonUnit, R );

        if (A.op() == Op::NoTrans) {
            // Solve A X = (QR) X = B.
            // Least squares solution X = R^{-1} Y = R^{-1} (Q^H B).
            // A and Q are m-by-n, R is n-by-n, X and Y are n-by-nrhs,
            // B is m-by-nrhs, m >= n.

            Matrix<scalar_t> QH = conj_transpose( A );

            // X is first n rows of BX. Y is also n rows.
            auto X = BX.slice( 0, n-1, 0, nrhs-1 );
            auto Y = X.emptyLike();
            Y.insertLocalTiles();

            // Y = Q^H B
            gemm( one, QH, BX, zero, Y );

            // Copy back the result
            copy( Y, X );

            // X = R^{-1} Y
            trsm( Side::Left, one, R_U, X, opts );
        }
        else {
            // Solve A X = A0^H X = (QR)^H X = B.
            // Minimum norm solution X = Q Y = Q (R^{-H} B).
            // A is m-by-n, A0 and Q are n-by-m, R is m-by-m, X is n-by-nrhs,
            // B and Y are m-by-nrhs, m < n.

            // B is first m rows of BX. Y is also m rows.
            auto B = BX.slice( 0, m-1, 0, nrhs-1 );
            auto Y = B.emptyLike();
            Y.insertLocalTiles();
            copy( B, Y );

            // Y = R^{-H} B
            auto RH = conj_transpose( R_U );
            trsm( Side::Left, one, RH, Y, opts );

            // X = Q Y, with Q stored in A0.
            gemm( one, A0, Y, zero, BX );
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
void gels_cholqr<float>(
    Matrix<float>& A,
    Matrix<float>& R,
    Matrix<float>& B,
    Options const& opts);

template
void gels_cholqr<double>(
    Matrix<double>& A,
    Matrix<double>& R,
    Matrix<double>& B,
    Options const& opts);

template
void gels_cholqr< std::complex<float> >(
    Matrix< std::complex<float> >& A,
    Matrix< std::complex<float> >& R,
    Matrix< std::complex<float> >& B,
    Options const& opts);

template
void gels_cholqr< std::complex<double> >(
    Matrix< std::complex<double> >& A,
    Matrix< std::complex<double> >& R,
    Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
