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
void gels(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& BX,
    Options const& opts)
{
    Method method = get_option( opts, Option::MethodGels, MethodGels::Cholqr );

    if (method == MethodGels::Auto)
        method = MethodGels::select_algo( A, BX, opts );

    switch (method) {
        case MethodGels::Geqrf: {
            TriangularFactors<scalar_t> T;
            gels_qr( A, T, BX, opts );
            break;
        }
        case MethodGels::Cholqr: {
            Matrix<scalar_t> R;
            gels_cholqr( A, R, BX, opts );
            break;
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gels<float>(
    Matrix<float>& A,
    Matrix<float>& B,
    Options const& opts);

template
void gels<double>(
    Matrix<double>& A,
    Matrix<double>& B,
    Options const& opts);

template
void gels< std::complex<float> >(
    Matrix< std::complex<float> >& A,
    Matrix< std::complex<float> >& B,
    Options const& opts);

template
void gels< std::complex<double> >(
    Matrix< std::complex<double> >& A,
    Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
