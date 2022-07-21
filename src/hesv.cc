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
/// Distributed parallel Hermitian indefinite $LTL^T$ factorization and solve.
///
/// Computes the solution to a system of linear equations
/// \[
///    A X = B,
/// \]
/// where $A$ is an n-by-n Hermitian matrix and $X$ and $B$ are n-by-nrhs
/// matrices.
///
/// Aasen's 2-stage algorithm is used to factor $A$ as
/// \[
///     A = L T L^H,
/// \]
/// if $A$ is stored lower, or
/// \[
///     A = U^H T U,
/// \]
/// if $A$ is stored upper.
/// $U$ (or $L$) is a product of permutation and unit upper (lower)
/// triangular matrices, and $T$ is Hermitian and banded. The matrix $T$ is
/// then LU-factored with partial pivoting. The factored form of $A$
/// is then used to solve the system of equations $A X = B$.
///
/// This is the blocked version of the algorithm, calling Level 3 BLAS.
///
/// Complexity (in real): $\approx \frac{1}{3} n^{3} + 2 n^{2} r$ flops.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the n-by-n Hermitian matrix $A$.
///     On exit, if return value = 0, overwritten by the factor $U$ or $L$ from
///     the factorization $A = U^H T U$ or $A = L T L^H$.
///     If scalar_t is real, $A$ can be a SymmetricMatrix object.
///
/// @param[out] pivots
///     On exit, details of the interchanges applied to $A$, i.e.,
///     row and column k of $A$ were swapped with row and column pivots(k).
///
/// @param[out] T
///     On exit, details of the LU factorization of the band matrix.
///
/// @param[out] pivots2
///     On exit, details of the interchanges applied to $T$, i.e.,
///     row and column k of $T$ were swapped with row and column pivots2(k).
///
/// @param[out] H
///     Auxiliary matrix used during the factorization.
///     TODO: can this be made internal?
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
/// @retval >0 for return value = $i$, the band LU factorization failed on the
///         $i$-th column.
///
/// @ingroup hesv
///
template <typename scalar_t>
void hesv(HermitianMatrix<scalar_t>& A, Pivots& pivots,
               BandMatrix<scalar_t>& T, Pivots& pivots2,
                   Matrix<scalar_t>& H,
          Matrix<scalar_t>& B,
          Options const& opts)
{
    assert(B.mt() == A.mt());

    auto A_ = A;  // local shallow copy to transpose

    // if upper, change to lower
    if (A_.uplo() == Uplo::Upper)
        A_ = conjTranspose(A_);

    // factorization
    hetrf(A_, pivots, T, pivots2, H, opts);

    // solve
    hetrs(A_, pivots, T, pivots2, B, opts);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void hesv<float>(
    HermitianMatrix<float>& A, Pivots& pivots,
         BandMatrix<float>& T, Pivots& pivots2,
             Matrix<float>& H,
    Matrix<float>& B,
    Options const& opts);

template
void hesv<double>(
    HermitianMatrix<double>& A, Pivots& pivots,
         BandMatrix<double>& T, Pivots& pivots2,
             Matrix<double>& H,
    Matrix<double>& B,
    Options const& opts);

template
void hesv< std::complex<float> >(
    HermitianMatrix< std::complex<float> >& A, Pivots& pivots,
         BandMatrix< std::complex<float> >& T, Pivots& pivots2,
             Matrix< std::complex<float> >& H,
    Matrix< std::complex<float> >& B,
    Options const& opts);

template
void hesv< std::complex<double> >(
    HermitianMatrix< std::complex<double> >& A, Pivots& pivots,
         BandMatrix< std::complex<double> >& T, Pivots& pivots2,
             Matrix< std::complex<double> >& H,
    Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
