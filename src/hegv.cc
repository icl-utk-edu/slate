// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/HermitianBandMatrix.hh"
#include "internal/internal.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Distributed parallel computation of all the eigenvalues, and optionally, the
/// eigenvectors of a complex generalized Hermitian-definite eigenproblem, of
/// the form:
///
/// itype      |  Problem
/// ---------- | ----------------------
/// itype = 1  |  $A   z = \lambda B z$
/// itype = 2  |  $A B z = \lambda   z$
/// itype = 3  |  $B A z = \lambda   z$
///
/// Here A and B are assumed to be Hermitian and B is also positive definite.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] itype
///     - itype = 1: Compute $A   z = \lambda B z$;
///     - itype = 2: Compute $A B z = \lambda   z$;
///     - itype = 3: Compute $B A z = \lambda   z$.
///
/// @param[in,out] A
///     On entry, the n-by-n Hermitian matrix $A$.
///     On exit, contents are destroyed.
///
/// @param[in,out] B
///     On entry, the n-by-n Hermitian positive definite matrix $B$.
///     On exit, B is overwritten by the triangular factor U or L from
///     the Cholesky factorization $B = U^H U$ or $B = L L^H$.
///
/// @param[out] Lambda
///     The vector Lambda of length n.
///     If successful, the eigenvalues in ascending order.
///
/// @param[out] Z
///     On entry, if Z is empty, does not compute eigenvectors.
///     On exit, orthonormal eigenvectors of the matrix A.
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
///
/// @ingroup hegv
///
template <typename scalar_t>
void hegv(
    int64_t itype,
    HermitianMatrix<scalar_t>& A,
    HermitianMatrix<scalar_t>& B,
    std::vector< blas::real_type<scalar_t> >& Lambda,
    Matrix<scalar_t>& Z,
    Options const& opts)
{
    // Constants
    const scalar_t one = 1.0;

    bool wantz = (Z.mt() > 0);

    // 1. Form a Cholesky factorization of B.
    potrf( B, opts );

    // 2. Transform problem to standard eigenvalue problem.
    hegst( itype, A, B, opts );

    // 3. Solve the standard eigenvalue problem and solve.
    heev( A, Lambda, Z, opts );

    if (wantz) {
        // 4. Backtransform eigenvectors to the original problem.
        auto L = TriangularMatrix<scalar_t>( Diag::NonUnit, B );
        if (itype == 1 || itype == 2) {
            // For A x = lambda B x and A B x = lambda x,
            // backtransform eigenvectors: x = inv(L)^H y.
            auto LH = conj_transpose( L );
            trsm( Side::Left, one, LH, Z, opts );
        }
        else {
            // For B A x = lambda x,
            // backtransform eigenvectors: x = L y.
            trmm( Side::Left, one, L, Z, opts );
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void hegv<float>(
    int64_t itype,
    HermitianMatrix<float>& A,
    HermitianMatrix<float>& B,
    std::vector<float>& Lambda,
    Matrix<float>& Z,
    Options const& opts);

template
void hegv<double>(
    int64_t itype,
    HermitianMatrix<double>& A,
    HermitianMatrix<double>& B,
    std::vector<double>& Lambda,
    Matrix<double>& Z,
    Options const& opts);

template
void hegv< std::complex<float> >(
    int64_t itype,
    HermitianMatrix< std::complex<float> >& A,
    HermitianMatrix< std::complex<float> >& B,
    std::vector<float>& Lambda,
    Matrix< std::complex<float> >& Z,
    Options const& opts);

template
void hegv< std::complex<double> >(
    int64_t itype,
    HermitianMatrix< std::complex<double> >& A,
    HermitianMatrix< std::complex<double> >& B,
    std::vector<double>& Lambda,
    Matrix< std::complex<double> >& Z,
    Options const& opts);

} // namespace slate
