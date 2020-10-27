// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "aux/Debug.hh"
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
/// itype = 1  |  $A   x = \lambda B x$
/// itype = 2  |  $A B x = \lambda   x$
/// itype = 3  |  $B A x = \lambda   x$
///
/// Here A and B are assumed to be Hermitian and B is also positive definite.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] itype
///     - itype = 1: Compute $A   x = \lambda B x$;
///     - itype = 2: Compute $A B x = \lambda   x$;
///     - itype = 3: Compute $B A x = \lambda   x$.
///
/// @param[in] jobz
///     - jobz = lapack::Job::NoVec: Compute eigenvalues only;
///     - jobz = lapack::Job::Vec:   Compute eigenvalues and eigenvectors.
///
/// @param[in,out] A
///     On entry, the n-by-n Hermitian matrix A.
///     On exit, if jobz = Vec, then if successful, A contains the
///     orthonormal eigenvectors of the matrix A.
///     If jobz = NoVec, then on exit the lower triangle (if uplo=Lower)
///     or the upper triangle (if uplo=Upper) of A, including the
///     diagonal, is destroyed.
///
/// @param[in, out] B
///     On entry, the n-by-n Hermitian positive definite matrix $A$.
///     On exit, if jobz = Vec, then if successful, the part of B containing the
///     matrix is overwritten by the triangular factor U or L from the Cholesky
///     factorization $B = U^H U$ or $B = L L^H$.
///
/// @param[out] W
///     The vector W of length n.
///     If successful, the eigenvalues in ascending order.
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
void hegv(int64_t itype,
          lapack::Job jobz,
          HermitianMatrix<scalar_t>& A,
          HermitianMatrix<scalar_t>& B,
          std::vector<blas::real_type<scalar_t>>& W,
          Matrix<scalar_t>& V,
          Options const& opts)
{
    // 1. Form a Cholesky factorization of B.
    potrf(B, opts);

    // 2. Transform problem to standard eigenvalue problem.
    hegst(itype, A, B, opts);

    // 3. Solve the standard eigenvalue problem and solve.
    heev(jobz, A, W, V, opts);

    if (jobz == lapack::Job::Vec) {
        // 4. Backtransform eigenvectors to the original problem.
        auto L = TriangularMatrix<scalar_t>(Diag::NonUnit, B);
        scalar_t one = 1.0;
        if (itype == 1 || itype == 2) {
            trsm(Side::Left, one, L, V, opts);
        }
        else {
            trmm(Side::Left, one, L, V, opts);
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void hegv<float>(
    int64_t itype,
    lapack::Job jobz,
    HermitianMatrix<float>& A,
    HermitianMatrix<float>& B,
    std::vector<float>& W,
    Matrix<float>& V,
    Options const& opts);

template
void hegv<double>(
    int64_t itype,
    lapack::Job jobz,
    HermitianMatrix<double>& A,
    HermitianMatrix<double>& B,
    std::vector<double>& W,
    Matrix<double>& V,
    Options const& opts);

template
void hegv<std::complex<float>>(
    int64_t itype,
    lapack::Job jobz,
    HermitianMatrix<std::complex<float>>& A,
    HermitianMatrix<std::complex<float>>& B,
    std::vector<float>& W,
    Matrix<std::complex<float>>& V,
    Options const& opts);

template
void hegv<std::complex<double>>(
    int64_t itype,
    lapack::Job jobz,
    HermitianMatrix<std::complex<double>>& A,
    HermitianMatrix<std::complex<double>>& B,
    std::vector<double>& W,
    Matrix<std::complex<double>>& V,
    Options const& opts);

} // namespace slate
