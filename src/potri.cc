// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Distributed parallel Cholesky inversion.
///
/// Computes the inverse of a complex Hermitian positive definite matrix $A$
/// using the Cholesky factorization $A = U^H U$ or $A = L L^H$  computed by
/// `potrf`.
///
/// Complexity (in real): $\approx \frac{2}{3} n^{3}$ flops.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the triangular factor $U$ or $L$ from the Cholesky
///     factorization $A = U^H U$ or $A = L L^H$, as computed by `potrf`.
///     On exit, the upper or lower triangle of the (Hermitian) inverse of $A$,
///     overwriting the input factor $U$ or $L$.
///     If scalar_t is real, $A$ can be a SymmetricMatrix object.
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
/// @retval >0 for return value = $i$, $A(i,i)$ is exactly zero. The triangular
///         matrix is singular and its inverse can not be computed.
///
/// @ingroup posv_computational
///
template <typename scalar_t>
void potri(HermitianMatrix<scalar_t>& A,
           Options const& opts)
{
    auto T = TriangularMatrix<scalar_t>(lapack::Diag::NonUnit, A);

    // triangular inversion
    trtri(T, opts);

    // triangular multiply
    trtrm(T, opts);

    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void potri<float>(
    HermitianMatrix<float>& A,
    Options const& opts);

template
void potri<double>(
    HermitianMatrix<double>& A,
    Options const& opts);

template
void potri< std::complex<float> >(
    HermitianMatrix< std::complex<float> >& A,
    Options const& opts);

template
void potri< std::complex<double> >(
    HermitianMatrix< std::complex<double> >& A,
    Options const& opts);

} // namespace slate
