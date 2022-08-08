// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/TriangularBandMatrix.hh"
#include "internal/internal.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Distributed parallel triangular band matrix-matrix solve.
/// Solves one of the triangular matrix equations
/// \[
///     A X = \alpha B,
/// \]
/// or
/// \[
///     X A = \alpha B,
/// \]
/// where alpha is a scalar, B is an m-by-n matrix and A is a unit or non-unit,
/// upper or lower triangular band matrix. The matrix X overwrites B.
/// Pivoting from tbtrf is applied during the solve.
/// The matrices can be transposed or conjugate-transposed beforehand, e.g.,
///
///     auto AT = slate::transpose( A );
///     slate::tbsm( Side::Left, alpha, AT, B );
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///         One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] side
///         Whether A appears on the left or on the right of X:
///         - Side::Left:  solve $A X = \alpha B$
///         - Side::Right: solve $X A = \alpha B$
///
/// @param[in] alpha
///         The scalar alpha.
///
/// @param[in] A
///         - If side = left,  the m-by-m triangular matrix A;
///         - if side = right, the n-by-n triangular matrix A.
///
/// @param[in,out] B
///         On entry, the m-by-n matrix B.
///         On exit, overwritten by the result X.
///
/// @param[in] opts
///         Additional options, as map of name = value pairs. Possible options:
///         - Option::Lookahead:
///           Number of panels to overlap with matrix updates.
///           lookahead >= 0. Default 1.
///         - Option::Target:
///           Implementation to target. Possible values:
///           - HostTask:  OpenMP tasks on CPU host [default].
///           - HostNest:  nested OpenMP parallel for loop on CPU host.
///           - HostBatch: batched BLAS on CPU host.
///           - Devices:   batched BLAS on GPU device.
///
/// @ingroup tbsm
///
template <typename scalar_t>
void tbsm(Side side, scalar_t alpha,
          TriangularBandMatrix<scalar_t>& A,
                        Matrix<scalar_t>& B,
          Options const& opts)
{
    Pivots no_pivots;
    tbsm(side, alpha, A, no_pivots, B, opts);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void tbsm<float>(
    blas::Side side,
    float alpha,
    TriangularBandMatrix<float>& A,
                  Matrix<float>& B,
    Options const& opts);

template
void tbsm<double>(
    blas::Side side,
    double alpha,
    TriangularBandMatrix<double>& A,
                  Matrix<double>& B,
    Options const& opts);

template
void tbsm< std::complex<float> >(
    blas::Side side,
    std::complex<float> alpha,
    TriangularBandMatrix< std::complex<float> >& A,
                  Matrix< std::complex<float> >& B,
    Options const& opts);

template
void tbsm< std::complex<double> >(
    blas::Side side,
    std::complex<double> alpha,
    TriangularBandMatrix< std::complex<double> >& A,
                  Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
