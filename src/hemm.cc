// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Distributed parallel Hermitian matrix-matrix multiplication.
/// Performs one of the matrix-matrix operations
/// \[
///     C = \alpha A B + \beta C
/// \]
/// or
/// \[
///     C = \alpha B A + \beta C
/// \]
/// where alpha and beta are scalars, A is a Hermitian matrix and B and
/// C are m-by-n matrices.
///
/// Complexity (in real): $2 m^{2} n$ flops.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///         One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] side
///         Whether the Hermitian matrix A appears on the left or right:
///         - Side::Left:  $C = \alpha A B + \beta C$
///         - Side::Right: $C = \alpha B A + \beta C$
///
/// @param[in] alpha
///         The scalar alpha.
///
/// @param[in] A
///         - If side = left,  the m-by-m Hermitian matrix A;
///         - if side = right, the n-by-n Hermitian matrix A.
///
/// @param[in] B
///         The m-by-n matrix B.
///
/// @param[in] beta
///         The scalar beta.
///
/// @param[in,out] C
///         On entry, the m-by-n matrix C.
///         On exit, overwritten by the result
///         $\alpha A B + \beta C$ or $\alpha B A + \beta C$.
///
/// @param[in] opts
///         Additional options, as map of name = value pairs. Possible options:
///         - Option::Lookahead:
///           Number of blocks to overlap communication and computation.
///           lookahead >= 0. Default 1.
///         - Option::MethodHemm:
///           Select the right routine to call. Possible values:
///           - Auto: let the routine decides [default]
///           - hemmA: select hemmA routine
///           - hemmC: select hemmC routine
///         - Option::Target:
///           Implementation to target. Possible values:
///           - HostTask:  OpenMP tasks on CPU host [default].
///           - HostNest:  nested OpenMP parallel for loop on CPU host.
///           - HostBatch: batched BLAS on CPU host.
///           - Devices:   batched BLAS on GPU device.
///
/// @ingroup hemm
///
template <typename scalar_t>
void hemm(Side side,
          scalar_t alpha, HermitianMatrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
          Options const& opts)
{
    Method method = get_option(
        opts, Option::MethodHemm, MethodHemm::Auto );

    if (method == MethodHemm::Auto)
        method = MethodHemm::select_algo( A, B, opts );

    switch (method) {
        case MethodHemm::HemmA:
            hemmA( side, alpha, A, B, beta, C, opts );
            break;
        case MethodHemm::HemmC:
            hemmC( side, alpha, A, B, beta, C, opts );
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void hemm<float>(
    Side side,
    float alpha, HermitianMatrix<float>& A,
                 Matrix<float>& B,
    float beta,  Matrix<float>& C,
    Options const& opts);

template
void hemm<double>(
    Side side,
    double alpha, HermitianMatrix<double>& A,
                  Matrix<double>& B,
    double beta,  Matrix<double>& C,
    Options const& opts);

template
void hemm< std::complex<float> >(
    Side side,
    std::complex<float> alpha, HermitianMatrix< std::complex<float> >& A,
                               Matrix< std::complex<float> >& B,
    std::complex<float> beta,  Matrix< std::complex<float> >& C,
    Options const& opts);

template
void hemm< std::complex<double> >(
    Side side,
    std::complex<double> alpha, HermitianMatrix< std::complex<double> >& A,
                                Matrix< std::complex<double> >& B,
    std::complex<double> beta,  Matrix< std::complex<double> >& C,
    Options const& opts);

} // namespace slate
