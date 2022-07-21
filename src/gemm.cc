// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Distributed parallel general matrix-matrix multiplication.
/// Performs the matrix-matrix operation
/// \[
///     C = \alpha A B + \beta C,
/// \]
/// where alpha and beta are scalars, and $A$, $B$, and $C$ are matrices, with
/// $A$ an m-by-k matrix, $B$ a k-by-n matrix, and $C$ an m-by-n matrix.
/// The matrices can be transposed or conjugate-transposed beforehand, e.g.,
///
///     auto AT = slate::transpose( A );
///     auto BT = slate::conjTranspose( B );
///     slate::gemm( alpha, AT, BT, beta, C );
///
/// Complexity (in real): $2 m n k$ flops.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///         One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] alpha
///         The scalar alpha.
///
/// @param[in] A
///         The m-by-k matrix A.
///
/// @param[in] B
///         The k-by-n matrix B.
///
/// @param[in] beta
///         The scalar beta.
///
/// @param[in,out] C
///         On entry, the m-by-n matrix C.
///         On exit, overwritten by the result $\alpha A B + \beta C$.
///
/// @param[in] opts
///         Additional options, as map of name = value pairs. Possible options:
///         - Option::Lookahead:
///           Number of blocks to overlap communication and computation.
///           lookahead >= 0. Default 1.
///         - Option::MethodGemm:
///           Select the right routine to call. Possible values:
///           - Auto: let the routine decides [default]
///           - gemmA: select gemmA routine
///           - gemmC: select gemmC routine
///         - Option::Target:
///           Implementation to target. Possible values:
///           - HostTask:  OpenMP tasks on CPU host [default].
///           - HostNest:  nested OpenMP parallel for loop on CPU host.
///           - HostBatch: batched BLAS on CPU host.
///           - Devices:   batched BLAS on GPU device.
///
/// @ingroup gemm
///
template <typename scalar_t>
void gemm(scalar_t alpha, Matrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
          Options const& opts)
{
    Method method = get_option(
        opts, Option::MethodGemm, MethodGemm::Auto );

    // Select_algo can also change target.
    Options tuned_opts = opts;

    if (method == MethodGemm::Auto)
        method = MethodGemm::select_algo( A, B, tuned_opts );

    switch (method) {
        case MethodGemm::GemmA:
            gemmA( alpha, A, B, beta, C, tuned_opts );
            break;
        case MethodGemm::GemmC:
            gemmC( alpha, A, B, beta, C, tuned_opts );
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gemm<float>(
    float alpha, Matrix<float>& A,
                 Matrix<float>& B,
    float beta,  Matrix<float>& C,
    Options const& opts);

template
void gemm<double>(
    double alpha, Matrix<double>& A,
                  Matrix<double>& B,
    double beta,  Matrix<double>& C,
    Options const& opts);

template
void gemm< std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >& A,
                               Matrix< std::complex<float> >& B,
    std::complex<float> beta,  Matrix< std::complex<float> >& C,
    Options const& opts);

template
void gemm< std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >& A,
                                Matrix< std::complex<double> >& B,
    std::complex<double> beta,  Matrix< std::complex<double> >& C,
    Options const& opts);

} // namespace slate
