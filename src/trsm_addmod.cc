// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"

namespace slate {

// TODO docs
template <typename scalar_t>
void trsm_addmod(blas::Side side, blas::Uplo uplo,
                 scalar_t alpha, AddModFactors<scalar_t>& W,
                                       Matrix<scalar_t>& B,
          Options const& opts)
{
    Method method = get_option(
        opts, Option::MethodTrsm, MethodTrsm::Auto );

    if (method == MethodTrsm::Auto)
        method = MethodTrsm::select_algo( W.A, B, side, opts );

    switch (method) {
        case MethodTrsm::TrsmA:
            trsmA_addmod( side, uplo, alpha, W, B, opts );
            break;
        case MethodTrsm::TrsmB:
            trsmB_addmod( side, uplo, alpha, W, B, opts );
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void trsm_addmod<float>(
    blas::Side side, blas::Uplo uplo,
    float alpha, AddModFactors<float>& A,
                          Matrix<float>& B,
    Options const& opts);

template
void trsm_addmod<double>(
    blas::Side side, blas::Uplo uplo,
    double alpha, AddModFactors<double>& A,
                           Matrix<double>& B,
    Options const& opts);

template
void trsm_addmod< std::complex<float> >(
    blas::Side side, blas::Uplo uplo,
    std::complex<float> alpha, AddModFactors< std::complex<float> >& A,
                                        Matrix< std::complex<float> >& B,
    Options const& opts);

template
void trsm_addmod< std::complex<double> >(
    blas::Side side, blas::Uplo uplo,
    std::complex<double> alpha, AddModFactors< std::complex<double> >& A,
                                         Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
