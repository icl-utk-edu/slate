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

// TODO docs
template <typename scalar_t>
void gesv_addmod(Matrix<scalar_t>& A, AddModFactors<scalar_t>& W,
          Matrix<scalar_t>& B,
          Options const& opts)
{
    slate_assert(A.mt() == A.nt());  // square
    slate_assert(B.mt() == A.mt());

    // factorization
    getrf_addmod(A, W, opts);

    // solve
    getrs_addmod(W, B, opts);

    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gesv_addmod<float>(
    Matrix<float>& A, AddModFactors<float>& W,
    Matrix<float>& B,
    Options const& opts);

template
void gesv_addmod<double>(
    Matrix<double>& A, AddModFactors<double>& W,
    Matrix<double>& B,
    Options const& opts);

template
void gesv_addmod< std::complex<float> >(
    Matrix< std::complex<float> >& A, AddModFactors< std::complex<float> >& W,
    Matrix< std::complex<float> >& B,
    Options const& opts);

template
void gesv_addmod< std::complex<double> >(
    Matrix< std::complex<double> >& A, AddModFactors< std::complex<double> >& W,
    Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
