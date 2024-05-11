// Copyright (c) 2022-2023, University of Tennessee. All rights reserved.
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
void getrs_addmod(AddModFactors<scalar_t>& W,
                  Matrix<scalar_t>& B,
                  Options const& opts)
{
    // Constants
    const scalar_t one  = 1;
    const scalar_t zero = 0;

    Matrix<scalar_t>& A = W.A;

    assert(A.mt() == A.nt());
    assert(B.mt() == A.mt());

    if (A.op() != Op::NoTrans) {
        slate_not_implemented("Transposed matrices not yet supported");
    }


    // Forward substitution, Y = L^{-1} P B.
    trsm_addmod(Side::Left, Uplo::Lower, scalar_t(1.0), W, B, opts);

    // Woodbury correction
    if (W.num_modifications > 0) {
        // Create temporary vector for Woodbury formula
        auto A_tileMb     = W.A.tileMbFunc();
        auto A_tileRank   = W.A.tileRankFunc();
        auto A_tileDevice = W.A.tileDeviceFunc();
        Matrix<scalar_t> temp(W.num_modifications, B.n(),
                              A_tileMb, A_tileMb,
                              A_tileRank, A_tileDevice, W.A.mpiComm());
        temp.insertLocalTiles();


        gemm(one,  W.S_VT_Rinv, B,
             zero, temp,
             opts);

        getrs(W.capacitance_matrix, W.capacitance_pivots, temp, opts);

        gemm(one, W.Linv_U, temp,
             one, B,
             opts);
    }

    // Backward substitution, X = U^{-1} Y.
    trsm_addmod(Side::Left, Uplo::Upper, scalar_t(1.0), W, B, opts);

    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void getrs_addmod<float>(
    AddModFactors<float>& W,
    Matrix<float>& B,
    Options const& opts);

template
void getrs_addmod<double>(
    AddModFactors<double>& W,
    Matrix<double>& B,
    Options const& opts);

template
void getrs_addmod< std::complex<float> >(
    AddModFactors< std::complex<float> >& W,
    Matrix< std::complex<float> >& B,
    Options const& opts);

template
void getrs_addmod< std::complex<double> >(
    AddModFactors< std::complex<double> >& W,
    Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
