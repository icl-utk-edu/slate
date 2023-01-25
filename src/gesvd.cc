// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/TriangularBandMatrix.hh"
#include "internal/internal.hh"

namespace slate {

//------------------------------------------------------------------------------
///
/// todo: document
///
/// @ingroup svd
///
/// Note A is passed by value, so we can transpose if needed
/// without affecting caller.
///
template <typename scalar_t>
void gesvd(
    Matrix<scalar_t> A,
    std::vector< blas::real_type<scalar_t> >& S,
    Options const& opts)
{
    using real_t = blas::real_type<scalar_t>;
    using std::swap;

    scalar_t zero = 0;

    int64_t m = A.m();
    int64_t n = A.n();

    bool flip = m < n; // Flip for fat matrix.
    if (flip) {
        slate_not_implemented("m < n not yet supported");
        swap(m, n);
        A = conjTranspose(A);
    }

    // Scale matrix to allowable range, if necessary.
    // todo

    // 0. If m >> n, use QR factorization to reduce to square.
    // Theoretical thresholds based on flops:
    // m >=  5/3 n for no vectors,
    // m >= 16/9 n for QR iteration with vectors,
    // m >= 10/3 n for Divide & Conquer with vectors.
    // Different in practice because stages have different flop rates.
    double threshold = 5/3.;
    bool qr_path = m > threshold*n;
    Matrix<scalar_t> Ahat;
    TriangularFactors<scalar_t> TQ;
    if (qr_path) {
        geqrf(A, TQ, opts);

        auto R_ = A.slice(0, n-1, 0, n-1);
        TriangularMatrix<scalar_t> R(Uplo::Upper, Diag::NonUnit, R_);

        Ahat = R_.emptyLike();
        Ahat.insertLocalTiles();
        set(zero, Ahat);  // todo: only lower

        TriangularMatrix<scalar_t> Ahat_tr(Uplo::Upper, Diag::NonUnit, Ahat);
        copy(R, Ahat_tr);
    }
    else {
        Ahat = A;
    }

    // 1. Reduce to band form.
    TriangularFactors<scalar_t> TU, TV;
    ge2tb(Ahat, TU, TV, opts);

    // Copy band.
    // Currently, gathers band matrix to rank 0.
    TriangularBandMatrix<scalar_t> Aband(Uplo::Upper, Diag::NonUnit,
                                         n, A.tileNb(0), A.tileNb(0),
                                         1, 1, A.mpiComm());
    Aband.insertLocalTiles();
    Aband.ge2tbGather(Ahat);

    // Currently, hb2st and sterf are run on a single node.
    slate::Matrix<scalar_t> U;
    slate::Matrix<scalar_t> VT;

    S.resize(n);
    if (A.mpiRank() == 0) {
        // 2. Reduce band to bi-diagonal.
        tb2bd(Aband, opts);

        // Copy diagonal and super-diagonal to vectors.
        std::vector<real_t> E(n - 1);
        internal::copytb2bd(Aband, S, E);

        // 3. Bi-diagonal SVD solver.
        // QR iteration
        bdsqr<scalar_t>(slate::Job::NoVec, slate::Job::NoVec, S, E, U, VT, opts);
    }

    // If matrix was scaled, then rescale singular values appropriately.
    // todo

    // todo: bcast S.

    if (qr_path) {
        // When initial QR was used.
        // U = Q*U;
    }

    if (flip) {
        // todo: swap(U, V);
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gesvd<float>(
     Matrix<float> A,
     std::vector<float>& S,
     Options const& opts);

template
void gesvd<double>(
     Matrix<double> A,
     std::vector<double>& S,
     Options const& opts);

template
void gesvd< std::complex<float> >(
     Matrix< std::complex<float> > A,
     std::vector<float>& S,
     Options const& opts);

template
void gesvd< std::complex<double> >(
     Matrix< std::complex<double> > A,
     std::vector<double>& S,
     Options const& opts);

} // namespace slate
