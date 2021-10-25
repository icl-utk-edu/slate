// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
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
///
/// todo: document
///
/// @ingroup heev
///
template <typename scalar_t>
void heev(lapack::Job jobz,
          HermitianMatrix<scalar_t>& A,
          std::vector<blas::real_type<scalar_t>>& W,
          Matrix<scalar_t>& Z,
          Options const& opts)
{
    using real_t = blas::real_type<scalar_t>;

    int64_t n = A.n();
    bool wantz = (jobz == Job::Vec);

    // Scale matrix to allowable range, if necessary.
    // todo

    // 1. Reduce to band form.
    TriangularFactors<scalar_t> T;
    he2hb(A, T, opts);

    // Copy band.
    // Currently, gathers band matrix to rank 0.
    int64_t nb = A.tileNb(0);
    HermitianBandMatrix<scalar_t> Aband(A.uplo(), n, nb, nb, 1, 1, A.mpiComm());
    Aband.insertLocalTiles();
    Aband.he2hbGather(A);

    // Currently, hb2st and sterf are run on a single node.
    W.resize(n);
    std::vector<real_t> E(n - 1);
    if (A.mpiRank() == 0) {
        // Matrix to store Householder vectors.
        // Could pack into a lower triangular matrix, but we store each
        // parallelogram in a 2nb-by-nb tile, with nt(nt + 1)/2 tiles.
        int64_t vm = 2*nb;
        int64_t nt = A.nt();
        int64_t vn = nt*(nt + 1)/2*nb;
        Matrix<scalar_t> V(vm, vn, vm, nb, 1, 1, A.mpiComm());
        V.insertLocalTiles();

        // 2. Reduce band to real symmetric tri-diagonal.
        hb2st(Aband, V, opts);

        // Copy diagonal and super-diagonal to vectors.
        internal::copyhb2st(Aband, W, E);
    }

    // 3. Tri-diagonal eigenvalue solver.
    if (wantz) {
        // Bcast the W and E vectors
        MPI_Bcast( &W[0], n, mpi_type<blas::real_type<scalar_t>>::value, 0, A.mpiComm() );
        MPI_Bcast( &E[0], n-1, mpi_type<blas::real_type<scalar_t>>::value, 0, A.mpiComm() );
        // QR iteration
        steqr2(jobz, W, E, Z);
    }
    else {
        if (A.mpiRank() == 0) {
            // QR iteration
            sterf<real_t>(W, E, opts);
            // Bcast the vectors of the eigenvalues W
        }
        MPI_Bcast( &W[0], n, mpi_type<blas::real_type<scalar_t>>::value, 0, A.mpiComm() );
    }
    // todo: If matrix was scaled, then rescale eigenvalues appropriately.
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void heev<float>(
    lapack::Job jobz,
    HermitianMatrix<float>& A,
    std::vector<float>& W,
    Matrix<float>& Z,
    Options const& opts);

template
void heev<double>(
    lapack::Job jobz,
    HermitianMatrix<double>& A,
    std::vector<double>& W,
    Matrix<double>& Z,
    Options const& opts);

template
void heev<std::complex<float>>(
    lapack::Job jobz,
    HermitianMatrix<std::complex<float>>& A,
    std::vector<float>& W,
    Matrix<std::complex<float>>& Z,
    Options const& opts);

template
void heev<std::complex<double>>(
    lapack::Job jobz,
    HermitianMatrix<std::complex<double>>& A,
    std::vector<double>& W,
    Matrix<std::complex<double>>& Z,
    Options const& opts);

} // namespace slate
