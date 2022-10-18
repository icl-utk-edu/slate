// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "slate/Matrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/HermitianBandMatrix.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"
#include "slate_steqr2.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Computes all eigenvalues/eigenvectors of a symmetric tridiagonal matrix
/// using the Pal-Walker-Kahan variant of the QL or QR algorithm.
///
/// ATTENTION: only host computation supported for now
///
/// @ingroup svd_computational
///
template <typename scalar_t>
void steqr2(
    Job jobz,
    std::vector< blas::real_type<scalar_t> >& D,
    std::vector< blas::real_type<scalar_t> >& E,
    Matrix<scalar_t>& Z,
    Options const& opts )
{
    trace::Block trace_block("lapack::steqr2");

    using blas::max;

    int64_t nb;
    int64_t n = D.size();

    int mpi_size;
    int64_t info = 0;
    int64_t nrc, ldc;

    int myrow;
    int izero = 0;
    scalar_t zero = 0.0, one = 1.0;

    bool wantz = (jobz == Job::Vec);


    nrc = 0;
    ldc = 1;
    std::vector<scalar_t> Q(1);
    std::vector< blas::real_type<scalar_t> > work(max( 1, 2*n-2 ));

    // Compute the local number of the eigenvectors.
    // Build the matrix Z using 1-dim grid.
    slate::Matrix<scalar_t> Z1d;
    if (wantz) {
        // Find the total number of processors.
        slate_mpi_call(
            MPI_Comm_size(Z.mpiComm(), &mpi_size));
        n = Z.n();
        nb = Z.tileNb(0);
        myrow = Z.mpiRank();
        nrc = numberLocalRowOrCol(n, nb, myrow, izero, mpi_size);
        ldc = max( 1, nrc );
        Q.resize(nrc*n);
        Z1d = slate::Matrix<scalar_t>::fromScaLAPACK(
              n, n, &Q[0], nrc, nb, mpi_size, 1, Z.mpiComm());
        set(zero, one, Z1d);
    }

    // Call the eigensolver.
    slate_steqr2( jobz, n, &D[0], &E[0], &Q[0], ldc, nrc, &work[0], &info);

    // Redstribute the 1-dim eigenvector matrix into 2-dim matrix.
    if (wantz) {
        Z.redistribute(Z1d);
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void steqr2<float>(
    Job job,
    std::vector<float>& D,
    std::vector<float>& E,
    Matrix<float>& Z,
    Options const& opts);

template
void steqr2<double>(
    Job job,
    std::vector<double>& D,
    std::vector<double>& E,
    Matrix<double>& Z,
    Options const& opts);

template
void steqr2< std::complex<float> >(
    Job job,
    std::vector<float>& D,
    std::vector<float>& E,
    Matrix< std::complex<float> >& Z,
    Options const& opts);

template
void steqr2< std::complex<double> >(
    Job job,
    std::vector<double>& D,
    std::vector<double>& E,
    Matrix< std::complex<double> >& Z,
    Options const& opts);

} // namespace slate
