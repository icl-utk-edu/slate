// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
// #include "auxiliary/Debug.hh"
#include "slate/Tile_blas.hh"
#include "slate/TriangularBandMatrix.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"

#include <atomic>

namespace slate {

//------------------------------------------------------------------------------
/// Computes the singular values and, optionally, the right and/or
/// left singular vectors from the singular value decomposition (SVD) of
/// a real (upper or lower) bidiagonal matrix.
/// Generic implementation for any target.
///
/// ATTENTION: only singular values computed for now, no singular vectors.
/// Only host computation supported for now.
///
/// @ingroup svd_computational
///
template <typename scalar_t>
void bdsqr(
    lapack::Job jobu, lapack::Job jobvt,
    std::vector< blas::real_type<scalar_t> >& D,
    std::vector< blas::real_type<scalar_t> >& E,
    Matrix<scalar_t>& U,
    Matrix<scalar_t>& VT,
    Options const& opts )
{
    trace::Block trace_block("slate::bdsqr");

    using blas::max;

    int64_t m, n, nb, mb;
    int64_t min_mn = D.size();
    //assert(m >= n);

    int mpi_size;

    scalar_t zero = 0.0, one = 1.0;

    int myrow, mycol;
    int izero = 0;

    int64_t nru  = 0;
    int64_t ncvt = 0;

    int64_t ldu = 1;
    int64_t ldvt = 1;

    std::vector<scalar_t> u1d(1);
    std::vector<scalar_t> vt1d(1);
    scalar_t dummy[1];

    bool wantu  = (jobu  == Job::Vec ||
                   jobu  == Job::AllVec ||
                   jobu  == Job::SomeVec );
    bool wantvt = (jobvt == Job::Vec ||
                   jobvt == Job::AllVec ||
                   jobvt == Job::SomeVec );

    // Compute the local number of the eigenvectors.
    // Build the 1-dim distributed U and VT
    slate::Matrix<scalar_t> U1d;
    slate::Matrix<scalar_t> VT1d;
    if (wantu) {
        m = U.m();
        mb = U.tileMb(0);
        nb = U.tileNb(0);
        // Find the total number of processors.
        slate_mpi_call(
            MPI_Comm_size(U.mpiComm(), &mpi_size));
        myrow = U.mpiRank();
        nru  = numberLocalRowOrCol(m, mb, myrow, izero, mpi_size);
        ldu = max( 1, nru );
        u1d.resize(ldu*min_mn);
        U1d = slate::Matrix<scalar_t>::fromScaLAPACK(
              m, min_mn, &u1d[0], ldu, nb, mpi_size, 1, U.mpiComm());
        set(zero, one, U1d);
    }
    if (wantvt) {
        n = VT.n();
        nb = VT.tileNb(0);
        // Find the total number of processors.
        slate_mpi_call(
            MPI_Comm_size(VT.mpiComm(), &mpi_size));
        mycol = VT.mpiRank();
        ncvt = numberLocalRowOrCol(n, nb, mycol, izero, mpi_size);
        ldvt = max( 1, min_mn );
        vt1d.resize(ldvt*ncvt);
        VT1d = slate::Matrix<scalar_t>::fromScaLAPACK(
               min_mn, n, &vt1d[0], ldvt, nb, 1, mpi_size, VT.mpiComm());
        set(zero, one, VT1d);
    }

    // Call the SVD
    lapack::bdsqr(Uplo::Upper, min_mn, ncvt, nru, 0,
                  &D[0], &E[0], &vt1d[0], min_mn, &u1d[0], ldu, dummy, 1);

    // Redistribute the 1-dim distributed U and VT into 2-dim matrices
    if (wantu) {
        U.redistribute(U1d);
    }
    if (wantvt) {
        VT.redistribute(VT1d);
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void bdsqr<float>(
    lapack::Job jobu,
    lapack::Job jobvt,
    std::vector<float>& D,
    std::vector<float>& E,
    Matrix<float>& U,
    Matrix<float>& VT,
    Options const& opts);

template
void bdsqr<double>(
    lapack::Job jobu,
    lapack::Job jobvt,
    std::vector<double>& D,
    std::vector<double>& E,
    Matrix<double>& U,
    Matrix<double>& VT,
    Options const& opts);

template
void bdsqr< std::complex<float> >(
    lapack::Job jobu,
    lapack::Job jobvt,
    std::vector<float>& D,
    std::vector<float>& E,
    Matrix< std::complex<float> >& U,
    Matrix< std::complex<float> >& VT,
    Options const& opts);

template
void bdsqr< std::complex<double> >(
    lapack::Job jobu,
    lapack::Job jobvt,
    std::vector<double>& D,
    std::vector<double>& E,
    Matrix< std::complex<double> >& U,
    Matrix< std::complex<double> >& VT,
    Options const& opts);

} // namespace slate
