// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "slate/Matrix.hh"
#include "slate/HermitianBandMatrix.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Computes all eigenvalues and eigenvectors of a symmetric tridiagonal
/// matrix using the implicit QL or QR iteration algorithm, as
/// implemented in LAPACK's `steqr`.
///
/// For computing eigenvalues only, uses the Pal-Walker-Kahan variant of
/// the QL or QR iteration algorithm, implemented in LAPACK's `sterf`.
///
/// ATTENTION: only host computation supported for now
///
/// @ingroup heev_computational
///
template <typename scalar_t>
void steqr(
    Job jobz,
    std::vector< blas::real_type<scalar_t> >& D,
    std::vector< blas::real_type<scalar_t> >& E,
    Matrix<scalar_t>& Z,
    Options const& opts )
{
    trace::Block trace_block( "lapack::steqr" );

    using blas::max;

    // Constants
    const scalar_t zero = 0.0, one = 1.0;

    int64_t nb;
    int64_t n = D.size();

    bool wantz = (jobz == Job::Vec);

    int64_t nrows = 0;
    int64_t ldz = 1;
    std::vector<scalar_t> Z1d_data( 1 );
    std::vector< blas::real_type<scalar_t> > work( 1 );

    // Compute the local number of rows of the eigenvectors.
    // Build the matrix Z using 1-dim grid.
    slate::Matrix<scalar_t> Z1d;
    if (wantz) {
        int64_t lwork = max( 1, 2*n-2 );
        work.resize( lwork );

        // Find the total number of processors.
        int mpi_size;
        slate_mpi_call(
            MPI_Comm_size( Z.mpiComm(), &mpi_size ) );
        n = Z.n();
        nb = Z.tileNb( 0 );
        int myrow = Z.mpiRank();
        nrows = num_local_rows_cols( n, nb, myrow, 0, mpi_size );
        ldz = max( 1, nrows );
        Z1d_data.resize( ldz*n );
        Z1d = slate::Matrix<scalar_t>::fromScaLAPACK(
              n, n, &Z1d_data[0], nrows, nb, mpi_size, 1, Z.mpiComm() );
        set( zero, one, Z1d );
    }

    // Call the eigensolver.
    int64_t info = steqr(
        n, &D[0], &E[0], &Z1d_data[0], ldz, nrows, &work[0], work.size() );
    slate_assert( info == 0 ); // todo: throw errors

    // Redstribute the 1-dim eigenvector matrix into 2-dim matrix.
    if (wantz) {
        redistribute( Z1d, Z, opts );
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void steqr<float>(
    Job job,
    std::vector<float>& D,
    std::vector<float>& E,
    Matrix<float>& Z,
    Options const& opts);

template
void steqr<double>(
    Job job,
    std::vector<double>& D,
    std::vector<double>& E,
    Matrix<double>& Z,
    Options const& opts);

template
void steqr< std::complex<float> >(
    Job job,
    std::vector<float>& D,
    std::vector<float>& E,
    Matrix< std::complex<float> >& Z,
    Options const& opts);

template
void steqr< std::complex<double> >(
    Job job,
    std::vector<double>& D,
    std::vector<double>& E,
    Matrix< std::complex<double> >& Z,
    Options const& opts);

} // namespace slate
