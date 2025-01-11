// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "blas.hh"
#include "test.hh"
#include "print_matrix.hh"

#include "band_utils.hh"
#include "grid_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_steqr_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;
    using blas::max;

    // Constants
    const scalar_t zero = 0.0;
    const scalar_t one  = 1.0;

    // get & mark input values
    int64_t n = params.dim.n();
    int64_t nb = params.nb();
    int p = params.grid.m();
    int q = params.grid.n();
    lapack::Job jobz = params.jobz();
    bool check = params.check() == 'y';
    bool trace = params.trace() == 'y';
    slate::Origin origin = params.origin();

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();
    params.ortho();

    bool wantz = (jobz == slate::Job::Vec);

    if (! run)
        return;

    // MPI variables
    int mpi_size, mpi_rank, myrow, mycol;
    MPI_Comm_size( MPI_COMM_WORLD, &mpi_size );
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    // Matrix Z: figure out local size.
    int64_t mlocZ = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocZ = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldZ  = max( 1, mlocZ ); // local leading dimension of Z
    std::vector<scalar_t> Z_data(1);

    // Initialize the diagonal and subdiagonal
    std::vector<real_t> D(n), E(n - 1);
    int64_t idist = 3; // normal
    int64_t iseed[4] = { 0, 0, 0, 3 };
    //int64_t iseed[4] = { 0, myrow, mycol, 3 };
    lapack::larnv(idist, iseed, D.size(), D.data());
    lapack::larnv(idist, iseed, E.size(), E.data());
    std::vector<real_t> Dref = D;
    std::vector<real_t> Eref = E;

    if (mpi_rank == 0) {
        print_vector( "D", D, params );
        print_vector( "E", E, params );
    }

    slate::Matrix<scalar_t> A; // To check the orth of the eigenvectors
    if (check) {
        A = slate::Matrix<scalar_t>(n, n, nb, p, q, MPI_COMM_WORLD);
        A.insertLocalTiles();
    }

    slate::Matrix<scalar_t> Z; // Matrix of the eigenvectors
    if (wantz) {
        if (origin != slate::Origin::ScaLAPACK) {
            Z = slate::Matrix<scalar_t>(
                    n, n, nb, p, q, MPI_COMM_WORLD);
            Z.insertLocalTiles(origin2target(origin));
        }
        else {
            Z_data.resize(lldZ*nlocZ);
            Z = slate::Matrix<scalar_t>::fromScaLAPACK(
                    n, n, &Z_data[0], lldZ, nb, p, q, MPI_COMM_WORLD);
        }
        // note slate::steqr sets Z = Identity, unlike ScaLAPACK steqr2.
    }

    // Check low-level lwork query.
    std::vector<real_t> work( 1 );
    int64_t nrows = num_local_rows_cols( n, nb, mpi_rank, mpi_size );
    int64_t ldz = max( 1, nrows );
    int64_t lwork;
    // Unless p-by-q is mpi_size-by-1, Z is the wrong size here,
    // but we're just testing lwork query.
    int64_t info;
    // Eigenvalues only.
    info = slate::steqr( n, &D[0], &E[0], &Z_data[0], 1, 0, &work[0], -1 );
    lwork = int64_t( work[ 0 ] );
    slate_assert( info == 0 );
    slate_assert( lwork == 1 );
    // Eigenvectors.
    info = slate::steqr( n, &D[0], &E[0], &Z_data[0], ldz, nrows, &work[0], -1 );
    lwork = int64_t( work[ 0 ] );
    slate_assert( info == 0 );
    slate_assert( lwork == max( 1, 2*n - 2 ) || (nrows == 0 && lwork == 1) );

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    double time = barrier_get_wtime(MPI_COMM_WORLD);

    //==================================================
    // Run SLATE test.
    //==================================================
    steqr( jobz, D, E, Z );

    params.time() = barrier_get_wtime(MPI_COMM_WORLD) - time;

    if (trace)
        slate::trace::Trace::finish();

    if (mpi_rank == 0) {
        print_vector( "D_out   ", D, params );
    }
    print_matrix( "Z_out", Z, params );

    if (check) {
        //==================================================
        // Test results
        //==================================================
        real_t tol = params.tol() * 0.5 * std::numeric_limits<real_t>::epsilon();

        #ifdef SLATE_HAVE_SCALAPACK
            // Set Zref = Identity, distributed on 1D mpi_size-by-1 grid.
            std::vector<scalar_t> Zref_data( ldz*n );
            auto Zref = slate::Matrix<scalar_t>::fromScaLAPACK(
                n, n, &Zref_data[0], ldz, nb, mpi_size, 1, MPI_COMM_WORLD );
            set( zero, one, Zref );

            lwork = max( 1, 2*n - 2 );
            work.resize( lwork );

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            time = barrier_get_wtime(MPI_COMM_WORLD);

            scalapack::steqr2(
                jobz, n, &Dref[0], &Eref[0], &Zref_data[0], ldz, nrows,
                &work[0], &info );
            assert( info == 0 );

            params.ref_time() = barrier_get_wtime(MPI_COMM_WORLD) - time;

            if (mpi_rank == 0) {
                print_vector( "Dref_out", Dref, params );
            }
            print_matrix( "Zref_out", Zref, params );

            // Relative forward error: || D - Dref || / || Dref ||.
            real_t Dnorm = blas::nrm2( n, &Dref[0], 1 );
            blas::axpy( n, -1.0, &D[0], 1, &Dref[0], 1 );
            params.error() = blas::nrm2( n, &Dref[0], 1 ) / Dnorm;
            params.okay() = (params.error() <= tol);
        #endif

        //==================================================
        // Test results by checking the orthogonality of Z
        //
        //     || Z^H Z - I ||_f
        //     ----------------- < tol * epsilon
        //           n
        //
        //==================================================
        if (wantz) {
            auto ZT = conj_transpose( Z );
            set(zero, one, A);
            slate::gemm(one, ZT, Z, -one, A);
            params.ortho() = slate::norm(slate::Norm::Fro, A) / n;
            params.okay() = params.okay() && (params.ortho() <= tol);
        }
    }
}

// -----------------------------------------------------------------------------
void test_steqr(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_steqr_work<float>( params, run );
            break;

        case testsweeper::DataType::Double:
            test_steqr_work<double>( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_steqr_work<std::complex<float>>( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_steqr_work<std::complex<double>>( params, run );
            break;

        default:
            throw std::runtime_error( "unknown datatype" );
            break;
    }
}
