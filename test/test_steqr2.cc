// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "blas.hh"
#include "test.hh"
#include "print_matrix.hh"

#include "scalapack_support_routines.hh"
#include "band_utils.hh"
#include "grid_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_steqr2_work(Params& params, bool run)
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
    int verbose = params.verbose();
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
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    // Matrix Z: figure out local size.
    int64_t mlocZ = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocZ = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldZ  = blas::max(1, mlocZ); // local leading dimension of Z
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

    slate::Matrix<scalar_t> A; // To check the orth of the eigenvectors
    if (check) {
        A = slate::Matrix<scalar_t>(n, n, nb, p, q, MPI_COMM_WORLD);
        A.insertLocalTiles();
    }

    slate::Matrix<scalar_t> Z; // Matrix of the eigenvectors
    if (origin != slate::Origin::ScaLAPACK) {
        if (wantz) {
            Z = slate::Matrix<scalar_t>(
                    n, n, nb, p, q, MPI_COMM_WORLD);
            Z.insertLocalTiles(origin2target(origin));
        }
    }
    else {
        if (wantz) {
            Z_data.resize(lldZ*nlocZ);
            Z = slate::Matrix<scalar_t>::fromScaLAPACK(
                    n, n, &Z_data[0], lldZ, nb, p, q, MPI_COMM_WORLD);
        }
    }
    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    double time = barrier_get_wtime(MPI_COMM_WORLD);

    //==================================================
    // Run SLATE test.
    //==================================================
    //slate::sterf(D, E);
    steqr2(jobz, D, E, Z);

    params.time() = barrier_get_wtime(MPI_COMM_WORLD) - time;

    if (trace)
        slate::trace::Trace::finish();

    if (check) {
        //==================================================
        // Test results
        //==================================================
        real_t tol = params.tol() * 0.5 * std::numeric_limits<real_t>::epsilon();

        //==================================================
        // Run LAPACK reference routine.
        //==================================================
        time = barrier_get_wtime(MPI_COMM_WORLD);

        lapack::sterf(n, &Dref[0], &Eref[0]);

        params.ref_time() = barrier_get_wtime(MPI_COMM_WORLD) - time;

        if (verbose) {
            // Print first 20 and last 20 rows.
            printf( "%9s  %9s\n", "D", "Dref" );
            for (int64_t i = 0; i < n; ++i) {
                if (i < 20 || i > n-20) {
                    bool okay = std::abs( D[i] - Dref[i] ) < tol;
                    printf( "%9.6f  %9.6f%s\n",
                            D[i], Dref[i], (okay ? "" : " !!") );
                }
            }
            printf( "\n" );
        }

        // Relative forward error: || D - Dref || / || Dref ||.
        real_t err = blas::nrm2(Dref.size(), &Dref[0], 1);
        blas::axpy(D.size(), -1.0, &D[0], 1, &Dref[0], 1);
        params.error() = blas::nrm2(Dref.size(), &Dref[0], 1) / err;
        params.okay() = (params.error() <= tol);

        //==================================================
        // Test results by checking the orthogonality of Q
        //
        //     || Q^H Q - I ||_f
        //     ----------------- < tol * epsilon
        //           n
        //
        //==================================================
        if (wantz) {
            auto ZT = conjTranspose(Z);
            set(zero, one, A);
            slate::gemm(one, ZT, Z, -one, A);
            params.ortho() = slate::norm(slate::Norm::Fro, A) / n;
            params.okay() = params.okay() && (params.ortho() <= tol);
        }
    }
}

// -----------------------------------------------------------------------------
void test_steqr2(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_steqr2_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_steqr2_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_steqr2_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_steqr2_work<std::complex<double>> (params, run);
            break;
    }
}
