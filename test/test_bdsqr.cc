// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "blas.hh"
#include "test.hh"
#include "print_matrix.hh"
#include "scalapack_support_routines.hh"
#include "internal/internal.hh"
#include "band_utils.hh"
#include "grid_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_bdsqr_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;

    // Constants
    const scalar_t zero = 0.0, one = 1.0;

    // get & mark input values
    slate::Uplo uplo = params.uplo();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t nb = params.nb();
    int p = params.grid.m();
    int q = params.grid.n();
    lapack::Job jobu = params.jobu();
    lapack::Job jobvt = params.jobvt();
    bool check = params.check() == 'y';
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Origin origin = params.origin();

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();
    params.ortho_U();
    params.ortho_V();

    slate_assert(m >= n);

    if (! run)
        return;

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    int64_t min_mn = std::min(m, n);

    // Matrix U: figure out local size.
    int64_t mlocU = num_local_rows_cols(m, nb, myrow, p);
    int64_t nlocU = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldU  = blas::max(1, mlocU); // local leading dimension of U
    std::vector<scalar_t> U_data(1);

    // Matrix VT: figure out local size.
    int64_t mlocVT = num_local_rows_cols(min_mn, nb, myrow, p);
    int64_t nlocVT = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldVT  = blas::max(1, mlocVT); // local leading dimension of VT
    std::vector<scalar_t> VT_data(1);

    // initialize D and E to call the bidiagonal svd solver
    std::vector<real_t> D(n), E(n - 1);
    int64_t idist = 3; // normal
    int64_t iseed[4] = { 0, 0, 0, 3 };
    lapack::larnv(idist, iseed, D.size(), D.data());
    lapack::larnv(idist, iseed, E.size(), E.data());
    std::vector<real_t> Dref = D;
    std::vector<real_t> Eref = E;

    bool wantu  = (jobu  == slate::Job::Vec || jobu  == slate::Job::AllVec
                   || jobu  == slate::Job::SomeVec);
    bool wantvt = (jobvt == slate::Job::Vec || jobvt == slate::Job::AllVec
                   || jobvt == slate::Job::SomeVec);

    slate::Matrix<scalar_t> U, VT;
    if (origin != slate::Origin::ScaLAPACK) {
        // SLATE allocates CPU or GPU tiles.
        if (wantu) {
            U = slate::Matrix<scalar_t>(m, min_mn, nb, p, q, MPI_COMM_WORLD);
            U.insertLocalTiles();
        }
        if (wantvt) {
            VT = slate::Matrix<scalar_t>(min_mn, n, nb, p, q, MPI_COMM_WORLD);
            VT.insertLocalTiles();
        }
    }
    else {
        // create SLATE matrices from the ScaLAPACK layouts
        if (wantu) {
            U_data.resize(lldU*nlocU);
            U = slate::Matrix<scalar_t>::fromScaLAPACK(
                    m, min_mn, &U_data[0], lldU, nb, p, q, MPI_COMM_WORLD);
        }
        if (wantvt) {
            VT_data.resize(lldVT*nlocVT);
            VT = slate::Matrix<scalar_t>::fromScaLAPACK(
                     min_mn, n, &VT_data[0], lldVT, nb, p, q, MPI_COMM_WORLD);
        }
    }

    //---------
    // run test
    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    double time = barrier_get_wtime(MPI_COMM_WORLD);

    //==================================================
    // Run SLATE test.
    //==================================================
    slate::bdsqr<scalar_t>(jobu, jobvt, D, E, U, VT);

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

        scalar_t dummy[1];  // U, VT, C not needed for NoVec
        lapack::bdsqr(uplo, n, 0, 0, 0,
                      &Dref[0], &Eref[0], dummy, 1, dummy, 1, dummy, 1);

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
        blas::axpy(D.size(), -1.0, &Dref[0], 1, &D[0], 1);
        params.error() = blas::nrm2(D.size(), &D[0], 1)
                       / blas::nrm2(Dref.size(), &Dref[0], 1);
        params.okay() = params.error() <= tol;

        //==================================================
        // Test results by checking the orthogonality of Q
        //
        //     || Q^H Q - I ||_f
        //     ----------------- < tol * epsilon
        //           n
        //
        //==================================================
        if (wantu) {
            slate::Matrix<scalar_t> Id(min_mn, min_mn, nb, p, q, MPI_COMM_WORLD);
            Id.insertLocalTiles();
            set(zero, one, Id);

            auto UT = conjTranspose(U);
            slate::gemm(one, UT, U, -one, Id);
            params.ortho_U() = slate::norm(slate::Norm::Fro, Id) / m;
            params.okay() = params.okay() && (params.ortho_U() <= tol);
        }
        // If we flip the fat matrix, then no need for Id
        if (wantvt) {
            slate::Matrix<scalar_t> Id(n, n, nb, p, q, MPI_COMM_WORLD);
            Id.insertLocalTiles();
            set(zero, one, Id);

            auto VTT = conjTranspose(VT);
            slate::gemm(one, VTT, VT, -one, Id);
            params.ortho_V() = slate::norm(slate::Norm::Fro, Id) / n;
            params.okay() = params.okay() && (params.ortho_V() <= tol);
        }
    }
}

// -----------------------------------------------------------------------------
void test_bdsqr(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_bdsqr_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_bdsqr_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_bdsqr_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_bdsqr_work<std::complex<double>> (params, run);
            break;
    }
}
