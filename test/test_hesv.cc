// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"
#include "blas/flops.hh"
#include "lapack/flops.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"
#include "grid_utils.hh"
#include "print_matrix.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_hesv_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;

    // Constants
    const scalar_t one = 1.0;

    //---------------------
    // get & mark input values
    slate::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs();
    int p = params.grid.m();
    int q = params.grid.n();
    int64_t nb = params.nb();
    int64_t lookahead = params.lookahead();
    int64_t panel_threads = params.panel_threads();
    bool check = params.check() == 'y';
    bool trace = params.trace() == 'y';
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    params.matrix.mark();
    params.matrixB.mark();

    //---------------------
    // mark non-standard output values
    params.time();
    params.gflops();

    if (! run)
        return;

    if (origin != slate::Origin::ScaLAPACK) {
        params.msg() = "skipping: currently only origin=scalapack is supported";
        return;
    }
    if (target == slate::Target::Devices) {
        params.msg() = "skipping: currently target=devices is not supported";
        return;
    }
    if (n % nb != 0) {
        params.msg() = "skipping: currently only (n %% nb == 0) is supported";
        return;
    }
    if (uplo != slate::Uplo::Lower) {
        params.msg() = "skipping: currently only uplo=lower is supported";
        return;
    }

    slate::Options const opts =  {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads}
    };

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    // Matrix A: figure out local size.
    int64_t mlocA = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocA = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldA  = blas::max(1, mlocA); // local leading dimension of A
    std::vector<scalar_t> A_data(lldA*nlocA);

    // Matrix B: figure out local size.
    int64_t mlocB = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocB = num_local_rows_cols(nrhs, nb, mycol, q);
    int64_t lldB  = blas::max(1, mlocB); // local leading dimension of B
    std::vector<scalar_t> B_data(lldB*nlocB);

    //---------------------
    // Create SLATE matrix from the ScaLAPACK layouts
    auto A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                 uplo, n, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(
                 n, nrhs, &B_data[0], lldB, nb, p, q, MPI_COMM_WORLD);
    real_t A_norm, X_norm;

    slate::generate_matrix( params.matrix, A );
    slate::generate_matrix( params.matrixB, B );

    slate::Pivots pivots;

    print_matrix( "A", A, params );
    print_matrix( "B", B, params );

    //---------------------
    // band matrix
    int64_t kl = nb;
    int64_t ku = nb;
    slate::Pivots pivots2;
    auto T = slate::BandMatrix<scalar_t>(n, n, kl, ku, nb, p, q, MPI_COMM_WORLD);

    //---------------------
    // auxiliary matrices
    auto H = slate::Matrix<scalar_t>(n, n, nb, p, q, MPI_COMM_WORLD);

    //---------------------
    // right-hand-side and solution vectors
    std::vector<scalar_t> Bref_data;

    //---------------------
    // if check is required, copy test data.
    slate::HermitianMatrix<scalar_t> Aref;
    slate::Matrix<scalar_t> Bref;
    std::vector<scalar_t> Aref_data;
    if (check) {
        Aref_data.resize( A_data.size() );
        Aref = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                   uplo, n, &Aref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        slate::copy( A, Aref );

        Bref_data.resize( B_data.size() );
        Bref = slate::Matrix<scalar_t>::fromScaLAPACK(
                   n, nrhs, &Bref_data[0], lldB, nb, p, q, MPI_COMM_WORLD);
        slate::copy( B, Bref );
    }

    if (params.routine == "hetrs") {
        slate::indefinite_factor(A, pivots, T, pivots2, H, opts);
        // Using traditional BLAS/LAPACK name
        // slate::hetrf(A, pivots, T, pivots2, H, opts);
    }

    //==================================================
    // Run SLATE test.
    // One of:
    // hetrf: Factor A = LTL^H or A = U^H TU.
    // hetrs: Solve AX = B, after factoring A above.
    // hesv:  Solve AX = B, including factoring A.
    //==================================================
    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    double time = barrier_get_wtime(MPI_COMM_WORLD);

    if (params.routine == "hetrf") {
        slate::indefinite_factor(A, pivots, T, pivots2, H, opts);
        // Using traditional BLAS/LAPACK name
        // slate::hetrf(A, pivots, T, pivots2, H, opts);
    }
    else if (params.routine == "hetrs") {
        slate::indefinite_solve_using_factor(A, pivots, T, pivots2, B, opts);
        // Using traditional BLAS/LAPACK name
        // slate::hetrs(A, pivots, T, pivots2, B, opts);
    }
    else {
        slate::indefinite_solve(A, B, opts);
        // Using traditional BLAS/LAPACK name
        // slate::hesv(A, pivots, T, pivots2, H, B, opts);
    }

    time = barrier_get_wtime(MPI_COMM_WORLD) - time;

    if (trace) slate::trace::Trace::finish();

    print_matrix( "Aout", A, params );
    print_matrix( "Bout", B, params );

    //---------------------
    // compute and save timing/performance
    double gflop;
    if (params.routine == "hetrf")
        gflop = lapack::Gflop<scalar_t>::potrf(n);
    else if (params.routine == "hetrs")
        gflop = lapack::Gflop<scalar_t>::potrs(n, nrhs);
    else
        gflop = lapack::Gflop<scalar_t>::posv(n, nrhs);
    params.time() = time;
    params.gflops() = gflop / time;

    if (check) {
        #ifdef SLATE_HAVE_SCALAPACK
            //---------------------
            // BLACS/MPI variables
            int ictxt, p_, q_, myrow_, mycol_, info;
            int A_desc[9], Aref_desc[9];
            int B_desc[9], Bref_desc[9];
            int mpi_rank_ = 0, nprocs = 1;

            //---------------------
            // initialize BLACS and ScaLAPACK
            Cblacs_pinfo(&mpi_rank_, &nprocs);
            slate_assert( mpi_rank_ == mpi_rank );
            slate_assert(p*q <= nprocs);
            Cblacs_get(-1, 0, &ictxt);
            Cblacs_gridinit(&ictxt, "Col", p, q);
            Cblacs_gridinfo(ictxt, &p_, &q_, &myrow_, &mycol_);
            slate_assert( p == p_ );
            slate_assert( q == q_ );
            slate_assert( myrow == myrow_ );
            slate_assert( mycol == mycol_ );

            scalapack_descinit(A_desc, n, n, nb, nb, 0, 0, ictxt, mlocA, &info);
            slate_assert(info == 0);

            scalapack_descinit(Aref_desc, n, n, nb, nb, 0, 0, ictxt, mlocA, &info);
            slate_assert(info == 0);

            scalapack_descinit(B_desc, n, nrhs, nb, nb, 0, 0, ictxt, mlocB, &info);
            slate_assert(info == 0);

            scalapack_descinit(Bref_desc, n, nrhs, nb, nb, 0, 0, ictxt, mlocB, &info);
            slate_assert(info == 0);

            copy( A, &A_data[0], A_desc );
            copy( B, &B_data[0], B_desc );
            copy( Aref, &Aref_data[0], Aref_desc );
            copy( Bref, &Bref_data[0], Bref_desc );

            if (params.routine == "hetrf") {
                // solve
                slate::indefinite_solve_using_factor(A, pivots, T, pivots2, B, opts);
                // Using traditional BLAS/LAPACK name
                // slate::hetrs(A, pivots, T, pivots2, B, opts);
            }

            // allocate work space
            std::vector<real_t> worklangeA(std::max(mlocA, nlocA));
            std::vector<real_t> worklangeB(std::max(mlocB, nlocB));

            // Norm of the orig matrix: || A ||
            A_norm = scalapack_plange("1", n, n, &Aref_data[0], 1, 1, Aref_desc, &worklangeA[0]);
            // norm of updated rhs matrix: || X ||
            X_norm = scalapack_plange("1", n, nrhs, &B_data[0], 1, 1, B_desc, &worklangeB[0]);

            // Bref_data -= Aref*B_data
            scalapack_phemm("Left", "Lower",
                            n, nrhs,
                            -one,
                            &Aref_data[0], 1, 1, Aref_desc,
                            &B_data[0], 1, 1, B_desc,
                            one,
                            &Bref_data[0], 1, 1, Bref_desc);

            // || B - AX ||
            real_t R_norm = scalapack_plange("1", n, nrhs, &Bref_data[0], 1, 1, Bref_desc, &worklangeB[0]);

            double residual = R_norm / (n*A_norm*X_norm);
            params.error() = residual;

            real_t tol = params.tol() * 0.5 * std::numeric_limits<real_t>::epsilon();
            params.okay() = (params.error() <= tol);

            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering
        #else  // not SLATE_HAVE_SCALAPACK
            SLATE_UNUSED( one );
            SLATE_UNUSED( A_norm );
            SLATE_UNUSED( X_norm );
            if (mpi_rank == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }
}

// -----------------------------------------------------------------------------
void test_hesv(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_hesv_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_hesv_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_hesv_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_hesv_work<std::complex<double>> (params, run);
            break;
    }
}
