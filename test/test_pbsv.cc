// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "slate/HermitianBandMatrix.hh"
#include "test.hh"
#include "blas/flops.hh"
#include "lapack/flops.hh"

#include "print_matrix.hh"
#include "grid_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
//------------------------------------------------------------------------------
template <typename scalar_t>
void test_pbsv_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;

    // Constants
    const scalar_t one = 1.0;

    // get & mark input values
    slate::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs();
    int64_t kd = params.kd();
    int p = params.grid.m();
    int q = params.grid.n();
    int64_t nb = params.nb();
    int64_t lookahead = params.lookahead();
    bool ref_only = params.ref() == 'o';
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    params.matrix.mark();

    // mark non-standard output values
    params.time();
    params.gflops();
    //params.ref_time();
    //params.ref_gflops();

    if (! run)
        return;

    if (origin != slate::Origin::ScaLAPACK) {
        params.msg() = "skipping: currently only origin=scalapack is supported";
        return;
    }

    slate::Options const opts =  {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    };

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    // Matrix B: figure out local size.
    int64_t mlocB = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocB = num_local_rows_cols(nrhs, nb, mycol, q);
    int64_t lldB  = blas::max(1, mlocB); // local leading dimension of B
    std::vector<scalar_t> B_data(lldB*nlocB);

    // Create SLATE matrix from the ScaLAPACK layouts
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(
                 n, nrhs, &B_data[0], lldB, nb, p, q, MPI_COMM_WORLD);

    slate::generate_matrix(params.matrix, B);

    int64_t iseeds[4] = { myrow, mycol, 2, 3 };
    auto A = slate::HermitianBandMatrix<scalar_t>(
                 uplo, n, kd, nb, p, q, MPI_COMM_WORLD);
    auto Aorig = slate::HermitianBandMatrix<scalar_t>(
                     uplo, n, kd, nb, p, q, MPI_COMM_WORLD);

    int64_t kdt = slate::ceildiv(kd, nb);
    int64_t jj = 0;
    for (int64_t j = 0; j < A.nt(); ++j) {
        int64_t jb = A.tileNb(j);
        int64_t ii = 0;
        for (int64_t i = 0; i < A.mt(); ++i) {
            int64_t ib = A.tileMb(i);
            if (A.tileIsLocal(i, j)) {
                if ((A.uplo() == slate::Uplo::Lower && i <= j + kdt && j <= i) ||
                    (A.uplo() == slate::Uplo::Upper && i >= j - kdt && j >= i)) {
                    A.tileInsert(i, j);
                    Aorig.tileInsert(i, j);
                    auto T = A(i, j);
                    lapack::larnv(2, iseeds, T.size(), T.data());
                    for (int64_t tj = jj; tj < jj + T.nb(); ++tj) {
                        for (int64_t ti = ii; ti < ii + T.mb(); ++ti) {
                            if ((A.uplo() == slate::Uplo::Lower && -kd     > tj - ti) ||
                                (A.uplo() == slate::Uplo::Upper && tj - ti > kd     )) {
                                T.at(ti - ii, tj - jj) = 0;
                            }
                        }
                    }
                    if (i == j) {
                        for (int64_t tj = jj; tj < jj + T.nb(); ++tj) {
                            for (int64_t ti = ii; ti < ii + T.mb(); ++ti) {
                                if ((ti - ii) == (tj - jj)) {
                                    T.at(ti - ii, tj - jj) = std::abs(T.at(ti - ii, tj - jj)) + n;
                                }
                            }
                        }
                    }
                    auto T2 = Aorig(i, j);
                    lapack::lacpy(lapack::MatrixType::General, T.mb(), T.nb(),
                                  T.data(), T.stride(),
                                  T2.data(), T2.stride());
                }
            }
            ii += ib;
        }
        jj += jb;
    }

    if (verbose > 1) {
        printf("%% rank %d A\n", A.mpiRank());
        //printf("%% rank %d A kd %lld\n", A.mpiRank(), llong( A.bandwidth()));
    }
    print_matrix("A", A, params);
    print_matrix("B", B, params);


    // if check is required, copy test data and create a descriptor for it
    slate::Matrix<scalar_t> Bref;
    if (check || ref) {
        Bref = slate::Matrix<scalar_t>(n, nrhs, nb, p, q, MPI_COMM_WORLD);
        Bref.insertLocalTiles();
        slate::copy( B, Bref );
    }

    double gflop;
    if (params.routine == "pbtrf")
        gflop = lapack::Gflop<scalar_t>::pbtrf(n, kd);
    else if (params.routine == "potrs")
        gflop = lapack::Gflop<scalar_t>::pbtrs(n, nrhs, kd);
    else
        gflop = lapack::Gflop<scalar_t>::pbsv(n, nrhs, kd);

    if (! ref_only) {
        if (params.routine == "pbtrs") {
            // Factor matrix A.
            slate::chol_factor(A, opts);
            // Using traditional BLAS/LAPACK name
            // slate::pbtrf(A, opts);
        }

        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        double time = barrier_get_wtime(MPI_COMM_WORLD);

        //==================================================
        // Run SLATE test.
        // One of:
        // pbtrf: Factor A = LL^U or A = U^H U.
        // pbtrs: Solve AX = B, after factoring A above.
        // pbsv:  Solve AX = B, including factoring A.
        //==================================================
        if (params.routine == "pbtrf") {
            slate::chol_factor(A, opts);
            // Using traditional BLAS/LAPACK name
            // slate::pbtrf(A, opts);
        }
        else if (params.routine == "pbtrs") {
            slate::chol_solve_using_factor(A, B, opts);
            // Using traditional BLAS/LAPACK name
            // slate::pbtrs(A, B, opts);
        }
        else {
            slate::chol_solve(A, B, opts);

            // slate::pbsv(A, B, opts);
        }

        time = barrier_get_wtime(MPI_COMM_WORLD) - time;

        if (trace)
            slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time;
        params.gflops() = gflop / time;

        if (verbose > 1) {
            printf("%% rank %d A2\n", A.mpiRank());
            //printf("%% rank %d A2 kd %lld\n",
            //       A.mpiRank(), llong( A.bandwidth( )));
            //printf( "nb = %lld;\n", llong( nb ) );
        }
        print_matrix("A2", A, params);
        print_matrix("B2", B, params);
    }
    if (check) {
        //==================================================
        // Test results by checking the residual
        //
        //           || B - AX ||_1
        //     --------------------------- < tol * epsilon
        //      || A ||_1 * || X ||_1 * N
        //
        //==================================================

        // LAPACK (dget02) does
        // max_j || A * x_j - b_j ||_1 / (|| A ||_1 * || x_j ||_1).
        // No N?

        if (params.routine == "pbtrf") {
            // Solve AX = B.
            slate::chol_solve_using_factor(A, B, opts);
            // Using traditional BLAS/LAPACK name
            // slate::pbtrs(A, B, opts);
        }

        // Norm of original matrix: || A ||_1
        real_t A_norm = slate::norm(slate::Norm::One, Aorig);
        // Norm of updated rhs matrix: || X ||_1
        real_t X_norm = slate::norm( slate::Norm::One, B );

        // Bref -= Aref*B
        slate::multiply(-one, Aorig, B, one, Bref);
        // Using traditional BLAS/LAPACK name
        // slate::hbmm(blas::Side::Left, -one, Aorig, B, one, Bref);

        // Norm of residual: || B - AX ||_1
        real_t R_norm = slate::norm( slate::Norm::One, Bref );

        double residual = R_norm / (n*A_norm*X_norm);
        params.error() = residual;

        real_t tol = params.tol() * 0.5 * std::numeric_limits<real_t>::epsilon();
        params.okay() = (params.error() <= tol);

        if (verbose > 0) {
            printf("Anorm = %.4e; Xnorm = %.4e; Rnorm = %.4e; error = %.4e;\n",
                   A_norm, X_norm, R_norm, residual);
        }
        print_matrix("Residual", Bref, params);
    }
    // todo: reference solution requires setting up band matrix in ScaLAPACK's
    // band storage format.
}

// -----------------------------------------------------------------------------
void test_pbsv(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_pbsv_work<float>(params, run);
            break;

        case testsweeper::DataType::Double:
            test_pbsv_work<double>(params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_pbsv_work<std::complex<float>>(params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_pbsv_work<std::complex<double>>(params, run);
            break;
    }
}
