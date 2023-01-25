// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"
#include "blas/flops.hh"
#include "lapack/flops.hh"
#include "print_matrix.hh"
#include "grid_utils.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_geqrf_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;

    // Constants
    const scalar_t zero = 0;
    const scalar_t one = 1;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int p = params.grid.m();
    int q = params.grid.n();
    int64_t nb = params.nb();
    int64_t ib = params.ib();
    int64_t lookahead = params.lookahead();
    int64_t panel_threads = params.panel_threads();
    bool ref_only = params.ref() == 'o';
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    bool trace = params.trace() == 'y';
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    slate::Method methodCholQR = params.method_cholQR();
    params.matrix.mark();

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    if (! run)
        return;

    if (params.routine == "cholqr" && m < n) {
        params.msg() = "skipping: cholqr requires m >= n";
        return;
    }

    slate::Options const opts =  {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib},
        {slate::Option::MethodCholQR, methodCholQR}
    };

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    // Matrix A: figure out local size.
    int64_t mlocA = num_local_rows_cols(m, nb, myrow, p);
    int64_t nlocA = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldA  = blas::max(1, mlocA); // local leading dimension of A
    std::vector<scalar_t> A_data;

    slate::Matrix<scalar_t> A;
    if (origin != slate::Origin::ScaLAPACK) {
        // SLATE allocates CPU or GPU tiles.
        slate::Target origin_target = origin2target(origin);
        A = slate::Matrix<scalar_t>(m, n, nb, p, q, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);
    }
    else {
        // create SLATE matrix from the ScaLAPACK layouts
        A_data.resize( lldA * nlocA );
        A = slate::Matrix<scalar_t>::fromScaLAPACK( m, n, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD );
    }

    slate::generate_matrix(params.matrix, A);

    slate::TriangularFactors<scalar_t> T;

        print_matrix("A", A, params);

    // For checks, keep copy of original matrix A.
    slate::Matrix<scalar_t> Aref;
    std::vector<scalar_t> Aref_data;
    real_t A_norm = 0; //initialize to prevent compiler warning
    if (check || ref) {
        // Norm of original matrix: || A ||_1
        A_norm = slate::norm( slate::Norm::One, A, opts );

        // For simplicity, always use ScaLAPACK format for Aref.
        Aref_data.resize( lldA * nlocA );
        Aref = slate::Matrix<scalar_t>::fromScaLAPACK(
                   m, n, &Aref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        slate::copy(A, Aref);
    }

    double gflop = lapack::Gflop<scalar_t>::geqrf(m, n);

    slate::Matrix<scalar_t> R_chol;

    if (! ref_only) {
        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        double time = barrier_get_wtime(MPI_COMM_WORLD);

        //==================================================
        // Run SLATE test.
        //==================================================
        if (params.routine == "cholqr") {
            slate::Target origin_target = origin2target( origin );
            R_chol = slate::Matrix<scalar_t>(
                n, n, nb, p, q, MPI_COMM_WORLD );
            R_chol.insertLocalTiles( origin_target );

            slate::cholqr( A, R_chol, opts );
        }
        else {
            slate::qr_factor( A, T, opts );
        }
        // Using traditional BLAS/LAPACK name
        // slate::geqrf(A, T, opts);

        time = barrier_get_wtime(MPI_COMM_WORLD) - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time;
        params.gflops() = gflop / time;

        print_matrix("A_factored", A, params);
        print_matrix("Tlocal",  T[0], params);
        print_matrix("Treduce", T[1], params);
    }

    if (check) {
        //==================================================
        // Test results by checking backwards error
        //
        //      || QR - A ||_1
        //     ---------------- < tol * epsilon
        //      || A ||_1 * m
        //
        //==================================================

        std::vector<scalar_t> QR_data(Aref_data.size(), zero);
        slate::Matrix<scalar_t> QR = slate::Matrix<scalar_t>::fromScaLAPACK(
                                         m, n, &QR_data[0], lldA, nb, p, q, MPI_COMM_WORLD);

        if (params.routine == "cholqr") {
            // Copy A in QR that will be overwritten by the Q matrix
            slate::copy(A, QR);

            auto R = slate::TriangularMatrix<scalar_t>(
                slate::Uplo::Upper, slate::Diag::NonUnit, R_chol );
            slate::triangular_multiply(one, QR, R, opts);
        }
        else {
            // R is the upper part of A matrix.
            slate::TrapezoidMatrix<scalar_t> R(slate::Uplo::Upper, slate::Diag::NonUnit, A);

            // R1 is the upper part of QR matrix.
            slate::TrapezoidMatrix<scalar_t> R1(slate::Uplo::Upper, slate::Diag::NonUnit, QR);

            // Copy A's upper trapezoid R to QR's upper trapezoid R1.
            slate::copy(R, R1);

            print_matrix("R", QR, params);

            // Multiply QR by Q (implicitly stored in A and T).
            // Form QR, where Q's representation is in A and T, and R is in QR.
            slate::qr_multiply_by_q(
                slate::Side::Left, slate::Op::NoTrans, A, T, QR, opts);
            // Using traditional BLAS/LAPACK name
            // slate::unmqr(
            //     slate::Side::Left, slate::Op::NoTrans, A, T, QR, opts);
        }

        print_matrix("QR", QR, params);

        // QR should now have the product Q*R, which should equal the original A.
        // Subtract the original Aref from QR.
        // Form QR - A, where A is in Aref.
        // todo: slate::add(-one, Aref, QR);
        // using axpy assumes Aref_data and QR_data have same lda.
        blas::axpy(QR_data.size(), -one, &Aref_data[0], 1, &QR_data[0], 1);
        print_matrix("QR - A", QR, params);

        // Norm of backwards error: || QR - A ||_1
        real_t R_norm = slate::norm(slate::Norm::One, QR);

        double residual = R_norm / (m*A_norm);
        params.error() = residual;
        real_t tol = params.tol() * 0.5 * std::numeric_limits<real_t>::epsilon();
        params.okay() = (params.error() <= tol);
    }

    if (ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            // A comparison with a reference routine from ScaLAPACK for timing only

            // BLACS/MPI variables
            int ictxt, myrow_, mycol_, info, p_, q_;
            int Aref_desc[9];
            int mpi_rank_ = 0, nprocs = 1;
            // initialize BLACS and ScaLAPACK
            Cblacs_pinfo(&mpi_rank_, &nprocs);
            slate_assert(p*q <= nprocs);
            Cblacs_get(-1, 0, &ictxt);
            Cblacs_gridinit(&ictxt, "Col", p, q);
            Cblacs_gridinfo(ictxt, &p_, &q_, &myrow_, &mycol_);
            slate_assert( p == p_ );
            slate_assert( q == q_ );
            slate_assert( myrow == myrow_ );
            slate_assert( mycol == mycol_ );

            scalapack_descinit(Aref_desc, m, n, nb, nb, 0, 0, ictxt, mlocA, &info);
            slate_assert(info == 0);

            // tau vector for ScaLAPACK
            int64_t ltau = num_local_rows_cols(std::min(m, n), nb, mycol, q);
            std::vector<scalar_t> tau(ltau);

            // workspace for ScaLAPACK
            int64_t lwork;
            std::vector<scalar_t> work(1);
            //---------------

            int64_t info_ref = 0;

            if (check) {
                // Copy original A for ScaLAPACK check
                slate::copy(Aref, A);
            }

            // query for workspace size
            scalar_t dummy;
            scalapack_pgeqrf(m, n, &Aref_data[0], 1, 1, Aref_desc, tau.data(),
                             &dummy, -1, &info_ref);
            lwork = int64_t( real( dummy ) );
            work.resize(lwork);

            //==================================================
            // Run ScaLAPACK reference test.
            //==================================================
            double time = barrier_get_wtime(MPI_COMM_WORLD);
            scalapack_pgeqrf(m, n, &Aref_data[0], 1, 1, Aref_desc, tau.data(),
                             work.data(), lwork, &info_ref);
            slate_assert(info_ref == 0);
            time = barrier_get_wtime(MPI_COMM_WORLD) - time;

            if (0) {
                //==================================================
                // Test results by checking backwards error
                // within ScaLAPACK implementation
                //
                //      || QR - A ||_1
                //     ---------------- < tol * epsilon
                //      || A ||_1 * m
                //
                //==================================================

                // R is the upper part of A matrix.
                slate::TrapezoidMatrix<scalar_t> scala_R(slate::Uplo::Upper, slate::Diag::NonUnit, Aref);

                std::vector<scalar_t> scala_QR_data(Aref_data.size(), zero);
                slate::Matrix<scalar_t> scala_QR = slate::Matrix<scalar_t>::fromScaLAPACK(
                                                     m, n, &scala_QR_data[0], lldA, nb, p, q, MPI_COMM_WORLD);

                slate::TrapezoidMatrix<scalar_t> scala_R1(slate::Uplo::Upper, slate::Diag::NonUnit, scala_QR);

                // Copy A's upper trapezoid R to QR's upper trapezoid R1.
                slate::copy(scala_R, scala_R1);

                // Apply Q to R-factor
                scalapack_punmqr(side2str(blas::Side::Left), op2str(slate::Op::NoTrans), m, n, n,
                                 &Aref_data[0], 1, 1, Aref_desc, tau.data(),
                                 &scala_QR_data[0], 1, 1, Aref_desc,
                                 &dummy, -1, &info_ref);
                lwork = int64_t( real( dummy ) );
                work.resize(lwork);
                scalapack_punmqr(side2str(blas::Side::Left), op2str(slate::Op::NoTrans), m, n, n,
                                 &Aref_data[0], 1, 1, Aref_desc, tau.data(),
                                 &scala_QR_data[0], 1, 1, Aref_desc,
                                 work.data(), lwork, &info_ref);
                slate_assert(info_ref == 0);

                print_matrix("QR", scala_QR, params);

                slate::add(-one, A, one, scala_QR, opts);
                print_matrix("QR - A", scala_QR, params);

                // Norm of backwards error: || QR - A ||_1
                real_t scala_R_norm = slate::norm(slate::Norm::One, scala_QR);

                double residual = scala_R_norm / (m*A_norm);
                if (mpi_rank == 0)
                    printf("\nScaLAPACK comparision: ||A - QR|| / ||A|| = %3.2e\n",residual);
            }

            params.ref_time() = time;
            params.ref_gflops() = gflop / time;

            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering
        #else  // not SLATE_HAVE_SCALAPACK
            if (mpi_rank == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }
}
// -----------------------------------------------------------------------------
void test_geqrf(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_geqrf_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_geqrf_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_geqrf_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_geqrf_work<std::complex<double>> (params, run);
            break;
    }
}
