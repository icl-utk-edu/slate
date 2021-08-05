// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "lapack/flops.hh"
#include "print_matrix.hh"
#include "grid_utils.hh"
//------------------------------------------------------------------------------
template <typename scalar_t>
void test_unmqr_work(Params& params, bool run)
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
    bool check = params.check() == 'y';
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    params.matrix.mark();

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    if (! run)
        return;

    slate::Options const opts =  {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
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
    slate::Target origin_target = origin2target(origin);
    if (origin != slate::Origin::ScaLAPACK) {
        // SLATE allocates CPU or GPU tiles.
        A = slate::Matrix<scalar_t>(m, n, nb, p, q, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);
    }
    else {
        // create SLATE matrix from the ScaLAPACK layouts
        A_data.resize( lldA * nlocA );
        A = slate::Matrix<scalar_t>::fromScaLAPACK( m, n, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD );
    }

    slate::generate_matrix(params.matrix, A);
    if (verbose > 1) {
        print_matrix("A", A);
    }

    slate::TriangularFactors<scalar_t> T;
    // For checks, keep copy of original matrix A.
    slate::Matrix<scalar_t> Aref;
    std::vector<scalar_t> Aref_data;
    real_t A_norm = 0; //initialize to prevent compiler warning
    if (check) {
        // Norm of original matrix: || A ||_1
        A_norm = slate::norm( slate::Norm::One, A, opts );

        // For simplicity, always use ScaLAPACK format for Aref.
        Aref_data.resize( lldA * nlocA );
        Aref = slate::Matrix<scalar_t>::fromScaLAPACK(
                   m, n, &Aref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        slate::copy(A, Aref);
    }

    //==================================================
    // Run SLATE test.
    //==================================================
    slate::qr_factor(A, T, opts);
    // Using traditional BLAS/LAPACK name
    // slate::geqrf(A, T, opts);

    if (verbose > 1) {
        print_matrix("A_factored", A);
        print_matrix("Tlocal",  T[0]);
        print_matrix("Treduce", T[1]);
    }

    //==================================================
    // Test results by checking backwards error
    //
    //      || QR - A ||_1
    //     ---------------- < tol * epsilon
    //      || A ||_1 * m
    //
    //==================================================
    // R is the upper part of A matrix.
    slate::TrapezoidMatrix<scalar_t> R(slate::Uplo::Upper, slate::Diag::NonUnit, A);

    std::vector<scalar_t> QR_data(Aref_data.size(), zero);
    slate::Matrix<scalar_t> QR = slate::Matrix<scalar_t>::fromScaLAPACK(
                                     m, n, &QR_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
    QR.insertLocalTiles(origin_target);

    // R1 is the upper part of QR matrix.
    slate::TrapezoidMatrix<scalar_t> R1(slate::Uplo::Upper, slate::Diag::NonUnit, QR);

    // Copy A's upper trapezoid R to QR's upper trapezoid R1.
    slate::copy(R, R1);
    
    if (verbose > 1) {
        print_matrix("R", QR);
    }
    
    if (! check) {
        // prefetch all matrices to devices
        // when performance of unmqr is measured
        A.tileGetAllForReadingOnDevices(slate::LayoutConvert::None);
        QR.tileGetAllForWritingOnDevices(slate::LayoutConvert::None); 
    }

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time = testsweeper::get_wtime();

    // Multiply QR by Q (implicitly stored in A and T).
    // Form QR, where Q's representation is in A and T, and R is in QR.
    slate::qr_multiply_by_q(
        slate::Side::Left, slate::Op::NoTrans, A, T, QR, opts);
    // Using traditional BLAS/LAPACK name
    // slate::unmqr(
    //     slate::Side::Left, slate::Op::NoTrans, A, T, QR, opts);

    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time_tst = testsweeper::get_wtime() - time;

    if (trace) slate::trace::Trace::finish();

    // compute and save timing/performance
    params.time() = time_tst;
    double gflop = lapack::Gflop<scalar_t>::unmqr(lapack::Side::Left, m, n, n);
    params.gflops() = gflop / time_tst;

    if (verbose > 1) {
        print_matrix("QR", QR);
    }

    if (check) {
        // QR should now have the product Q*R, which should equal the original A.
        // Subtract the original Aref from QR.
        // Form QR - A, where A is in Aref.
        // todo: slate::add(-one, Aref, QR);
        // using axpy assumes Aref_data and QR_data have same lda.
        blas::axpy(QR_data.size(), -one, &Aref_data[0], 1, &QR_data[0], 1);
        if (verbose > 1) {
            print_matrix("QR - A", QR);
        }

        // Norm of backwards error: || QR - A ||_1
        real_t R_norm = slate::norm(slate::Norm::One, QR);

        double residual = R_norm / (m*A_norm);
        params.error() = residual;
        real_t tol = params.tol() * 0.5 * std::numeric_limits<real_t>::epsilon();
        params.okay() = (params.error() <= tol);
    }
}

// -----------------------------------------------------------------------------
void test_unmqr(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_unmqr_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_unmqr_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_unmqr_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_unmqr_work<std::complex<double>> (params, run);
            break;
    }
}
