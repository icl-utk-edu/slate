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
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    params.matrix.mark();

    // mark non-standard output values
    params.time();
    params.gflops();
    params.time2();
    params.time2.name( "QR time (s)" );
    params.time2.width( 12 );
    params.gflops2();
    params.gflops2.name( "QR gflop/s" );

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

    // Keep copy of original matrix A.
    slate::Matrix<scalar_t> Aref;
    std::vector<scalar_t> Aref_data;
    real_t A_norm = 0; //initialize to prevent compiler warning
    if (check) {
        // Norm of original matrix: || A ||_1
        A_norm = slate::norm( slate::Norm::One, A, opts );
    }

    // For simplicity, always use ScaLAPACK format for Aref.
    Aref_data.resize( lldA * nlocA );
    Aref = slate::Matrix<scalar_t>::fromScaLAPACK(
               m, n, &Aref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
    slate::copy(A, Aref);


    double time_qr = barrier_get_wtime(MPI_COMM_WORLD);

    slate::qr_factor(A, T, opts);
    // Using traditional BLAS/LAPACK name
    // slate::geqrf(A, T, opts);

    time_qr = barrier_get_wtime(MPI_COMM_WORLD) - time_qr;

    double gflops_qr = lapack::Gflop<scalar_t>::geqrf(m, n);

    // compute and save timing/performance
    params.time2() = time_qr;
    params.gflops2() = gflops_qr / time_qr;

    print_matrix("A_factored", A, params);
    print_matrix("Tlocal",  T[0], params);
    print_matrix("Treduce", T[1], params);

    //==================================================
    // Check backwards error
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

    // R1 is the upper part of QR matrix.
    slate::TrapezoidMatrix<scalar_t> R1(slate::Uplo::Upper, slate::Diag::NonUnit, QR);

    // Copy A's upper trapezoid R to QR's upper trapezoid R1.
    slate::copy(R, R1);

    print_matrix("R", QR, params);

    if (trace) slate::trace::Trace::on();

    double time_unmqr = barrier_get_wtime(MPI_COMM_WORLD);

    //==================================================
    // Run SLATE test.
    //==================================================
    // Multiply QR by Q (implicitly stored in A and T).
    // Form QR, where Q's representation is in A and T, and R is in QR.
    slate::qr_multiply_by_q(
        slate::Side::Left, slate::Op::NoTrans, A, T, QR, opts);
    // Using traditional BLAS/LAPACK name
    // slate::unmqr(
    //     slate::Side::Left, slate::Op::NoTrans, A, T, QR, opts);

    time_unmqr = barrier_get_wtime(MPI_COMM_WORLD) - time_unmqr;

    if (trace) slate::trace::Trace::finish();

    print_matrix("QR", QR, params);

    double gflops_unmqr = lapack::Gflop<scalar_t>::unmqr(lapack::Side::Left, m, n, n);

    // compute and save timing/performance
    params.time() = time_unmqr;
    params.gflops() = gflops_unmqr / time_unmqr;

    if (check) {
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
