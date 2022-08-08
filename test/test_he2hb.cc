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
#include "matrix_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_he2hb_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    // using blas::real;
    // using blas::conj;

    // Constants
    const scalar_t one = 1;

    // get & mark input values
    slate::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t p = params.grid.m();
    int64_t q = params.grid.n();
    int64_t nb = params.nb();
    int64_t ib = params.ib();
    int64_t panel_threads = params.panel_threads();
    bool check = params.check() == 'y';
    bool trace = params.trace() == 'y';
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    params.matrix.mark();

    // mark non-standard output values
    params.time();
    params.gflops();

    if (! run)
        return;

    slate::Options const opts =  {
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    };

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    // Skip invalid or unimplemented options.
    if (uplo == slate::Uplo::Upper) {
        params.msg() = "skipping: Uplo::Upper isn't supported.";
        return;
    }
    if (p != q) {
        params.msg() = "skipping: requires square process grid (p == q).";
        return;
    }

    // Matrix A: figure out local size.
    int64_t mlocal = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocal = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldA   = blas::max(1, mlocal); // local leading dimension of A
    std::vector<scalar_t> A_data;

    slate::HermitianMatrix<scalar_t> A;
    if (origin != slate::Origin::ScaLAPACK) {
        // SLATE allocates CPU or GPU tiles.
        A = slate::HermitianMatrix<scalar_t>(uplo, n, nb, p, q, MPI_COMM_WORLD);
        A.insertLocalTiles(origin2target(origin));
    }
    else {
        // Create SLATE matrices from the ScaLAPACK layouts.
        A_data.resize( lldA*nlocal );
        A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                uplo, n, A_data.data(), lldA, nb, p, q, MPI_COMM_WORLD);
    }
    slate::TriangularFactors<scalar_t> T;

    slate::generate_matrix( params.matrix, A );

    print_matrix("A", A, params);

    // Copy test data for check.
    slate::HermitianMatrix<scalar_t> Aref(uplo, n, nb, p, q, MPI_COMM_WORLD);
    Aref.insertLocalTiles();
    slate::copy(A, Aref);

    // compute and save timing/performance
    //double gflop = lapack::Gflop<scalar_t>::he2hb(n, n);
    double gflop = lapack::Gflop<scalar_t>::hetrd( n );

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    double time = barrier_get_wtime(MPI_COMM_WORLD);

    //==================================================
    // Run SLATE test.
    //==================================================
    slate::he2hb(A, T, opts);

    time = barrier_get_wtime(MPI_COMM_WORLD) - time;

    if (trace) slate::trace::Trace::finish();

    // compute and save timing/performance
    params.time() = time;
    params.gflops() = gflop / time;

    print_matrix("A_factored", A, params);
    print_matrix("Tlocal",  T[0], params);
    print_matrix("Treduce", T[1], params);

    if (check) {
        //==================================================
        // Test results by checking backwards error
        //
        //      || QBQ^H - A ||_1
        //     ------------------- < tol * epsilon
        //      || A ||_1 * n
        //
        //==================================================

        // Norm of original matrix: || A ||_1
        real_t A_norm = slate::norm(slate::Norm::One, Aref);

        slate::Matrix<scalar_t> B(n, n, nb, p, q, MPI_COMM_WORLD);
        B.insertLocalTiles();
        he2gb(A, B);
        print_matrix("B", B, params);

        slate::unmtr_he2hb(slate::Side::Left, slate::Op::NoTrans,
                           A, T, B, opts);
        print_matrix("Q^H B", B, params);

        slate::unmtr_he2hb(slate::Side::Right, slate::Op::ConjTrans,
                           A, T, B, opts);
        print_matrix("Q^H B Q", B, params);

        // Form QBQ^H - A, where A is in Aref.
        // todo: slate::tradd(-one, TriangularMatrix(Aref),
        //                     one, TriangularMatrix(B));
        for (int64_t j = 0; j < A.nt(); ++j) {
            for (int64_t i = j; i < A.nt(); ++i) {
                if (Aref.tileIsLocal(i, j)) {
                    auto Aij = Aref(i, j);
                    auto Bij = B(i, j);
                    // if i == j, Aij was Lower; set it to General for axpy.
                    Aij.uplo(slate::Uplo::General);
                    slate::tile::add( -one, Aij, Bij );
                }
            }
        }
        slate::HermitianMatrix<scalar_t> B_he(uplo, B);
        print_matrix("QBQ^H - A", B_he, params);

        // Norm of backwards error: || QBQ^H - A ||_1
        params.error() = slate::norm(slate::Norm::One, B_he) / (n * A_norm);
        real_t tol = params.tol() * std::numeric_limits<real_t>::epsilon()/2;
        params.okay() = (params.error() <= tol);
    }
}

// -----------------------------------------------------------------------------
void test_he2hb(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_he2hb_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_he2hb_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_he2hb_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_he2hb_work<std::complex<double>> (params, run);
            break;
    }
}
