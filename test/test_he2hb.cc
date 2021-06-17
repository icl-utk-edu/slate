// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
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
    // using llong = long long;

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
    int verbose = params.verbose();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();

    origin = slate::Origin::ScaLAPACK;  // todo: for now

    // mark non-standard output values
    params.time();
    //params.gflops();

    if (! run)
        return;

    slate::Options const opts =  {
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    };

    // Requires a square processing grid.
    slate_assert(p == q);
    slate_assert(uplo == slate::Uplo::Lower);  // only lower for now.

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    // matrix A, figure out local size, allocate, initialize
    int64_t mlocal = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocal = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldA   = mlocal;
    std::vector<scalar_t> A_data(lldA*nlocal);
    int64_t idist = 3; // normal
    int64_t iseed[4] = { 0, myrow, mycol, 3 };
    lapack::larnv(idist, iseed, A_data.size(), A_data.data());

    slate::HermitianMatrix<scalar_t> A;
    if (origin != slate::Origin::ScaLAPACK) {
        // Copy local ScaLAPACK data to GPU or CPU tiles.
        A = slate::HermitianMatrix<scalar_t>(uplo, n, nb, p, q, MPI_COMM_WORLD);
        A.insertLocalTiles(origin2target(origin));
        // todo: need ScaLAPACK descriptor for copy. hmpf!
        //copy(A_data.data(), descA_tst, A);
        assert(false);
    }
    else {
        // Create SLATE matrices from the ScaLAPACK layouts.
        A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                uplo, n, A_data.data(), lldA, nb, p, q, MPI_COMM_WORLD);
    }
    slate::TriangularFactors<scalar_t> T;

    if (verbose > 1) {
        print_matrix("A", A);
    }

    // Copy test data for check.
    slate::HermitianMatrix<scalar_t> A_ref(uplo, n, nb, p, q, MPI_COMM_WORLD);
    A_ref.insertLocalTiles();
    slate::copy(A, A_ref);

    // todo
    //double gflop = lapack::Gflop<scalar_t>::he2hb(n, n);

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time = testsweeper::get_wtime();

    //==================================================
    // Run SLATE test.
    //==================================================
    slate::he2hb(A, T, opts);

    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time_tst = testsweeper::get_wtime() - time;

    if (trace) slate::trace::Trace::finish();

    // compute and save timing/performance
    params.time() = time_tst;
    //params.gflops() = gflop / time_tst;

    if (verbose > 1) {
        print_matrix("A_factored", A);
        print_matrix("Tlocal",  T[0]);
        print_matrix("Treduce", T[1]);
    }

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
        real_t A_norm = slate::norm(slate::Norm::One, A_ref);

        slate::Matrix<scalar_t> B(n, n, nb, p, q, MPI_COMM_WORLD);
        B.insertLocalTiles();
        he2gb(A, B);
        if (verbose > 1) {
            print_matrix("B", B);
        }

        slate::unmtr_he2hb(slate::Side::Left, slate::Op::NoTrans,
                           A, T, B, opts);
        if (verbose > 1) {
            print_matrix("Q^H B", B);
        }

        slate::unmtr_he2hb(slate::Side::Right, slate::Op::ConjTrans,
                           A, T, B, opts);
        if (verbose > 1) {
            print_matrix("Q^H B Q", B);
        }

        // Form QBQ^H - A, where A is in A_ref.
        // todo: slate::tradd(-one, TriangularMatrix(A_ref),
        //                     one, TriangularMatrix(B));
        for (int64_t j = 0; j < A.nt(); ++j) {
            for (int64_t i = j; i < A.nt(); ++i) {
                if (A_ref.tileIsLocal(i, j)) {
                    auto Aij = A_ref(i, j);
                    auto Bij = B(i, j);
                    // if i == j, Aij was Lower; set it to General for axpy.
                    Aij.uplo(slate::Uplo::General);
                    axpy(-one, Aij, Bij);
                }
            }
        }
        slate::HermitianMatrix<scalar_t> B_he(uplo, B);
        if (verbose > 1) {
            print_matrix("QBQ^H - A", B_he);
        }

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
