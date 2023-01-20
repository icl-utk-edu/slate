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
void test_gecondest_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;

    // Constants
    const scalar_t zero = 0;
    const scalar_t one = 1;
    real_t rone = 1.;

    auto method = params.method_lu();

    // get & mark input values
    slate::Norm norm = params.norm();
    int64_t n = params.dim.n();
    int64_t p = params.grid.m();
    int64_t q = params.grid.n();
    int64_t nb = params.nb();
    int64_t ib = params.ib();
    int64_t lookahead = params.lookahead();
    int64_t panel_threads = params.panel_threads();
    bool ref_only = params.ref() == 'o';
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    SLATE_UNUSED(verbose);
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    slate::GridOrder grid_order = params.grid_order();
    params.matrix.mark();
    params.matrixB.mark();

    // NoPiv and CALU ignore threshold.
    double pivot_threshold = params.pivot_threshold();

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();
    params.time2();
    params.time2.name( "lu time (s)" );
    params.time2.width( 12 );
    params.gflops2();
    params.gflops2.name( "lu gflop/s" );

    params.error();
    params.error.name( "slate-exact" );
    params.error2();
    params.error2.name( "scl-exact" );
    params.error3();
    params.error3.name( "slate-scl" );

    params.value();
    params.value.name( "slate" );
    params.value2();
    params.value2.name( "exact" );
    params.value3();
    params.value3.name( "scl" );

    params.pivot_threshold.width( 0 );

    if (! run) {
        params.tol() = 0.75;
        return;
    }

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo( mpi_rank, grid_order, p, q, &myrow, &mycol );

    slate::Options const opts =  {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib},
        {slate::Option::PivotThreshold, pivot_threshold},
        {slate::Option::MethodLU, method},
    };

    // Matrix A: figure out local size.
    int64_t mlocA = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocA = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldA  = blas::max(1, mlocA); // local leading dimension of A

    std::vector<scalar_t> A_data;
    slate::Matrix<scalar_t> A;
    if (origin != slate::Origin::ScaLAPACK) {
        // SLATE allocates CPU or GPU tiles.
        slate::Target origin_target = origin2target(origin);
        A = slate::Matrix<scalar_t>(
                n, n,    nb, p, q, MPI_COMM_WORLD );
        A.insertLocalTiles(origin_target);
    }
    else {
        // Create SLATE matrix from the ScaLAPACK layouts
        A_data.resize( lldA * nlocA );
        A = slate::Matrix<scalar_t>::fromScaLAPACK(
            n, n, &A_data[0], lldA, nb, nb, grid_order, p, q, MPI_COMM_WORLD );
    }

    slate::Matrix<scalar_t> Id;
    if (check) {
        slate::Target origin_target = origin2target(origin);
        Id = slate::Matrix<scalar_t>(
                n, n,    nb, p, q, MPI_COMM_WORLD );
        Id.insertLocalTiles(origin_target);
    }

    slate::Pivots pivots;

    slate::generate_matrix(params.matrix,  A);
    print_matrix("A", A, params);

    // If ref is required, copy test data.
    slate::Matrix<scalar_t> Aref;
    std::vector<scalar_t> Aref_data;
    if (ref) {
        Aref_data.resize( lldA* nlocA );
        Aref = slate::Matrix<scalar_t>::fromScaLAPACK(
                n, n, &Aref_data[0], lldA, nb, nb,
                grid_order, p, q, MPI_COMM_WORLD );

        slate::copy(A, Aref);
    }

    double gflop = lapack::Gflop<scalar_t>::getrf(n, n);

    // Compute the matrix norm
    real_t Anorm = 0;
    Anorm = slate::norm(norm, A, opts);
    real_t slate_rcond = 0., scl_rcond = 0., exact_rcond = 0.;

    if (! ref_only) {

        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        //==================================================
        // Run SLATE test: getrf followed by gecondest
        // getrf: Factor PA = LU.
        // gecondest:  Solve AX = B, including factoring A.
        //==================================================

        double time2 = barrier_get_wtime(MPI_COMM_WORLD);
        slate::lu_factor(A, pivots, opts);
        // Using traditional BLAS/LAPACK name
        // slate::getrf(A, pivots, opts);
        // compute and save timing/performance
        time2 = barrier_get_wtime(MPI_COMM_WORLD) - time2;
        params.time2() = time2;
        params.gflops2() = gflop / time2;



        double time = barrier_get_wtime(MPI_COMM_WORLD);

        slate::gecondest(norm, A, &Anorm, &slate_rcond, opts);
        time = barrier_get_wtime(MPI_COMM_WORLD) - time;
        // compute and save timing/performance
        params.time() = time;
        params.gflops() = gflop / time;

        if (trace) slate::trace::Trace::finish();
    }

    if (check) {
        // Find the exact condition number:
        // A * A^-1  = Id
        // LU * A^-1 = Id
        // U * A^-1 = trsm(L, Id), Id will be overwritten by the first solve
        // A^-1     = trsm(U, Id), Id will be overwritten by the second solve
        // Id is the inverse of the matrix A, norm(A^-1) = norm(Id)

        // Set the Id matrix to identity
        set(zero, one, Id);
        lu_solve_using_factor( A, pivots, Id, opts );
        exact_rcond = (1. / slate::norm(norm, Id, opts)) / Anorm;
    }

    if (ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            // A comparison with a reference routine from ScaLAPACK for timing only

            // BLACS/MPI variables
            int ictxt, myrow_, mycol_, info, p_, q_;
            int mpi_rank_ = 0, nprocs = 1;
            // initialize BLACS and ScaLAPACK
            Cblacs_pinfo(&mpi_rank_, &nprocs);
            slate_assert(p*q <= nprocs);
            Cblacs_get(-1, 0, &ictxt);
            Cblacs_gridinit( &ictxt, grid_order2str( grid_order ), p, q );
            Cblacs_gridinfo(ictxt, &p_, &q_, &myrow_, &mycol_);
            slate_assert( p == p_ );
            slate_assert( q == q_ );
            slate_assert( myrow == myrow_ );
            slate_assert( mycol == mycol_ );

            int64_t info_ref = 0;

            // ScaLAPACK descriptor for the reference matrix
            int Aref_desc[9];
            scalapack_descinit(Aref_desc, n, n, nb, nb, 0, 0, ictxt, mlocA, &info);
            slate_assert(info == 0);

            // ScaLAPACK data for pivots.
            std::vector<int> ipiv_ref(lldA + nb);

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            scalapack_pgetrf(n, n,
                    &Aref_data[0], 1, 1, Aref_desc, &ipiv_ref[0], &info_ref);

            // query for workspace size for pgecon
            int64_t info_ref_cond = 0;
            int64_t lwork = -1;
            int64_t liwork = -1;
            scalar_t dummy;
            int  idummy;
            scalapack_pgecon( norm2str(norm), n,
                              &Aref_data[0], 1, 1, Aref_desc,
                              &Anorm, &scl_rcond,
                              &dummy, lwork, &idummy, liwork, info_ref_cond);
            lwork = (int64_t)( real( dummy ) );
            liwork = (int64_t)( real( idummy ) );

            // Compute the condition number using scalapack
            std::vector<scalar_t> work(lwork);
            std::vector<int> iwork(liwork);

            // todo: ScaLAPCK pzgecon has a seg fault

            double time = barrier_get_wtime(MPI_COMM_WORLD);
            scalapack_pgecon( norm2str(norm), n,
                              &Aref_data[0], 1, 1, Aref_desc,
                              &Anorm, &scl_rcond,
                              &work[0], lwork, &iwork[0], liwork, info_ref_cond);
            slate_assert(info_ref_cond == 0);
            time = barrier_get_wtime(MPI_COMM_WORLD) - time;

            params.ref_time() = time;
            params.ref_gflops() = gflop / time;

            Cblacs_gridexit(ictxt);

        #else  // not SLATE_HAVE_SCALAPACK
            if (mpi_rank == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }

    // Compute the error
    params.error()  = std::abs( rone/slate_rcond - rone/exact_rcond ) / (rone/exact_rcond);
    params.error2() = std::abs( rone/scl_rcond - rone/exact_rcond ) / (rone/exact_rcond);
    params.error3() = std::abs(slate_rcond - scl_rcond);

    // Printf out the condest
    params.value()  = slate_rcond;
    params.value2() = exact_rcond;
    params.value3() = scl_rcond;

    real_t tol = params.tol();
    params.okay() = (params.error() <= tol);

}

// -----------------------------------------------------------------------------
void test_gecondest(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_gecondest_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_gecondest_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_gecondest_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gecondest_work<std::complex<double>> (params, run);
            break;
    }
}
