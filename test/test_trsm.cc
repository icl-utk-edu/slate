// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"
#include "blas/flops.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"
#include "print_matrix.hh"
#include "grid_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template< typename scalar_t >
void test_trsm_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using slate::Op;
    using slate::Norm;
    using blas::real;

    // Constants
    const scalar_t one = 1;

    // get & mark input values
    slate::Side side = params.side();
    slate::Uplo uplo = params.uplo();
    slate::Op transA = params.transA();
    slate::Diag diag = params.diag();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    scalar_t alpha = params.alpha.get<scalar_t>();
    int p = params.grid.m();
    int q = params.grid.n();
    int64_t nb = params.nb();
    int64_t lookahead = params.lookahead();
    slate::Norm norm = params.norm();
    bool check = params.check() == 'y';
    bool ref = params.ref() == 'y';
    bool trace = params.trace() == 'y';
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    params.matrix.mark();
    params.matrixB.mark();

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    // Suppress norm from output; it's only for checks.
    params.norm.width( 0 );

    if (! run) {
        params.matrix.kind.set_default( "rand_dominant" );
        return;
    }

    slate::Options const opts =  {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    };

    // Error analysis applies in these norms.
    slate_assert(norm == Norm::One || norm == Norm::Inf || norm == Norm::Fro);

    // setup so B is m-by-n
    int64_t An  = (side == slate::Side::Left ? m : n);
    int64_t Am  = An;
    int64_t Bm  = m;
    int64_t Bn  = n;

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    // Matrix A: figure out local size.
    int64_t mlocA = num_local_rows_cols(Am, nb, myrow, p);
    int64_t nlocA = num_local_rows_cols(An, nb, mycol, q);
    int64_t lldA  = blas::max(1, mlocA); // local leading dimension of A

    // Matrix B: figure out local size.
    int64_t mlocB = num_local_rows_cols(Bm, nb, myrow, p);
    int64_t nlocB = num_local_rows_cols(Bn, nb, mycol, q);
    int64_t lldB  = blas::max(1, mlocB); // local leading dimension of B

    // Allocate ScaLAPACK data if needed.
    std::vector<scalar_t> A_data, B_data;
    if (ref || origin == slate::Origin::ScaLAPACK) {
        A_data.resize( lldA * nlocA );
    }

    slate::TriangularMatrix<scalar_t> A;
    slate::Matrix<scalar_t> B;
    if (origin != slate::Origin::ScaLAPACK) {
        // SLATE allocates CPU or GPU tiles.
        slate::Target origin_target = origin2target(origin);
        A = slate::TriangularMatrix<scalar_t>(
                uplo, diag, An, nb, p, q, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);

        B = slate::Matrix<scalar_t>(
                Bm, Bn, nb, p, q, MPI_COMM_WORLD);
        B.insertLocalTiles(origin_target);
    }
    else {
        // create SLATE matrices from the ScaLAPACK layouts
        A = slate::TriangularMatrix<scalar_t>::fromScaLAPACK(
                uplo, diag, An, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD);

        B_data.resize( lldB * nlocB );
        B = slate::Matrix<scalar_t>::fromScaLAPACK(
                Bm, Bn, &B_data[0], lldB, nb, p, q, MPI_COMM_WORLD);
    }

    slate::generate_matrix( params.matrix, A );
    slate::generate_matrix( params.matrixB, B );

    // Cholesky factor of A to get a well conditioned triangular matrix.
    // Even when we replace the diagonal with unit diagonal,
    // it seems to still be well conditioned.
    auto AH = slate::HermitianMatrix<scalar_t>( A );
    slate::potrf( AH, opts );

    // if check is required, copy test data
    std::vector< scalar_t > Bref_data;
    slate::Matrix<scalar_t> Bref;
    if (check || ref) {
        Bref_data.resize( lldB * nlocB );
        Bref = slate::Matrix<scalar_t>::fromScaLAPACK(
                   Bm, Bn, &Bref_data[0], lldB, nb, p, q, MPI_COMM_WORLD);
        slate::copy( B, Bref );
    }

    print_matrix( "A", A, params );
    print_matrix( "B", B, params );

    auto opA = A;
    if (transA == Op::Trans)
        opA = transpose(A);
    else if (transA == Op::ConjTrans)
        opA = conjTranspose(A);

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    double time = barrier_get_wtime(MPI_COMM_WORLD);

    //==================================================
    // Run SLATE test.
    // Solve AX = alpha B (left) or XA = alpha B (right).
    //==================================================
    if (side == slate::Side::Left) {
        if (params.routine == "trsmA")
            slate::trsmA(side, alpha, opA, B, opts);
        else
            slate::triangular_solve(alpha, opA, B, opts);
    }
    else if (side == slate::Side::Right) {
        if (params.routine == "trsmA")
            slate::trsmA(side, alpha, opA, B, opts);
        else
            slate::triangular_solve(alpha, B, opA, opts);
    }
    else
        throw slate::Exception("unknown side");
    // Using traditional BLAS/LAPACK name
    // slate::trsm(side, alpha, A, B, opts);

    time = barrier_get_wtime(MPI_COMM_WORLD) - time;

    if (trace) slate::trace::Trace::finish();

    // Compute and save timing/performance
    double gflop = blas::Gflop<scalar_t>::trsm(side, m, n);
    params.time() = time;
    params.gflops() = gflop / time;

    print_matrix( "B_out", B, params );

    if (check) {
        //==================================================
        // Test results by checking the residual
        //
        //      || B - 1/alpha AX ||_1
        //     ------------------------ < epsilon
        //      || A ||_1 * N
        //
        //==================================================

        // get norms of the original data
        // todo: add TriangularMatrix norm
        auto AZ = static_cast< slate::TrapezoidMatrix<scalar_t> >( opA );
        real_t A_norm = slate::norm(norm, AZ);

        slate::trmm(side, one/alpha, opA, B);
        slate::add(-one, Bref, one, B);
        real_t error = slate::norm(norm, B);
        error = error / (Am * A_norm);
        params.error() = error;

        // Allow 3*eps; complex needs 2*sqrt(2) factor; see Higham, 2002, sec. 3.6.
        real_t eps = std::numeric_limits<real_t>::epsilon();
        params.okay() = (params.error() <= 3*eps);
    }

    if (ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            // comparison with reference routine from ScaLAPACK for timing only

            // BLACS/MPI variables
            int ictxt, p_, q_, myrow_, mycol_, info;
            int A_desc[9], B_desc[9], Bref_desc[9];
            int mpi_rank_ = 0, nprocs = 1;

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

            scalapack_descinit(A_desc, Am, An, nb, nb, 0, 0, ictxt, mlocA, &info);
            slate_assert(info == 0);

            scalapack_descinit(B_desc, Bm, Bn, nb, nb, 0, 0, ictxt, mlocB, &info);
            slate_assert(info == 0);

            scalapack_descinit(Bref_desc, Bm, Bn, nb, nb, 0, 0, ictxt, mlocB, &info);
            slate_assert(info == 0);

            copy( A, &A_data[0], A_desc );

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            time = barrier_get_wtime(MPI_COMM_WORLD);
            scalapack_ptrsm(side2str(side), uplo2str(uplo), op2str(transA), diag2str(diag),
                            m, n, alpha,
                            &A_data[0], 1, 1, A_desc,
                            &Bref_data[0], 1, 1, Bref_desc);
            time = barrier_get_wtime(MPI_COMM_WORLD) - time;

            print_matrix( "Bref", Bref, params );

            params.ref_time() = time;
            params.ref_gflops() = gflop / time;

            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering.
        #else  // not SLATE_HAVE_SCALAPACK
            if (mpi_rank == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }
}

// -----------------------------------------------------------------------------
void test_trsm(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_trsm_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_trsm_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_trsm_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_trsm_work<std::complex<double>> (params, run);
            break;
    }
}
