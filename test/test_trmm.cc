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
#include "grid_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template<typename scalar_t>
void test_trmm_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using slate::Op;
    using slate::Norm;

    // Constants
    const scalar_t zero = 0.0, one = 1.0;

    // get & mark input values
    slate::Side side = params.side();
    slate::Uplo uplo = params.uplo();
    slate::Op transA = params.transA();
    slate::Diag diag = params.diag();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs();
    scalar_t alpha = params.alpha.get<scalar_t>();
    slate::Op transB = params.transB();
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

    // Suppress norm, nrhs from output; they're only for checks.
    params.norm.width( 0 );
    params.nrhs.width( 0 );

    if (! run) {
        return;
    }

    slate::Options const opts =  {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    };

    // Error analysis applies in these norms.
    slate_assert(norm == Norm::One || norm == Norm::Inf || norm == Norm::Fro);

    // setup so op(B) is m-by-n
    int64_t An = (side == slate::Side::Left ? m : n);
    int64_t Am = An;
    int64_t Bm = (transB == slate::Op::NoTrans ? m : n);
    int64_t Bn = (transB == slate::Op::NoTrans ? n : m);

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
        B_data.resize( lldB * nlocB );
    }

    slate::TriangularMatrix<scalar_t> A;
    slate::Matrix<scalar_t> B;
    slate::Target origin_target = origin2target(origin);
    if (origin != slate::Origin::ScaLAPACK) {
        // SLATE allocates CPU or GPU tiles.
        A = slate::TriangularMatrix<scalar_t>(uplo, diag, An, nb, p, q, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);

        B = slate::Matrix<scalar_t>(Bm, Bn, nb, p, q, MPI_COMM_WORLD);
        B.insertLocalTiles(origin_target);
    }
    else {
        // Create SLATE matrices from the ScaLAPACK layouts.
        A = slate::TriangularMatrix<scalar_t>::fromScaLAPACK(
                uplo, diag, An, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        B = slate::Matrix<scalar_t>::fromScaLAPACK(
                Bm, Bn, &B_data[0], lldB, nb, p, q, MPI_COMM_WORLD);
    }

    generate_matrix( params.matrix, A );
    generate_matrix( params.matrixB, B );

    #ifdef SLATE_HAVE_SCALAPACK
        // if reference run is required, copy test data.
        std::vector<scalar_t> Bref_data;
        if (ref) {
            Bref_data.resize( B_data.size() );
            auto Bref = slate::Matrix<scalar_t>::fromScaLAPACK(
                            Bm, Bn, &Bref_data[0], lldB, nb, p, q, MPI_COMM_WORLD);
            slate::copy( B, Bref );
        }
    #endif

    // Keep the original untransposed A matrix,
    // and make a shallow copy of it for transposing.
    auto opA = A;
    if (transA == Op::Trans)
        opA = transpose(A);
    else if (transA == Op::ConjTrans)
        opA = conjTranspose(A);

    if (transB == Op::Trans)
        B = transpose(B);
    else if (transB == Op::ConjTrans)
        B = conjTranspose(B);

    // If check run, perform first half of SLATE residual check.
    slate::Matrix<scalar_t> X, X2, Y;
    if (check && ! ref) {
        X = slate::Matrix<scalar_t>( n, nrhs, nb, p, q, MPI_COMM_WORLD );
        X.insertLocalTiles(origin_target);
        X2 = slate::Matrix<scalar_t>( n, nrhs, nb, p, q, MPI_COMM_WORLD );
        X2.insertLocalTiles(origin_target);
        Y = slate::Matrix<scalar_t>( m, nrhs, nb, p, q, MPI_COMM_WORLD );
        Y.insertLocalTiles(origin_target);
        MatrixParams mp;
        mp.kind.set_default( "rand" );
        generate_matrix( mp, X );

        if (side == slate::Side::Left ) {
            // Compute Y = alpha A * (B * X).
            // Y = B * X;
            slate::multiply( one, B, X, zero, Y, opts );
            // Y = alpha * A * Y;
            slate::triangular_multiply( alpha, opA, Y, opts );
        }
        else if (side == slate::Side::Right) {
            // Compute Y = alpha B * (A * X).
            slate::copy( X, X2 );
            // X2 = A * X2;
            slate::triangular_multiply( one, opA, X2, opts );
            // Y = alpha * B * X2;
            slate::multiply( alpha, B, X2, zero, Y, opts );
        }
        else
            throw slate::Exception("unknown side");
    }

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    double time = barrier_get_wtime(MPI_COMM_WORLD);

    //==================================================
    // Run SLATE test.
    // B = alpha AB (left) or B = alpha BA (right).
    //==================================================
    if (side == slate::Side::Left)
        slate::triangular_multiply(alpha, opA, B, opts);
    else if (side == slate::Side::Right)
        slate::triangular_multiply(alpha, B, opA, opts);
    else
        throw slate::Exception("unknown side");
    // Using traditional BLAS/LAPACK name
    // slate::trmm(side, alpha, A, B, opts);

    time = barrier_get_wtime(MPI_COMM_WORLD) - time;

    if (trace) slate::trace::Trace::finish();

    // compute and save timing/performance
    double gflop = blas::Gflop<scalar_t>::trmm(side, m, n);
    params.time() = time;
    params.gflops() = gflop / time;

    if (check && ! ref) {
        // SLATE residual check.
        // Check error, B*X - Y.
        real_t y_norm = slate::norm( norm, Y, opts );
        // Y = B * X - Y
        slate::multiply( one, B, X, -one, Y );
        // error = norm( Y ) / y_norm
        real_t error = slate::norm( norm, Y, opts )/y_norm;
        params.error() = error;

        // Allow 3*eps; complex needs 2*sqrt(2) factor; see Higham, 2002, sec. 3.6.
        real_t eps = std::numeric_limits<real_t>::epsilon();
        params.okay() = (params.error() <= 3*eps);
    }

    if (ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            // comparison with reference routine from ScaLAPACK
            // BLACS/MPI variables
            int ictxt, p_, q_, myrow_, mycol_, info;
            int A_desc[9], B_desc[9], Bref_desc[9];
            int mpi_rank_ = 0, nprocs = 1;

            // initialize BLACS and ScaLAPACK
            Cblacs_pinfo(&mpi_rank_, &nprocs);
            slate_assert( mpi_rank == mpi_rank_ );
            slate_assert(p*q <= nprocs);
            Cblacs_get(-1, 0, &ictxt);
            Cblacs_gridinit(&ictxt, "Col", p, q);
            Cblacs_gridinfo(ictxt, &p_, &q_, &myrow_, &mycol_);
            slate_assert(p == p_ && q == q_);
            slate_assert( myrow == myrow_ );
            slate_assert( mycol == mycol_ );

            scalapack_descinit(A_desc, Am, An, nb, nb, 0, 0, ictxt, mlocA, &info);
            slate_assert(info == 0);

            scalapack_descinit(B_desc, Bm, Bn, nb, nb, 0, 0, ictxt, mlocB, &info);
            slate_assert(info == 0);

            scalapack_descinit(Bref_desc, Bm, Bn, nb, nb, 0, 0, ictxt, mlocB, &info);
            slate_assert(info == 0);

            if (origin != slate::Origin::ScaLAPACK) {
                // Copy SLATE result back from GPU or CPU tiles.
                copy( A, &A_data[0], A_desc );
                copy( B, &B_data[0], B_desc );
            }

            // allocate workspace for norms
            std::vector<real_t> worklantr(std::max(mlocA, nlocA));
            std::vector<real_t> worklange(std::max(mlocB, nlocB));

            // get norms of the original data
            real_t A_norm = scalapack_plantr(
                norm2str(norm), uplo2str(uplo), diag2str(diag), Am, An, &A_data[0],
                1, 1, A_desc, &worklantr[0]);
            real_t B_orig_norm = scalapack_plange(
                norm2str(norm), Bm, Bn, &Bref_data[0], 1, 1, B_desc, &worklange[0]);

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            time = barrier_get_wtime(MPI_COMM_WORLD);
            scalapack_ptrmm(side2str(side), uplo2str(uplo), op2str(transA), diag2str(diag),
                            m, n, alpha,
                            &A_data[0], 1, 1, A_desc,
                            &Bref_data[0], 1, 1, Bref_desc);
            time = barrier_get_wtime(MPI_COMM_WORLD) - time;

            // Local operation: error = Bref_data - B_data
            blas::axpy(Bref_data.size(), -1.0, &B_data[0], 1, &Bref_data[0], 1);

            // norm(Bref_data - B_data)
            real_t B_diff_norm = scalapack_plange(norm2str(norm), Bm, Bn, &Bref_data[0],
                                                  1, 1, Bref_desc, &worklange[0]);

            real_t error = B_diff_norm
                         / (sqrt(real_t(Am) + 2) * std::abs(alpha) * A_norm * B_orig_norm);

            params.ref_time() = time;
            params.ref_gflops() = gflop / time;
            params.error() = error;

            // Allow 3*eps; complex needs 2*sqrt(2) factor; see Higham, 2002, sec. 3.6.
            real_t eps = std::numeric_limits<real_t>::epsilon();
            params.okay() = (params.error() <= 3*eps);

            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering
        #else  // not SLATE_HAVE_SCALAPACK
            if (mpi_rank == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }
}

// -----------------------------------------------------------------------------
void test_trmm(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_trmm_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_trmm_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_trmm_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_trmm_work<std::complex<double>> (params, run);
            break;
    }
}
