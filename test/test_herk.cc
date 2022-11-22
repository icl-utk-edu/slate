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
void test_herk_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using slate::Norm;

    // Constants
    const scalar_t zero = 0.0, one = 1.0;

    // get & mark input values
    slate::Uplo uplo = params.uplo();
    slate::Op transA = params.trans();
    int64_t n = params.dim.n();
    int64_t k = params.dim.k();
    real_t alpha = params.alpha.get<real_t>();
    real_t beta = params.beta.get<real_t>();
    int p = params.grid.m();
    int q = params.grid.n();
    int64_t nrhs = params.nrhs();
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

    if (! run)
        return;

    slate::Options const opts =  {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        // TODO fix gemmA on device
        {slate::Option::MethodGemm, slate::MethodGemm::GemmC}
    };

    // Error analysis applies in these norms.
    slate_assert(norm == Norm::One || norm == Norm::Inf || norm == Norm::Fro);

    // setup so op(A) is n-by-k
    int64_t Am = (transA == slate::Op::NoTrans ? n : k);
    int64_t An = (transA == slate::Op::NoTrans ? k : n);
    int64_t Cm = n;
    int64_t Cn = n;

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    // Matrix A: figure out local size.
    int64_t mlocA = num_local_rows_cols(Am, nb, myrow, p);
    int64_t nlocA = num_local_rows_cols(An, nb, mycol, q);
    int64_t lldA  = blas::max(1, mlocA); // local leading dimension of A

    // Matrix C: figure out local size.
    int64_t mlocC = num_local_rows_cols(Cm, nb, myrow, p);
    int64_t nlocC = num_local_rows_cols(Cn, nb, mycol, q);
    int64_t lldC  = blas::max(1, mlocC); // local leading dimension of C

    // Allocate ScaLAPACK data if needed.
    std::vector<scalar_t> A_data, C_data;
    if (ref || origin == slate::Origin::ScaLAPACK) {
        A_data.resize( lldA * nlocA );
        C_data.resize( lldC * nlocC );
    }

    slate::Matrix<scalar_t> A;
    slate::HermitianMatrix<scalar_t> C;
    slate::Target origin_target = origin2target(origin);
    if (origin != slate::Origin::ScaLAPACK) {
        // SLATE allocates CPU or GPU tiles.
        A = slate::Matrix<scalar_t>(Am, An, nb, p, q, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);

        C = slate::HermitianMatrix<scalar_t>(uplo, Cn, nb, p, q, MPI_COMM_WORLD);
        C.insertLocalTiles(origin_target);
    }
    else {
        // Create SLATE matrices from the ScaLAPACK layouts
        A = slate::Matrix<scalar_t>::fromScaLAPACK(
                Am, An, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        C = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                uplo, Cn, &C_data[0], lldC, nb, p, q, MPI_COMM_WORLD);
    }

    slate::generate_matrix( params.matrix, A );
    slate::generate_matrix( params.matrixB, C );

    #ifdef SLATE_HAVE_SCALAPACK
        // if reference run is required, copy test data.
        slate::HermitianMatrix<scalar_t> Cref;
        std::vector<scalar_t> Cref_data;
        if (ref) {
            Cref_data.resize( C_data.size() );
            Cref = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                       uplo, Cn, &Cref_data[0], lldC, nb, p, q, MPI_COMM_WORLD);
            slate::copy( C, Cref );
        }
    #endif

    // Keep the original untransposed A matrix,
    // and make a shallow copy of it for transposing.
    auto opA = A;
    if (transA == slate::Op::Trans)
        opA = transpose(A);
    else if (transA == slate::Op::ConjTrans)
        opA = conjTranspose(A);
    slate_assert(opA.mt() == C.mt());

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    // If check run, perform first half of SLATE residual check.
    slate::Matrix<scalar_t> X, Y, Z;
    if (check && ! ref) {
        X = slate::Matrix<scalar_t>( An, nrhs, nb, p, q, MPI_COMM_WORLD );
        X.insertLocalTiles(origin_target);
        Y = slate::Matrix<scalar_t>( Am, nrhs, nb, p, q, MPI_COMM_WORLD );
        Y.insertLocalTiles(origin_target);
        Z = slate::Matrix<scalar_t>( Am, nrhs, nb, p, q, MPI_COMM_WORLD);
        Z.insertLocalTiles(origin_target);
        MatrixParams mp;
        mp.kind.set_default( "rand" );
        generate_matrix( mp, X );

        // Compute Y = alpha A (A^H X) + (beta C X).
        // Y = beta C X
        slate::multiply( scalar_t(beta), C, X, zero, Y, opts );
        // Z = A^H X
        auto AH = conjTranspose( opA );
        slate::multiply( one, AH, X, zero, Z, opts );
        // Y = alpha A Z + Y
        slate::multiply( scalar_t(alpha), opA, Z, one, Y, opts );
    }

    double time = barrier_get_wtime(MPI_COMM_WORLD);

    //==================================================
    // Run SLATE test.
    // C = alpha A A^H + beta C.
    //==================================================
    slate::rank_k_update(alpha, opA, beta, C, opts);
    // Using traditional BLAS/LAPACK name
    // slate::herk(alpha, A, beta, C, opts);

    time = barrier_get_wtime(MPI_COMM_WORLD) - time;

    if (trace) slate::trace::Trace::finish();

    // Compute and save timing/performance
    double gflop = blas::Gflop<scalar_t>::herk(n, k);
    params.time() = time;
    params.gflops() = gflop / time;

    if (check && ! ref) {
        // SLATE residual check.
        // Check error, C*X - Y.
        real_t y_norm = slate::norm( norm, Y, opts );
        // Y = C * X - Y
        slate::multiply( one, C, X, -one, Y, opts );
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
            int A_desc[9], C_desc[9], Cref_desc[9];
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

            scalapack_descinit(C_desc, Cm, Cn, nb, nb, 0, 0, ictxt, mlocC, &info);
            slate_assert(info == 0);

            scalapack_descinit(Cref_desc, Cm, Cn, nb, nb, 0, 0, ictxt, mlocC, &info);
            slate_assert(info == 0);

            copy( A, &A_data[0], A_desc );
            copy( C, &C_data[0], C_desc );

            // allocate workspace for norms
            size_t ldw = nb*ceil(ceil(mlocC / (double) nb) / (scalapack_ilcm(&p, &q) / p));
            std::vector<real_t> worklansy(2*nlocC + mlocC + ldw);
            std::vector<real_t> worklange(std::max(mlocA, nlocA));

            // get norms of the original data
            real_t A_norm = scalapack_plange(norm2str(norm), Am, An, &A_data[0], 1, 1,
                                             A_desc, &worklange[0]);
            real_t C_orig_norm = scalapack_plansy(norm2str(norm), uplo2str(uplo), Cn,
                                                  &Cref_data[0], 1, 1, Cref_desc,
                                                  &worklansy[0]);

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            time = barrier_get_wtime(MPI_COMM_WORLD);
            scalapack_pherk(uplo2str(uplo), op2str(transA), n, k, alpha,
                            &A_data[0], 1, 1, A_desc, beta,
                            &Cref_data[0], 1, 1, Cref_desc);
            time = barrier_get_wtime(MPI_COMM_WORLD) - time;

            // local operation: error = Cref_data - C_data
            blas::axpy(Cref_data.size(), -1.0, &C_data[0], 1, &Cref_data[0], 1);

            // norm(Cref_data - C_data)
            real_t C_diff_norm = scalapack_plansy(norm2str(norm), uplo2str(uplo), Cn,
                                                  &Cref_data[0], 1, 1, Cref_desc,
                                                  &worklansy[0]);

            real_t error = C_diff_norm
                         / (sqrt(real_t(k) + 2) * std::abs(alpha) * A_norm * A_norm
                            + 2 * std::abs(beta) * C_orig_norm);

            params.ref_time() = time;
            params.ref_gflops() = gflop / time;
            params.error() = error;

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
void test_herk(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_herk_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_herk_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_herk_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_herk_work<std::complex<double>> (params, run);
            break;
    }
}
