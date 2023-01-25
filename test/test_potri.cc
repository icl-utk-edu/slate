// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"
#include "blas/flops.hh"
#include "lapack/flops.hh"
#include "print_matrix.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"
#include "grid_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_potri_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;

    // Constants
    const scalar_t zero = 0;
    const scalar_t one  = 1;

    // get & mark input values
    slate::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs();
    int p = params.grid.m();
    int q = params.grid.n();
    int64_t nb = params.nb();
    int64_t lookahead = params.lookahead();
    bool ref_only = params.ref() == 'o';
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    SLATE_UNUSED(verbose);
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    params.matrix.mark();
    params.matrixB.mark();

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    // Suppress nrhs from output; it's only for checks.
    params.nrhs.width( 0 );

    if (! run) {
        params.matrix.kind.set_default( "rand_dominant" );
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

    // Matrix A: figure out local size.
    int64_t mlocA = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocA = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldA  = blas::max(1, mlocA); // local leading dimension of A

    // Allocate ScaLAPACK data if needed.
    std::vector<scalar_t> A_data;
    if (ref || origin != slate::Origin::Devices) {
        A_data.resize( lldA * nlocA );
    }

    // todo: work-around to initialize BaseMatrix::num_devices_
    slate::HermitianMatrix<scalar_t> A0(uplo, n, nb, p, q, MPI_COMM_WORLD);

    slate::HermitianMatrix<scalar_t> A;
    if (origin == slate::Origin::Devices) {
        // SLATE allocates CPU or GPU tiles.
        slate::Target origin_target = origin2target(origin);
        A = slate::HermitianMatrix<scalar_t>(uplo, n, nb, p, q, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);
    }
    else {
        // Create SLATE matrix from the ScaLAPACK layouts
        A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                uplo, n, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
    }

    slate::generate_matrix(params.matrix, A);

    // if check (or ref) is required, copy test data
    slate::HermitianMatrix<scalar_t> Aref;
    std::vector<scalar_t> Aref_data;
    if (check || ref) {
        Aref_data.resize( lldA * nlocA );
        Aref = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                   uplo, n, &Aref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        slate::copy( A, Aref );
    }

    double gflop = 0.0;
    // 1/3 n^3 + 1/2 n^2 flops for Cholesky factorization
    // 2/3 n^3 + 1/2 n^2 flops for Cholesky inversion
    gflop = lapack::Gflop<scalar_t>::potrf(n)
          + lapack::Gflop<scalar_t>::potri(n);

    if (! ref_only) {

        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        double time = barrier_get_wtime(MPI_COMM_WORLD);

        //==================================================
        // Run SLATE test.
        //==================================================
        // factor then invert; measure time for both
        slate::chol_factor(A, opts);
        slate::chol_inverse_using_factor(A, opts);
        // Using traditional BLAS/LAPACK name
        // slate::potrf(A, opts);
        // slate::potri(A, opts);

        time = barrier_get_wtime(MPI_COMM_WORLD) - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time;
        params.gflops() = gflop / time;
    }

    // Test using only SLATE routines for a residual check.
    if (check) {
        //==================================================
        // Check || X - A^{-1} A X || / (n || X ||) < tol
        //==================================================
        slate::Matrix<scalar_t> X, Y;
        X = slate::Matrix<scalar_t>( n, nrhs, nb, p, q, MPI_COMM_WORLD);
        Y = slate::Matrix<scalar_t>( n, nrhs, nb, p, q, MPI_COMM_WORLD);
        slate::Target origin_target = origin2target(origin);
        X.insertLocalTiles(origin_target);
        Y.insertLocalTiles(origin_target);
        generate_matrix( params.matrixB, X );  // rand
        real_t normX = slate::norm( slate::Norm::One, X );
        slate::multiply(  one, Aref, X, zero, Y );  // hemm: Y = Aref X;
        slate::multiply( -one, A, Y, one, X );      // hemm: X = X - A^{-1} Y
        real_t error = slate::norm( slate::Norm::One, X ) / (n * normX);
        params.error() = error;
        real_t tol = params.tol() * std::numeric_limits<real_t>::epsilon();
        params.okay() = (params.error() <= tol);
    }

    // TODO: Enable the SLATE_HAVE_SCALAPACK check after a SLATE hehemm
    // routine is created, or after a SLATE symmetrize routine is
    // created to transform a Hermitian/Symmetric matrix into a general matrix.
    if (check) {
        #if 0 // #ifdef SLATE_HAVE_SCALAPACK

            // BLACS/MPI variables
            int ictxt, p_, q_, myrow_, mycol_, info;
            int A_desc[9], Aref_desc[9];
            int Cchk_desc[9];
            int mpi_rank_ = 0, nprocs = 1;

            // initialize BLACS and ScaLAPACK
            Cblacs_pinfo(&mpi_rank_, &nprocs);
            slate_assert( mpi_rank == mpi_rank_ );
            slate_assert(p*q <= nprocs);
            Cblacs_get(-1, 0, &ictxt);
            Cblacs_gridinit(&ictxt, "Col", p, q);
            Cblacs_gridinfo(ictxt, &p_, &q_, &myrow_, &mycol_);
            slate_assert( p == p_ );
            slate_assert( q == q_ );
            slate_assert( myrow == myrow_ );
            slate_assert( mycol == mycol_ );

            scalapack_descinit(A_desc, n, n, nb, nb, 0, 0, ictxt, mlocA, &info);
            slate_assert(info == 0);

            scalapack_descinit(Aref_desc, n, n, nb, nb, 0, 0, ictxt, mlocA, &info);
            slate_assert(info == 0);

            scalapack_descinit(Cchk_desc, n, n, nb, nb, 0, 0, ictxt, mlocA, &info);
            slate_assert(info == 0);

            // Check  || I - inv(A)*A || / ( || A || * N ) <=  tol * eps

            if (origin != slate::Origin::ScaLAPACK) {
                // Copy SLATE result back from GPU or CPU tiles.
                copy(A, &A_data[0], A_desc);
            }

            // make diagonal real
            for (int64_t i = 0; i < Aref.nt(); ++i) {
                if (Aref.tileIsLocal(i, i)) {
                    auto T = Aref(i, i);
                    for (int ii = 0; ii < Aref.tileMb(i); ++ii) {
                        T.at(ii, ii) = std::real( T.at(ii, ii) );
                    }
                }
            }

            // Make Cchk_data into an identity matrix to check the result of
            // multiplying A and A_inv.
            // Cchk_data starts with the same size/dimensions as A_data.
            std::vector<scalar_t> Cchk_data( A_data.size() );
            scalapack_plaset("All", n, n, zero, one, &Cchk_data[0], 1, 1, Cchk_desc);

            // Cchk_data has been setup as an identity matrix; Cchk_data = C_chk - inv(A)*A
            // A should have real diagonal. potrf and potri ignore the img part on the diagonal
            scalapack_phemm("Left", uplo2str(uplo), n, n, -one,
                            &A_data[0], 1, 1, A_desc,
                            &Aref_data[0], 1, 1, Aref_desc, one,
                            &Cchk_data[0], 1, 1, Cchk_desc);

            // Norm of Cchk_data ( = I - inv(A) * A )
            // allocate work space for lange and lanhe
            int lcm = scalapack_ilcm(&p, &q);
            int ldw = nb*slate::ceildiv(int(slate::ceildiv(nlocA, nb)), (lcm / p));
            int lwork = std::max(n, 2*mlocA + nlocA + ldw);
            std::vector<real_t> worknorm(lwork);
            real_t C_norm = scalapack_plange(
                                "One", n, n, &Cchk_data[0], 1, 1, Cchk_desc, &worknorm[0]);

            real_t A_inv_norm = scalapack_planhe(
                                    "One", uplo2str(A.uplo()),
                                    n, &A_data[0], 1, 1, A_desc, &worknorm[0]);

            double residual = C_norm / (A_norm * n * A_inv_norm);
            params.error() = residual;

            real_t tol = params.tol() * std::numeric_limits<real_t>::epsilon();
            params.okay() = (params.error() <= tol);

            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering
        #else  // not SLATE_HAVE_SCALAPACK
            if (mpi_rank == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }

    if (ref) {
        // todo: call to reference potri from ScaLAPACK not implemented
    }
}

// -----------------------------------------------------------------------------
void test_potri(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_potri_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_potri_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_potri_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_potri_work<std::complex<double>> (params, run);
            break;
    }
}
