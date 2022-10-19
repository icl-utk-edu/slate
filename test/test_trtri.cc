// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"
#include "blas/flops.hh"
#include "lapack/flops.hh"

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
template <typename scalar_t>
void test_trtri_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;

    // Constants
    const scalar_t one = 1.0, zero = 0.0;

    // get & mark input values
    slate::Uplo uplo = params.uplo();
    slate::Diag diag = params.diag();
    int64_t n = params.dim.n();
    int p = params.grid.m();
    int q = params.grid.n();
    int64_t nb = params.nb();
    int64_t lookahead = params.lookahead();
    bool ref_only = params.ref() == 'o';
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    bool trace = params.trace() == 'y';
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    params.matrix.mark();

    // mark non-standard output values
    params.time();
    params.gflops();
    //params.ref_time();
    //params.ref_gflops();

    if (! run) {
        // AH Hermitian matrix needs to be rand_dominant (or other SPD)
        // for the Cholesky factorization.
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
    int64_t lldA  = blas::max(1, mlocA); // local leading dimension of A_band
    std::vector<scalar_t> A_data(lldA*nlocA);

    // todo: work-around to initialize BaseMatrix::num_devices_
    slate::Matrix<scalar_t> A0(n, n, nb, p, q, MPI_COMM_WORLD);

    // Cholesky factor of AH to get a well conditioned triangular matrix
    // Even when we replace the diagonal with unit diagonal,
    // it seems to still be well conditioned.
    auto AH = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                  uplo, n, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
    print_matrix( "AH", AH, params );

    slate::generate_matrix(params.matrix, AH);
    slate::potrf(AH);

    // Setup SLATE triangular matrix A from Cholesky factor in AH.
    slate::TriangularMatrix<scalar_t> A( diag, AH );

    print_matrix( "A", A, params );

    // if check (or ref) is required, copy test data and create a descriptor for it
    std::vector<scalar_t> Aref_data;
    if (check || ref) {
        Aref_data = A_data;
    }

    // If check is required, record the norm of the original triangular matrix
    real_t A_norm = 0.0;
    if (check) {
        // todo: add TriangularMatrix norm
        auto AZ = static_cast< slate::TrapezoidMatrix<scalar_t> >( A );
        slate::Norm norm = slate::Norm::One;
        A_norm = slate::norm(norm, AZ);
    }

    // if check is required, create matrix to hold the result of multiplying A and A_inv
    std::vector<scalar_t> Cchk_data;
    if (check) {
        // Cchk_data starts with the same size/dimensions as A_data
        Cchk_data = A_data;
    }

    // Create SLATE matrix from the ScaLAPACK layouts
    slate::Matrix<scalar_t> C;
    if (check) {
        C = slate::Matrix<scalar_t>::fromScaLAPACK(
                n, n, &Cchk_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
    }

    // trtri flop count
    double gflop = lapack::Gflop<scalar_t>::trtri(n);

    if (! ref_only) {

        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        double time = barrier_get_wtime(MPI_COMM_WORLD);

        //==================================================
        // Run SLATE test.
        //==================================================
        // invert and measure time
        slate::trtri(A, opts);

        time = barrier_get_wtime(MPI_COMM_WORLD) - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time;
        params.gflops() = gflop / time;

        print_matrix( "Ainv", A, params );
    }

    if (check) {
        #ifdef SLATE_HAVE_SCALAPACK
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

            //==================================================
            // Check  || I - inv(A)*A || / ( || A || * N ) <=  tol * eps
            //==================================================
            if (origin != slate::Origin::ScaLAPACK) {
                // Copy data back from CPU/GPUs to ScaLAPACK layout
                copy(A, &A_data[0], A_desc);
            }

            // Setup full nxn SLATE matrix in Aref on CPU pointing to ScaLAPACK
            // data in Aref_data
            auto Aref = slate::Matrix<scalar_t>::fromScaLAPACK(
                            n, n, &Aref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
            print_matrix( "Aref_", Aref, params );

            // Zero out unused opposite lower/upper triangle.
            // For diag=unit, set diag to 1.0.
            // todo: how can slate::set work on A( 1:n-1, 2:n )? Does slicing work?
            slate::Uplo lo_up = (uplo == slate::Uplo::Lower
                                 ? slate::Uplo::Upper
                                 : slate::Uplo::Lower);
            if (uplo == slate::Uplo::Lower) {
                if (diag == slate::Diag::Unit) {
                    scalapack_plaset( uplo2str(lo_up), n, n, zero, one,
                                      &Aref_data[0], 1, 1, Aref_desc );
                }
                else {
                    scalapack_plaset( uplo2str(lo_up), n-1, n-1, zero, zero,
                                      &Aref_data[0], 1, 2, Aref_desc );
                }
            }
            else {
                if (diag == slate::Diag::Unit) {
                    scalapack_plaset( uplo2str(lo_up), n, n, zero, one,
                                      &Aref_data[0], 1, 1, Aref_desc );
                }
                else {
                    scalapack_plaset( uplo2str(lo_up), n-1, n-1, zero, zero,
                                      &Aref_data[0], 2, 1, Aref_desc );
                }
            }
            print_matrix( "Aref", Aref, params );

            // Aref_data = inv(A) * Aref_data
            scalapack_ptrmm("left", uplo2str(uplo), "notrans", diag2str(diag),
                            n, n, one,
                            &A_data[0], 1, 1, A_desc,
                            &Aref_data[0], 1, 1, Aref_desc);

            // Make Cchk_data into an identity matrix
            slate::set(zero, one, C);
            print_matrix( "C", C, params );

            // C = C - A; note Aref is a general n-by-n SLATE matrix pointing to Aref_data data
            slate::add(-one, Aref, one, C);
            print_matrix( "Cdiff", C, params );

            // Norm of Cchk_data ( = I - inv(A) * A )
            //// real_t C_norm = slate::norm(slate::norm::One, C);
            std::vector<real_t> worklange(n);
            real_t C_norm
                = scalapack_plange(
                      "1", n, n, &Cchk_data[0], 1, 1, Cchk_desc, &worklange[0]);

            double residual = C_norm / (A_norm * n);
            params.error() = residual;

            real_t tol = params.tol() * std::numeric_limits<real_t>::epsilon();
            params.okay() = (params.error() <= tol);
            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering
        #else  // not SLATE_HAVE_SCALAPACK
            SLATE_UNUSED( A_norm );
            SLATE_UNUSED( one );
            SLATE_UNUSED( zero );
            SLATE_UNUSED( origin );
            if (mpi_rank == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }

    if (ref) {
        // todo: call to reference trtri from ScaLAPACK not implemented
    }
}

// -----------------------------------------------------------------------------
void test_trtri(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_trtri_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_trtri_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_trtri_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_trtri_work<std::complex<double>> (params, run);
            break;
    }
}
