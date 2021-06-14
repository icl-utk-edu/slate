// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
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
#include "aux/Debug.hh"
#include "grid_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#define SLATE_HAVE_SCALAPACK
//------------------------------------------------------------------------------
template <typename scalar_t>
void test_posv_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;

    // get & mark input values
    slate::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs();
    int64_t p = params.grid.m();
    int64_t q = params.grid.n();
    int64_t nb = params.nb();
    int64_t lookahead = params.lookahead();
    bool ref_only = params.ref() == 'o';
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    slate::Dist dev_dist = params.dev_dist();
    params.matrix.mark();
    params.matrixB.mark();

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    if (params.routine == "posvMixed") {
        params.iters();
    }

    if (! run) {
        params.matrix.kind.set_default( "rand_dominant" );
        return;
    }

    slate::Options const opts =  {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    };

    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (target != slate::Target::Devices && dev_dist != slate::Dist::Col) {
        if (mpi_rank == 0)
            printf("skipping: dev_dist = Row applies only to target devices\n");
        return;
    }

    if (params.routine == "posvMixed") {
        if (! std::is_same<real_t, double>::value) {
            if (mpi_rank == 0) {
                printf("Unsupported mixed precision\n");
            }
            return;
        }
    }

    // Constants
    const scalar_t one = 1.0;
    const int izero = 0, ione = 1;

    // Local values
    int myrow, mycol;
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    // matrix A, figure out local size, allocate, create descriptor, initialize
    int64_t mlocA = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocA = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldA  = blas::max(1, mlocA); // local leading dimension of A
    std::vector<scalar_t> A_data(lldA*nlocA);

    // matrix B, figure out local size, allocate, create descriptor, initialize
    int64_t mlocB = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocB = num_local_rows_cols(nrhs, nb, mycol, q);
    int64_t lldB  = blas::max(1, mlocB); // local leading dimension of B
    std::vector<scalar_t> B_data(lldB*nlocB);

    // todo: work-around to initialize BaseMatrix::num_devices_
    slate::HermitianMatrix<scalar_t> A0(uplo, n, nb, p, q, MPI_COMM_WORLD);

    slate::HermitianMatrix<scalar_t> A;
    slate::Matrix<scalar_t> B, X;
    std::vector<scalar_t> X_data;
    if (origin != slate::Origin::ScaLAPACK) {
        if (dev_dist == slate::Dist::Row && target == slate::Target::Devices) {
            // slate_assert(target == slate::Target::Devices);
            // todo: doesn't work when lookahead is greater than 2
            // slate_assert(lookahead < 3);
            // std::function<int64_t (int64_t i)> tileMb = [nrhs, nb] (int64_t i)
            //    { return (i + 1)*mb > nrhs ? nrhs%mb : mb; };
            std::function<int64_t (int64_t j)> tileNb = [n, nb] (int64_t j) {
                return (j + 1)*nb > n ? n%nb : nb;
            };

            std::function<int (std::tuple<int64_t, int64_t> ij)>
            tileRank = [p, q](std::tuple<int64_t, int64_t> ij) {
                int64_t i = std::get<0>(ij);
                int64_t j = std::get<1>(ij);
                return int(i%p + (j%q)*p);
            };

            int num_devices = blas::get_device_count();
            slate_assert(num_devices > 0);

            std::function<int (std::tuple<int64_t, int64_t> ij)>
            tileDevice = [p, num_devices](std::tuple<int64_t, int64_t> ij) {
                int64_t i = std::get<0>(ij);
                return int(i/p)%num_devices;
            };

            A = slate::HermitianMatrix<scalar_t>(
                    uplo, n, tileNb, tileRank, tileDevice, MPI_COMM_WORLD);
            B = slate::Matrix<scalar_t>(
                    n, nrhs, tileNb, tileNb, tileRank, tileDevice, MPI_COMM_WORLD);
        }
        else {
            // A
            A = slate::HermitianMatrix<scalar_t>(
                    uplo, n, nb, p, q, MPI_COMM_WORLD);
            // B
            B = slate::Matrix<scalar_t>(
                    n, nrhs, nb, p, q, MPI_COMM_WORLD);
        }

        // Copy local ScaLAPACK data to GPU or CPU tiles.
        slate::Target origin_target = origin2target(origin);
        A.insertLocalTiles(origin_target);

        B.insertLocalTiles(origin_target);

        if (params.routine == "posvMixed") {
            if (std::is_same<real_t, double>::value) {
                X_data.resize(lldB*nlocB);
                X = slate::Matrix<scalar_t>(n, nrhs, nb, p, q, MPI_COMM_WORLD);
                X.insertLocalTiles(origin_target);
            }
        }
    }
    else {
        // Create SLATE matrix from the ScaLAPACK layouts
        A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(uplo, n, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        B = slate::Matrix<scalar_t>::fromScaLAPACK(n, nrhs, &B_data[0], lldB, nb, p, q, MPI_COMM_WORLD);
        if (params.routine == "posvMixed") {
            if (std::is_same<real_t, double>::value) {
                X_data.resize(lldB*nlocB);
                X = slate::Matrix<scalar_t>::fromScaLAPACK(n, nrhs, &X_data[0], lldB, nb, p, q, MPI_COMM_WORLD);
            }
        }
    }

    slate::generate_matrix(params.matrix, A);
    slate::generate_matrix(params.matrixB, B);
    if (verbose >= 2) {
        print_matrix("A", A);
        print_matrix("B", B);
    }

    // if check is required, copy test data and create a descriptor for it
    std::vector<scalar_t> Aref_data(lldA*nlocA);
    std::vector<scalar_t> Bref_data(lldB*nlocB);
    std::vector<scalar_t> B_orig;
    slate::HermitianMatrix<scalar_t> Aref;
    slate::Matrix<scalar_t> Bref;
    if (check || ref) {
        // SLATE matrix wrappers for the reference data
        Aref = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                   uplo, n, &Aref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        Bref = slate::Matrix<scalar_t>::fromScaLAPACK(
                   n, nrhs, &Bref_data[0], lldB, nb, p, q, MPI_COMM_WORLD);

        slate::copy( A, Aref );
        slate::copy( B, Bref );
        if (origin != slate::Origin::ScaLAPACK) {
            A_data = Aref_data;
            B_data = Bref_data;
        }

        if (check && ref)
            B_orig = Bref_data;
    }

    int iters = 0;

    double gflop;
    if (params.routine == "potrf")
        gflop = lapack::Gflop<scalar_t>::potrf(n);
    else if (params.routine == "potrs")
        gflop = lapack::Gflop<scalar_t>::potrs(n, nrhs);
    else
        gflop = lapack::Gflop<scalar_t>::posv(n, nrhs);

    if (! ref_only) {
        if (params.routine == "potrs") {
            // Factor matrix A.
            slate::chol_factor(A, opts);
            // Using traditional BLAS/LAPACK name
            // slate::potrf(A, opts);
        }

        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        double time = barrier_get_wtime(MPI_COMM_WORLD);
        //==================================================
        // Run SLATE test.
        // One of:
        // potrf: Factor A = LL^H or A = U^H U.
        // potrs: Solve AX = B, after factoring A above.
        // posv:  Solve AX = B, including factoring A.
        //==================================================
        if (params.routine == "potrf") {
            slate::chol_factor(A, opts);
            // Using traditional BLAS/LAPACK name
            // slate::potrf(A, opts);
        }
        else if (params.routine == "potrs") {
            slate::chol_solve_using_factor(A, B, opts);
            // Using traditional BLAS/LAPACK name
            // slate::potrs(A, B, opts);
        }
        else if (params.routine == "posv") {
            slate::chol_solve(A, B, opts);
            // Using traditional BLAS/LAPACK name
            // slate::posv(A, B, opts);
        }
        else if (params.routine == "posvMixed") {
            if (std::is_same<real_t, double>::value) {
                slate::posvMixed(A, B, X, iters, opts);
            }
        }
        else {
            slate_error("Unknown routine!");
        }

        time = barrier_get_wtime(MPI_COMM_WORLD) - time;

        if (trace) slate::trace::Trace::finish();

        if (params.routine == "posvMixed") {
            params.iters() = iters;
        }

        // compute and save timing/performance
        params.time() = time;
        params.gflops() = gflop / time;
    }

    if (check) {
        //==================================================
        // Test results by checking the residual
        //
        //           || B - AX ||_1
        //     --------------------------- < tol * epsilon
        //      || A ||_1 * || X ||_1 * N
        //
        //==================================================

        if (params.routine == "potrf") {
            // Solve AX = B.
            slate::chol_solve_using_factor(A, B, opts);
            // Using traditional BLAS/LAPACK name
            // slate::potrs(A, B, opts);
        }

        // Norm of original matrix: || A ||_1
        real_t A_norm = slate::norm(slate::Norm::One, Aref);

        // Norm of updated-rhs/solution matrix: || X ||_1
        real_t X_norm;
        if (params.routine == "posvMixed")
            X_norm = slate::norm(slate::Norm::One, X);
        else
            X_norm = slate::norm(slate::Norm::One, B);

        // Bref_data -= Aref*B_data
        if (params.routine == "posvMixed") {
            if (std::is_same<real_t, double>::value)
                slate::multiply(-one, Aref, X, one, Bref);
            // Using traditional BLAS/LAPACK name
            // slate::hemm(slate::Side::Left, -one, Aref, X, one, Bref);
        }
        else {
            slate::multiply(-one, Aref, B, one, Bref);
            // Using traditional BLAS/LAPACK name
            // slate::hemm(slate::Side::Left, -one, Aref, B, one, Bref);
        }

        // Norm of residual: || B - AX ||_1
        real_t R_norm = slate::norm(slate::Norm::One, Bref);
        double residual = R_norm / (n*A_norm*X_norm);
        params.error() = residual;

        real_t tol = params.tol() * 0.5 * std::numeric_limits<real_t>::epsilon();
        params.okay() = (params.error() <= tol);
    }

    if (ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            // A comparison with a reference routine from ScaLAPACK for timing only
            // BLACS/MPI variables
            int ictxt, p_, q_, myrow_, mycol_, info;
            int A_desc[9], Aref_desc[9];
            int B_desc[9], Bref_desc[9];
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

            scalapack_descinit(A_desc, n, n, nb, nb, izero, izero, ictxt, mlocA, &info);
            slate_assert(info == 0);

            scalapack_descinit(B_desc, n, nrhs, nb, nb, izero, izero, ictxt, mlocB, &info);
            slate_assert(info == 0);

            scalapack_descinit(Aref_desc, n, n, nb, nb, izero, izero, ictxt, mlocA, &info);
            slate_assert(info == 0);

            scalapack_descinit(Bref_desc, n, nrhs, nb, nb, izero, izero, ictxt, mlocB, &info);
            slate_assert(info == 0);

            // set MKL num threads appropriately for parallel BLAS
            int omp_num_threads;
            #pragma omp parallel
            { omp_num_threads = omp_get_num_threads(); }
            int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);

            if (check) {
                // restore Bref_data
                Bref_data = B_orig;
                //scalapack_descinit(Bref_desc, n, nrhs, nb, nb, izero, izero, ictxt, mlocB, &info);
                //slate_assert(info == 0);
            }

            if (params.routine == "potrs") {
                // Factor matrix A.
                scalapack_ppotrf(uplo2str(uplo), n, &Aref_data[0], ione, ione, Aref_desc, &info);
                slate_assert(info == 0);
            }

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            double time = barrier_get_wtime(MPI_COMM_WORLD);
            if (params.routine == "potrf") {
                scalapack_ppotrf(uplo2str(uplo), n, &Aref_data[0], ione, ione, Aref_desc, &info);
            }
            else if (params.routine == "potrs") {
                scalapack_ppotrs(uplo2str(uplo), n, nrhs, &Aref_data[0], ione, ione, Aref_desc, &Bref_data[0], ione, ione, Bref_desc, &info);
            }
            else {
                scalapack_pposv(uplo2str(uplo), n, nrhs, &Aref_data[0], ione, ione, Aref_desc, &Bref_data[0], ione, ione, Bref_desc, &info);
            }
            slate_assert(info == 0);
            time = barrier_get_wtime(MPI_COMM_WORLD) - time;

            params.ref_time() = time;
            params.ref_gflops() = gflop / time;

            slate_set_num_blas_threads(saved_num_threads);

            if (verbose > 2) {
                if (origin == slate::Origin::ScaLAPACK) {
                    slate::Debug::diffLapackMatrices<scalar_t>(n, n, &A_data[0], lldA, &Aref_data[0], lldA, nb, nb);
                    if (params.routine != "potrf") {
                        slate::Debug::diffLapackMatrices<scalar_t>(n, nrhs, &B_data[0], lldB, &Bref_data[0], lldB, nb, nb);
                    }
                }
            }
            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering
        #else
            if (mpi_rank == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }
}

// -----------------------------------------------------------------------------
void test_posv(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_posv_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_posv_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_posv_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_posv_work<std::complex<double>> (params, run);
            break;
    }
}
