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
#include "auxiliary/Debug.hh"
#include "grid_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_posv_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;

    // Constants
    const scalar_t one = 1.0;

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
    bool hold_local_workspace = params.hold_local_workspace() == 'y';
    int verbose = params.verbose();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    slate::Dist dev_dist = params.dev_dist();
    slate::TileReleaseStrategy tile_release_strategy = params.tile_release_strategy();
    params.matrix.mark();
    params.matrixB.mark();
    slate::Method methodTrsm = params.method_trsm();
    slate::Method methodHemm = params.method_hemm();

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();
    params.time2();
    params.time2.name( "trs time (s)" );
    params.time2.width( 12 );
    params.gflops2();
    params.gflops2.name( "trs gflop/s" );

    bool do_potrs = (
        (check && params.routine == "potrf") || params.routine == "potrs");

    if (params.routine == "posvMixed") {
        params.iters();
    }

    if (! run) {
        params.matrix.kind.set_default( "rand_dominant" );
        return;
    }

    slate::Options const opts =  {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::TileReleaseStrategy, tile_release_strategy},
        {slate::Option::HoldLocalWorkspace, hold_local_workspace},
        {slate::Option::MethodTrsm, methodTrsm},
        {slate::Option::MethodHemm, methodHemm},
    };

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    if (target != slate::Target::Devices && dev_dist != slate::Dist::Col) {
        params.msg() = "skipping: dev_dist = Row applies only to target devices";
        return;
    }

    if (params.routine == "posvMixed"
        && ! std::is_same<real_t, double>::value) {
        params.msg() = "skipping: unsupported mixed precision; must be type=d or z";
        return;
    }

    if (params.routine == "posvMixed"
        && target == slate::Target::Devices) {
        params.msg() = "skipping: unsupported mixed precision; no devices support";
        return;
    }

    // Matrix A: figure out local size.
    int64_t mlocA = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocA = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldA  = blas::max(1, mlocA); // local leading dimension of A

    // Matrix B: figure out local size.
    int64_t mlocB = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocB = num_local_rows_cols(nrhs, nb, mycol, q);
    int64_t lldB  = blas::max(1, mlocB); // local leading dimension of B

    // ScaLAPACK data if needed.
    std::vector<scalar_t> A_data, B_data;

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
            A = slate::HermitianMatrix<scalar_t>(
                    uplo, n, nb, p, q, MPI_COMM_WORLD);
            B = slate::Matrix<scalar_t>(
                    n, nrhs, nb, p, q, MPI_COMM_WORLD);
        }

        // SLATE allocates CPU or GPU tiles.
        slate::Target origin_target = origin2target(origin);
        A.insertLocalTiles(origin_target);

        B.insertLocalTiles(origin_target);

        if (params.routine == "posvMixed") {
            X_data.resize(lldB*nlocB);
            X = slate::Matrix<scalar_t>(n, nrhs, nb, p, q, MPI_COMM_WORLD);
            X.insertLocalTiles(origin_target);
        }
    }
    else {
        // Create SLATE matrix from the ScaLAPACK layouts
        A_data.resize( lldA * nlocA );
        B_data.resize( lldB * nlocB );
        A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                uplo, n, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        B = slate::Matrix<scalar_t>::fromScaLAPACK(
                n, nrhs, &B_data[0], lldB, nb, p, q, MPI_COMM_WORLD);
        if (params.routine == "posvMixed") {
            X_data.resize(lldB*nlocB);
            X = slate::Matrix<scalar_t>::fromScaLAPACK(
                    n, nrhs, &X_data[0], lldB, nb, p, q, MPI_COMM_WORLD);
        }
    }

    slate::generate_matrix(params.matrix, A);
    slate::generate_matrix(params.matrixB, B);
    print_matrix("A", A, params);
    print_matrix("B", B, params);

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

        if (check && ref)
            B_orig = Bref_data;
    }

    double gflop;
    if (params.routine == "posv" || params.routine == "posvMixed")
        gflop = lapack::Gflop<scalar_t>::posv(n, nrhs);
    else
        gflop = lapack::Gflop<scalar_t>::potrf(n);

    if (! ref_only) {

        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        //==================================================
        // Run SLATE test: potrf or posv
        // potrf: Factor A = LL^H or A = U^H U.
        // posv:  Solve AX = B, including factoring A.
        //==================================================
        double time = barrier_get_wtime(MPI_COMM_WORLD);

        if (params.routine == "potrf" || params.routine == "potrs") {
            // Factor matrix A.
            slate::chol_factor(A, opts);
            // Using traditional BLAS/LAPACK name
            // slate::potrf(A, opts);
        }
        else if (params.routine == "posv") {
            slate::chol_solve(A, B, opts);
            // Using traditional BLAS/LAPACK name
            // slate::posv(A, B, opts);
        }
        else if (params.routine == "posvMixed") {
            if constexpr (std::is_same<real_t, double>::value) {
                int iters = 0;
                slate::posvMixed(A, B, X, iters, opts);
                params.iters() = iters;
            }
        }
        time = barrier_get_wtime(MPI_COMM_WORLD) - time;
        // compute and save timing/performance
        params.time() = time;
        params.gflops() = gflop / time;

        //==================================================
        // Run SLATE test: potrs
        // potrs: Solve AX = B, after factoring A above.
        //==================================================
        if (do_potrs) {
            double time2 = barrier_get_wtime(MPI_COMM_WORLD);

            if ((check && params.routine == "potrf")
                || params.routine == "potrs")
            {
                slate::chol_solve_using_factor(A, B, opts);
                // Using traditional BLAS/LAPACK name
                // slate::potrs(A, B, opts);
            }
            else {
                slate_error("Unknown routine!");
            }
            time2 = barrier_get_wtime(MPI_COMM_WORLD) - time2;
            // compute and save timing/performance
            params.time2() = time2;
            params.gflops2() = lapack::Gflop<scalar_t>::potrs(n, nrhs) / time2;
        }

        if (trace) slate::trace::Trace::finish();
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

        // Norm of original matrix: || A ||_1
        real_t A_norm = slate::norm(slate::Norm::One, Aref);

        // Norm of updated-rhs/solution matrix: || X ||_1
        real_t X_norm;
        if (params.routine == "posvMixed")
            X_norm = slate::norm(slate::Norm::One, X);
        else
            X_norm = slate::norm(slate::Norm::One, B);

        // Bref -= Aref*B
        if (params.routine == "posvMixed") {
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
        if (params.routine == "posvMixed")
            params.okay() = params.okay() && params.iters() >= 0;
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

            scalapack_descinit(A_desc, n, n, nb, nb, 0, 0, ictxt, mlocA, &info);
            slate_assert(info == 0);

            scalapack_descinit(B_desc, n, nrhs, nb, nb, 0, 0, ictxt, mlocB, &info);
            slate_assert(info == 0);

            scalapack_descinit(Aref_desc, n, n, nb, nb, 0, 0, ictxt, mlocA, &info);
            slate_assert(info == 0);

            scalapack_descinit(Bref_desc, n, nrhs, nb, nb, 0, 0, ictxt, mlocB, &info);
            slate_assert(info == 0);

            if (check) {
                // restore Bref_data
                Bref_data = B_orig;
                //scalapack_descinit(Bref_desc, n, nrhs, nb, nb, 0, 0, ictxt, mlocB, &info);
                //slate_assert(info == 0);
            }

            if (params.routine == "potrs") {
                // Factor matrix A.
                scalapack_ppotrf(uplo2str(uplo), n, &Aref_data[0], 1, 1, Aref_desc, &info);
                slate_assert(info == 0);
            }

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            double time = barrier_get_wtime(MPI_COMM_WORLD);
            if (params.routine == "potrf") {
                scalapack_ppotrf(uplo2str(uplo), n, &Aref_data[0], 1, 1, Aref_desc, &info);
            }
            else if (params.routine == "potrs") {
                scalapack_ppotrs(uplo2str(uplo), n, nrhs, &Aref_data[0], 1, 1, Aref_desc, &Bref_data[0], 1, 1, Bref_desc, &info);
            }
            else {
                scalapack_pposv(uplo2str(uplo), n, nrhs, &Aref_data[0], 1, 1, Aref_desc, &Bref_data[0], 1, 1, Bref_desc, &info);
            }
            slate_assert(info == 0);
            time = barrier_get_wtime(MPI_COMM_WORLD) - time;

            params.ref_time() = time;
            params.ref_gflops() = gflop / time;

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
        #else  // not SLATE_HAVE_SCALAPACK
            SLATE_UNUSED( verbose );
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
