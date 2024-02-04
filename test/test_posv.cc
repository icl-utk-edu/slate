// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"
#include "blas/flops.hh"
#include "lapack/flops.hh"
#include "print_matrix.hh"

#include "matrix_utils.hh"
#include "test_utils.hh"

#include "scalapack_wrappers.hh"
#include "auxiliary/Debug.hh"

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
    int64_t nb = params.nb();
    int64_t lookahead = params.lookahead();
    bool ref_only = params.ref() == 'o';
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    bool trace = params.trace() == 'y';
    bool hold_local_workspace = params.hold_local_workspace() == 'y';
    int verbose = params.verbose();
    int timer_level = params.timer_level();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    params.matrix.mark();
    params.matrixB.mark();
    slate::Method methodTrsm = params.method_trsm();
    slate::Method methodHemm = params.method_hemm();

    mark_params_for_test_HermitianMatrix( params );
    mark_params_for_test_Matrix( params );

    // Currently only posv* supports timer_level >= 2.
    std::vector<std::string> timer_lvl_support{ "posv", "posv_mixed",
                                                "posv_mixed_gmres" };
    bool supported = std::find( timer_lvl_support.begin(),
                                timer_lvl_support.end(), params.routine )
                     != timer_lvl_support.end();

    if (! supported)
        timer_level = 1;

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    bool do_potrs = params.routine == "potrs"
                    || (check && params.routine == "potrf");

    if (do_potrs) {
        params.time2();
        params.time2.name( "trs time (s)" );
        params.time2.width( 12 );
        params.gflops2();
        params.gflops2.name( "trs gflop/s" );
    }
    if (timer_level >= 2 && params.routine == "posv") {
        params.time2();
        params.time3();
        params.time2.name( "potrf (s)" );
        params.time3.name( "potrs (s)" );
    }
    else if (timer_level >=2 && (params.routine == "posv_mixed"
                                 || params.routine == "posv_mixed_gmres")) {
        params.time2();
        params.time3();
        params.time4();
        params.time5();
        params.time6();
        params.time7();
        params.time2.name( "potrf_lo (s)" );
        params.time3.name( "potrs_lo (s)" );
        params.time4.name( "hemm_hi (s)" );
        params.time5.name( "add_hi (s)" );
        params.time6.name( "potrf_hi (s)" );
        params.time7.name( "potrs_hi (s)" );
        if (params.routine == "posv_mixed_gmres") {
            params.time8();
            params.time9();
            params.time10();
            params.time8.name( "rotations (s)" );
            params.time9.name( "trsm_hi (s)" );
            params.time10.name( "gemm_hi (s)" );
        }
    }

    bool is_iterative = params.routine == "posv_mixed"
                        || params.routine == "posv_mixed_gmres";

    int64_t itermax = 0;
    bool fallback = true;
    if (is_iterative) {
        params.iters();
        fallback = params.fallback() == 'y';
        itermax = params.itermax();
    }

    if (! run) {
        params.matrix.kind.set_default( "rand_dominant" );
        return;
    }

    // Check for common invalid combinations
    if (is_invalid_parameters( params )) {
        return;
    }

    slate::Options const opts =  {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::HoldLocalWorkspace, hold_local_workspace},
        {slate::Option::MethodTrsm, methodTrsm},
        {slate::Option::MethodHemm, methodHemm},
        {slate::Option::MaxIterations, itermax},
        {slate::Option::UseFallbackSolver, fallback},
    };

    if ((params.routine == "posv_mixed" || params.routine == "posv_mixed_gmres")
        && ! std::is_same<real_t, double>::value) {
        params.msg() = "skipping: unsupported mixed precision; must be type=d or z";
        return;
    }

    if ((params.routine == "posv_mixed" || params.routine == "posv_mixed_gmres")
        && target == slate::Target::Devices) {
        params.msg() = "skipping: unsupported devices support";
        return;
    }

    int64_t info = 0;

    auto A_alloc = allocate_test_HermitianMatrix<scalar_t>( check || ref, true, n, params );
    auto B_alloc = allocate_test_Matrix<scalar_t>( check || ref, true, n, nrhs, params );
    TestMatrix<slate::Matrix<scalar_t>> X_alloc;
    if (is_iterative) {
        X_alloc = allocate_test_Matrix<scalar_t>( false, true, n, nrhs, params );
    }

    auto& A         = A_alloc.A;
    auto& A_data    = A_alloc.A_data;
    auto& Aref      = A_alloc.Aref;
    auto& Aref_data = A_alloc.Aref_data;
    auto& B         = B_alloc.A;
    auto& B_data    = B_alloc.A_data;
    auto& Bref      = B_alloc.Aref;
    auto& Bref_data = B_alloc.Aref_data;
    auto& X         = X_alloc.A;

    slate::generate_matrix(params.matrix, A);
    slate::generate_matrix(params.matrixB, B);
    print_matrix("A", A, params);
    print_matrix("B", B, params);

    // if check is required, copy test data and create a descriptor for it
    std::vector<scalar_t> B_orig;
    if (check || ref) {
        slate::copy( A, Aref );
        slate::copy( B, Bref );

        if (check && ref)
            B_orig = Bref_data;
    }

    double gflop;
    if (params.routine == "posv"
        || params.routine == "posv_mixed"
        || params.routine == "posv_mixed_gmres")
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
            info = slate::chol_factor( A, opts );
            // Using traditional BLAS/LAPACK name
            // slate::potrf(A, opts);
        }
        else if (params.routine == "posv") {
            info = slate::chol_solve( A, B, opts );
            // Using traditional BLAS/LAPACK name
            // slate::posv(A, B, opts);
        }
        else if (params.routine == "posv_mixed") {
            if constexpr (std::is_same<real_t, double>::value) {
                int iters = 0;
                info = slate::posv_mixed( A, B, X, iters, opts );
                params.iters() = iters;
            }
        }
        else if (params.routine == "posv_mixed_gmres") {
            if constexpr (std::is_same<real_t, double>::value) {
                int iters = 0;
                info = slate::posv_mixed_gmres(A, B, X, iters, opts);
                params.iters() = iters;
            }
        }
        time = barrier_get_wtime(MPI_COMM_WORLD) - time;
        // compute and save timing/performance
        params.time() = time;
        params.gflops() = gflop / time;

        if (timer_level >= 2 && params.routine == "posv") {
            params.time2() = slate::timers[ "posv::potrf" ];
            params.time3() = slate::timers[ "posv::potrs" ];
        }
        else if (timer_level >= 2 && params.routine == "posv_mixed") {
            params.time2() = slate::timers[ "posv_mixed::potrf_lo" ];
            params.time3() = slate::timers[ "posv_mixed::potrs_lo" ];
            params.time4() = slate::timers[ "posv_mixed::hemm_hi" ];
            params.time5() = slate::timers[ "posv_mixed::add_hi" ];
            params.time6() = slate::timers[ "posv_mixed::potrf_hi" ];
            params.time7() = slate::timers[ "posv_mixed::potrs_hi" ];
        }
        else if (timer_level >= 2 && params.routine == "posv_mixed_gmres") {
            params.time2() = slate::timers[ "posv_mixed_gmres::potrf_lo" ];
            params.time3() = slate::timers[ "posv_mixed_gmres::potrs_lo" ];
            params.time4() = slate::timers[ "posv_mixed_gmres::hemm_hi" ];
            params.time5() = slate::timers[ "posv_mixed_gmres::add_hi" ];
            params.time6() = slate::timers[ "posv_mixed_gmres::potrf_hi" ];
            params.time7() = slate::timers[ "posv_mixed_gmres::potrs_hi" ];
            params.time8() = slate::timers[ "posv_mixed_gmres::rotations" ];
            params.time9() = slate::timers[ "posv_mixed_gmres::trsm_hi" ];
            params.time10() = slate::timers[ "posv_mixed_gmres::gemm_hi" ];
        }

        //==================================================
        // Run SLATE test: potrs
        // potrs: Solve AX = B, after factoring A above.
        //==================================================
        if (do_potrs && info == 0) {
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

        if (info != 0) {
            char buf[ 80 ];
            snprintf( buf, sizeof(buf), "info = %lld, cond = %.2e",
                      llong( info ), params.matrix.cond_actual() );
            params.msg() = buf;
        }
    }
    print_matrix( "X_out", X, params );

    if (info != 0 || std::isinf( params.matrix.cond_actual() )) {
        // info != 0 if and only if cond == inf (singular matrix).
        // Matrices with unknown cond (nan) that are singular are marked failed.
        params.okay() = info != 0 && std::isinf( params.matrix.cond_actual() );
    }
    else if (check) {
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
        if (is_iterative)
            X_norm = slate::norm(slate::Norm::One, X);
        else
            X_norm = slate::norm(slate::Norm::One, B);

        // Bref -= Aref*B
        if (is_iterative) {
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
        if (is_iterative)
            params.okay() = params.okay() && params.iters() >= 0;
    }

    if (ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            // A comparison with a reference routine from ScaLAPACK for timing only

            // initialize BLACS and ScaLAPACK
            blas_int ictxt, Aref_desc[9], Bref_desc[9];
            A_alloc.create_ScaLAPACK_context( &ictxt );
            A_alloc.ScaLAPACK_descriptor( ictxt, Aref_desc );
            B_alloc.ScaLAPACK_descriptor( ictxt, Bref_desc );

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
                    slate::Debug::diffLapackMatrices<scalar_t>(
                        n, n, &A_data[0], A_alloc.lld,
                        &Aref_data[0], A_alloc.lld, nb, nb);
                    if (params.routine != "potrf") {
                        slate::Debug::diffLapackMatrices<scalar_t>(
                            n, nrhs, &B_data[0], B_alloc.lld,
                            &Bref_data[0], B_alloc.lld, nb, nb);
                    }
                }
            }
            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering
        #else  // not SLATE_HAVE_SCALAPACK
            SLATE_UNUSED( verbose );
            if (A.mpiRank() == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }
}

// -----------------------------------------------------------------------------
void test_posv(Params& params, bool run)
{
    switch (params.datatype()) {
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

        default:
            throw std::runtime_error( "unknown datatype" );
            break;
    }
}
