// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
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
void test_gesv_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;

    // Constants
    const scalar_t one = 1.0;

    // If routine is gesv, getrf, getrs (without suffix), the first time
    // this is called, with run = false, method = PartialPiv because the
    // command line hasn't been parsed yet.

    // Decode routine, setting method and chopping off _tntpiv or _nopiv suffix.
    if (ends_with( params.routine, "_tntpiv" )) {
        params.routine = params.routine.substr( 0, params.routine.size() - 7 );
        params.method_lu() = slate::MethodLU::CALU;
    }
    else if (ends_with( params.routine, "_nopiv" )) {
        params.routine = params.routine.substr( 0, params.routine.size() - 6 );
        params.method_lu() = slate::MethodLU::NoPiv;
    }
    auto method_lu   = params.method_lu();
    auto methodTrsm = params.method_trsm();
    auto methodGemm = params.method_gemm();

    // get & mark input values
    slate::Op trans = slate::Op::NoTrans;
    if (params.routine == "getrs" || params.routine == "getrf_addmod")
        trans = params.trans();

    int64_t m;
    if (params.routine == "getrf" || params.routine == "getrf_addmod")
        m = params.dim.m();
    else
        m = params.dim.n();  // square, n-by-n

    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs();
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
    bool nonuniform_nb = params.nonuniform_nb() == 'y';
    int verbose = params.verbose();
    int timer_level = params.timer_level();
    SLATE_UNUSED(verbose);
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    slate::GridOrder grid_order = params.grid_order();
    params.matrix.mark();
    params.matrixB.mark();

    // Currently only gesv* supports timer_level >= 2.
    std::vector<std::string> timer_lvl_support{ "gesv", "gesv_mixed",
                                                "gesv_mixed_gmres"};
    bool supported = std::find( timer_lvl_support.begin(),
                                timer_lvl_support.end(), params.routine )
                     != timer_lvl_support.end();
    if (! supported)
        timer_level = 1;

    slate::BlockFactor blockFactorType = slate::BlockFactor::SVD;
    if (ends_with( params.routine, "_addmod" )) {
        blockFactorType = params.blockfactor();
    }

    // NoPiv and CALU ignore threshold.
    double pivot_threshold = params.pivot_threshold();
    double add_tol = 0.0;
    bool useWoodbury = false;
    if (params.routine == "getrf_addmod" || params.routine == "gesv_addmod"
        || params.routine == "gesv_addmod_ir") {
        add_tol = params.add_tol();
        useWoodbury = params.woodbury() == 'y';
    }

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    bool do_getrs = params.routine == "getrs" || params.routine == "getrs_addmod"
                    || (check && (params.routine == "getrf" || params.routine == "getrf_addmod"));

    if (do_getrs) {
        params.time2();
        params.time2.name( "trs time (s)" );
        params.time2.width( 12 );
        params.gflops2();
        params.gflops2.name( "trs gflop/s" );
    }
    if (timer_level >= 2 && params.routine == "gesv") {
        params.time2();
        params.time3();
        params.time2.name( "getrf (s)" );
        params.time3.name( "getrs (s)" );
    }
    else if (timer_level >=2 && (params.routine == "gesv_mixed"
                                 || params.routine == "gesv_mixed_gmres")) {
        params.time2();
        params.time3();
        params.time4();
        params.time5();
        params.time6();
        params.time7();
        params.time2.name( "getrf_lo (s)" );
        params.time3.name( "getrs_lo (s)" );
        params.time4.name( "gemm_hi (s)" );
        params.time5.name( "add_hi (s)" );
        params.time6.name( "getrf_hi (s)" );
        params.time7.name( "getrs_hi (s)" );
        if (params.routine == "gesv_mixed_gmres") {
            params.time8();
            params.time9();
            params.time8.name( "rotations (s)" );
            params.time9.name( "trsm_hi (s)" );
        }
    }

    bool is_iterative = params.routine == "gesv_mixed"
                        || params.routine == "gesv_mixed_gmres"
                        || params.routine == "gesv_rbt"
                        || params.routine == "gesv_addmod_ir";

    int64_t itermax = 0;
    bool fallback = true;
    if (is_iterative) {
        params.iters();
        fallback = params.fallback() == 'y';
        itermax = params.itermax();
    }

    int64_t depth = 0;
    if (params.routine == "gesv_rbt") {
        depth = params.depth();
    }

    if (! run)
        return;

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo( mpi_rank, grid_order, p, q, &myrow, &mycol );

    if (nonuniform_nb) {
        if (ref || origin == slate::Origin::ScaLAPACK) {
            params.msg() = "skipping: nonuniform tile not supported with ScaLAPACK";
            return;
        }
        params.ref() = 'n';
        params.origin() = slate::Origin::Host;
        ref = false;
        origin = slate::Origin::Host;
    }

    if ((params.routine == "gesv_mixed" || params.routine == "gesv_mixed_gmres")
        && ! std::is_same<real_t, double>::value) {
        params.msg() = "skipping: unsupported mixed precision; must be type=d or z";
        return;
    }

    slate::Options const opts =  {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::BlockFactor, blockFactorType},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib},
        {slate::Option::PivotThreshold, pivot_threshold},
        {slate::Option::MethodLU, method_lu},
        {slate::Option::MethodGemm, methodGemm},
        {slate::Option::MethodTrsm, methodTrsm},
        {slate::Option::Depth, depth},
        {slate::Option::MaxIterations, itermax},
        {slate::Option::UseFallbackSolver, fallback},
        {slate::Option::AdditiveTolerance, add_tol},
        {slate::Option::UseWoodbury, useWoodbury},
    };

    int64_t info = 0;

    // Matrix A: figure out local size.
    int64_t mlocA = num_local_rows_cols(m, nb, myrow, p);
    int64_t nlocA = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldA  = blas::max(1, mlocA); // local leading dimension of A

    // Matrix B: figure out local size.
    int64_t mlocB = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocB = num_local_rows_cols(nrhs, nb, mycol, q);
    int64_t lldB  = blas::max(1, mlocB); // local leading dimension of B

    // To generate matrix with non-uniform tile size using the lambda constructor
    std::function< int64_t (int64_t j) >
    tileNb = [nb](int64_t j) {
        // for non-uniform tile size
        return (j % 2 != 0 ? nb/2 : nb);
    };

    // rank and device functions for using the lambda constructor
    auto tileRank = slate::func::process_2d_grid( slate::GridOrder::Col, p, q );
    int num_devices_ = blas::get_device_count();
    auto tileDevice = slate::func::device_1d_grid( slate::GridOrder::Col,
                                                   p, num_devices_ );

    std::vector<scalar_t> A_data, B_data, X_data;
    slate::Matrix<scalar_t> A, B, X;
    if (origin != slate::Origin::ScaLAPACK) {
        // SLATE allocates CPU or GPU tiles.
        slate::Target origin_target = origin2target(origin);
        if (nonuniform_nb) {
            A = slate::Matrix<scalar_t>(m, n, tileNb, tileNb, tileRank,
                                        tileDevice, MPI_COMM_WORLD);
            B = slate::Matrix<scalar_t>(n, nrhs, tileNb, tileNb, tileRank,
                                        tileDevice, MPI_COMM_WORLD);
        }
        else {
            A = slate::Matrix<scalar_t>(
                m, n,    nb, nb, grid_order, p, q, MPI_COMM_WORLD );
            B = slate::Matrix<scalar_t>(
                n, nrhs, nb, nb, grid_order, p, q, MPI_COMM_WORLD );
        }
        A.insertLocalTiles(origin_target);
        B.insertLocalTiles(origin_target);

        if (is_iterative) {
            X_data.resize(lldB*nlocB);
            if (nonuniform_nb) {
                X = slate::Matrix<scalar_t>(n, nrhs, tileNb, tileNb, tileRank,
                                            tileDevice, MPI_COMM_WORLD);
            }
            else {
                X = slate::Matrix<scalar_t>(
                    n, nrhs, nb, nb, grid_order, p, q, MPI_COMM_WORLD );
            }
            X.insertLocalTiles(origin_target);
        }
    }
    else {
        // Create SLATE matrix from the ScaLAPACK layouts
        A_data.resize( lldA * nlocA );
        A = slate::Matrix<scalar_t>::fromScaLAPACK(
            m, n, &A_data[0], lldA, nb, nb, grid_order, p, q, MPI_COMM_WORLD );

        B_data.resize( lldB * nlocB );
        B = slate::Matrix<scalar_t>::fromScaLAPACK(
            n, nrhs, &B_data[0], lldB, nb, nb, grid_order, p, q, MPI_COMM_WORLD );

        if (is_iterative) {
            X_data.resize(lldB*nlocB);
            X = slate::Matrix<scalar_t>::fromScaLAPACK(
                n, nrhs, &X_data[0], lldB, nb, nb, grid_order, p, q, MPI_COMM_WORLD );
        }
    }

    slate::Pivots pivots;
    slate::AddModFactors<scalar_t> amfactors;

    slate::Options matgen_opts = {{slate::Option::Target, target}};
    slate::generate_matrix(params.matrix,  A, matgen_opts);
    slate::generate_matrix(params.matrixB, B, matgen_opts);

    // If check/ref is required, copy test data.
    slate::Matrix<scalar_t> Aref, Bref;
    std::vector<scalar_t> Aref_data, Bref_data;
    if (check || ref) {
        if (nonuniform_nb) {
            Aref = slate::Matrix<scalar_t>(m, n, tileNb, tileNb, tileRank,
                                           tileDevice, MPI_COMM_WORLD);
            Bref = slate::Matrix<scalar_t>(n, nrhs, tileNb, tileNb, tileRank,
                                           tileDevice, MPI_COMM_WORLD);
            Aref.insertLocalTiles( slate::Target::Host );
            Bref.insertLocalTiles( slate::Target::Host );
        }
        else {
            Aref_data.resize( lldA* nlocA );
            Bref_data.resize( lldB* nlocB );
            Aref = slate::Matrix<scalar_t>::fromScaLAPACK(
                       m, n, &Aref_data[0], lldA, nb, nb,
                       grid_order, p, q, MPI_COMM_WORLD );
            Bref = slate::Matrix<scalar_t>::fromScaLAPACK(
                       n, nrhs, &Bref_data[0], lldB, nb, nb,
                       grid_order, p, q, MPI_COMM_WORLD );
        }

        slate::copy(A, Aref);
        slate::copy(B, Bref);
    }

    print_matrix( "A", A, params );
    print_matrix( "B", B, params );

    double gflop;
    if (params.routine == "gesv"
        || params.routine == "gesv_mixed"
        || params.routine == "gesv_mixed_gmres"
        || params.routine == "gesv_rbt"
        || params.routine == "gesv_addmod"
        || params.routine == "gesv_addmod_ir")
        gflop = lapack::Gflop<scalar_t>::gesv(n, nrhs);
    else
        gflop = lapack::Gflop<scalar_t>::getrf(m, n);

    if (! ref_only) {

        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        //==================================================
        // Run SLATE test: getrf or gesv
        // getrf: Factor PA = LU.
        // gesv:  Solve AX = B, including factoring A.
        //==================================================
        double time = barrier_get_wtime(MPI_COMM_WORLD);

        if (params.routine == "getrf" || params.routine == "getrs") {
            info = slate::lu_factor(A, pivots, opts);
            // Using traditional BLAS/LAPACK name
            // slate::getrf(A, pivots, opts);
        }
        else if (params.routine == "gesv") {
            info = slate::lu_solve(A, B, opts);
            // Using traditional BLAS/LAPACK name
            // slate::gesv(A, pivots, B, opts);
        }
        else if (params.routine == "gesv_mixed") {
            if constexpr (std::is_same<real_t, double>::value) {
                int iters = 0;
                info = slate::gesv_mixed( A, pivots, B, X, iters, opts );
                params.iters() = iters;
            }
        }
        else if (params.routine == "gesv_mixed_gmres") {
            if constexpr (std::is_same<real_t, double>::value) {
                int iters = 0;
                info = slate::gesv_mixed_gmres( A, pivots, B, X, iters, opts );
                params.iters() = iters;
            }
        }
        else if (params.routine == "gesv_rbt") {
            int iters = 0;
            slate::gesv_rbt(A, B, X, iters, opts);
            params.iters() = iters;
        }
        else if (params.routine == "gesv_addmod") {
            // slate::lu_solve_addmod(A, B, opts);
            // Using traditional BLAS/LAPACK name
            slate::gesv_addmod(A, amfactors, B, opts);
        }
        else if (params.routine == "gesv_addmod_ir") {
            int iters = 0;
            slate::gesv_addmod_ir(A, amfactors, B, X, iters, opts);
            params.iters() = iters;
        }
        else if (params.routine == "getrf_addmod" || params.routine == "getrs_addmod") {
            slate::getrf_addmod(A, amfactors, opts);
        }
        time = barrier_get_wtime(MPI_COMM_WORLD) - time;
        // compute and save timing/performance
        params.time() = time;
        params.gflops() = gflop / time;

        if (timer_level >= 2 && params.routine == "gesv") {
            params.time2() = slate::timers[ "gesv::getrf" ];
            params.time3() = slate::timers[ "gesv::getrs" ];
        }
        else if (timer_level >= 2 && params.routine == "gesv_mixed") {
            params.time2() = slate::timers[ "gesv_mixed::getrf_lo" ];
            params.time3() = slate::timers[ "gesv_mixed::getrs_lo" ];
            params.time4() = slate::timers[ "gesv_mixed::gemm_hi" ];
            params.time5() = slate::timers[ "gesv_mixed::add_hi" ];
            params.time6() = slate::timers[ "gesv_mixed::getrf_hi" ];
            params.time7() = slate::timers[ "gesv_mixed::getrs_hi" ];
        }
        else if (timer_level >= 2 && params.routine == "gesv_mixed_gmres") {
            params.time2() = slate::timers[ "gesv_mixed_gmres::getrf_lo" ];
            params.time3() = slate::timers[ "gesv_mixed_gmres::getrs_lo" ];
            params.time4() = slate::timers[ "gesv_mixed_gmres::gemm_hi" ];
            params.time5() = slate::timers[ "gesv_mixed_gmres::add_hi" ];
            params.time6() = slate::timers[ "gesv_mixed_gmres::getrf_hi" ];
            params.time7() = slate::timers[ "gesv_mixed_gmres::getrs_hi" ];
            params.time8() = slate::timers[ "gesv_mixed_gmres::rotations" ];
            params.time9() = slate::timers[ "gesv_mixed_gmres::trsm_hi" ];
        }

        //==================================================
        // Run SLATE test: getrs
        // getrs: Solve AX = B after factoring A above.
        //==================================================
        if (do_getrs && info == 0) {
            double time2 = barrier_get_wtime(MPI_COMM_WORLD);

            auto opA = A;
            if (trans == slate::Op::Trans)
                opA = transpose(A);
            else if (trans == slate::Op::ConjTrans)
                opA = conj_transpose( A );

            if (params.routine == "getrs"
                || params.routine == "getrf")
            {
                slate::lu_solve_using_factor( opA, pivots, B, opts );
                // Using traditional BLAS/LAPACK name
                // slate::getrs(opA, pivots, B, opts);
            }
            else if ((check && params.routine == "getrf_addmod")
                     || params.routine == "getrs_addmod")
            {
                slate::getrs_addmod(amfactors, B, opts);
            }
            else {
                slate_error("Unknown routine!");
            }

            // compute and save timing/performance
            time2 = barrier_get_wtime(MPI_COMM_WORLD) - time2;
            params.time2() = time2;
            params.gflops2() = lapack::Gflop<scalar_t>::getrs(n, nrhs) / time2;
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

        // Norm of updated-rhs/solution matrix: || X ||_1
        real_t X_norm;
        if (is_iterative)
            X_norm = slate::norm(slate::Norm::One, X);
        else
            X_norm = slate::norm(slate::Norm::One, B);

        // Norm of original A matrix
        real_t A_norm = slate::norm(slate::Norm::One, Aref);

        // Apply transpose operations to the A matrix
        slate::Matrix<scalar_t> opAref;
        if (trans == slate::Op::Trans)
            opAref = slate::transpose(Aref);
        else if (trans == slate::Op::ConjTrans)
            opAref = slate::conj_transpose( Aref );
        else
            opAref = Aref;

        // Bref -= op(Aref)*B
        if (is_iterative) {
            slate::multiply(-one, opAref, X, one, Bref);
            // Using traditional BLAS/LAPACK name
            // slate::gemm(-one, opAref, X, one, Bref);
        }
        else {
            slate::multiply(-one, opAref, B, one, Bref);
            // Using traditional BLAS/LAPACK name
            // slate::gemm(-one, opAref, B, one, Bref);
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
            if (nonuniform_nb) {
                printf("Unsupported to test nonuniform tile size using scalapack\n");
                return;
            }

            // BLACS/MPI variables
            blas_int ictxt, myrow_, mycol_, p_, q_;
            blas_int mpi_rank_ = 0, nprocs = 1;
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

            // ScaLAPACK descriptor for the reference matrix
            blas_int Aref_desc[9];
            scalapack_descinit(Aref_desc, m, n, nb, nb, 0, 0, ictxt, mlocA, &info);
            slate_assert(info == 0);

            blas_int Bref_desc[9];
            scalapack_descinit(Bref_desc, n, nrhs, nb, nb, 0, 0, ictxt, mlocB, &info);
            slate_assert(info == 0);

            // ScaLAPACK data for pivots.
            std::vector<blas_int> ipiv_ref(lldA + nb);

            if (params.routine == "getrs") {
                // Factor matrix A.
                scalapack_pgetrf(m, n,
                                 &Aref_data[0], 1, 1, Aref_desc, &ipiv_ref[0], &info);
                slate_assert( info == 0 );
            }

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            double time = barrier_get_wtime(MPI_COMM_WORLD);
            if (params.routine == "getrf") {
                scalapack_pgetrf(m, n,
                                 &Aref_data[0], 1, 1, Aref_desc, &ipiv_ref[0], &info);
            }
            else if (params.routine == "getrs") {
                scalapack_pgetrs(op2str(trans), n, nrhs,
                                 &Aref_data[0], 1, 1, Aref_desc, &ipiv_ref[0],
                                 &Bref_data[0], 1, 1, Bref_desc, &info);
            }
            else {
                scalapack_pgesv(n, nrhs,
                                &Aref_data[0], 1, 1, Aref_desc, &ipiv_ref[0],
                                &Bref_data[0], 1, 1, Bref_desc, &info);
            }
            slate_assert( info == 0 );
            time = barrier_get_wtime(MPI_COMM_WORLD) - time;

            params.ref_time() = time;
            params.ref_gflops() = gflop / time;

            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering
        #else  // not SLATE_HAVE_SCALAPACK
            if (mpi_rank == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }
}

// -----------------------------------------------------------------------------
void test_gesv(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_gesv_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_gesv_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_gesv_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gesv_work<std::complex<double>> (params, run);
            break;

        default:
            throw std::runtime_error( "unknown datatype" );
            break;
    }
}
