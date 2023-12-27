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
#include "scalapack_copy.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_svd_work( Params& params, bool run )
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;

    // Constants
    const scalar_t zero = 0;
    const scalar_t one  = 1;

    // get & mark input values
    lapack::Job jobu = params.jobu();
    lapack::Job jobvt = params.jobvt();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t ib = params.ib();
    int64_t panel_threads = params.panel_threads();
    int64_t lookahead = params.lookahead();
    bool ref_only = params.ref() == 'o';
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    int timer_level = params.timer_level();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    params.matrix.mark();

    mark_params_for_test_Matrix( params );
    // nonuniform nb is not always supported in the reduction to band
    params.nonuniform_nb.used( false );

    params.time();
    params.ref_time();
    params.error2();
    params.ortho_U();
    params.ortho_V();
    params.error.name( "S - Sref" );
    params.error2.name( "Backward" );
    if (timer_level >= 2) {
        params.time2();
        params.time3();
        params.time4();
        params.time5();
        params.time6();
        params.time7();
        params.time8();
        params.time9();
        params.time10();
        params.time11();
        params.time12();
        params.time2.name( "geqrf (s)" );
        params.time3.name( "gelqf (s)" );
        params.time4.name( "ge2tb (s)" );
        params.time5.name( "tb2bd (s)" );
        params.time6.name( "bdsvd (s)" );
        params.time7.name( "unmbr_tb2bd_U (s)" );
        params.time8.name( "unmbr_ge2tb_U (s)" );
        params.time9.name( "unmqr (s) " );
        params.time10.name( "unmbr_tb2bd_V (s)" );
        params.time11.name( "unmbr_ge2tb_V (s)" );
        params.time12.name( "unmlq (s)" );
    }

    if (! run)
        return;

    // Check for common invalid combinations
    if (is_invalid_parameters( params )) {
        return;
    }

    if (check && ! ref
        && (jobu == slate::Job::NoVec || jobvt == slate::Job::NoVec)) {
        params.msg() = "job = NoVec requires --ref y to check singular values";
    }

    slate::Options const opts =  {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    };

    bool wantu  = (jobu  == slate::Job::Vec
                   || jobu  == slate::Job::AllVec
                   || jobu  == slate::Job::SomeVec);
    bool wantvt = (jobvt == slate::Job::Vec
                   || jobvt == slate::Job::AllVec
                   || jobvt == slate::Job::SomeVec);

    int64_t min_mn = std::min(m, n);

    // U  is either m-by-min( m, n ) for some vec, or m-by-m for all vec;
    // VT is either min( m, n )-by-n for some vec, or n-by-n for all vec.
    int64_t Um  = wantu  ? m : 0;
    int64_t Un  = wantu  ? (jobu  == slate::Job::AllVec ? m : min_mn) : 0;
    int64_t VTm = wantvt ? (jobvt == slate::Job::AllVec ? n : min_mn) : 0;
    int64_t VTn = wantvt ? n : 0;

    // array Sigma (global output), singular values of A
    std::vector<real_t> Sigma(min_mn);

    auto A_alloc = allocate_test_Matrix<scalar_t>( check || ref, true, m, n, params );
    auto U_alloc = allocate_test_Matrix<scalar_t>( false, true, Um, Un, params );
    auto VT_alloc = allocate_test_Matrix<scalar_t>( false, true, VTm, VTn, params );
    // TODO Acpy isn't always needed
    auto Acpy_alloc = allocate_test_Matrix<scalar_t>( false, true, m, n, params );

    auto& A         = A_alloc.A;
    auto& U         = U_alloc.A;
    auto& VT        = VT_alloc.A;
    auto& Acpy      = Acpy_alloc.A;

    if (verbose >= 1) {
        printf( "%% A   %6lld-by-%6lld\n", llong(   A.m() ), llong(   A.n() ) );
        printf( "%% U   %6lld-by-%6lld\n", llong(   U.m() ), llong(   U.n() ) );
        printf( "%% VT  %6lld-by-%6lld\n", llong(  VT.m() ), llong(  VT.n() ) );
    }

    real_t tol = params.tol() * 0.5 * std::numeric_limits<real_t>::epsilon();

    //params.matrix.kind.set_default("svd");
    //params.matrix.cond.set_default(1.e16);

    slate::generate_matrix( params.matrix, A);
    print_matrix( "A",  A, params );

    std::vector<real_t> Sigma_ref;
    if (check || ref) {
        Sigma_ref.resize( min_mn );
        slate::copy( A, A_alloc.Aref );
        slate::copy( A, Acpy );
    }

    if (! ref_only) {
        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        double time = barrier_get_wtime(MPI_COMM_WORLD);

        //==================================================
        // Run SLATE test.
        //==================================================
        if (wantu || wantvt) {
            slate::svd( A, Sigma, U, VT, opts );
        }
        else {
            slate::svd_vals( A, Sigma, opts );
        }

        time = barrier_get_wtime(MPI_COMM_WORLD) - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time;


        if (timer_level >= 2) {
            params.time2() = slate::timers[ "svd::geqrf" ];
            params.time3() = slate::timers[ "svd::gelqf" ];
            params.time4() = slate::timers[ "svd::ge2tb" ];
            params.time5() = slate::timers[ "svd::tb2bd" ];
            params.time6() = slate::timers[ "svd::bdsvd" ];
            params.time7() = slate::timers[ "svd::unmbr_tb2bd_U" ];
            params.time8() = slate::timers[ "svd::unmbr_ge2tb_U" ];
            params.time9() = slate::timers[ "svd::unmqr" ];
            params.time10() = slate::timers[ "svd::unmbr_tb2bd_V" ];
            params.time11() = slate::timers[ "svd::unmbr_ge2tb_V" ];
            params.time12() = slate::timers[ "svd::unmlq" ];
        }

        print_matrix("D", 1, min_mn,   &Sigma[0], 1, params);
        if (wantu) {
            print_matrix( "U",  U, params );
        }
        if (wantvt) {
            print_matrix( "VT", VT, params );
        }
    }

    // Initialize okay if any check will be run.
    if (check && (wantu || wantvt || ref)) {
        params.okay() = true;
    }

    if (check && (wantu || wantvt)) {
        // Residual matrix.
        int64_t Rm = min_mn;
        if (jobu == slate::Job::AllVec)
            Rm = blas::max( Rm, m );
        if (jobvt == slate::Job::AllVec)
            Rm = blas::max( Rm, n );
        auto R_alloc = allocate_test_Matrix<scalar_t>( false, true, Rm, Rm, params );
        auto R = R_alloc.A;

        if (wantu) {
            //==================================================
            // Test results by checking orthogonality of U
            //
            //      || I - U^H U ||_1
            //     ------------------- < tol * epsilon
            //              N
            //==================================================
            slate::Matrix<scalar_t> Ru;
            if (jobu == slate::Job::AllVec)
                Ru = R.slice( 0, m-1, 0, m-1 );
            else
                Ru = R.slice( 0, min_mn-1, 0, min_mn-1 );

            slate::set( zero, one, Ru ); // identity
            auto UH = conj_transpose( U );
            slate::gemm( -one, UH, U, one, Ru );
            params.ortho_U() = slate::norm( slate::Norm::One, Ru ) / n;
            params.okay() = params.okay() && (params.ortho_U() <= tol);
        }

        if (wantvt) {
            //==================================================
            // Test results by checking orthogonality of VT
            //
            //      || I - V^H V ||_1
            //     ------------------- < tol * epsilon
            //              N
            //==================================================
            slate::Matrix<scalar_t> Rv;
            if (jobvt == slate::Job::AllVec)
                Rv = R.slice( 0, n-1, 0, n-1 );
            else
                Rv = R.slice( 0, min_mn-1, 0, min_mn-1 );

            slate::set( zero, one, Rv ); // identity
            auto V = conj_transpose( VT );
            slate::gemm( -one, VT, V, one, Rv );
            params.ortho_V() = slate::norm( slate::Norm::One, Rv ) / n;
            params.okay() = params.okay() && (params.ortho_V() <= tol);
        }

        if (wantu && wantvt) {
            //==================================================
            // Test results by checking backwards error
            //
            //      || Acpy - U Sigma VT ||_1
            //     --------------------------- < tol * epsilon
            //            || A ||_1 * N
            //
            // using only economy-size U, VT.
            //==================================================
            slate::Matrix<scalar_t>  Ueco =  U.slice( 0, m-1, 0, min_mn-1 );
            slate::Matrix<scalar_t> VTeco = VT.slice( 0, min_mn-1, 0, n-1 );

            // U = U Sigma
            slate::scale_row_col( slate::Equed::Col, Sigma, Sigma, Ueco );

            real_t Anorm = slate::norm( slate::Norm::One, Acpy );
            slate::gemm( -one, Ueco, VTeco, one, Acpy );
            params.error2() = slate::norm( slate::Norm::One, Acpy ) / (Anorm * n);
            params.okay() = params.okay() && (params.error2() <= tol);
        }
    }

    if (ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            // Run reference routine from ScaLAPACK

            // initialize BLACS and ScaLAPACK
            blas_int ictxt, A_desc[9], U_desc[9], VT_desc[9];
            A_alloc.create_ScaLAPACK_context( &ictxt );

            A_alloc.ScaLAPACK_descriptor( ictxt, A_desc );
            U_alloc.ScaLAPACK_descriptor( ictxt, U_desc );
            VT_alloc.ScaLAPACK_descriptor( ictxt, VT_desc );

            auto& Aref_data = A_alloc.Aref_data;
            auto& U_data = U_alloc.A_data;
            auto& VT_data = VT_alloc.A_data;

            if (origin != slate::Origin::ScaLAPACK) {
                U_data.resize( U_alloc.lld * U_alloc.nloc );
                VT_data.resize( VT_alloc.lld * VT_alloc.nloc );
            }

            // ScaLAPACK uses job = N and V (same as S);
            // it doesn't support LAPACK's job = S, A, O options.
            // Warn if that makes a smaller U or V than AllVec would.
            const char* jobu_str  = jobu  == slate::Job::NoVec ? "N" : "V";
            const char* jobvt_str = jobvt == slate::Job::NoVec ? "N" : "V";
            if ((jobu == slate::Job::AllVec && m > n)
                || (jobvt == slate::Job::AllVec && n > m)) {
                params.msg() = "ScaLAPACK doesn't support AllVec; using SomeVec.";
            }

            // query for workspace size
            int64_t info_ref = 0;
            scalar_t dummy_work;
            real_t dummy_rwork;
            scalapack_pgesvd(
                jobu_str, jobvt_str, m, n,
                &Aref_data[0],  1, 1, A_desc, &Sigma_ref[0],
                &U_data[0],  1, 1, U_desc,
                &VT_data[0], 1, 1, VT_desc,
                &dummy_work, -1, &dummy_rwork, &info_ref );
            slate_assert(info_ref == 0);
            int64_t lwork  = int64_t( real( dummy_work ) );
            int64_t lrwork = int64_t( dummy_rwork );
            std::vector<scalar_t> work(lwork);
            std::vector<real_t> rwork(lrwork);

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            double time = barrier_get_wtime(MPI_COMM_WORLD);
            scalapack_pgesvd(
                jobu_str, jobvt_str, m, n,
                &Aref_data[0],  1, 1, A_desc, &Sigma_ref[0],
                &U_data[0],  1, 1, U_desc,
                &VT_data[0], 1, 1, VT_desc,
                &work[0], lwork, &rwork[0], &info_ref );
            slate_assert(info_ref == 0);
            time = barrier_get_wtime(MPI_COMM_WORLD) - time;

            params.ref_time() = time;

            if (! ref_only) {
                //==================================================
                // Test results by checking relative forward error
                //
                //      || Sigma_ref - Sigma ||
                //     ------------------------- < tol * epsilon
                //         || Sigma_ref ||
                //==================================================
                real_t Sigma_ref_norm = blas::asum( min_mn, &Sigma_ref[0], 1 );
                // Perform a local operation to get differences Sigma = Sigma - Sigma_ref
                blas::axpy( min_mn, -1.0, &Sigma[0], 1, &Sigma_ref[0], 1 );

                params.error() = blas::asum( min_mn, &Sigma_ref[0], 1 )
                               / Sigma_ref_norm;

                params.okay() = params.okay() && (params.error() <= tol);
            }

            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering
        #else  // not SLATE_HAVE_SCALAPACK
            if (A.mpiRank() == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }
}

// -----------------------------------------------------------------------------
void test_svd( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_svd_work<float>( params, run );
            break;

        case testsweeper::DataType::Double:
            test_svd_work<double>( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_svd_work<std::complex<float>>( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_svd_work<std::complex<double>>( params, run );
            break;

        default:
            throw std::runtime_error( "unknown datatype" );
            break;
    }
}
