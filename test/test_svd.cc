// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
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
    int64_t p = params.grid.m();
    int64_t q = params.grid.n();
    int64_t nb = params.nb();
    int64_t ib = params.ib();
    int64_t panel_threads = params.panel_threads();
    int64_t lookahead = params.lookahead();
    bool ref_only = params.ref() == 'o';
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    params.matrix.mark();

    params.time();
    params.ref_time();
    params.error2();
    params.ortho_U();
    params.ortho_V();
    params.error.name( "S - Sref" );
    params.error2.name( "Backward" );

    if (! run)
        return;

    slate::Options const opts =  {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    };

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    int64_t min_mn = std::min(m, n);

    // Figure out local size.
    // matrix A (local input), m-by-n
    int64_t mlocA = num_local_rows_cols(m, nb, myrow, p);
    int64_t nlocA = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldA  = blas::max(1, mlocA); // local leading dimension of A
    std::vector<scalar_t> A_data;
    std::vector<scalar_t> Acpy_data;

    // matrix U (local output), U(m, min_mn), singular values of A
    int64_t mlocU = num_local_rows_cols(m, nb, myrow, p);
    int64_t nlocU = num_local_rows_cols(min_mn, nb, mycol, q);
    int64_t lldU  = blas::max(1, mlocU); // local leading dimension of U
    std::vector<scalar_t> U_data(1);

    // matrix VT (local output), VT(min_mn, n)
    int64_t mlocVT = num_local_rows_cols(min_mn, nb, myrow, p);
    int64_t nlocVT = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldVT  = blas::max(1, mlocVT); // local leading dimension of VT
    std::vector<scalar_t> VT_data(1);

    // array Sigma (global output), singular values of A
    std::vector<real_t> Sigma(min_mn);

    slate::Matrix<scalar_t> A; // (m, n);
    slate::Matrix<scalar_t> U; // (m, min_mn);
    slate::Matrix<scalar_t> VT; // (min_mn, n);
    slate::Matrix<scalar_t> Acpy;

    bool wantu  = (jobu  == slate::Job::Vec
                   || jobu  == slate::Job::AllVec
                   || jobu  == slate::Job::SomeVec);
    bool wantvt = (jobvt == slate::Job::Vec
                   || jobvt == slate::Job::AllVec
                   || jobvt == slate::Job::SomeVec);

    if (origin != slate::Origin::ScaLAPACK) {
        // SLATE allocates CPU or GPU tiles.
        slate::Target origin_target = origin2target(origin);
        A = slate::Matrix<scalar_t>(m, n, nb, p, q, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);

        Acpy = slate::Matrix<scalar_t>(m, n, nb, p, q, MPI_COMM_WORLD);
        Acpy.insertLocalTiles(origin_target);

        if (wantu) {
            U = slate::Matrix<scalar_t>(m, min_mn, nb, p, q, MPI_COMM_WORLD);
            U.insertLocalTiles(origin_target);
        }
        if (wantvt) {
            VT = slate::Matrix<scalar_t>(min_mn, n, nb, p, q, MPI_COMM_WORLD);
            VT.insertLocalTiles(origin_target);
        }
    }
    else {
        // create SLATE matrices from the ScaLAPACK layouts
        A_data.resize( lldA * nlocA );
        A = slate::Matrix<scalar_t>::fromScaLAPACK(
                m, n, &A_data[0],  lldA,  nb, p, q, MPI_COMM_WORLD);

        Acpy_data.resize( lldA * nlocA );
        Acpy = slate::Matrix<scalar_t>::fromScaLAPACK(
                m, n, &Acpy_data[0],  lldA,  nb, p, q, MPI_COMM_WORLD);

        if (wantu) {
            U_data.resize(lldU*nlocU);
            U = slate::Matrix<scalar_t>::fromScaLAPACK(
                    m, min_mn, &U_data[0], lldU, nb, p, q, MPI_COMM_WORLD);
        }
        if (wantvt) {
            VT_data.resize(lldVT*nlocVT);
            VT = slate::Matrix<scalar_t>::fromScaLAPACK(
                     min_mn, n, &VT_data[0], lldVT, nb, p, q, MPI_COMM_WORLD);
        }
    }

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

    slate::Matrix<scalar_t> Aref;
    std::vector<real_t> Sigma_ref;
    std::vector<scalar_t> Aref_data;
    if (check || ref) {
        Sigma_ref.resize( min_mn );
        Aref_data.resize( lldA * nlocA );
        Aref = slate::Matrix<scalar_t>::fromScaLAPACK(
                   m, n, &Aref_data[0], lldA, nb, p, q, MPI_COMM_WORLD );
        slate::copy( A, Aref );
        slate::copy( A, Acpy );
    }

    if (! ref_only) {
        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        double time = barrier_get_wtime(MPI_COMM_WORLD);

        //==================================================
        // Run SLATE test.
        //==================================================
        if (wantu && wantvt) {
            slate::svd( A, Sigma, U, VT, opts );
        }
        else {
            slate::svd_vals( A, Sigma, opts );
        }

        time = barrier_get_wtime(MPI_COMM_WORLD) - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time;

        print_matrix("D", 1, min_mn,   &Sigma[0], 1, params);
        if (wantu) {
            print_matrix( "U",  U, params );
        }
        if (wantvt) {
            print_matrix( "VT", VT, params );
        }
    }

    if (check || ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            // Run reference routine from ScaLAPACK

            // BLACS/MPI variables
            int ictxt, p_, q_, myrow_, mycol_, info;
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

            int A_desc[9];
            scalapack_descinit(A_desc, m, n, nb, nb, 0, 0, ictxt, mlocA, &info);
            slate_assert(info == 0);

            int U_desc[9];
            scalapack_descinit(U_desc, m, min_mn, nb, nb, 0, 0, ictxt, mlocU, &info);
            slate_assert(info == 0);

            int VT_desc[9];
            scalapack_descinit(VT_desc, min_mn, n, nb, nb, 0, 0, ictxt, mlocVT, &info);
            slate_assert(info == 0);

            // todo: if will check on the vectors computed by scalapack,
            // then compute U and VT. But, for now we call scalapck just to check on
            // singular values.
            // Allocate if not already allocated.
            //if (wantu) {
            //    U_data.resize( lldU * nlocU );
            //}
            //if (wantvt) {
            //    VT_data.resize( lldVT * nlocVT );
            //}

            // query for workspace size
            int64_t info_ref = 0;
            scalar_t dummy_work;
            real_t dummy_rwork;
            scalapack_pgesvd(job2str(slate::Job::NoVec), job2str(slate::Job::NoVec), m, n,
                             &Aref_data[0],  1, 1, A_desc, &Sigma_ref[0],
                             &U_data[0],  1, 1, U_desc,
                             &VT_data[0], 1, 1, VT_desc,
                             &dummy_work, -1, &dummy_rwork, &info_ref);
            slate_assert(info_ref == 0);
            int64_t lwork  = int64_t( real( dummy_work ) );
            int64_t lrwork = int64_t( dummy_rwork );
            std::vector<scalar_t> work(lwork);
            std::vector<real_t> rwork(lrwork);

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            double time = barrier_get_wtime(MPI_COMM_WORLD);
            scalapack_pgesvd(job2str(slate::Job::NoVec), job2str(slate::Job::NoVec), m, n,
                             &Aref_data[0],  1, 1, A_desc, &Sigma_ref[0],
                             &U_data[0],  1, 1, U_desc,
                             &VT_data[0], 1, 1, VT_desc,
                             &work[0], lwork, &rwork[0], &info_ref);
            slate_assert(info_ref == 0);
            time = barrier_get_wtime(MPI_COMM_WORLD) - time;

            params.ref_time() = time;

            //==================================================
            // Test results by checking relative forward error
            //
            //      || Sigma_ref - Sigma ||
            //     ------------------------- < tol * epsilon
            //         || Sigma_ref ||
            //==================================================
            real_t Sigma_ref_norm = blas::asum(Sigma_ref.size(), &Sigma_ref[0], 1);
            // Perform a local operation to get differences Sigma = Sigma - Sigma_ref
            blas::axpy(Sigma_ref.size(), -1.0, &Sigma[0], 1, &Sigma_ref[0], 1);

            params.error() = blas::asum(Sigma.size(), &Sigma_ref[0], 1)
                           / Sigma_ref_norm;

            params.okay() = (params.error() <= tol);
            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering
        #else  // not SLATE_HAVE_SCALAPACK
            if (mpi_rank == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }

    if (check && (wantu || wantvt)) {
        // Residual matrix.
        slate::Matrix<scalar_t> R( min_mn, min_mn, nb, p, q, MPI_COMM_WORLD );
        R.insertLocalTiles();

        if (wantu) {
            //==================================================
            // Test results by checking orthogonality of U
            //
            //      || I - U^H U ||_1
            //     ------------------- < tol * epsilon
            //              N
            //==================================================
            slate::set( zero, one, R ); // identity
            auto UH = conj_transpose( U );
            slate::gemm( -one, UH, U, one, R );
            params.ortho_U() = slate::norm( slate::Norm::One, R ) / n;
            params.okay() = params.okay() && (params.ortho_U() <= tol);
        }

        if (wantvt) {
            //==================================================
            // Test results by checking orthogonality of VT
            //
            //      || I - V V^H ||_1
            //     ------------------- < tol * epsilon
            //              N
            //==================================================
            slate::set( zero, one, R ); // identity
            auto V = conj_transpose( VT );
            slate::gemm( -one, VT, V, one, R );
            params.ortho_V() = slate::norm( slate::Norm::One, R ) / n;
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
            //==================================================
            // todo:
            slate::scale_row_col( slate::Equed::Col, Sigma, Sigma, U );

            real_t Anorm = slate::norm( slate::Norm::One, Acpy );
            slate::gemm( -one, U, VT, one, Acpy );
            params.error2() = slate::norm( slate::Norm::One, Acpy ) / (Anorm * n);
            params.okay() = params.okay() && (params.error2() <= tol);
        }
    }
}

// -----------------------------------------------------------------------------
void test_svd( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

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
    }
}
