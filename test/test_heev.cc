// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"
#include "print_matrix.hh"
#include "grid_utils.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_heev_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;

    // Constants
    const scalar_t zero = 0;
    const scalar_t one  = 1;
    const real_t eps = std::numeric_limits<real_t>::epsilon();
    const real_t tol = params.tol() * 0.5 * eps;

    // get & mark input values
    slate::Job jobz = params.jobz();
    slate::Uplo uplo = params.uplo();
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

    // mark non-standard output values
    params.time();
    params.ref_time();
    params.error2();
    params.ortho();
    params.error.name( "value err" );
    params.error2.name( "back err" );
    params.ortho.name( "Z orth." );

    if (! run)
        return;

    slate::Options const opts = {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    };

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    // Skip invalid or unimplemented options.
    if (uplo == slate::Uplo::Upper) {
        params.msg() = "skipping: Uplo::Upper isn't supported.";
        return;
    }
    if (p != q) {
        params.msg() = "skipping: requires square process grid (p == q).";
        return;
    }

    // Figure out local size.
    // matrix A (local input), m-by-n, symmetric matrix
    int64_t mlocA = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocA = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldA  = blas::max(1, mlocA); // local leading dimension of A

    std::vector<scalar_t> A_data;

    // matrix Lambda (global output) gets eigenvalues in decending order
    std::vector<real_t> Lambda(n);

    // matrix Z (local output), Z(n,n), gets orthonormal eigenvectors
    // corresponding to Lambda of the reference scalapack
    int64_t mlocZ = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocZ = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldZ  = blas::max(1, mlocZ); // local leading dimension of Z
    std::vector<scalar_t> Z_data( lldZ * nlocZ );

    // Initialize SLATE data structures
    slate::HermitianMatrix<scalar_t> A;
    if (origin != slate::Origin::ScaLAPACK) {
        // SLATE allocates CPU or GPU tiles.
        slate::Target origin_target = origin2target(origin);
        A = slate::HermitianMatrix<scalar_t>(uplo, n, nb, p, q, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);
    }
    else {
        // create SLATE matrices from the ScaLAPACK layouts
        A_data.resize( lldA * nlocA );
        A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                uplo, n, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
    }

    slate::generate_matrix( params.matrix, A );

    // Z is currently used for ScaLAPACK heev call and can also be used
    // for SLATE heev call when slate::eig_vals takes Z
    auto Z = slate::Matrix<scalar_t>::fromScaLAPACK(
                 n, n, &Z_data[0], lldZ, nb, p, q, MPI_COMM_WORLD);

    if (verbose >= 1) {
        printf( "%% A %6lld-by-%6lld\n", llong( A.m() ), llong( A.n() ) );
        printf( "%% Z %6lld-by-%6lld\n", llong( Z.m() ), llong( Z.n() ) );
    }

    print_matrix( "A", A, params );

    std::vector<scalar_t> Aref_data;
    std::vector<real_t> Lambda_ref;
    slate::HermitianMatrix<scalar_t> Aref;
    slate::Matrix<scalar_t> Aref_gen;
    if (check || ref) {
        Aref_data.resize( lldA * nlocA );
        Lambda_ref.resize( Lambda.size() );
        Aref = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                   uplo, n, &Aref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        Aref_gen = slate::Matrix<scalar_t>::fromScaLAPACK(
                   n, n, &Aref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        slate::copy( A, Aref );
    }

    // SLATE test
    if (! ref_only) {
        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        double time = barrier_get_wtime(MPI_COMM_WORLD);

        //==================================================
        // Run SLATE test.
        //==================================================
        if (jobz == slate::Job::NoVec) {
            slate::eig_vals( A, Lambda, opts );
            // Or slate::eig( A, Lambda, opts );
            // Using traditional BLAS/LAPACK name
            // slate::heev( A, Lambda, opts );
        }
        else {
            slate::eig( A, Lambda, Z, opts );
            // Using traditional BLAS/LAPACK name
            // slate::heev( A, Lambda, Z, opts );
        }

        time = barrier_get_wtime(MPI_COMM_WORLD) - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time;

        if (check && jobz == slate::Job::Vec) {
            //==================================================
            // Test results by checking backwards error
            //
            //      || A - Z Lambda Z^H ||_1
            //     --------------------------- < tol * epsilon
            //            || A ||_1 * N
            //
            // and orthogonality
            //
            //      || I - Z Z^H ||_1
            //     ------------------- < tol * epsilon
            //              N
            //==================================================

            // Compute Z_Lambda = Z Lambda.
            // todo Z.copy()
            auto Z_Lambda = Z.emptyLike();
            Z_Lambda.insertLocalTiles();
            slate::copy( Z, Z_Lambda );

            // todo: refactor column scaling
            int64_t mt = Z.mt();
            int64_t nt = Z.nt();
            int64_t jj = 0;
            for (int64_t j = 0; j < nt; ++j) {
                #pragma omp parallel for slate_omp_default_none \
                    firstprivate( mt, j, jj ) shared( Z_Lambda, Lambda )
                for (int64_t i = 0; i < mt; ++i) {
                    if (Z_Lambda.tileIsLocal( i, j )) {
                        auto T = Z_Lambda( i, j );
                        scalar_t* T_data = T.data();
                        int64_t ldt = T.stride();
                        int64_t mb  = T.mb();
                        int64_t nb2  = T.nb();
                        for (int64_t tj = 0; tj < nb2; ++tj)
                            for (int64_t ti = 0; ti < mb; ++ti)
                                T_data[ ti + tj*ldt ] *= Lambda[ jj + tj ];
                    }
                }
                jj += Z_Lambda.tileNb( j );
            }

            // Restore A.
            copy( Aref, A );

            // A - Z_Lambda Z^H
            // Aref_gen and Aref point to the same data.
            // todo: implement herkx
            auto ZH = conj_transpose( Z );
            slate::gemm( -one, Z_Lambda, ZH, one, Aref_gen );
            real_t Anorm = slate::norm( slate::Norm::One, A );
            params.error2() = slate::norm( slate::Norm::One, Aref ) / (Anorm * n);
            params.okay() = (params.error2() <= tol);

            // I - Z^H Z
            slate::set( zero, one, Aref_gen );
            slate::gemm( -one, ZH, Z, one, Aref_gen );
            params.ortho() = slate::norm( slate::Norm::One, Aref_gen ) / n;
            params.okay() = params.okay() && (params.ortho() <= tol);

            // Restore Aref.
            copy( A, Aref );
        }

        print_matrix( "A_out", A, params );
        print_matrix( "Z_out", Z, params ); // Relevant when slate::eig_vals takes Z
    }

    if (ref) {
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
            scalapack_descinit(A_desc, n, n, nb, nb, 0, 0, ictxt, mlocA, &info);
            slate_assert(info == 0);

            int Z_desc[9];
            scalapack_descinit(Z_desc, n, n, nb, nb, 0, 0, ictxt, mlocZ, &info);
            slate_assert(info == 0);

            // query for workspace size
            int64_t info_tst = 0;
            int64_t lwork = -1, lrwork = -1;
            std::vector<scalar_t> work(1);
            std::vector<real_t> rwork(1);
            scalapack_pheev(job2str(jobz), uplo2str(uplo), n,
                            &Aref_data[0], 1, 1, A_desc,
                            &Lambda_ref[0], // global output
                            &Z_data[0], 1, 1, Z_desc,
                            &work[0], -1, &rwork[0], -1, &info_tst);
            slate_assert(info_tst == 0);
            lwork = int64_t( real( work[0] ) );
            work.resize(lwork);
            // The lrwork, rwork parameters are only valid for complex
            if (slate::is_complex<scalar_t>::value) {
                lrwork = int64_t( real( rwork[0] ) );
                rwork.resize(lrwork);
            }

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            double time = barrier_get_wtime(MPI_COMM_WORLD);
            scalapack_pheev(job2str(jobz), uplo2str(uplo), n,
                            &Aref_data[0], 1, 1, A_desc,
                            &Lambda_ref[0],
                            &Z_data[0], 1, 1, Z_desc,
                            &work[0], lwork, &rwork[0], lrwork, &info_tst);
            slate_assert(info_tst == 0);
            time = barrier_get_wtime(MPI_COMM_WORLD) - time;

            params.ref_time() = time;

            // Reference Scalapack was run, check reference against test
            // Perform a local operation to get differences Lambda = Lambda - Lambda_ref
            blas::axpy( n, -1.0, &Lambda_ref[0], 1, &Lambda[0], 1 );

            // Relative forward error: || Lambda_ref - Lambda || / || Lambda_ref ||.
            params.error() = blas::asum( n, &Lambda[0], 1 )
                           / blas::asum( n, &Lambda_ref[0], 1 );

            params.okay() = params.okay() && (params.error() <= tol);

            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering
        #else  // not SLATE_HAVE_SCALAPACK
            if (mpi_rank == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }
}

// -----------------------------------------------------------------------------
void test_heev(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_heev_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_heev_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_heev_work< std::complex<float> > (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_heev_work< std::complex<double> > (params, run);
            break;
    }
}
