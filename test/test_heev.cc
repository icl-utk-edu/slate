// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
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
#include <limits>
#include <utility>
#define SLATE_HAVE_SCALAPACK

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_heev_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using llong = long long;

    // get & mark input values
    lapack::Job jobz = params.jobz();
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

    params.time();
    params.ref_time();
    // params.gflops();
    // params.ref_gflops();

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

    // Skip invalid or unimplemented options.
    if (uplo == slate::Uplo::Upper) {
        if (mpi_rank == 0)
            params.msg() = "skipping: Uplo::Upper isn't supported.";
        return;
    }
    if (p != q) {
        if (mpi_rank == 0)
            params.msg() = "skipping: requires square process grid (p == q).";
        return;
    }
    if (jobz != lapack::Job::NoVec) {
        if (mpi_rank == 0)
            params.msg() = "skipping: only supports Job::NoVec.";
        return;
    }

    // Figure out local size.
    // matrix A (local input), m-by-n, symmetric matrix
    int64_t mlocA = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocA = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldA  = blas::max(1, mlocA); // local leading dimension of A

    std::vector<scalar_t> A_data(lldA*nlocA);

    // matrix Lambda (global output) gets eigenvalues in decending order
    std::vector<real_t> Lambda(n);

    // matrix Z (local output), Z(n,n), gets orthonormal eigenvectors
    // corresponding to Lambda of the reference scalapack
    int64_t mlocZ = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocZ = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldZ  = blas::max(1, mlocZ); // local leading dimension of Z
    std::vector<scalar_t> Z_data(lldZ * nlocZ, 0);

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
        A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                uplo, n, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
    }

    slate::generate_matrix( params.matrix, A);

    // Z is currently used for ScaLAPACK heev call and can also be used
    // for SLATE heev call when slate::eig_vals takes Z
    auto Z = slate::Matrix<scalar_t>::fromScaLAPACK(
                 n, n, &Z_data[0], lldZ, nb, p, q, MPI_COMM_WORLD);

    if (verbose >= 1) {
        printf( "%% A   %6lld-by-%6lld\n", llong( A.m() ), llong( A.n() ) );

        printf( "%% Z   %6lld-by-%6lld\n", llong( Z.m() ), llong( Z.n() ) );
    }

    if (verbose > 1) {
        print_matrix( "A",  A  );
    }

    std::vector<scalar_t> Aref_data;
    std::vector<real_t> Lambda_ref;
    slate::HermitianMatrix<scalar_t> Aref;
    if (check || ref) {
        Aref_data.resize( A_data.size() );
        Lambda_ref.resize( Lambda.size() );
        Aref = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                   uplo, n, &Aref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
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
            slate::eig_vals(A, Lambda, opts);
        }
        // else {
            // todo: slate::Job::Vec
        // }
        // Using traditional BLAS/LAPACK name
        // slate::heev(jobz, A, Lambda, Z, opts);

        time = barrier_get_wtime(MPI_COMM_WORLD) - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time;

        if (verbose > 1) {
            print_matrix( "A",  A  );
            print_matrix( "Z",  Z  ); //Relevant when slate::eig_vals takes Z
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
            scalapack_descinit(A_desc, n, n, nb, nb, 0, 0, ictxt, mlocA, &info);
            slate_assert(info == 0);

            int Z_desc[9];
            scalapack_descinit(Z_desc, n, n, nb, nb, 0, 0, ictxt, mlocZ, &info);
            slate_assert(info == 0);

            // set num threads appropriately for parallel BLAS if possible
            int omp_num_threads = 1;
            #pragma omp parallel
            { omp_num_threads = omp_get_num_threads(); }
            int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);

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

            // Reset omp thread number
            slate_set_num_blas_threads(saved_num_threads);

            // Reference Scalapack was run, check reference against test
            // Perform a local operation to get differences Lambda = Lambda - Lambda_ref
            blas::axpy(Lambda_ref.size(), -1.0, &Lambda_ref[0], 1, &Lambda[0], 1);

            // Relative forward error: || Lambda_ref - Lambda || / || Lambda_ref ||.
            params.error() = blas::asum(n, &Lambda[0], 1)
                           / blas::asum(n, &Lambda_ref[0], 1);

            real_t tol = params.tol() * 0.5 * std::numeric_limits<real_t>::epsilon();
            params.okay() = (params.error() <= tol);

            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering
        #else
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
            test_heev_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_heev_work<std::complex<double>> (params, run);
            break;
    }
}
