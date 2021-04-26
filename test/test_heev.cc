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
    slate::Norm norm = params.norm();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    params.matrix.mark();

    slate_assert(p == q);  // heev requires square process grid.

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

    // Constants
    const int izero = 0, ione = 1;

    // Local values
    int myrow, mycol;
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

#if 0
    bool wantz = (jobz == slate::Job::Vec);
#endif

    // figure out local size, allocate, create descriptor, initialize
    // matrix A (local input), m-by-n, symmetric matrix
    int64_t mlocA = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocA = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldA  = blas::max(1, mlocA); // local leading dimension of A

    std::vector<scalar_t> A_data(lldA*nlocA);

    // matrix W (global output), W(n), gets eigenvalues in decending order
    std::vector<real_t> W_data(n);


    // matrix Z (local output), Z(n,n), gets orthonormal eigenvectors corresponding to W of the reference scalapack
    int64_t mlocZ = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocZ = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldZ  = blas::max(1, mlocZ); // local leading dimension of Z
#if 0
    std::vector<scalar_t> Z_data(lldZ * nlocZ, 0);
#endif

#if 0
    // matrix Q (local output), Q(n,n), gets orthonormal eigenvectors corresponding to W of slate heev
    int64_t mlocQ = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocQ = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldQ  = blas::max(1, mlocQ); // local leading dimension of Q
    std::vector<scalar_t> Q_data(lldQ * nlocQ, 0);
#endif

    // Initialize SLATE data structures
    slate::HermitianMatrix<scalar_t> A;
    std::vector<real_t> W(n);
#if 0
    //Don't need Z
    slate::Matrix<scalar_t> Z;
#endif
#if 0
    //Don't need Q
    slate::Matrix<scalar_t> Q;
#endif
    if (origin != slate::Origin::ScaLAPACK) {
        // Copy local ScaLAPACK data to GPU or CPU tiles.
        slate::Target origin_target = origin2target(origin);
        A = slate::HermitianMatrix<scalar_t>(uplo, n, nb, p, q, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);

#if 0
        Z = slate::Matrix<scalar_t>(n, n, nb, p, q, MPI_COMM_WORLD);
        Z.insertLocalTiles(origin_target);
#endif

#if 0
        if (wantz) {
            Q = slate::Matrix<scalar_t>(
                n, n, nb, p, q, MPI_COMM_WORLD);
            Q.insertLocalTiles(origin2target(origin));
        }
#endif
    }
    else {
        // create SLATE matrices from the ScaLAPACK layouts
        A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(uplo, n, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
#if 0
        Z = slate::Matrix<scalar_t>::fromScaLAPACK(n, n, &Z_data[0], lldZ, nb, p, q, MPI_COMM_WORLD);
#endif
#if 0
        if (wantz) {
            Q_data.resize(lldQ*nlocQ);
            Q = slate::Matrix<scalar_t>::fromScaLAPACK(
                n, n, &Q_data[0], lldQ, nb, p, q, MPI_COMM_WORLD);
        }
#endif
    }

    //lapack::TestMatrixType type = lapack::TestMatrixType::heev;
    //params.matrix.kind.set_default("heev");
    //params.matrix.cond.set_default(1e4);
//    slate::generate_matrix( params.matrix, A)

#if 0
    slate::generate_matrix( params.matrix, Z);
    A = slate::HermitianMatrix<scalar_t>(
               uplo, Z );
#endif
    slate::generate_matrix( params.matrix, A);

    if (verbose >= 1) {
        printf( "%% A   %6lld-by-%6lld\n", llong(   A.m() ), llong(   A.n() ) );
#if 0
        printf( "%% Z   %6lld-by-%6lld\n", llong(   Z.m() ), llong(   Z.n() ) );
#endif
    }

    if (verbose > 1) {
        print_matrix( "A",  A  );
    }

    std::vector<scalar_t> Aref_data, Zref_data;
    std::vector<real_t> Wref_data;
    slate::HermitianMatrix<scalar_t> Aref;
    if (check || ref) {
        Aref_data.resize( A_data.size() );
#if 0
        Zref_data.resize( Z_data.size(), 0 );
#endif
        Zref_data.resize( lldZ * nlocZ, 0 );
        Wref_data.resize( W_data.size() );
        Aref = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(uplo, n, &Aref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
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
            slate::eig_vals(A, W_data, opts);
        }
        // else {
            // todo: slate::Job::Vec
        // }

        //---------------------
        // Using traditional BLAS/LAPACK name
        // slate::heev(jobz, A, W_data, Q, opts);

        time = barrier_get_wtime(MPI_COMM_WORLD) - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time;

        if (verbose > 1) {
            print_matrix( "A",  A  );
#if 0
            print_matrix( "Z",  Z  );
#endif
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
            scalapack_descinit(A_desc, n, n, nb, nb, izero, izero, ictxt, mlocA, &info);
            slate_assert(info == 0);

            int Z_desc[9];
            scalapack_descinit(Z_desc, n, n, nb, nb, izero, izero, ictxt, mlocZ, &info);
            slate_assert(info == 0);

#if 0
            int Q_desc[9];
            scalapack_descinit(Q_desc, n, n, nb, nb, izero, izero, ictxt, mlocQ, &info);
            slate_assert(info == 0);
#endif
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
                            &Aref_data[0], ione, ione, A_desc,
                            &Wref_data[0], // global output
                            &Zref_data[0], // local output
                            ione, ione, Z_desc,
                            &work[0], -1, &rwork[0], -1, &info_tst);
            slate_assert(info_tst == 0);
            lwork = int64_t( real( work[0] ) );
            work.resize(lwork);
            // The lrwork, rwork parameters are only valid for complex
            if (slate::is_complex<scalar_t>::value) {
                lrwork = int64_t( real( rwork[0] ) );
                rwork.resize(lrwork);
            }
            // Run ScaLAPACK reference routine.
            double time = barrier_get_wtime(MPI_COMM_WORLD);
            scalapack_pheev(job2str(jobz), uplo2str(uplo), n,
                            &Aref_data[0], ione, ione, A_desc,
                            &Wref_data[0],
                            &Zref_data[0], ione, ione, Z_desc,
                            &work[0], lwork, &rwork[0], lrwork, &info_tst);
            slate_assert(info_tst == 0);
            time = barrier_get_wtime(MPI_COMM_WORLD) - time;

            params.ref_time() = time;

            // Reset omp thread number
            slate_set_num_blas_threads(saved_num_threads);

            // Reference Scalapack was run, check reference against test
            // Perform a local operation to get differences W_data = W_data - Wref_data
            blas::axpy(Wref_data.size(), -1.0, &Wref_data[0], 1, &W_data[0], 1);

            real_t reduced_error;
            real_t local_error;
            // Relative forward error: || Wref_data - W_data || / || Wref_data ||
            local_error = lapack::lange(norm, 1, W_data.size(), &W_data[0], 1)
                        / lapack::lange(norm, 1, Wref_data.size(), &Wref_data[0], 1);

            real_t tol = params.tol() * 0.5 * std::numeric_limits<real_t>::epsilon();

            if (local_error > tol) {
                printf("\nOn MPI Rank = %d, the eigenvalues are suspicious, the error is  %e \n",
                    A.mpiRank(), params.error());
                //for (int64_t i = 0; i < n; i++) {
                //    printf("\n %f", W_data[i]);
                //}
            }

            slate_mpi_call(
                MPI_Allreduce( &local_error, &reduced_error,
                            1, slate::mpi_type<real_t>::value,
                            MPI_MAX, A.mpiComm()));

            params.error() = reduced_error;
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
