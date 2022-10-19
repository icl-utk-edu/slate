// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"
#include "blas/flops.hh"
#include "lapack/flops.hh"
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
void test_getri_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;

    // Constants
    const scalar_t zero = 0.0;
    const scalar_t one  = 1.0;

    // get & mark input values
    int64_t n = params.dim.n();
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
    int verbose = params.verbose();
    SLATE_UNUSED(verbose);
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    params.matrix.mark();

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();

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

    // Matrix A: figure out local size.
    int64_t mlocA = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocA = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldA  = blas::max(1, mlocA); // local leading dimension of A

    // Allocate ScaLAPACK data if needed.
    std::vector<scalar_t> A_data;
    if (check || ref || origin == slate::Origin::ScaLAPACK) {
        A_data.resize( lldA * nlocA );
    }
    // todo: work-around to initialize BaseMatrix::num_devices_
    slate::Matrix<scalar_t> A0(n, n, nb, p, q, MPI_COMM_WORLD);

    // Setup SLATE matrix A based on scalapack matrix/data in A_data
    slate::Matrix<scalar_t> A;
    if (origin != slate::Origin::ScaLAPACK) {
        // SLATE allocates CPU or GPU tiles.
        slate::Target origin_target = origin2target(origin);
        A = slate::Matrix<scalar_t>(n, n, nb, p, q, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);
    }
    else {
        // Create SLATE matrix from the ScaLAPACK layouts
        A = slate::Matrix<scalar_t>::fromScaLAPACK(
                n, n, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
    }
    slate::generate_matrix(params.matrix, A);

    // Create pivot structure to store pivots after factoring
    slate::Pivots pivots;

    // if check (or ref) is required, copy test data and create a descriptor for it
    std::vector<scalar_t> Aref_data;
    slate::Matrix<scalar_t> Aref;
    if (check || ref) {
        // For simplicity, always use ScaLAPACK format for ref matrices.
        Aref_data.resize( lldA*nlocA );
        Aref = slate::Matrix<scalar_t>::fromScaLAPACK(
                   n, n, &Aref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        slate::copy(A, Aref);
    }

    // If check is required: record the norm(A original)
    real_t A_norm = 0.0;
    if (check)
        A_norm = slate::norm(slate::Norm::One, A);

    // initialize Cchk_data; space to hold A*inv(A);
    // also used for out-of-place algorithm
    std::vector<scalar_t> Cchk_data( lldA*nlocA );

    // C will be used as storage for out-of-place algorithm
    slate::Matrix<scalar_t> C;
    // todo: Select correct times to use out-of-place getri, currently always use
    if (params.routine == "getriOOP") {
        // setup SLATE matrix C based on scalapack matrix/data in Cchk_data
        if (origin != slate::Origin::ScaLAPACK) {
            // SLATE allocates CPU or GPU tiles.
            slate::Target origin_target = origin2target(origin);
            C = slate::Matrix<scalar_t>(n, n, nb, p, q, MPI_COMM_WORLD);
            C.insertLocalTiles(origin_target);
            slate::copy( A, C );
        }
        else {
            // Create SLATE matrix from the ScaLAPACK layouts
            C = slate::Matrix<scalar_t>::fromScaLAPACK(
                    n, n, &Cchk_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        }
    }

    // the timing includes getrf and getri
    double gflop = lapack::Gflop<scalar_t>::getrf(n, n)
                 + lapack::Gflop<scalar_t>::getri(n);

    if (! ref_only) {

        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        double time = barrier_get_wtime(MPI_COMM_WORLD);

        //==================================================
        // Run SLATE test.
        //==================================================
        // factor then invert; measure time for both
        slate::lu_factor(A, pivots, opts);
        // Using traditional BLAS/LAPACK name
        // slate::getrf(A, pivots, opts);

        if (params.routine == "getri") {
            // call in-place inversion
            slate::lu_inverse_using_factor(A, pivots, opts);
            // Using traditional BLAS/LAPACK name
            // slate::getri(A, pivots, opts);
        }
        else if (params.routine == "getriOOP") {
            // Call the out-of-place version; on exit, C = inv(A), A unchanged
            slate::lu_inverse_using_factor_out_of_place(A, pivots, C, opts);
            // Using traditional BLAS/LAPACK name
            // slate::getri(A, pivots, C, opts);
        }

        time = barrier_get_wtime(MPI_COMM_WORLD) - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time;
        params.gflops() = gflop / time;
    }

    if (check) {
        #ifdef SLATE_HAVE_SCALAPACK
            //==================================================
            // Check  || I - inv(A)*A || / ( || A || * N ) <=  tol * eps
            // TODO: implement check using SLATE.

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

            if (origin != slate::Origin::ScaLAPACK) {
                // Copy SLATE result back from GPU or CPU tiles.
                copy(A, &A_data[0], A_desc);
                copy(C, &Cchk_data[0], Cchk_desc);
            }

            // Copy inv(A) from oop vector storage Cchk_data to expected location A_data
            // After this, A_data contains the inv(A)
            if (params.routine == "getriOOP") {
                A_data = Cchk_data;
            }

            // For check make Cchk_data a identity matrix to check the result of multiplying A and A_inv
            scalapack_plaset("All", n, n, zero, one, &Cchk_data[0], 1, 1, Cchk_desc);

            // Cchk_data has been setup as an identity matrix; C_chk = C_chk - inv(A)*A
            scalapack_pgemm("notrans", "notrans", n, n, n, -one,
                            &A_data[0], 1, 1, A_desc,
                            &Aref_data[0], 1, 1, Aref_desc, one,
                            &Cchk_data[0], 1, 1, Cchk_desc);

            // Norm of Cchk_data ( = I - inv(A) * A )
            std::vector<real_t> worklange(n);
            real_t C_norm = scalapack_plange("One", n, n, &Cchk_data[0], 1, 1, Cchk_desc, &worklange[0]);

            real_t A_inv_norm = scalapack_plange("One", n, n, &A_data[0], 1, 1, A_desc, &worklange[0]);

            double residual = C_norm / (A_norm * n * A_inv_norm);
            params.error() = residual;

            real_t tol = params.tol() * std::numeric_limits<real_t>::epsilon();
            params.okay() = (params.error() <= tol);

            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering
        #else  // not SLATE_HAVE_SCALAPACK
            SLATE_UNUSED(zero);
            SLATE_UNUSED(one);
            SLATE_UNUSED(A_norm);
            if (mpi_rank == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }

    if (ref) {
        // todo: call to reference getri from ScaLAPACK not implemented
    }
}

// -----------------------------------------------------------------------------
void test_getri(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_getri_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_getri_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_getri_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_getri_work<std::complex<double>> (params, run);
            break;
    }
}
