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
#include <utility>
//------------------------------------------------------------------------------
double barrier_get_wtime(MPI_Comm comm)
{
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(comm);
        return testsweeper::get_wtime();
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_geqrf_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int p = params.grid.m();
    int q = params.grid.n();
    int64_t nb = params.nb();
    int64_t ib = params.ib();
    int64_t lookahead = params.lookahead();
    int64_t panel_threads = params.panel_threads();
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

    // constants
    const scalar_t zero = 0;
    const scalar_t one = 1;

    // Local values
    int myrow, mycol;
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    // matrix A, figure out local size, allocate, initialize
    int64_t mlocA = num_local_rows_cols(m, nb, myrow, p);
    int64_t nlocA = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldA  = std::max(int64_t(1), mlocA); // local leading dimension of A

    std::vector<scalar_t> A_data;
    slate::Matrix<scalar_t> A;
    if (origin != slate::Origin::ScaLAPACK) {
        slate::Target origin_target = origin2target(origin);
        A = slate::Matrix<scalar_t>(m, n, nb, p, q, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);
    }
    else {
        // create SLATE matrix from the ScaLAPACK layouts
        A_data.resize( mlocA * nlocA );
        A = slate::Matrix<scalar_t>::fromScaLAPACK( m, n, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD );
    }

    slate::generate_matrix(params.matrix, A);

    slate::TriangularFactors<scalar_t> T;

    if (verbose > 1) {
        print_matrix("A", A);
    }

    // For checks, keep copy of original matrix A.
    slate::Matrix<scalar_t> Aref;
    std::vector<scalar_t> Aref_data;
    real_t A_norm;
    if (check || ref) {
        // Norm of original matrix: || A ||_1
        A_norm = slate::norm( slate::Norm::One, A, opts );

        // For simplicity, always use ScaLAPACK format for Aref.
        Aref_data.resize( mlocA * nlocA );
        Aref = slate::Matrix<scalar_t>::fromScaLAPACK(
            m, n, &Aref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        slate::copy(A, Aref);
    }

    double gflop = lapack::Gflop<scalar_t>::geqrf(m, n);

    if (! ref_only) {
        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        double time = barrier_get_wtime(MPI_COMM_WORLD);

        //==================================================
        // Run SLATE test.
        //==================================================
        slate::qr_factor(A, T, opts);
        // Using traditional BLAS/LAPACK name
        // slate::geqrf(A, T, opts);

       double time_tst = barrier_get_wtime(MPI_COMM_WORLD) - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time_tst;
        params.gflops() = gflop / time_tst;

        if (verbose > 1) {
            print_matrix("A_factored", A);
            print_matrix("Tlocal",  T[0]);
            print_matrix("Treduce", T[1]);
        }
    }

    if (check) {
        //==================================================
        // Test results by checking backwards error
        //
        //      || QR - A ||_1
        //     ---------------- < tol * epsilon
        //      || A ||_1 * m
        //
        //==================================================

        // R is the upper part of A matrix.
        slate::TrapezoidMatrix<scalar_t> R(slate::Uplo::Upper, slate::Diag::NonUnit, A);

        std::vector<scalar_t> QR_data;
        slate::Matrix<scalar_t> QR;
        QR_data = std::vector<scalar_t>(Aref_data.size(), zero);
        QR = slate::Matrix<scalar_t>::fromScaLAPACK(
            m, n, &QR_data[0], lldA, nb, p, q, MPI_COMM_WORLD);

        // R1 is the upper part of QR matrix.
        slate::TrapezoidMatrix<scalar_t> R1(slate::Uplo::Upper, slate::Diag::NonUnit, QR);

        // Zero out QR matrix.
        slate::set(zero, QR); //Already zero when QR_data is initialized above?
        // Copy A's upper trapezoid R to QR's upper trapezoid R1.
        slate::copy(R, R1);

        if (verbose > 1) {
            print_matrix("R", QR);
        }

        // Multiply QR by Q (implicitly stored in A and T).
        // Form QR, where Q's representation is in A and T, and R is in QR.
        slate::qr_multiply_by_q(
            slate::Side::Left, slate::Op::NoTrans, A, T, QR,
            {{slate::Option::Target, target}}
        );
        //---------------------
        // Using traditional BLAS/LAPACK name
        // slate::unmqr(slate::Side::Left, slate::Op::NoTrans, A, T, QR, {
        //     {slate::Option::Target, target}
        // });

        if (verbose > 1) {
            print_matrix("QR", QR);
        }

        // QR should now have the product Q*R, which should equal the original A.
        // Subtract the original A_ref from QR.
        // Form QR - A, where A is in Aref.
        // todo: slate::geadd(-one, Aref, QR);
        // using axpy assumes Aref_data and QR_data have same lda.
        blas::axpy(QR_data.size(), -one, &Aref_data[0], 1, &QR_data[0], 1);
        if (verbose > 1) {
            print_matrix("QR - A", QR);
        }

        // Norm of backwards error: || QR - A ||_1
        real_t R_norm = slate::norm(slate::Norm::One, QR);

        double residual = R_norm / (m*A_norm);
        params.error() = residual;
        real_t tol = params.tol() * 0.5 * std::numeric_limits<real_t>::epsilon();
        params.okay() = (params.error() <= tol);
    }

    if (ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            // A comparison with a reference routine from ScaLAPACK for timing only

            // BLACS/MPI variables
            int ictxt, myrow, mycol, info, p_, q_;
            int Aref_desc[9], QR_desc[9];
            int iam = 0, nprocs = 1;
            // initialize BLACS and ScaLAPACK
            Cblacs_pinfo(&iam, &nprocs);
            slate_assert(p*q <= nprocs);
            Cblacs_get(-1, 0, &ictxt);
            Cblacs_gridinit(&ictxt, "Col", p, q);
            Cblacs_gridinfo(ictxt, &p_, &q_, &myrow, &mycol);
            assert( p == p_ );
            assert( q == q_ );

            scalapack_descinit(Aref_desc, m, n, nb, nb, 0, 0, ictxt, mlocA, &info);
            slate_assert(info == 0);
            int64_t lldA = (int64_t)Aref_desc[8];
            std::vector<scalar_t> Aref_data(lldA*nlocA);

            // matrix QR, for checking result
            std::vector<scalar_t> QR_data(1);
            scalapack_descinit(QR_desc, m, n, nb, nb, 0, 0, ictxt, mlocA, &info);
            slate_assert(info == 0);

            // Form QR - A, where A is in Aref.
            // todo: slate::geadd(-one, Aref, QR);
            // using axpy assumes Aref_data and QR_data have same lda.
            blas::axpy(QR_data.size(), -one, &Aref_data[0], 1, &QR_data[0], 1);

            // tau vector for ScaLAPACK
            int64_t ltau = num_local_rows_cols(std::min(m, n), nb, mycol, q);
            std::vector<scalar_t> tau(ltau);

            // workspace for ScaLAPACK
            int64_t lwork;
            std::vector<scalar_t> work(1);
            //---------------

            // set MKL num threads appropriately for parallel BLAS
            int omp_num_threads;
            #pragma omp parallel
            { omp_num_threads = omp_get_num_threads(); }
            int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);
            int64_t info_ref = 0;

            // query for workspace size
            scalar_t dummy;
            scalapack_pgeqrf(m, n, &Aref_data[0], 1, 1, Aref_desc, tau.data(),
                            &dummy, -1, &info_ref);
            lwork = int64_t( real( dummy ) );
            work.resize(lwork);

            //==================================================
            // Run ScaLAPACK reference test.
            //==================================================
            double time = barrier_get_wtime(MPI_COMM_WORLD);
            scalapack_pgeqrf(m, n, &Aref_data[0], 1, 1, Aref_desc, tau.data(),
                            work.data(), lwork, &info_ref);
            slate_assert(info_ref == 0);
            double time_ref = barrier_get_wtime(MPI_COMM_WORLD) - time;

            params.ref_time() = time_ref;
            params.ref_gflops() = gflop / time_ref;

            slate_set_num_blas_threads(saved_num_threads);
            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering
        #else
            if (mpi_rank == 0)
               printf( "ScaLAPACK not available\n" );
        #endif
    }

}
// -----------------------------------------------------------------------------
void test_geqrf(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_geqrf_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_geqrf_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_geqrf_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_geqrf_work<std::complex<double>> (params, run);
            break;
    }
}
