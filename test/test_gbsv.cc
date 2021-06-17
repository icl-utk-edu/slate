// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "slate/BandMatrix.hh"
#include "test.hh"
#include "blas/flops.hh"
#include "lapack/flops.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "print_matrix.hh"
#include "grid_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#define SLATE_HAVE_SCALAPACK
//------------------------------------------------------------------------------
template <typename scalar_t>
void test_gbsv_work(Params& params, bool run)
{
    using blas::max;
    using blas::real;
    using real_t = blas::real_type<scalar_t>;
    using llong = long long;

    // Constants
    const int izero = 0, ione = 1;
    const scalar_t one = 1.0;

    // get & mark input values
    slate::Op trans = slate::Op::NoTrans;
    if (params.routine == "gbtrs")
        trans = params.trans();

    int64_t m;
    if (params.routine == "gbtrf")
        m = params.dim.m();
    else
        m = params.dim.n();  // square, n-by-n

    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs();
    int64_t kl = params.kl();
    int64_t ku = params.ku();
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
    //params.gflops();
    //params.ref_time();
    //params.ref_gflops();

    if (! run)
        return;

    if (origin != slate::Origin::ScaLAPACK) {
        printf("skipping: currently only origin=scalapack is supported\n");
        return;
    }

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

    // matrix B, figure out local size, allocate, create descriptor, initialize
    int64_t mlocB = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocB = num_local_rows_cols(nrhs, nb, mycol, q);
    int64_t lldB  = blas::max(1, mlocB); // local leading dimension of B
    std::vector<scalar_t> B_data(lldB*nlocB);

    // Create SLATE matrix from the ScaLAPACK layouts
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(
                 n, nrhs, &B_data[0], lldB, nb, p, q, MPI_COMM_WORLD);

    slate::generate_matrix(params.matrix, B);

    int64_t iseeds[4] = { myrow, mycol, 2, 3 };
    auto A     = slate::BandMatrix<scalar_t>(m, n, kl, ku, nb, p, q, MPI_COMM_WORLD);
    auto Aorig = slate::BandMatrix<scalar_t>(m, n, kl, ku, nb, p, q, MPI_COMM_WORLD);
    slate::Pivots pivots;

    int64_t klt = slate::ceildiv(kl, nb);
    int64_t kut = slate::ceildiv(ku, nb);
    int64_t jj = 0;
    for (int64_t j = 0; j < A.nt(); ++j) {
        int64_t ii = 0;
        for (int64_t i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal(i, j) && i >= j - kut && i <= j + klt) {
                A.tileInsert(i, j);
                Aorig.tileInsert(i, j);
                auto T = A(i, j);
                lapack::larnv(2, iseeds, T.size(), T.data());
                for (int64_t tj = jj; tj < jj + T.nb(); ++tj) {
                    for (int64_t ti = ii; ti < ii + T.mb(); ++ti) {
                        if (-kl > tj-ti || tj-ti > ku) {
                            // set outside band to zero
                            T.at(ti - ii, tj - jj) = 0;
                        }
                    }
                }
                auto T2 = Aorig(i, j);
                lapack::lacpy(lapack::MatrixType::General, T.mb(), T.nb(),
                              T.data(), T.stride(),
                              T2.data(), T2.stride());
            }
            ii += A.tileMb(i);
        }
        jj += A.tileNb(j);
    }

    if (verbose > 1) {
        printf("%% rank %d A kl %lld, ku %lld\n",
               A.mpiRank(), llong( A.lowerBandwidth() ), llong( A.upperBandwidth() ));
        print_matrix("A", A);
        print_matrix("B", B);
    }

    // if check is required, copy test data
    slate::Matrix<scalar_t> Bref;
    std::vector<scalar_t> Bref_data;
    if (check || ref) {
        Bref_data.resize( B_data.size() );
        Bref = slate::Matrix<scalar_t>::fromScaLAPACK(
                   n, nrhs, &Bref_data[0], lldB, nb, p, q, MPI_COMM_WORLD);
        slate::copy( B, Bref);
    }

    // todo: gflops formula for band.
    //double gflop;
    //if (params.routine == "getrf")
    //    gflop = lapack::Gflop<scalar_t>::getrf(m, n);
    //else if (params.routine == "getrs")
    //    gflop = lapack::Gflop<scalar_t>::getrs(n, nrhs);
    //else
    //    gflop = lapack::Gflop<scalar_t>::gesv(n, nrhs);

    if (! ref_only) {
        if (params.routine == "gbtrs") {
            // Factor matrix A.
            slate::lu_factor(A, pivots, opts);
            // Using traditional BLAS/LAPACK name
            // slate::gbtrf(A, pivots, opts);
        }

        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        double time = barrier_get_wtime(MPI_COMM_WORLD);

        //==================================================
        // Run SLATE test.
        // One of:
        // gbtrf: Factor PA = LU.
        // gbtrs: Solve AX = B, after factoring A above.
        // gbsv:  Solve AX = B, including factoring A.
        //==================================================
        if (params.routine == "gbtrf") {
            slate::lu_factor(A, pivots, opts);
            // Using traditional BLAS/LAPACK name
            // slate::gbtrf(A, pivots, opts);
        }
        else if (params.routine == "gbtrs") {
            auto opA = A;
            if (trans == slate::Op::Trans)
                opA = transpose(A);
            else if (trans == slate::Op::ConjTrans)
                opA = conjTranspose(A);

            slate::lu_solve_using_factor(opA, pivots, B, opts);
            // Using traditional BLAS/LAPACK name
            // slate::gbtrs(opA, pivots, B, opts);
        }
        else {
            slate::lu_solve(A, B, opts);
            // Using traditional BLAS/LAPACK name
            // slate::gbsv(A, pivots, B, opts);
        }

        time = barrier_get_wtime(MPI_COMM_WORLD) - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time;
        ///params.gflops() = gflop / time;

        if (verbose > 1) {
            printf("%% rank %d A2 kl %lld, ku %lld\n",
                   A.mpiRank(), llong( A.lowerBandwidth() ), llong( A.upperBandwidth() ));
            print_matrix("A2", A);
            print_matrix("B2", B);
            printf( "nb = %lld;\n", llong( nb ) );
            printf( "pivots = [\n" );
            int ii = 0;
            for (auto pivot: pivots) {
                int i = ii / nb;
                for (auto piv: pivot) {
                    printf( "  %d*nb + %lld*nb + %lld\n", i,
                            llong( piv.tileIndex() ), llong( piv.elementOffset() ) );
                    ++ii;
                }
                printf( "\n" );
            }
            printf( "] + 1;\n" );
        }
    }

    if (check) {
        #ifdef SLATE_HAVE_SCALAPACK

            // BLACS/MPI variables
            int ictxt, p_, q_, myrow_, mycol_, info;
            int B_desc[9], Bref_desc[9];
            int mpi_rank_ = 0, nprocs = 1;

            // initialize BLACS and ScaLAPACK
            Cblacs_pinfo(&mpi_rank, &nprocs);
            slate_assert( mpi_rank_ == mpi_rank );
            slate_assert(p*q <= nprocs);
            Cblacs_get(-1, 0, &ictxt);
            Cblacs_gridinit(&ictxt, "Col", p, q);
            Cblacs_gridinfo(ictxt, &p_, &q_, &myrow_, &mycol_);
            slate_assert( p == p_ );
            slate_assert( q == q_ );
            slate_assert( myrow == myrow_ );
            slate_assert( mycol == mycol_ );

            scalapack_descinit(B_desc, n, nrhs, nb, nb, izero, izero, ictxt, mlocB, &info);
            slate_assert(info == 0);

            scalapack_descinit(Bref_desc, n, nrhs, nb, nb, izero, izero, ictxt, mlocB, &info);
            slate_assert(info == 0);

            //==================================================
            // Test results by checking the residual
            //
            //           || B - AX ||_1
            //     --------------------------- < tol * epsilon
            //      || A ||_1 * || X ||_1 * N
            //
            //==================================================

            // LAPACK (dget02) does
            // max_j || A * x_j - b_j ||_1 / (|| A ||_1 * || x_j ||_1).
            // No N?

            if (params.routine == "gbtrf") {
                // Solve AX = B.
                slate::lu_solve_using_factor(A, pivots, B, opts);
                // Using traditional BLAS/LAPACK name
                // slate::gbtrs(A, pivots, B, opts);
            }

            // allocate work space
            std::vector<real_t> worklangeB(std::max(mlocB, nlocB));

            // Norm of original matrix: || A ||_1
            real_t A_norm = slate::norm(slate::Norm::One, Aorig);
            // Norm of updated rhs matrix: || X ||_1
            real_t X_norm = scalapack_plange("1", n, nrhs, &B_data[0], ione, ione, B_desc, &worklangeB[0]);

            // Bref_data -= op(Aref)*B_data
            auto opAorig = Aorig;
            if (trans == slate::Op::Trans)
                opAorig = transpose(Aorig);
            else if (trans == slate::Op::ConjTrans)
                opAorig = conjTranspose(Aorig);
            slate::multiply(-one, opAorig, B, one, Bref);
            // Using traditional BLAS/LAPACK name
            // slate::gbmm(-one, opAorig, B, one, Bref);

            // Norm of residual: || B - AX ||_1
            real_t R_norm = scalapack_plange("1", n, nrhs, &Bref_data[0], ione, ione, Bref_desc, &worklangeB[0]);
            double residual = R_norm / (n*A_norm*X_norm);
            params.error() = residual;

            real_t tol = params.tol() * 0.5 * std::numeric_limits<real_t>::epsilon();
            params.okay() = (params.error() <= tol);

            if (verbose > 0) {
                printf("Anorm = %.4e; Xnorm = %.4e; Rnorm = %.4e; error = %.4e;\n",
                       A_norm, X_norm, R_norm, residual);
            }
            if (verbose > 1) {
                print_matrix("Residual", n, nrhs, &Bref_data[0], lldB, p, q, MPI_COMM_WORLD);
            }

            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering
        #else
            if (mpi_rank == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }

    // todo: reference solution requires setting up band matrix in ScaLAPACK's
    // band storage format.
}

// -----------------------------------------------------------------------------
void test_gbsv(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_gbsv_work<float>(params, run);
            break;

        case testsweeper::DataType::Double:
            test_gbsv_work<double>(params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_gbsv_work<std::complex<float>>(params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gbsv_work<std::complex<double>>(params, run);
            break;
    }
}
