// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"
#include "blas/flops.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "print_matrix.hh"
#include "band_utils.hh"
#include "grid_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#undef PIN_MATRICES
#define SLATE_HAVE_SCALAPACK
//------------------------------------------------------------------------------
template<typename scalar_t>
void test_tbsm_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;
    using slate::Norm;
    //using llong = long long;

    // get & mark input values
    slate::Side side = params.side();
    slate::Uplo uplo = params.uplo();
    slate::Op transA = params.transA();
    // ref. code to check can't do transB; disable for now.
    //slate::Op transB = params.transB();
    slate::Diag diag = params.diag();
    scalar_t alpha = params.alpha.get<scalar_t>();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t kd = params.kd();
    int64_t nb = params.nb();
    int p = params.grid.m();
    int q = params.grid.n();
    int64_t lookahead = params.lookahead();
    slate::Norm norm = params.norm();
    bool check = params.check() == 'y';
    bool ref = params.ref() == 'y';
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    params.matrix.mark();
    params.matrixB.mark();

    // mark non-standard output values
    params.time();
    //params.gflops();
    params.ref_time();
    //params.ref_gflops();

    if (! run) {
        // Note is printed before table header.
        printf("%% Note this does NOT test pivoting in tbsm. See gbtrs for that.\n");
        return;
    }

    if (origin != slate::Origin::ScaLAPACK) {
        printf("skipping: currently only origin=scalapack is supported\n");
        return;
    }

    slate::Options const opts =  {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    };

    // Error analysis applies in these norms.
    slate_assert(norm == Norm::One || norm == Norm::Inf || norm == Norm::Fro);

    // setup so trans(B) is m-by-n
    int64_t An  = (side == slate::Side::Left ? m : n);
    int64_t Am  = An;
    int64_t Bm  = m;  //(transB == slate::Op::NoTrans ? m : n);
    int64_t Bn  = n;  //(transB == slate::Op::NoTrans ? n : m);

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    // matrix A, figure out local size, allocate, create descriptor, initialize
    int64_t mlocA = num_local_rows_cols(Am, nb, myrow, p);
    int64_t nlocA = num_local_rows_cols(An, nb, mycol, q);
    int64_t lldA  = blas::max(1, mlocA); // local leading dimension of A_band
    std::vector<scalar_t> A_data(lldA*nlocA);

    // matrix B, figure out local size, allocate, create descriptor, initialize
    int64_t mlocB = num_local_rows_cols(Bm, nb, myrow, p);
    int64_t nlocB = num_local_rows_cols(Bn, nb, mycol, q);
    int64_t lldB  = blas::max(1, mlocB); // local leading dimension of B
    std::vector<scalar_t> B_data(lldB*nlocB);

    // create SLATE matrices from the ScaLAPACK layouts
    auto Aref = slate::Matrix<scalar_t>::fromScaLAPACK(
                    Am, An, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD );

    slate::generate_matrix(params.matrix, Aref);
    zeroOutsideBand(&A_data[0], Am, An, kd, kd, nb, nb, myrow, mycol, p, q, mlocA);

    auto Aband = BandFromScaLAPACK(
                     Am, An, kd, kd, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD);

    auto A = slate::TriangularBandMatrix<scalar_t>(uplo, diag, Aband);
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(
                 Bm, Bn, &B_data[0], lldB, nb, p, q, MPI_COMM_WORLD);
    slate::Pivots pivots;

    slate::generate_matrix(params.matrixB, B);

    // if check is required, copy test data and create a descriptor for it
    std::vector< scalar_t > Bref_data;
    if (check || ref) {
        Bref_data = B_data;
    }

    // Make A diagonally dominant to be reasonably well conditioned.
    // tbsm seems to pass with unit diagonal, even without diagonal dominance.
    for (int i = 0; i < A.mt(); ++i) {
        if (A.tileIsLocal(i, i)) {
            auto T = A(i, i);
            for (int ii = 0; ii < T.mb(); ++ii) {
                T.at(ii, ii) += Am;
            }
        }
    }

    if (transA == slate::Op::Trans)
        A = transpose(A);
    else if (transA == slate::Op::ConjTrans)
        A = conjTranspose(A);

    //if (transB == slate::Op::Trans)
    //    B = transpose(B);
    //else if (transB == slate::Op::ConjTrans)
    //    B = conjTranspose(B);

    if (verbose > 1) {
        // todo: print_matrix( A ) calls Matrix version;
        // need TriangularBandMatrix version.
        printf("alpha = %10.6f + %10.6fi;\n", real(alpha), imag(alpha));
        print_matrix("A_data", mlocA, nlocA, &A_data[0], lldA, p, q, MPI_COMM_WORLD);
        print_matrix("B_data", mlocB, nlocB, &B_data[0], lldB, p, q, MPI_COMM_WORLD);
        print_matrix("A", Aband);
        print_matrix("B", B);
    }

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    double time = barrier_get_wtime(MPI_COMM_WORLD);

    //==================================================
    // Run SLATE test.
    // Solve AX = alpha B (left) or XA = alpha B (right).
    //==================================================
    if (side == slate::Side::Left)
        slate::triangular_solve(alpha, A, B, opts);
    else if (side == slate::Side::Right)
        slate::triangular_solve(alpha, B, A, opts);
    else
        throw slate::Exception("unknown side");
    // Using traditional BLAS/LAPACK name
    // slate::tbsm(side, alpha, A, pivots, B, opts);

    time = barrier_get_wtime(MPI_COMM_WORLD) - time;

    if (trace) slate::trace::Trace::finish();

    // compute and save timing/performance
    //double gflop = blas::Gflop<scalar_t>::gemm(m, n, k);
    params.time() = time;
    //params.gflops() = gflop / time;

    if (verbose > 1) {
        print_matrix("B2", B);
        print_matrix("B2_tst", mlocB, nlocB, &B_data[0], lldB, p, q, MPI_COMM_WORLD);
    }

    if (check || ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            //printf("%% check & ref\n");
            // comparison with reference routine from ScaLAPACK

            // BLACS/MPI variables
            int ictxt, p_, q_, myrow_, mycol_, info;
            int A_desc[9], B_desc[9], Bref_desc[9];
            int mpi_rank_ = 0, nprocs = 1;
            //int iseed = 1;

            // initialize BLACS and ScaLAPACK
            Cblacs_pinfo(&mpi_rank_, &nprocs);
            slate_assert( mpi_rank_ == mpi_rank );
            slate_assert(p*q <= nprocs);
            Cblacs_get(-1, 0, &ictxt);
            Cblacs_gridinit(&ictxt, "Col", p, q);
            Cblacs_gridinfo(ictxt, &p_, &q_, &myrow_, &mycol_);
            slate_assert( p == p_ );
            slate_assert( q == q_ );
            slate_assert( myrow == myrow_ );
            slate_assert( mycol == mycol_ );

            scalapack_descinit(A_desc, Am, An, nb, nb, 0, 0, ictxt, mlocA, &info);
            slate_assert(info == 0);

            scalapack_descinit(B_desc, Bm, Bn, nb, nb, 0, 0, ictxt, mlocB, &info);
            slate_assert(info == 0);

            scalapack_descinit(Bref_desc, Bm, Bn, nb, nb, 0, 0, ictxt, mlocB, &info);
            slate_assert(info == 0);

            // set MKL num threads appropriately for parallel BLAS
            int omp_num_threads;
            #pragma omp parallel
            { omp_num_threads = omp_get_num_threads(); }
            int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);

            std::vector<real_t> worklantr(std::max(mlocA, nlocA));
            std::vector<real_t> worklange(std::max(mlocB, nlocB));

            // get norms of the original data
            real_t A_norm = scalapack_plantr(norm2str(norm), uplo2str(uplo), diag2str(diag), Am, An, &A_data[0], 1, 1, A_desc, &worklantr[0]);
            real_t B_orig_norm = scalapack_plange(norm2str(norm), Bm, Bn, &B_data[0], 1, 1, B_desc, &worklange[0]);

            if (verbose > 1) {
                print_matrix("Bref_data", mlocB, nlocB, &Bref_data[0], lldB, p, q, MPI_COMM_WORLD);
            }

            //==================================================
            // Run ScaLAPACK reference routine.
            // Note this is on a FULL matrix, so ignore reference performance!
            //==================================================
            time = barrier_get_wtime(MPI_COMM_WORLD);
            scalapack_ptrsm(side2str(side), uplo2str(uplo), op2str(transA), diag2str(diag),
                            m, n, alpha,
                            &A_data[0], 1, 1, A_desc,
                            &Bref_data[0], 1, 1, Bref_desc);
            time = barrier_get_wtime(MPI_COMM_WORLD) - time;

            if (verbose > 1) {
                print_matrix("B2_ref", mlocB, nlocB, &Bref_data[0], lldB, p, q, MPI_COMM_WORLD);
            }
            // local operation: error = Bref_data - B_data
            blas::axpy(Bref_data.size(), -1.0, &B_data[0], 1, &Bref_data[0], 1);

            // norm(Bref_data - B_data)
            real_t B_diff_norm = scalapack_plange(norm2str(norm), Bm, Bn, &Bref_data[0], 1, 1, Bref_desc, &worklange[0]);

            if (verbose > 1) {
                print_matrix("B_diff", mlocB, nlocB, &Bref_data[0], lldB, p, q, MPI_COMM_WORLD);
            }
            real_t error = B_diff_norm
                         / (sqrt(real_t(Am) + 2) * std::abs(alpha) * A_norm * B_orig_norm);

            params.ref_time() = time;
            //params.ref_gflops() = gflop / time;
            params.error() = error;

            slate_set_num_blas_threads(saved_num_threads);

            // Allow 3*eps; complex needs 2*sqrt(2) factor; see Higham, 2002, sec. 3.6.
            real_t eps = std::numeric_limits<real_t>::epsilon();
            params.okay() = (params.error() <= 3*eps);

            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering
        #else
            if (mpi_rank == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }
    //printf("%% done\n");

    #ifdef PIN_MATRICES
        cuerror = cudaHostUnregister(&A_data[0]);
        cuerror = cudaHostUnregister(&B_data[0]);
    #endif
}

// -----------------------------------------------------------------------------
void test_tbsm(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_tbsm_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_tbsm_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_tbsm_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_tbsm_work<std::complex<double>> (params, run);
            break;
    }
}
