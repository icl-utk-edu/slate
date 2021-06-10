// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"
#include "blas/flops.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"
#include "grid_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#define SLATE_HAVE_SCALAPACK
//------------------------------------------------------------------------------
template<typename scalar_t>
void test_symm_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using slate::Norm;

    // get & mark input values
    slate::Side side = params.side();
    slate::Uplo uplo = params.uplo();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    scalar_t alpha = params.alpha.get<scalar_t>();
    scalar_t beta = params.beta.get<scalar_t>();
    int p = params.grid.m();
    int q = params.grid.n();
    int64_t nb = params.nb();
    int64_t lookahead = params.lookahead();
    slate::Norm norm = params.norm();
    bool check = params.check() == 'y';
    bool ref = params.ref() == 'y';
    bool trace = params.trace() == 'y';
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    params.matrix.mark();
    params.matrixB.mark();
    params.matrixC.mark();

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    if (! run)
        return;

    slate::Options const opts =  {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    };

    // Error analysis applies in these norms.
    slate_assert(norm == Norm::One || norm == Norm::Inf || norm == Norm::Fro);

    // sizes of data
    int64_t An = (side == slate::Side::Left ? m : n);
    int64_t Am = An;
    int64_t Bm = m;
    int64_t Bn = n;
    int64_t Cm = m;
    int64_t Cn = n;

    // Constants
    const int izero = 0, ione = 1;

    // Local values
    int myrow, mycol;
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    // matrix A, figure out local size, allocate, create descriptor, initialize
    int64_t mlocA = num_local_rows_cols(Am, nb, myrow, p);
    int64_t nlocA = num_local_rows_cols(An, nb, mycol, q);
    int64_t lldA  = blas::max(1, mlocA); // local leading dimension of A
    std::vector<scalar_t> A_data(lldA*nlocA);

    // matrix B, figure out local size, allocate, create descriptor, initialize
    int64_t mlocB = num_local_rows_cols(Bm, nb, myrow, p);
    int64_t nlocB = num_local_rows_cols(Bn, nb, mycol, q);
    int64_t lldB  = blas::max(1, mlocB); // local leading dimension of B
    std::vector<scalar_t> B_data(lldB*nlocB);

    // matrix C, figure out local size, allocate, create descriptor, initialize
    int64_t mlocC = num_local_rows_cols(Cm, nb, myrow, p);
    int64_t nlocC = num_local_rows_cols(Cn, nb, mycol, q);
    int64_t lldC  = blas::max(1, mlocC); // local leading dimension of C
    std::vector<scalar_t> C_data(lldC*nlocC);

    slate::SymmetricMatrix<scalar_t> A;
    slate::Matrix<scalar_t> B, C;
    if (origin != slate::Origin::ScaLAPACK) {
        // Copy local ScaLAPACK data to GPU or CPU tiles.
        slate::Target origin_target = origin2target(origin);
        A = slate::SymmetricMatrix<scalar_t>(uplo, An, nb, p, q, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);

        B = slate::Matrix<scalar_t>(Bm, Bn, nb, p, q, MPI_COMM_WORLD);
        B.insertLocalTiles(origin_target);

        C = slate::Matrix<scalar_t>(Cm, Cn, nb, p, q, MPI_COMM_WORLD);
        C.insertLocalTiles(origin_target);
    }
    else {
        // create SLATE matrices from the ScaLAPACK layouts
        A = slate::SymmetricMatrix<scalar_t>::fromScaLAPACK(uplo, An, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        B = slate::Matrix<scalar_t>::fromScaLAPACK(Bm, Bn, &B_data[0], lldB, nb, p, q, MPI_COMM_WORLD);
        C = slate::Matrix<scalar_t>::fromScaLAPACK(Cm, Cn, &C_data[0], lldC, nb, p, q, MPI_COMM_WORLD);
    }

    slate::generate_matrix( params.matrix, A);
    slate::generate_matrix( params.matrixB, B);
    slate::generate_matrix( params.matrixC, C);

    // if check is required, copy test data and create a descriptor for it
    slate::Matrix<scalar_t> Cref;
    std::vector<scalar_t> Cref_data;
    if (check || ref) {
        Cref_data.resize( C_data.size() );
        Cref = slate::Matrix<scalar_t>::fromScaLAPACK(Cm, Cn, &Cref_data[0], lldC, nb, p, q, MPI_COMM_WORLD);
        slate::copy( C, Cref );
    }

    if (side == slate::Side::Left)
        slate_assert(A.mt() == C.mt());
    else
        slate_assert(A.mt() == C.nt());
    slate_assert(B.mt() == C.mt());
    slate_assert(B.nt() == C.nt());

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    double time = barrier_get_wtime(MPI_COMM_WORLD);

    //==================================================
    // Run SLATE test.
    // C = alpha A B + beta C (left) or
    // C = alpha B A + beta C (right).
    //==================================================
    if (side == slate::Side::Left)
        slate::multiply(alpha, A, B, beta, C, opts);
    else if (side == slate::Side::Right)
        slate::multiply(alpha, B, A, beta, C, opts);
    else
        throw slate::Exception("unknown side");
    // Using traditional BLAS/LAPACK name
    // slate::symm(side, alpha, A, B, beta, C, opts);

    time = barrier_get_wtime(MPI_COMM_WORLD) - time;

    if (trace) slate::trace::Trace::finish();

    // Compute and save timing/performance
    double gflop = blas::Gflop<scalar_t>::symm(side, m, n);
    params.time() = time;
    params.gflops() = gflop / time;

    if (check || ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            // comparison with reference routine from ScaLAPACK

            // BLACS/MPI variables
            int ictxt, p_, q_, myrow_, mycol_, info;
            int A_desc[9], B_desc[9], C_desc[9], Cref_desc[9];
            int mpi_rank_ = 0, nprocs = 1;
            //int iseed = 1;

            // initialize BLACS and ScaLAPACK
            Cblacs_pinfo(&mpi_rank_, &nprocs);
            slate_assert( mpi_rank == mpi_rank_ );
            slate_assert(p*q <= nprocs);
            Cblacs_get(-1, 0, &ictxt);
            Cblacs_gridinit(&ictxt, "Col", p, q);
            Cblacs_gridinfo(ictxt, &p_, &q_, &myrow_, &mycol_);
            slate_assert(p == p_ && q == q_);
            slate_assert( myrow == myrow_ );
            slate_assert( mycol == mycol_ );

            scalapack_descinit(A_desc, Am, An, nb, nb, izero, izero, ictxt, mlocA, &info);
            slate_assert(info == 0);

            scalapack_descinit(B_desc, Bm, Bn, nb, nb, izero, izero, ictxt, mlocB, &info);
            slate_assert(info == 0);

            scalapack_descinit(C_desc, Cm, Cn, nb, nb, izero, izero, ictxt, mlocC, &info);
            slate_assert(info == 0);

            scalapack_descinit(Cref_desc, Cm, Cn, nb, nb, izero, izero, ictxt, mlocC, &info);
            slate_assert(info == 0);

            if (origin != slate::Origin::ScaLAPACK) {
                // Copy SLATE result back from GPU or CPU tiles.
                copy(A, &A_data[0], A_desc);
                copy(B, &B_data[0], B_desc);
                copy(C, &C_data[0], C_desc);
            }

            // set MKL num threads appropriately for parallel BLAS
            int omp_num_threads;
            #pragma omp parallel
            { omp_num_threads = omp_get_num_threads(); }
            int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);

            // allocate workspace for norms
            size_t ldw = nb*ceil(ceil(mlocA / (double) nb) / (scalapack_ilcm(&p, &q) / p));
            std::vector<real_t> worklansy(2*nlocA + mlocA + ldw);
            std::vector<real_t> worklange(std::max({mlocC, nlocC, mlocB, nlocB}));

            // get norms of the original data
            real_t A_norm = scalapack_plansy(norm2str(norm), uplo2str(uplo), An, &A_data[0], ione, ione, A_desc, &worklansy[0]);
            real_t B_norm = scalapack_plange(norm2str(norm), Bm, Bn, &B_data[0], ione, ione, B_desc, &worklange[0]);
            real_t C_orig_norm = scalapack_plange(norm2str(norm), Cm, Cn, &Cref_data[0], ione, ione, Cref_desc, &worklange[0]);

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            time = barrier_get_wtime(MPI_COMM_WORLD);
            scalapack_psymm(side2str(side), uplo2str(uplo), m, n, alpha,
                            &A_data[0], ione, ione, A_desc,
                            &B_data[0], ione, ione, B_desc, beta,
                            &Cref_data[0], ione, ione, Cref_desc);
            MPI_Barrier(MPI_COMM_WORLD);
            time = barrier_get_wtime(MPI_COMM_WORLD) - time;

            // Local operation: error = Cref_data - C_data
            blas::axpy(Cref_data.size(), -1.0, &C_data[0], 1, &Cref_data[0], 1);

            // norm(Cref_data - C_data)
            real_t C_diff_norm = scalapack_plange(norm2str(norm), Cm, Cn, &Cref_data[0], ione, ione, Cref_desc, &worklange[0]);

            real_t error = C_diff_norm
                         / (sqrt(real_t(An) + 2) * std::abs(alpha) * A_norm * B_norm
                            + 2 * std::abs(beta) * C_orig_norm);

            params.ref_time() = time;
            params.ref_gflops() = gflop / time;
            params.error() = error;

            slate_set_num_blas_threads(saved_num_threads);

            real_t eps = std::numeric_limits<real_t>::epsilon();
            params.okay() = (params.error() <= 3*eps);
            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering
        #else
            if (mpi_rank == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }
}

// -----------------------------------------------------------------------------
void test_symm(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_symm_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_symm_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_symm_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_symm_work<std::complex<double>> (params, run);
            break;
    }
}
