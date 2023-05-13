// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"
#include "blas/flops.hh"

#include "print_matrix.hh"
#include "band_utils.hh"
#include "grid_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template<typename scalar_t>
void test_gbmm_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;
    using blas::max;
    using blas::min;
    using slate::Norm;

    // Constants
    const scalar_t one = 1;

    // get & mark input values
    slate::Op transA = params.transA();
    slate::Op transB = params.transB();
    scalar_t alpha = params.alpha.get<scalar_t>();
    scalar_t beta = params.beta.get<scalar_t>();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t k = params.dim.k();
    int64_t kl = params.kl();
    int64_t ku = params.ku();
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
    params.matrixC.mark();

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    // Suppress norm, nrhs from output; they're only for checks.
    params.norm.width( 0 );
    params.nrhs.width( 0 );

    if (! run)
        return;

    if (origin != slate::Origin::ScaLAPACK) {
        params.msg() = "skipping: currently only origin=scalapack is supported";
        return;
    }

    slate::Options const opts =  {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    };

    // Error analysis applies in these norms.
    slate_assert(norm == Norm::One || norm == Norm::Inf || norm == Norm::Fro);

    // sizes of A and B
    int64_t Am = (transA == slate::Op::NoTrans ? m : k);
    int64_t An = (transA == slate::Op::NoTrans ? k : m);
    int64_t Bm = (transB == slate::Op::NoTrans ? k : n);
    int64_t Bn = (transB == slate::Op::NoTrans ? n : k);

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    // Matrix A: figure out local size.
    int64_t mlocA = num_local_rows_cols(Am, nb, myrow, p);
    int64_t nlocA = num_local_rows_cols(An, nb, mycol, q);
    int64_t lldA  = blas::max(1, mlocA); // local leading dimension of A_band
    std::vector<scalar_t> A_data(lldA*nlocA);

    // Matrix B: figure out local size.
    int64_t mlocB = num_local_rows_cols(Bm, nb, myrow, p);
    int64_t nlocB = num_local_rows_cols(Bn, nb, mycol, q);
    int64_t lldB  = blas::max(1, mlocB); // local leading dimension of B
    std::vector<scalar_t> B_data(lldB*nlocB);

    // Matrix C: figure out local size.
    int64_t mlocC = num_local_rows_cols(m, nb, myrow, p);
    int64_t nlocC = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldC  = blas::max(1, mlocC); // local leading dimension of C
    std::vector<scalar_t> C_data(lldC*nlocC);

    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(
                 Am, An, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD );
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(
                 Bm, Bn, &B_data[0], lldB, nb, p, q, MPI_COMM_WORLD);
    auto C = slate::Matrix<scalar_t>::fromScaLAPACK(
                 m, n, &C_data[0], lldC, nb, p, q, MPI_COMM_WORLD);
    slate::generate_matrix(params.matrix,  A);
    slate::generate_matrix(params.matrixB, B);
    slate::generate_matrix(params.matrixC, C);
    zeroOutsideBand(&A_data[0], Am, An, kl, ku, nb, nb, myrow, mycol, p, q, lldA);

    // create SLATE matrices from the ScaLAPACK layouts
    auto A_band = BandFromScaLAPACK(
                      Am, An, kl, ku, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD);

    // If check is required, copy test data.
    slate::Matrix<scalar_t> Cref;
    if (check || ref) {
        Cref = slate::Matrix<scalar_t>(m, n, nb, p, q, MPI_COMM_WORLD);
        Cref.insertLocalTiles();
        slate::copy( C, Cref );
    }

    print_matrix("A_band", A_band, params);
    print_matrix("B", B, params);
    print_matrix("C", C, params);

    if (verbose > 1 && mpi_rank == 0) {
        printf( "alpha = %.4f + %.4fi;\n"
                "beta  = %.4f + %.4fi;\n",
                real( alpha ), imag( alpha ),
                real( beta  ), imag( beta  ) );
    }

    //printf("%% trans\n");
    if (transA == slate::Op::Trans)
        A_band = transpose(A_band);
    else if (transA == slate::Op::ConjTrans)
        A_band = conj_transpose( A_band );

    if (transB == slate::Op::Trans)
        B = transpose(B);
    else if (transB == slate::Op::ConjTrans)
        B = conj_transpose( B );

    slate_assert(A_band.mt() == C.mt());
    slate_assert(B.nt() == C.nt());
    slate_assert(A_band.nt() == B.mt());

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    double time = barrier_get_wtime(MPI_COMM_WORLD);

    //==================================================
    // Run SLATE test.
    // C = alpha A_band B + beta C.
    //==================================================
    slate::multiply(alpha, A_band, B, beta, C, opts);
    // Using traditional BLAS/LAPACK name
    // slate::gbmm(alpha, A_band, B, beta, C, opts);

    time = barrier_get_wtime(MPI_COMM_WORLD) - time;

    if (trace) slate::trace::Trace::finish();

    // compute and save timing/performance
    double gflop = blas::Gflop<scalar_t>::gbmm(m, n, k, kl, ku);
    params.time() = time;
    params.gflops() = gflop / time;

    print_matrix( "C_out", C, params );

    if (check || ref) {
        //printf("%% check & ref\n");
        // comparison with SLATE non-band routine

        if (transA == slate::Op::Trans)
            A = transpose(A);
        else if (transA == slate::Op::ConjTrans)
            A = conj_transpose( A );

        print_matrix( "Cref_in", Cref, params );

        // Get norms of the original data.
        real_t A_norm = slate::norm( norm, A );
        real_t B_norm = slate::norm( norm, B );
        real_t Cref_norm = slate::norm( norm, Cref );

        //==================================================
        // Run SLATE non-band routine
        //==================================================
        time = barrier_get_wtime(MPI_COMM_WORLD);

        slate::multiply( alpha, A, B, beta, Cref, opts );

        time = barrier_get_wtime(MPI_COMM_WORLD) - time;

        print_matrix( "Cref", Cref, params );

        // get differences Cref = Cref - C
        slate::add( -one, C, one, Cref );
        real_t C_diff_norm = slate::norm( norm, Cref ); // norm of residual

        real_t error = C_diff_norm
                    / (sqrt(real_t(k) + 2) * std::abs(alpha) * A_norm * B_norm
                        + 2 * std::abs(beta) * Cref_norm);

        print_matrix( "C_diff", Cref, params );

        params.ref_time() = time;
        params.ref_gflops() = gflop / time;
        params.error() = error;

        // Allow 3*eps; complex needs 2*sqrt(2) factor; see Higham, 2002, sec. 3.6.
        real_t eps = std::numeric_limits<real_t>::epsilon();
        params.okay() = (params.error() <= 3*eps);
    }
}

// -----------------------------------------------------------------------------
void test_gbmm(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_gbmm_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_gbmm_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_gbmm_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gbmm_work<std::complex<double>> (params, run);
            break;
    }
}
