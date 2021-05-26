// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"
#include "blas/flops.hh"

#include "scalapack_support_routines.hh"
#include "print_matrix.hh"
#include "band_utils.hh"
#include "grid_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#undef PIN_MATRICES

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
    //using llong = long long;

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

    if (! run)
        return;

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

    // sizes of A and B
    int64_t Am = (transA == slate::Op::NoTrans ? m : k);
    int64_t An = (transA == slate::Op::NoTrans ? k : m);
    int64_t Bm = (transB == slate::Op::NoTrans ? k : n);
    int64_t Bn = (transB == slate::Op::NoTrans ? n : k);

    // constants
    const scalar_t one = 1;

    // Local values
    int myrow, mycol;
    int mpi_rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    // matrix A, figure out local size, allocate, initialize
    int64_t mlocA = num_local_rows_cols(Am, nb, myrow, p);
    int64_t nlocA = num_local_rows_cols(An, nb, mycol, q);
    int64_t lldA  = blas::max(1, mlocA); // local leading dimension of A_band
    std::vector<scalar_t> A_data(lldA*nlocA);

    if (verbose > 1) {
        print_matrix("A_data", mlocA, nlocA, &A_data[0], lldA, p, q, MPI_COMM_WORLD);
    }

    // matrix B, figure out local size, allocate, initialize
    int64_t mlocB = num_local_rows_cols(Bm, nb, myrow, p);
    int64_t nlocB = num_local_rows_cols(Bn, nb, mycol, q);
    int64_t lldB  = blas::max(1, mlocB); // local leading dimension of B
    std::vector<scalar_t> B_data(lldB*nlocB);

    // matrix C, figure out local size, allocate, initialize
    int64_t mlocC = num_local_rows_cols(m, nb, myrow, p);
    int64_t nlocC = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldC  = blas::max(1, mlocC); // local leading dimension of C
    std::vector<scalar_t> C_data(lldC*nlocC);

    #ifdef PIN_MATRICES
    int cuerror;
    cuerror = cudaHostRegister(&A_data[0], (size_t)size_A*sizeof(scalar_t), cudaHostRegisterDefault);
    cuerror = cudaHostRegister(&B_data[0], (size_t)size_A*sizeof(scalar_t), cudaHostRegisterDefault);
    cuerror = cudaHostRegister(&C_data[0], (size_t)size_A*sizeof(scalar_t), cudaHostRegisterDefault);
    #endif

    auto Aref = slate::Matrix<scalar_t>::fromScaLAPACK(
                Am, An, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD );
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(
                 Bm, Bn, &B_data[0], lldB, nb, p, q, MPI_COMM_WORLD);
    auto C = slate::Matrix<scalar_t>::fromScaLAPACK(
                 m, n, &C_data[0], lldC, nb, p, q, MPI_COMM_WORLD);
    slate::generate_matrix( params.matrix, Aref );
    slate::generate_matrix(params.matrixB, B);
    slate::generate_matrix(params.matrixC, C);
    zeroOutsideBand(&A_data[0], Am, An, kl, ku, nb, nb, myrow, mycol, p, q, lldA);

    // create SLATE matrices from the ScaLAPACK layouts
    auto A_band = BandFromScaLAPACK(
                 Am, An, kl, ku, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD);

    // if check is required, copy test data and create a descriptor for it
    slate::Matrix<scalar_t> Cref;
    std::vector<scalar_t> Cref_data;
    if (check || ref) {
        Cref_data.resize( C_data.size() );
        Cref = slate::Matrix<scalar_t>::fromScaLAPACK(
                 m, n, &Cref_data[0], lldC, nb, p, q, MPI_COMM_WORLD);
        slate::copy( C, Cref );
    }

    if (verbose > 1) {
        //printf("%% rank %d A2 kl %lld, ku %lld\n",
        //       A_band.mpiRank(), A_band.lowerBandwidth(), A_band.upperBandwidth());
        print_matrix("A_band", A_band);
        print_matrix("B", B);
        print_matrix("C", C);
        printf("alpha = %.4f + %.4fi;\nbeta  = %.4f + %.4fi;\n",
               real(alpha), imag(alpha),
               real(beta), imag(beta));
    }

    //printf("%% trans\n");
    if (transA == slate::Op::Trans)
        A_band = transpose(A_band);
    else if (transA == slate::Op::ConjTrans)
        A_band = conjTranspose(A_band);

    if (transB == slate::Op::Trans)
        B = transpose(B);
    else if (transB == slate::Op::ConjTrans)
        B = conjTranspose(B);

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

    //---------------------
    // Using traditional BLAS/LAPACK name
    // slate::gbmm(alpha, A_band, B, beta, C, opts);

    time = barrier_get_wtime(MPI_COMM_WORLD) - time;

    if (trace) slate::trace::Trace::finish();

    // compute and save timing/performance
    double gflop = blas::Gflop<scalar_t>::gbmm(m, n, k, kl, ku);
    params.time() = time;
    params.gflops() = gflop / time;

    if (verbose > 1) {
        print_matrix("C2", C);
        print_matrix("C_data", mlocC, nlocC, &C_data[0], lldC, p, q, MPI_COMM_WORLD);
    }

    if (check || ref) {
        //printf("%% check & ref\n");
        // comparison with SLATE non-band routine

        if (transA == slate::Op::Trans)
            Aref = transpose(Aref);
        else if (transA == slate::Op::ConjTrans)
            Aref = conjTranspose(Aref);

        // set MKL num threads appropriately for parallel BLAS
        int omp_num_threads;
        #pragma omp parallel
        { omp_num_threads = omp_get_num_threads(); }
        int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);

        //==================================================
        // Run SLATE non-band routine
        //==================================================
        if (verbose > 1) {
            print_matrix("Cref", Cref);
        }
        time = barrier_get_wtime(MPI_COMM_WORLD);
        slate::multiply( alpha, Aref, B, beta, Cref, opts );
        // get differences Cref_data = Cref_data - C_data
        slate::geadd( -one, C, one, Cref );
        real_t error = slate::norm( norm, Cref );
        time = barrier_get_wtime(MPI_COMM_WORLD) - time;

        if (verbose > 1) {
            print_matrix( "C_diff", Cref );
        }

        params.ref_time() = time;
        params.ref_gflops() = gflop / time;
        params.error() = error;

        slate_set_num_blas_threads(saved_num_threads);

        // Allow 3*eps; complex needs 2*sqrt(2) factor; see Higham, 2002, sec. 3.6.
        real_t eps = std::numeric_limits<real_t>::epsilon();
        params.okay() = (params.error() <= 3*eps);
    }
    //printf("%% done\n");

    #ifdef PIN_MATRICES
    cuerror = cudaHostUnregister(&A_data[0]);
    cuerror = cudaHostUnregister(&B_data[0]);
    cuerror = cudaHostUnregister(&C_data[0]);
    #endif
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
