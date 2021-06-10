// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"
#include "blas/flops.hh"
#include "lapack/flops.hh"
#include "print_matrix.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"
#include "aux/Debug.hh"
#include "grid_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#define SLATE_HAVE_SCALAPACK
//------------------------------------------------------------------------------
template <typename scalar_t>
void test_hegst_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;

    // constants
    const scalar_t one = 1.0;

    int64_t itype = params.itype();
    slate::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t p = params.grid.m();
    int64_t q = params.grid.n();
    int64_t nb = params.nb();
    int64_t lookahead = params.lookahead();
    bool ref_only = params.ref() == 'o';
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Target target = params.target();
    slate::Origin origin = params.origin();
    params.matrix.mark();
    params.matrixB.mark();

    params.time();
    params.ref_time();
    // params.gflops(); // todo
    // params.ref_gflops(); // todo

    origin = slate::Origin::ScaLAPACK;  // todo: for now

    if (! run) {
        params.matrix.kind.set_default( "rand_dominant" );
        params.matrixB.kind.set_default( "rand_dominant" );
        return;
    }

    slate::Options const opts =  {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    };

    if (origin != slate::Origin::ScaLAPACK) { // todo
        // Copy local ScaLAPACK data to GPU or CPU tiles.
        // auto A = slate::HermitianMatrix<scalar_t>(
        //                          uplo, n, nb, p, q, MPI_COMM_WORLD);
        // A.insertLocalTiles(origin2target(origin));
        // todo: need ScaLAPACK descriptor for copy.
        //copy(A_data.data(), A_desc, A);
        assert(false);
    }

    // MPI variables
    int mpi_rank;
    slate_mpi_call(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    int mpi_size;
    slate_mpi_call(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
    slate_assert(p*q <= mpi_size);

    int myrow = whoismyrow(mpi_rank, p);
    int mycol = whoismycol(mpi_rank, p);

    // Figure out local size, allocate, initialize
    int64_t mlocal = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocal = num_local_rows_cols(n, nb, mycol, q);
    int64_t lld   = mlocal;

    // Matrix A
    std::vector<scalar_t> A_data(lld*nlocal);
    auto A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                 uplo, n, A_data.data(), lld, nb, p, q, MPI_COMM_WORLD);

    slate::generate_matrix( params.matrix, A );

    if (verbose > 1) {
        print_matrix("A", A);
    }

    // Matrix Aref
    std::vector<scalar_t> Aref_data(lld*nlocal);
    auto Aref = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                    uplo, n, Aref_data.data(), lld, nb, p, q, MPI_COMM_WORLD);

    slate::copy( A, Aref );
    if (verbose > 2) {
        print_matrix("Aref", Aref);
    }

    // Matrix B
    std::vector<scalar_t> B_data(lld*nlocal);
    auto B = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                 uplo, n, B_data.data(), lld, nb, p, q, MPI_COMM_WORLD);

    slate::generate_matrix( params.matrixB, B );


    if (verbose > 1) {
        print_matrix("B", B);
    }

    // Factorize B
    slate::potrf(B, opts);

    if (verbose > 2) {
        print_matrix("B_factored", B);
    }

    if (! ref_only) {
        // todo
        //double gflop = lapack::Gflop<scalar_t>::hegst(n);

        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        double time = barrier_get_wtime(MPI_COMM_WORLD);

        //==================================================
        // Run SLATE test.
        //==================================================
        slate::hegst(itype, A, B, opts);
        time = barrier_get_wtime(MPI_COMM_WORLD) - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time;
        //params.gflops() = gflop / time_tst;

        if (verbose > 1) {
            print_matrix("A_hegst", A);
        }
    }

    if (check || ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            real_t A_norm = slate::norm(slate::Norm::One, Aref);

            int ictxt;
            Cblacs_get(-1, 0, &ictxt);
            Cblacs_gridinit(&ictxt, "Col", p, q);

            const int izero = 0;
            int A_desc[9], B_desc[9], info;
            scalapack_descinit(
                A_desc, n, n, nb, nb, izero, izero, ictxt, mlocal, &info);
            slate_assert(info == 0);
            scalapack_descinit(
                B_desc, n, n, nb, nb, izero, izero, ictxt, mlocal, &info);
            slate_assert(info == 0);
            const int64_t ione = 1;
            double scale;

            copy( A, &A_data[0], A_desc );
            copy( Aref, &Aref_data[0], A_desc );
            copy( B, &B_data[0], B_desc );

            // set MKL num threads appropriately for parallel BLAS
            int omp_num_threads;
            #pragma omp parallel
            { omp_num_threads = omp_get_num_threads(); }
            int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================

            double time = barrier_get_wtime(MPI_COMM_WORLD);

            scalapack_phegst(itype, uplo2str(uplo), n,
                             Aref_data.data(), ione, ione, A_desc,
                             B_data.data(),     ione, ione, B_desc,
                             &scale, &info);
            slate_assert(info == 0);

            time = barrier_get_wtime(MPI_COMM_WORLD) - time;

            params.ref_time() = time;
            // params.ref_gflops() = gflop / time_ref;

            if (verbose > 1) {
                print_matrix("Aref_hegst", Aref);
            }

            slate_set_num_blas_threads(saved_num_threads);

            if (! ref_only) {
                // Local operation: error = Aref - A
                blas::axpy(
                    Aref_data.size(), -one,
                    A_data.data(), 1,
                    Aref_data.data(), 1);

                params.error() = slate::norm(slate::Norm::One, Aref) / (n * A_norm);
                real_t tol = params.tol() * std::numeric_limits<real_t>::epsilon()/2;
                params.okay() = (params.error() <= tol);
            }
            Cblacs_gridexit(ictxt);
        #else
            if (mpi_rank == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }
}

// -----------------------------------------------------------------------------
void test_hegst(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_hegst_work<float>(params, run);
            break;

        case testsweeper::DataType::Double:
            test_hegst_work<double>(params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_hegst_work<std::complex<float>>(params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_hegst_work<std::complex<double>>(params, run);
            break;
    }
}
