// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
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
#include "auxiliary/Debug.hh"
#include "grid_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_hegst_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;

    // Constants
    const scalar_t one = 1.0;

    // get & mark input values
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
        // SLATE allocates CPU or GPU tiles.
        // A = slate::HermitianMatrix<scalar_t>(
        //                          uplo, n, nb, p, q, MPI_COMM_WORLD);
        // A.insertLocalTiles(origin2target(origin));
        assert(false);
    }

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    // Matrix A, B: figure out local size.
    int64_t mlocal = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocal = num_local_rows_cols(n, nb, mycol, q);
    int64_t lld    = blas::max(1, mlocal); // local leading dimension of A, B

    // Matrix A
    std::vector<scalar_t> A_data(lld*nlocal);
    auto A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                 uplo, n, A_data.data(), lld, nb, p, q, MPI_COMM_WORLD);
    real_t A_norm;

    slate::generate_matrix( params.matrix, A );

    print_matrix("A", A, params);

    // Matrix Aref
    std::vector<scalar_t> Aref_data(lld*nlocal);
    auto Aref = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                    uplo, n, Aref_data.data(), lld, nb, p, q, MPI_COMM_WORLD);

    slate::copy( A, Aref );
    print_matrix("Aref", Aref, params);

    // Matrix B
    std::vector<scalar_t> B_data(lld*nlocal);
    auto B = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                 uplo, n, B_data.data(), lld, nb, p, q, MPI_COMM_WORLD);

    slate::generate_matrix( params.matrixB, B );

    print_matrix("B", B, params);

    // Factorize B
    slate::potrf(B, opts);

    print_matrix("B_factored", B, params);

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
        //params.gflops() = gflop / time;

        print_matrix("A_hegst", A, params);
    }

    if (check || ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            A_norm = slate::norm(slate::Norm::One, Aref);

            int ictxt;
            Cblacs_get(-1, 0, &ictxt);
            Cblacs_gridinit(&ictxt, "Col", p, q);

            int A_desc[9], B_desc[9], info;
            scalapack_descinit(
                A_desc, n, n, nb, nb, 0, 0, ictxt, mlocal, &info);
            slate_assert(info == 0);
            scalapack_descinit(
                B_desc, n, n, nb, nb, 0, 0, ictxt, mlocal, &info);
            slate_assert(info == 0);
            double scale;

            copy( A, &A_data[0], A_desc );
            copy( Aref, &Aref_data[0], A_desc );
            copy( B, &B_data[0], B_desc );

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            double time = barrier_get_wtime(MPI_COMM_WORLD);

            scalapack_phegst(itype, uplo2str(uplo), n,
                             Aref_data.data(), 1, 1, A_desc,
                             B_data.data(),    1, 1, B_desc,
                             &scale, &info);
            slate_assert(info == 0);

            time = barrier_get_wtime(MPI_COMM_WORLD) - time;

            params.ref_time() = time;
            // params.ref_gflops() = gflop / time;

            print_matrix("Aref_hegst", Aref, params);

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
        #else  // not SLATE_HAVE_SCALAPACK
            SLATE_UNUSED( one );
            SLATE_UNUSED( A_norm );
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
