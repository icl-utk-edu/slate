// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"
#include "blas/flops.hh"
#include "lapack/flops.hh"
#include "print_matrix.hh"
#include "grid_utils.hh"
#include "matrix_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_unmbr_ge2tb_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;

    // Constants
    const scalar_t zero = 0;
    const scalar_t one = 1;

    // get & mark input values
    slate::Uplo uplo = params.uplo();
    slate::Side side = params.side();
    slate::Op trans = params.trans();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t p = params.grid.m();
    int64_t q = params.grid.n();
    int64_t nb = params.nb();
    bool check = params.check() == 'y';
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    params.matrix.mark();

    // mark non-standard output values
    params.time();
    params.error2();
    params.error3();
    //params.error.name( "Aref-UxBxVT" );
    params.error2.name( "Q orth" );
    params.error3.name( "AQ orth" );
    //params.gflops();

    if (! run)
        return;

    slate::Options const opts =  {
        {slate::Option::Target, target}
    };

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    // Skip invalid or unimplemented options.
    if (uplo == slate::Uplo::Upper) {
        params.msg() = "skipping: Uplo::Upper isn't supported.";
        return;
    }
    if (p != q) {
        params.msg() = "skipping: requires square process grid (p == q).";
        return;
    }

    slate::Target origin_target = origin2target(origin);

    // Matrix A: figure out local size.
    int64_t mlocal = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocal = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldA   = blas::max(1, mlocal); // local leading dimension of A
    std::vector<scalar_t> A_data;

    slate::Matrix<scalar_t> A;
    if (origin != slate::Origin::ScaLAPACK) {
        A = slate::Matrix<scalar_t>(m, n, nb, p, q, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);
    }
    else {
        // Create SLATE matrices from the ScaLAPACK layouts.
        A_data.resize( lldA*nlocal );
        A = slate::Matrix<scalar_t>::fromScaLAPACK( m, n, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD );
    }

    slate::generate_matrix(params.matrix, A);

    print_matrix("A", A, params);

    slate::Matrix<scalar_t> Aref, C, Q, Iden;
    if (check) {
        if ((side == slate::Side::Left  && trans == slate::Op::NoTrans) ||
            (side == slate::Side::Right && trans != slate::Op::NoTrans)) {
            Aref = slate::Matrix<scalar_t>(m, n, nb, p, q, MPI_COMM_WORLD);

            Aref.insertLocalTiles(origin_target);
            slate::copy(A, Aref, opts);

            print_matrix("Aref", Aref, params);
        }

        // todo: to check on the orthogonality of A * Q,
        // where Q is an orthogonal matrix,
        // and A is the out put of geqrf
        // generate random matrix
        C = slate::Matrix<scalar_t>(n, n, nb, p, q, MPI_COMM_WORLD);
        C.insertLocalTiles(origin_target);
        slate::generate_matrix(params.matrix, C);

        Q = slate::Matrix<scalar_t>(m, n, nb, p, q, MPI_COMM_WORLD);
        Q.insertLocalTiles(origin_target);
        set(zero, one, Q);

        Iden = slate::Matrix<scalar_t>(n, n, nb, p, q, MPI_COMM_WORLD);
        Iden.insertLocalTiles(origin_target);
        set(zero, one, Iden);

        // QR of C
        slate::TriangularFactors<scalar_t> T;
        slate::geqrf(C, T, opts);

        // Generate orthogonal matrix Q
        auto Qhat = Q.slice( 0, n-1, 0, n-1 );
        slate::unmqr(
                slate::Side::Left, slate::Op::NoTrans, C, T, Qhat, opts);
        print_matrix("Q", Q, params);

        // Check orthogonality of Q
        // Iden = QT * Q - Iden
        auto QT = conj_transpose(Q);
        slate::gemm(-one, QT, Q, one, Iden);
        real_t Orth_Q = slate::norm(slate::Norm::One, Iden);
        print_matrix("QTxQ_Iden", Iden, params);
        printf("\n Orth_Q %e \n", Orth_Q);
    }

    // Triangular Factors T
    slate::TriangularFactors<scalar_t> TU;
    slate::TriangularFactors<scalar_t> TV;
    //todo: to check on Aref - U * Aband * VT call ge2tb
    //slate::ge2tb(A, TU, TV, opts);
    //
    slate::geqrf(A, TU, opts);

    print_matrix("A_factored", A, params);
    //print_matrix("T_local",    T[0], params);
    //print_matrix("T_reduce",   T[1], params);

    //todo: to check on Aref - U * Aband * VT
    #if 0
    // Matrix Aband
    slate::Matrix<scalar_t> Aband = A.emptyLike();
    Aband.insertLocalTiles(origin_target);
    set(zero, Aband);
    int64_t min_mtnt = std::min(A.mt(), A.nt());
    for (int64_t i = 0; i < min_mtnt; ++i) {
        if (Aband.tileIsLocal(i, i)) {
            // diagonal tile
            A.tileGetForReading( i, i, slate::LayoutConvert::None );
            auto Aii = A(i, i);
            auto Bii = Aband(i, i);
            Aii.uplo(slate::Uplo::Upper);
            Bii.uplo(slate::Uplo::Upper);
            slate::tile::tzcopy( Aii, Bii );
        }
        if (i+1 < min_mtnt && Aband.tileIsLocal(i, i+1)) {
            // super-diagonal tile
            A.tileGetForReading( i, i+1, slate::LayoutConvert::None );
            auto Aii1 = A(i, i+1);
            auto Bii1 = Aband(i, i+1);
            Aii1.uplo(slate::Uplo::Lower);
            Bii1.uplo(slate::Uplo::Lower);
            slate::tile::tzcopy( Aii1, Bii1 );
        }
    }
    //ge2gb(A, Aband);


    // U Aband VT  - Acpy; Acpy is the original matrix before ge2tb

    print_matrix("Aband", Aband, params);

    // todo
    //double gflop = lapack::Gflop<scalar_t>::unmbr_ge2tb(n, n);

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    double time = barrier_get_wtime(MPI_COMM_WORLD);

    //==================================================
    // Run SLATE test.
    //==================================================
    auto Uhat = Aband.slice( 0, Aband.n()-1, 0, Aband.n()-1 );
    slate::Matrix<scalar_t> U1d(Uhat.m(), Uhat.n(), Uhat.tileNb(0), 1, p*q, MPI_COMM_WORLD);
    U1d.insertLocalTiles(origin_target);
    U1d.redistribute(Uhat);
    Uhat.redistribute(U1d);
    slate::unmbr_ge2tb(slate::Side::Left, slate::Op::NoTrans, A, TU, Aband, opts);
    slate::unmbr_ge2tb(slate::Side::Right, slate::Op::NoTrans, A, TV, Aband, opts);


    print_matrix("UBVT", Aband, params);

    time = barrier_get_wtime(MPI_COMM_WORLD) - time;

    if (trace) slate::trace::Trace::finish();

    // compute and save timing/performance
    params.time() = time;
    //params.gflops() = gflop / time;
    #endif

    if (check) {
        real_t tol = params.tol() * std::numeric_limits<real_t>::epsilon()/2;

        //todo: to check on Aref - U * Aband * VT
        //slate::add(-one, Aref, one, Aband);
        //print_matrix("UBV^H - A", Aband, params);
        // Norm of backwards error: || UBV^H - A ||_1
        //real_t A_norm = slate::norm(slate::Norm::One, Aref);
        //params.error() = slate::norm(slate::Norm::One, Aband) / (m * A_norm);
        //params.okay() = (params.error() <= tol);

        // Apply householder (from geqrf(A)) on the orthogonal matrix Q generated above.
        //slate::unmbr_ge2tb(slate::Side::Left, slate::Op::NoTrans, A, TU, Q, opts);
        slate::unmqr(slate::Side::Left, slate::Op::NoTrans, A, TU, Q, opts);
        // QT * Q - Iden
        set(zero, one, Iden);
        auto QT = conj_transpose(Q);
        slate::gemm(-one, QT, Q, one, Iden);
        params.error2() = slate::norm(slate::Norm::One, Iden);
        print_matrix("AQTxAQ_Iden", Iden, params);

        // Apply householder (from geqrf(A)) on identity
        // check on orth Q
        set(zero, one, Q);
        slate::unmqr(slate::Side::Left, slate::Op::NoTrans, A, TU, Q, opts);
        // QT * Q - Iden
        set(zero, one, Iden);
        auto QT2 = conj_transpose(Q);
        slate::gemm(-one, QT2, Q, one, Iden);
        params.error3() = slate::norm(slate::Norm::One, Iden);

        params.okay() = (params.error2() <= tol && params.error3() <= tol);
    }
}

// -----------------------------------------------------------------------------
void test_unmbr_ge2tb(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_unmbr_ge2tb_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_unmbr_ge2tb_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_unmbr_ge2tb_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_unmbr_ge2tb_work<std::complex<double>> (params, run);
            break;
    }
}
