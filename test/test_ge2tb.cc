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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_ge2tb_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;

    // Constants
    const scalar_t zero = 0;
    const scalar_t one = 1;

    // get & mark input values
    lapack::Job jobu = params.jobu();
    lapack::Job jobvt = params.jobvt();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t p = params.grid.m();
    int64_t q = params.grid.n();
    int64_t nb = params.nb();
    int64_t ib = params.ib();
    int64_t panel_threads = params.panel_threads();
    bool check = params.check() == 'y';
    bool trace = params.trace() == 'y';
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    params.matrix.mark();

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ortho_U();
    params.ortho_V();
    params.error.name( "UBV^H - A" );

    if (! run)
        return;

    slate::Options const opts =  {
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    };

    bool wantu  = (jobu  == slate::Job::Vec
                   || jobu  == slate::Job::AllVec
                   || jobu  == slate::Job::SomeVec);
    bool wantvt = (jobvt == slate::Job::Vec
                   || jobvt == slate::Job::AllVec
                   || jobvt == slate::Job::SomeVec);

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    // Matrix A: figure out local size.
    int64_t mlocal = num_local_rows_cols(m, nb, myrow, p);
    int64_t nlocal = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldA   = blas::max(1, mlocal); // local leading dimension of A
    std::vector<scalar_t> A_data;

    slate::Matrix<scalar_t> A;
    if (origin != slate::Origin::ScaLAPACK) {
        // SLATE allocates CPU or GPU tiles.
        A = slate::Matrix<scalar_t>(m, n, nb, p, q, MPI_COMM_WORLD);
        A.insertLocalTiles(origin2target(origin));
    }
    else {
        // Create SLATE matrices from the ScaLAPACK layouts.
        A_data.resize( lldA*nlocal );
        A = slate::Matrix<scalar_t>::fromScaLAPACK(
                m, n, A_data.data(), lldA, nb, p, q, MPI_COMM_WORLD);
    }
    slate::TriangularFactors<scalar_t> TU, TV;

    slate::generate_matrix( params.matrix, A );

    print_matrix("A", A, params);

    // Copy test data for check.
    slate::Matrix<scalar_t> Aref(m, n, nb, p, q, MPI_COMM_WORLD);
    Aref.insertLocalTiles();
    slate::copy(A, Aref);

    slate::Matrix<scalar_t> U, VT;
    // Create U and U1d. Set U to Identity.
    if (wantu) {
        U = slate::Matrix<scalar_t>(m, n, nb, p, q, MPI_COMM_WORLD);
        U.insertLocalTiles(target);
        set(zero, one, U);
    }

    // Create VT and V1d. Set VT to Identity.
    if (wantvt) {
        VT = slate::Matrix<scalar_t>(n, n, nb, p, q, MPI_COMM_WORLD);
        VT.insertLocalTiles(target);
        set(zero, one, VT);
    }

    // todo
    //double gflop = lapack::Gflop<scalar_t>::ge2tb(m, n);
    double gflop = lapack::Gflop<scalar_t>::gebrd(m, n);

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    double time = barrier_get_wtime(MPI_COMM_WORLD);

    //==================================================
    // Run SLATE test.
    //==================================================
    slate::ge2tb(A, TU, TV, opts);

    time = barrier_get_wtime(MPI_COMM_WORLD) - time;

    if (trace) slate::trace::Trace::finish();

    // compute and save timing/performance
    params.time() = time;
    params.gflops() = gflop / time;

    //==================================================
    // Back transform U and VT of the band matrix..
    //==================================================
    if (wantu) {
        slate::unmbr_ge2tb( slate::Side::Left, slate::Op::NoTrans, A, TU, U, opts );
    }
    if (wantvt) {
        slate::unmbr_ge2tb( slate::Side::Right, slate::Op::NoTrans, A, TV, VT, opts );
    }

    print_matrix("A_factored", A, params);
    print_matrix("TUlocal",  TU[0], params);
    print_matrix("TUreduce", TU[1], params);
    print_matrix("TVlocal",  TV[0], params);
    print_matrix("TVreduce", TV[1], params);

    if (check) {
        //==================================================
        // Test results by checking backwards error
        //
        //      || UBV^H - A ||_1
        //     ------------------- < tol * epsilon
        //      || A ||_1 * m
        //
        //==================================================

        // Norm of original matrix: || A ||_1
        real_t A_norm = slate::norm(slate::Norm::One, Aref);

        // Zero out B, then copy band matrix B from A.
        slate::Matrix<scalar_t> B = A.emptyLike();
        B.insertLocalTiles();
        set(zero, B);
        int64_t min_mtnt = std::min(A.mt(), A.nt());
        for (int64_t i = 0; i < min_mtnt; ++i) {
            if (B.tileIsLocal(i, i)) {
                // diagonal tile
                A.tileGetForReading( i, i, slate::LayoutConvert::None );
                auto Aii = A(i, i);
                auto Bii = B(i, i);
                Aii.uplo(slate::Uplo::Upper);
                Bii.uplo(slate::Uplo::Upper);
                slate::tile::tzcopy( Aii, Bii );
            }
            if (i+1 < min_mtnt && B.tileIsLocal(i, i+1)) {
                // super-diagonal tile
                A.tileGetForReading( i, i+1, slate::LayoutConvert::None );
                auto Aii1 = A(i, i+1);
                auto Bii1 = B(i, i+1);
                Aii1.uplo(slate::Uplo::Lower);
                Bii1.uplo(slate::Uplo::Lower);
                slate::tile::tzcopy( Aii1, Bii1 );
            }
        }
        print_matrix("B", B, params);

        // Form UB, where U's representation is in lower part of A and TU.
        slate::qr_multiply_by_q(
            slate::Side::Left, slate::Op::NoTrans, A, TU, B, opts);
        // Using traditional BLAS/LAPACK name
        // slate::unmqr(slate::Side::Left, slate::Op::NoTrans, A, TU, B, opts);

        print_matrix("UB", B, params);

        // Form (UB)V^H, where V's representation is above band in A and TV.
        auto Asub =  A.sub(0, A.mt()-1, 1, A.nt()-1);
        auto Bsub =  B.sub(0, B.mt()-1, 1, B.nt()-1);
        slate::TriangularFactors<scalar_t> TVsub = {
            TV[0].sub(0, TV[0].mt()-1, 1, TV[0].nt()-1),
            TV[1].sub(0, TV[1].mt()-1, 1, TV[1].nt()-1)
        };

        // Note V^H == Q, not Q^H.
        slate::lq_multiply_by_q(
            slate::Side::Right, slate::Op::NoTrans, Asub, TVsub, Bsub, opts);
        // Using traditional BLAS/LAPACK name
        // slate::unmlq(slate::Side::Right, slate::Op::NoTrans,
        //              Asub, TVsub, Bsub, opts);

        print_matrix("UBV^H", B, params);

        // Form UBV^H - A, where A is in Aref.
        slate::add(-one, Aref, one, B);
        print_matrix("UBV^H - A", B, params);

        // Norm of backwards error: || UBV^H - A ||_1
        params.error() = slate::norm(slate::Norm::One, B) / (m * A_norm);
        real_t tol = params.tol() * std::numeric_limits<real_t>::epsilon()/2;
        params.okay() = (params.error() <= tol);

        if (wantu || wantvt) {
            //==================================================
            // Test results orthogonality of U.
            // || I - U^H U || / n < tol
            //==================================================
            slate::Matrix<scalar_t> Iden( n, n, nb, p, q, MPI_COMM_WORLD );
            Iden.insertLocalTiles(target);
            set(zero, one, Iden);
            if (wantu) {
                set(zero, one, Iden);
                auto UH = conj_transpose( U );
                slate::gemm( -one, UH, U, one, Iden );
                params.ortho_U() = slate::norm( slate::Norm::One, Iden ) / n;
                params.okay() = params.okay() && (params.ortho_U() <= tol);
            }
            //==================================================
            // Test results orthogonality of VT.
            // || I - VT^H VT || / n < tol
            //==================================================
            if (wantvt) {
                set(zero, one, Iden);
                auto V = conj_transpose(VT);
                slate::gemm(-one, V, VT, one, Iden);
                params.ortho_V() = slate::norm(slate::Norm::One, Iden) / n;
                params.okay() = params.okay() && (params.ortho_V() <= tol);
            }
        }
    }
}

// -----------------------------------------------------------------------------
void test_ge2tb(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_ge2tb_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_ge2tb_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_ge2tb_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_ge2tb_work<std::complex<double>> (params, run);
            break;
    }
}
