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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_unmqr_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    // using llong = long long;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t p = params.grid.m();
    int64_t q = params.grid.n();
    int64_t nb = params.nb();
    int64_t ib = params.ib();
    int64_t lookahead = params.lookahead();
    int64_t panel_threads = params.panel_threads();
    bool check = params.check() == 'y';
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    if (! run)
        return;

    // Local values
    const scalar_t zero = 0;
    const scalar_t one = 1;

    // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol, info;
    int descA_tst[9], descA_ref[9], descQR_tst[9];
    int iam = 0, nprocs = 1;
    int iseed = 1;

    // initialize BLACS and ScaLAPACK
    Cblacs_pinfo(&iam, &nprocs);
    slate_assert(p*q <= nprocs);
    Cblacs_get(-1, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Col", p, q);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    // matrix A, figure out local size, allocate, create descriptor, initialize
    int64_t mlocA = scalapack_numroc(m, nb, myrow, 0, nprow);
    int64_t nlocA = scalapack_numroc(n, nb, mycol, 0, npcol);
    scalapack_descinit(descA_tst, m, n, nb, nb, 0, 0, ictxt, mlocA, &info);
    slate_assert(info == 0);
    int64_t lldA = (int64_t)descA_tst[8];
    std::vector<scalar_t> A_tst(lldA*nlocA);
    scalapack_pplrnt(&A_tst[0], m, n, nb, nb, myrow, mycol, nprow, npcol, mlocA, iseed + 1);

    // matrix QR, for unmqr
    std::vector<scalar_t> QR_tst(1);
    scalapack_descinit(descQR_tst, m, n, nb, nb, 0, 0, ictxt, mlocA, &info);
    slate_assert(info == 0);

    slate::Matrix<scalar_t> A;
    if (origin != slate::Origin::ScaLAPACK) {
        // Copy local ScaLAPACK data to GPU or CPU tiles.
        slate::Target origin_target = origin2target(origin);
        A = slate::Matrix<scalar_t>(m, n, nb, nprow, npcol, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);
        copy(&A_tst[0], descA_tst, A);
    }
    else {
        // create SLATE matrices from the ScaLAPACK layouts
        A = slate::Matrix<scalar_t>::fromScaLAPACK(m, n, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
    }
    slate::TriangularFactors<scalar_t> T;

    if (verbose > 1) {
        print_matrix("A", A);
    }

    // copy test data and create a descriptor for unmqr
    std::vector<scalar_t> A_ref;
    slate::Matrix<scalar_t> Aref;
    if (check) {
        A_ref = A_tst;
        scalapack_descinit(descA_ref, m, n, nb, nb, 0, 0, ictxt, mlocA, &info);
        slate_assert(info == 0);

        Aref = slate::Matrix<scalar_t>::fromScaLAPACK(
                m, n, &A_ref[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
    }

    //==================================================
    // Run SLATE test.
    //==================================================
    slate::qr_factor(A, T, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    });

    //---------------------
    // Using traditional BLAS/LAPACK name
    // slate::geqrf(A, T, {
    //     {slate::Option::Lookahead, lookahead},
    //     {slate::Option::Target, target},
    //     {slate::Option::MaxPanelThreads, panel_threads},
    //     {slate::Option::InnerBlocking, ib}
    // });

    if (verbose > 1) {
        print_matrix("A_factored", A);
        print_matrix("Tlocal",  T[0]);
        print_matrix("Treduce", T[1]);
    }

    //==================================================
    // Test results by checking backwards error
    //
    //      || QR - A ||_1
    //     ---------------- < tol * epsilon
    //      || A ||_1 * m
    //
    //==================================================

    if (origin != slate::Origin::ScaLAPACK) {
        // Copy SLATE result back from GPU or CPU tiles.
        copy(A, &A_tst[0], descA_tst);
    }

    // Norm of original matrix: || A ||_1
    real_t A_norm = 0.0;
    if (check) {
        A_norm = slate::norm(slate::Norm::One, Aref);
    }

    // Zero out QR, then copy R, stored in upper triangle of A_tst.
    // todo: replace with slate set/copy functions.
    QR_tst = std::vector<scalar_t>(A_tst.size(), zero);
    scalapack_placpy("Upper", std::min(m, n), n,
                     &A_tst[0], 1, 1, descA_tst,
                     &QR_tst[0], 1, 1, descQR_tst);
    if (! check) {
        A_tst.clear();
    }

    slate::Matrix<scalar_t> QR;
    if (origin != slate::Origin::ScaLAPACK) {
        // Copy local ScaLAPACK data to GPU or CPU tiles.
        slate::Target origin_target = origin2target(origin);
        QR = slate::Matrix<scalar_t>(m, n, nb, nprow, npcol, MPI_COMM_WORLD);
        QR.insertLocalTiles(origin_target);
        copy(&QR_tst[0], descQR_tst, QR);
    }
    else {
        QR = slate::Matrix<scalar_t>::fromScaLAPACK(
                m, n, &QR_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
    }
    
    if (verbose > 1) {
        print_matrix("R", QR);
    }
    
    if (! check) {
        QR_tst.clear();
    }
    
    if (! check) {
        // prefetch all matrices to devices
        // when performance of unmqr is measured
        A.tileGetAllForReadingOnDevices(slate::LayoutConvert::None);
        QR.tileGetAllForWritingOnDevices(slate::LayoutConvert::None); 
    }

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time = testsweeper::get_wtime();

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
    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time_tst = testsweeper::get_wtime() - time;

    if (trace) slate::trace::Trace::finish();

    // compute and save timing/performance
    params.time() = time_tst;
    double gflop = lapack::Gflop<scalar_t>::unmqr(lapack::Side::Left, m, n, n);
    params.gflops() = gflop / time_tst;

    if (verbose > 1) {
        print_matrix("QR", QR);
    }

    if (check) {
        if (origin != slate::Origin::ScaLAPACK) {
            copy(QR, &QR_tst[0], descQR_tst);
        }
        
        // Form QR - A, where A is in Aref.
        // todo: slate::geadd(-one, Aref, QR);
        // using axpy assumes A_ref and QR_tst have same lda.
        blas::axpy(QR_tst.size(), -one, &A_ref[0], 1, &QR_tst[0], 1);
        
        if (origin != slate::Origin::ScaLAPACK) {
            copy(&QR_tst[0], descQR_tst, QR);
        }

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

    Cblacs_gridexit(ictxt);
    //Cblacs_exit(1) does not handle re-entering
}

// -----------------------------------------------------------------------------
void test_unmqr(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_unmqr_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_unmqr_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_unmqr_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_unmqr_work<std::complex<double>> (params, run);
            break;
    }
}
