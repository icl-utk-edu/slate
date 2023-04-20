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

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
//------------------------------------------------------------------------------
template <typename scalar_t>
void test_qdwh_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;

    // Constants
    const scalar_t zero = 0;
    const scalar_t one = 1;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
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
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    params.matrix.mark();

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();
    params.ortho();

    if (! run)
        return;

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


    // BLACS/MPI variables
    // matrix A, figure out local size, allocate, create descriptor, initialize
    int64_t mlocA = num_local_rows_cols(m, nb, myrow, p);
    int64_t nlocA = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldA  = blas::max(1, mlocA); // local leading dimension of A
    std::vector<scalar_t> A_data;

    int64_t mlocH = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocH = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldH  = blas::max(1, mlocH); // local leading dimension of H
    std::vector<scalar_t> H_data;

    slate::Matrix<scalar_t> A;
    slate::Matrix<scalar_t> H;
    //slate::HermitianMatrix<scalar_t> H;
    if (origin != slate::Origin::ScaLAPACK) {
        // Copy local ScaLAPACK data to GPU or CPU tiles.
        slate::Target origin_target = origin2target(origin);
        A = slate::Matrix<scalar_t>(m, n, nb, p, q, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);
        //H = slate::HermitianMatrix<scalar_t>(slate::Uplo::Lower, n, nb, p, q, MPI_COMM_WORLD);
        //H.insertLocalTiles(origin_target);
        H = slate::Matrix<scalar_t>(n, n, nb, p, q, MPI_COMM_WORLD);
        H.insertLocalTiles(origin_target);
    }
    else {
        // create SLATE matrices from the ScaLAPACK layouts
        A_data.resize( lldA * nlocA );
        A = slate::Matrix<scalar_t>::fromScaLAPACK(m, n, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        //H_data.resize( lldH * nlocH );
        //H = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(slate::Uplo::Lower, n, &H_data[0], lldH, nb, p, q, MPI_COMM_WORLD);
        H_data.resize( lldH * nlocH );
        H = slate::Matrix<scalar_t>::fromScaLAPACK(n, n, &H_data[0], lldH, nb, p, q, MPI_COMM_WORLD);
    }

    real_t cond = 1 / std::numeric_limits<real_t>::epsilon();
    params.matrix.kind.set_default("svd");
    params.matrix.cond.set_default(cond);

    slate::generate_matrix( params.matrix, A);

    // if check is required, copy test data
    slate::Matrix<scalar_t> Aref;
    std::vector<scalar_t> Aref_data;
    if (check || ref) {
        Aref_data.resize( lldA * nlocA );
        Aref = slate::Matrix<scalar_t>::fromScaLAPACK(
            m, n, &Aref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        slate::copy(A, Aref);
    }

    print_matrix("A", A, params);

    // todo: how to compute gflops?

    if (! ref_only) {
        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        int itqr = 0, itpo = 0;

        double time = barrier_get_wtime(MPI_COMM_WORLD);

        //==================================================
        // Run SLATE test.
        //==================================================
        // A = AH,
        // A will be written by the orthogonal polar factor
        // H is the symmetric positive semidefinite polar factor
        slate::qdwh(A, H, itqr, itpo, opts);

        time = barrier_get_wtime(MPI_COMM_WORLD) - time;

        // gflop include the flops to estimate the condition
        // number, the flops for the qr_based iterations the
        // flops for the po_based iterations and the flops
        // to compute H (hermition polar factor).
        // gflop = trcon + itqr * (flop_per_qr_based_iter) +
        //         itpo * (flop_per_po_based_iter) +
        //         flop_compute_H
        double gflop_trcon = lapack::Gflop<scalar_t>::geqrf(m, n);

        // this flop count of the QR-based iteration is in case we are not taking advantage
        // of the identity trailing submatrix:
        //double gflop_itqr  = itqr * ( lapack::Gflop<scalar_t>::geqrf(m+n, n) +
        //                              lapack::Gflop<scalar_t>::unmqr(slate::Side::Left, m+n, n, n) +
        //                              blas::Gflop<scalar_t>::gemm(m, n, n) );
        // when take advantage of the matrix structure, the flops count will be reduced:
        double gflop_itqr  = itqr * ( 1e-9 * 2. * double(m*n*n) +
                                      1e-9 * 2. * double(m*n*n) +
                                      blas::Gflop<scalar_t>::gemm(m, n, n) );

        double gflop_itpo  = itpo * ( blas::Gflop<scalar_t>::gemm(m, n, n) +
                                      lapack::Gflop<scalar_t>::potrf(m)      +
                                      blas::Gflop<scalar_t>::trsm(slate::Side::Left, m, n) );

        double gflop_compute_H = blas::Gflop<scalar_t>::her2k(n, m);

        double gflop = gflop_trcon + gflop_itqr + gflop_itpo + gflop_compute_H;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time;
        params.gflops() = gflop / time;

        print_matrix("U: orthogonal polar factor", A, params);
        print_matrix("H: symmetric postive semi definite factor", H, params);
    }

    if (check) {
        //==================================================
        // Test results by checking backwards error
        //
        //      || A'H - Aref ||_f
        //     -------------------    < tol * epsilon
        //          ||Aref||_f
        //
        //==================================================
        real_t normA = slate::norm(slate::Norm::Fro, Aref, opts);
        slate::gemm(one, A, H, -one, Aref, opts);
        //auto H = HermitianMatrix<scalar_t>(
        //        slate::Uplo::Lower, B );
        //slate::hemm(slate::Side::Right, one, H, A, -one, Aref, opts);
        real_t berr = slate::norm(slate::Norm::Fro, Aref, opts);
        params.error() = berr / normA;

        //==================================================
        // Test results by checking the orthogonality of U factor
        //
        //      || A'A - I ||_f
        //     ---------------- < tol * epsilon
        //            n
        //
        //==================================================
        auto AT = conj_transpose(A);
        set(zero, one, Aref);
        slate::gemm(one, AT, A, -one, Aref, opts);
        real_t orth = slate::norm(slate::Norm::Fro, Aref);
        params.ortho() = orth / sqrt(n);

        real_t tol = params.tol() * 0.5 * std::numeric_limits<real_t>::epsilon();
        params.okay() = ((params.error() <= tol) && (params.ortho() <= tol));


    }
}

// -----------------------------------------------------------------------------
void test_qdwh(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_qdwh_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_qdwh_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_qdwh_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_qdwh_work<std::complex<double>> (params, run);
            break;
    }
}
