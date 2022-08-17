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
void test_unmtr_he2hb_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;

    // Constants
    const scalar_t one = 1;

    // get & mark input values
    slate::Uplo uplo = params.uplo();
    slate::Side side = params.side();
    slate::Op trans = params.trans();
    int64_t n = params.dim.n();
    int64_t p = params.grid.m();
    int64_t q = params.grid.n();
    int64_t nb = params.nb();
    bool check = params.check() == 'y';
    bool trace = params.trace() == 'y';
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    params.matrix.mark();

    // mark non-standard output values
    params.time();
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

    // Matrix A: figure out local size.
    int64_t mlocal = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocal = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldA   = blas::max(1, mlocal); // local leading dimension of A
    std::vector<scalar_t> A_data;

    slate::HermitianMatrix<scalar_t> A;
    if (origin != slate::Origin::ScaLAPACK) {
        slate::Target origin_target = origin2target(origin);
        A = slate::HermitianMatrix<scalar_t>(uplo, n, nb, p, q, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);
    }
    else {
        // Create SLATE matrices from the ScaLAPACK layouts.
        A_data.resize( lldA*nlocal );
        A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                uplo, n, A_data.data(), lldA, nb, p, q, MPI_COMM_WORLD);
    }

    slate::generate_matrix(params.matrix, A);

    print_matrix("A", A, params);

    // Matrix Aref
    slate::HermitianMatrix<scalar_t> Aref;
    if (check) {
        if ((side == slate::Side::Left  && trans == slate::Op::NoTrans) ||
            (side == slate::Side::Right && trans != slate::Op::NoTrans)) {
            Aref = slate::HermitianMatrix<scalar_t>(
                       uplo, n, nb, p, q, MPI_COMM_WORLD);

            Aref.insertLocalTiles();
            slate::copy(A, Aref);

            print_matrix("Aref", Aref, params);
        }
    }

    // Matrix Afull is full, symmetrized copy of Hermitian matrix.
    slate::Matrix<scalar_t> Afull;
    if ((side == slate::Side::Left  && trans != slate::Op::NoTrans) ||
        (side == slate::Side::Right && trans == slate::Op::NoTrans)) {
        Afull = slate::Matrix<scalar_t>(n, n, nb, p, q, MPI_COMM_WORLD);

        Afull.insertLocalTiles();
        he2ge(A, Afull);

        print_matrix("Afull", Afull, params);
    }

    // Triangular Factors T
    slate::TriangularFactors<scalar_t> T;
    slate::he2hb(A, T, opts);

    print_matrix("A_factored", A, params);
    print_matrix("T_local",    T[0], params);
    print_matrix("T_reduce",   T[1], params);

    // Matrix B
    slate::Matrix< scalar_t > B(n, n, nb, p, q, MPI_COMM_WORLD);

    B.insertLocalTiles();
    he2gb(A, B);

    print_matrix("B", B, params);

    // todo
    //double gflop = lapack::Gflop<scalar_t>::unmtr_he2hb(n, n);

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    double time = barrier_get_wtime(MPI_COMM_WORLD);

    //==================================================
    // Run SLATE test.
    //==================================================
    if ((side == slate::Side::Left  && trans == slate::Op::NoTrans) ||
        (side == slate::Side::Right && trans != slate::Op::NoTrans)) {
        slate::unmtr_he2hb(side, trans, A, T, B, opts);
    }
    else {
        // Left-(Conj)Trans or Right-NoTrans
        slate::unmtr_he2hb(side, trans, A, T, Afull, opts);
    }

    time = barrier_get_wtime(MPI_COMM_WORLD) - time;

    if (trace) slate::trace::Trace::finish();

    // compute and save timing/performance
    params.time() = time;
    //params.gflops() = gflop / time;

    if (check) {
        if ((side == slate::Side::Left  && trans == slate::Op::NoTrans) ||
            (side == slate::Side::Right && trans != slate::Op::NoTrans)) {
            //==================================================
            // Test results by checking backwards error
            //
            //      || A - QBQ^H ||_1
            //     ------------------- < tol * epsilon
            //      || A ||_1 * n
            //
            //==================================================

            if (trans == slate::Op::NoTrans) {
                // QB is already computed, we need (QB)Q^H
                // (QB)Q^H
                slate::unmtr_he2hb(slate::Side::Right,
                                   slate::Op::ConjTrans, A, T, B, opts);
            }
            else {
                // BQ^H is already computed, we need QB
                // (QB)Q^H
                slate::unmtr_he2hb(slate::Side::Left,
                                   slate::Op::NoTrans, A, T, B, opts);
            }

            // Norm of original matrix: || A ||_1, where A is in Aref
            real_t A_norm = slate::norm(slate::Norm::One, Aref);

            // Form A - QBQ^H, where A is in Aref.
            for (int64_t j = 0; j < Aref.nt(); ++j) {
                for (int64_t i = j; i < Aref.nt(); ++i) {
                    if (Aref.tileIsLocal(i, j)) {
                        auto Aref_ij = Aref(i, j);
                        auto Bij = B(i, j);
                        // if i == j, Aij was Lower; set it to General for axpy.
                        Aref_ij.uplo(slate::Uplo::General);
                        slate::tile::add( -one, Bij, Aref_ij );
                    }
                }
            }

            print_matrix("A - QBQ^H", Aref, params);

            // Norm of backwards error: || A - QBQ^H ||_1
            params.error() = slate::norm(slate::Norm::One, Aref)
                           / (n * A_norm);
        }
        else {
            // Left-(Conj)Trans or Right-NoTrans
            //==================================================
            // Test results by checking forward error
            //
            //      || Q^HAQ - B ||_1
            //     ------------------- < tol * epsilon
            //      || B ||_1 * n
            //
            //==================================================

            if (trans == slate::Op::NoTrans) {
                // AQ is already computed, we need Q^HA
                // (Q^HA)Q
                slate::unmtr_he2hb(slate::Side::Left,
                                   slate::Op::ConjTrans, A, T, Afull, opts);
            }
            else {
                // Q^HA is already computed, we need (Q^HA)Q
                // (Q^HA)Q
                slate::unmtr_he2hb(slate::Side::Right,
                                   slate::Op::NoTrans, A, T, Afull, opts);
            }

            // Norm of B matrix: || B ||_1
            real_t B_norm = slate::norm(slate::Norm::One, B);

            // Form Q^HAQ - B
            for (int64_t i = 0; i < Afull.nt(); ++i) {
                for (int64_t j = 0; j < Afull.mt(); ++j) {
                    if (Afull.tileIsLocal(i, j)) {
                        slate::tile::add( -one, B(i, j), Afull(i, j) );
                    }
                }
            }

            print_matrix("Q^HAQ - B", Afull, params);

            // Norm of backwards error: || Q^HAQ - B ||_1
            params.error() = slate::norm(slate::Norm::One, Afull)
                           / (n * B_norm);
        }

        real_t tol = params.tol() * std::numeric_limits<real_t>::epsilon() / 2;
        params.okay() = (params.error() <= tol);
    }
}

// -----------------------------------------------------------------------------
void test_unmtr_he2hb(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_unmtr_he2hb_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_unmtr_he2hb_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_unmtr_he2hb_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_unmtr_he2hb_work<std::complex<double>> (params, run);
            break;
    }
}
