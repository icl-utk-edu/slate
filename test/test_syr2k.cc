// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"
#include "print_matrix.hh"
#include "blas/flops.hh"

#include "matrix_utils.hh"
#include "test_utils.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_copy.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template< typename scalar_t >
void test_syr2k_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::conj;
    using slate::Op;
    using slate::Norm;
    using slate::ceildiv;

    // Constants
    const scalar_t zero = 0.0, one = 1.0;

    // get & mark input values
    slate::Uplo uplo = params.uplo();
    slate::Op trans = params.trans();
    int64_t n = params.dim.n();
    int64_t k = params.dim.k();
    scalar_t alpha = params.alpha.get<scalar_t>();
    scalar_t beta = params.beta.get<scalar_t>();
    int64_t nrhs = params.nrhs();
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

    mark_params_for_test_SymmetricMatrix( params );
    mark_params_for_test_Matrix( params );

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

    // Check for common invalid combinations
    if (is_invalid_parameters( params )) {
        return;
    }

    slate::Options const opts =  {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    };

    // Error analysis applies in these norms.
    slate_assert(norm == Norm::One || norm == Norm::Inf || norm == Norm::Fro);

    // setup so op(A) and op(B) are n-by-k
    int64_t Am = (trans == slate::Op::NoTrans ? n : k);
    int64_t An = (trans == slate::Op::NoTrans ? k : n);
    int64_t Bm = Am;
    int64_t Bn = An;
    int64_t Cn = n;

    auto A_alloc = allocate_test_Matrix<scalar_t>( false, true, Am, An, params );
    auto B_alloc = allocate_test_Matrix<scalar_t>( false, true, Bm, Bn, params );
    auto C_alloc = allocate_test_SymmetricMatrix<scalar_t>( ref, true, Cn, params );

    auto& A         = A_alloc.A;
    auto& B         = B_alloc.A;
    auto& C         = C_alloc.A;
    auto& Cref      = C_alloc.Aref;

    slate::generate_matrix( params.matrix, A );
    slate::generate_matrix( params.matrixB, B );
    slate::generate_matrix( params.matrixC, C );

    // If reference run is required, record norms to be used in the check/ref.
    real_t A_norm=0, B_norm=0, C_orig_norm=0;
    if (ref) {
        slate::copy( C, Cref );

        A_norm = slate::norm(norm, A);
        B_norm = slate::norm(norm, B);
        C_orig_norm = slate::norm(norm, Cref);
    }

    // Keep the original untransposed A and B matrices,
    // and make a shallow copy of them for transposing.
    auto opA = A;
    auto opB = B;
    if (trans == slate::Op::Trans) {
        opA = transpose(A);
        opB = transpose(B);
    }
    else if (trans == slate::Op::ConjTrans) {
        opA = conj_transpose( A );
        opB = conj_transpose( B );
    }
    slate_assert( opA.mt() == C.mt() );
    slate_assert( opB.mt() == C.mt() );
    slate_assert( opA.nt() == opB.nt() );

    print_matrix( "A", A, params );
    print_matrix( "B", B, params );
    print_matrix( "C", C, params );

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    // If check run, perform first half of SLATE residual check.
    TestMatrix<slate::Matrix<scalar_t>> X_alloc, Y_alloc, Z_alloc;
    if (check && ! ref) {
        X_alloc = allocate_test_Matrix<scalar_t>( false, true, n, nrhs, params );
        Y_alloc = allocate_test_Matrix<scalar_t>( false, true, n, nrhs, params );
        Z_alloc = allocate_test_Matrix<scalar_t>( false, true, k, nrhs, params );

        auto& X = X_alloc.A;
        auto& Y = Y_alloc.A;
        auto& Z = Z_alloc.A;

        MatrixParams mp;
        mp.kind.set_default( "rand" );
        slate::generate_matrix( mp, X );

        // Compute Y = (alpha A (B^T X)) + alpha B (A^T X)) + (beta C X).
        // Y = beta C X
        slate::multiply( beta, C, X, zero, Y, opts );
        // Z = A^T X
        auto AT = transpose( opA );
        slate::multiply( one, AT, X, zero, Z, opts );
        // Y = alpha B Z + Y
        slate::multiply( alpha, opB, Z, one, Y, opts );
        // Z = B^T X
        auto BT = transpose( opB );
        slate::multiply( one, BT, X, zero, Z, opts );
        // Y = alpha A Z + Y
        slate::multiply( alpha, opA, Z, one, Y, opts );
    }

    double time = barrier_get_wtime( MPI_COMM_WORLD );

    //==================================================
    // Run SLATE test.
    // C = alpha A B^T + alpha B A^T + beta C.
    //==================================================
    slate::rank_2k_update(alpha, opA, opB, beta, C, opts);
    // Using traditional BLAS/LAPACK name
    // slate::syr2k(alpha, A, B, beta, C, opts);

    time = barrier_get_wtime( MPI_COMM_WORLD ) - time;

    print_matrix( "C_out", C, params );

    if (trace) slate::trace::Trace::finish();

    // Compute and save timing/performance
    double gflop = blas::Gflop<scalar_t>::syr2k(n, k);
    params.time() = time;
    params.gflops() = gflop / time;

    if (check && ! ref) {
        auto& X = X_alloc.A;
        auto& Y = Y_alloc.A;

        // SLATE residual check.
        // Check error, C*X - Y.
        real_t y_norm = slate::norm( norm, Y, opts );
        // Y = C * X - Y
        slate::multiply( one, C, X, -one, Y );
        // error = norm( Y ) / y_norm
        real_t error = slate::norm( slate::Norm::One, Y, opts )/y_norm;
        params.error() = error;

        // Allow 3*eps; complex needs 2*sqrt(2) factor; see Higham, 2002, sec. 3.6.
        real_t eps = std::numeric_limits<real_t>::epsilon();
        params.okay() = (params.error() <= 3*eps);
    }

    if (ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            // comparison with reference routine from ScaLAPACK

            // initialize BLACS and ScaLAPACK
            blas_int ictxt, A_desc[9], B_desc[9], C_desc[9], Cref_desc[9];
            A_alloc.create_ScaLAPACK_context( &ictxt );

            A_alloc.ScaLAPACK_descriptor( ictxt, A_desc );
            B_alloc.ScaLAPACK_descriptor( ictxt, B_desc );
            C_alloc.ScaLAPACK_descriptor( ictxt, C_desc );
            C_alloc.ScaLAPACK_descriptor( ictxt, Cref_desc );

            auto& A_data = A_alloc.A_data;
            auto& B_data = B_alloc.A_data;
            auto& C_data = C_alloc.A_data;
            auto& Cref_data = C_alloc.Aref_data;

            if (origin != slate::Origin::ScaLAPACK) {
                A_data.resize( A_alloc.lld * A_alloc.nloc );
                B_data.resize( B_alloc.lld * B_alloc.nloc );
                C_data.resize( C_alloc.lld * C_alloc.nloc );

                // Copy SLATE result back from GPU or CPU tiles.
                copy(A, &A_data[0], A_desc);
                copy(B, &B_data[0], B_desc);
                copy(C, &C_data[0], C_desc);
            }

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            time = barrier_get_wtime( MPI_COMM_WORLD );
            scalapack::syr2k( uplo, trans, n, k,
                              alpha, &A_data[0], 1, 1, A_desc,
                                     &B_data[0], 1, 1, B_desc,
                              beta,  &Cref_data[0], 1, 1, Cref_desc );
            time = barrier_get_wtime( MPI_COMM_WORLD ) - time;

            print_matrix( "Cref_out", Cref, params );

            // get differences C = C - Cref
            slate::add(-one, Cref, one, C);

            // norm(C - Cref)
            real_t C_diff_norm = slate::norm(norm, C);

            real_t error = C_diff_norm
                         / (sqrt(real_t(2*k) + 2) * std::abs(alpha) * A_norm * B_norm
                            + 2 * std::abs(beta) * C_orig_norm);

            params.ref_time() = time;
            params.ref_gflops() = gflop / time;
            params.error() = error;

            // Allow 3*eps; complex needs 2*sqrt(2) factor; see Higham, 2002, sec. 3.6.
            real_t eps = std::numeric_limits<real_t>::epsilon();
            params.okay() = (params.error() <= 3*eps);

            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering
        #else   // not SLATE_HAVE_SCALAPACK
            if (A.mpiRank() == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }
}

// -----------------------------------------------------------------------------
void test_syr2k(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_syr2k_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_syr2k_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_syr2k_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_syr2k_work<std::complex<double>> (params, run);
            break;

        default:
            throw std::runtime_error( "unknown datatype" );
            break;
    }
}
