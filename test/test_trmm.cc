// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"
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
template<typename scalar_t>
void test_trmm_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using slate::Op;
    using slate::Norm;

    // Constants
    const scalar_t zero = 0.0, one = 1.0;

    // get & mark input values
    slate::Side side = params.side();
    slate::Uplo uplo = params.uplo();
    slate::Op transA = params.transA();
    slate::Diag diag = params.diag();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs();
    scalar_t alpha = params.alpha.get<scalar_t>();
    slate::Op transB = params.transB();
    int64_t lookahead = params.lookahead();
    slate::Norm norm = params.norm();
    bool check = params.check() == 'y';
    bool ref = params.ref() == 'y';
    bool trace = params.trace() == 'y';
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    params.matrix.mark();
    params.matrixB.mark();

    mark_params_for_test_TriangularMatrix( params );
    mark_params_for_test_Matrix( params );

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    // Suppress norm, nrhs from output; they're only for checks.
    params.norm.width( 0 );
    params.nrhs.width( 0 );

    if (! run) {
        return;
    }

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

    // setup so op(B) is m-by-n
    int64_t An = (side == slate::Side::Left ? m : n);
    int64_t Am = An;
    int64_t Bm = (transB == slate::Op::NoTrans ? m : n);
    int64_t Bn = (transB == slate::Op::NoTrans ? n : m);

    auto A_alloc = allocate_test_TriangularMatrix<scalar_t>( false, true, An, params );
    auto B_alloc = allocate_test_Matrix<scalar_t>( ref, true, Bm, Bn, params );

    auto& A         = A_alloc.A;
    auto& B         = B_alloc.A;
    auto& Bref      = B_alloc.Aref;

    generate_matrix( params.matrix, A );
    generate_matrix( params.matrixB, B );

    // If reference run is required, record norms to be used in the check/ref.
    real_t A_norm=0, B_orig_norm=0;
    if (ref) {
        slate::copy( B, Bref );

        A_norm = slate::norm(norm, A);
        B_orig_norm = slate::norm(norm, B);
    }

    // Keep the original untransposed A matrix,
    // and make a shallow copy of it for transposing.
    auto opA = A;
    if (transA == Op::Trans)
        opA = transpose(A);
    else if (transA == Op::ConjTrans)
        opA = conj_transpose( A );

    if (transB == Op::Trans)
        B = transpose(B);
    else if (transB == Op::ConjTrans)
        B = conj_transpose( B );

    // If check run, perform first half of SLATE residual check.
    TestMatrix<slate::Matrix<scalar_t>> X_alloc, X2_alloc, Y_alloc;
    if (check && ! ref) {
        X_alloc = allocate_test_Matrix<scalar_t>( false, true, n, nrhs, params );
        X2_alloc = allocate_test_Matrix<scalar_t>( false, true, n, nrhs, params );
        Y_alloc = allocate_test_Matrix<scalar_t>( false, true, m, nrhs, params );

        auto& X = X_alloc.A;
        auto& X2 = X2_alloc.A;
        auto& Y = Y_alloc.A;

        MatrixParams mp;
        mp.kind.set_default( "rand" );
        generate_matrix( mp, X );

        if (side == slate::Side::Left ) {
            // Compute Y = alpha A * (B * X).
            // Y = B * X;
            slate::multiply( one, B, X, zero, Y, opts );
            // Y = alpha * A * Y;
            slate::triangular_multiply( alpha, opA, Y, opts );
        }
        else if (side == slate::Side::Right) {
            // Compute Y = alpha B * (A * X).
            slate::copy( X, X2 );
            // X2 = A * X2;
            slate::triangular_multiply( one, opA, X2, opts );
            // Y = alpha * B * X2;
            slate::multiply( alpha, B, X2, zero, Y, opts );
        }
        else
            throw slate::Exception("unknown side");
    }

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    double time = barrier_get_wtime(MPI_COMM_WORLD);

    //==================================================
    // Run SLATE test.
    // B = alpha AB (left) or B = alpha BA (right).
    //==================================================
    if (side == slate::Side::Left)
        slate::triangular_multiply(alpha, opA, B, opts);
    else if (side == slate::Side::Right)
        slate::triangular_multiply(alpha, B, opA, opts);
    else
        throw slate::Exception("unknown side");
    // Using traditional BLAS/LAPACK name
    // slate::trmm(side, alpha, A, B, opts);

    time = barrier_get_wtime(MPI_COMM_WORLD) - time;

    if (trace) slate::trace::Trace::finish();

    // compute and save timing/performance
    double gflop = blas::Gflop<scalar_t>::trmm(side, m, n);
    params.time() = time;
    params.gflops() = gflop / time;

    if (check && ! ref) {
        auto& X = X_alloc.A;
        auto& Y = Y_alloc.A;

        // SLATE residual check.
        // Check error, B*X - Y.
        real_t y_norm = slate::norm( norm, Y, opts );
        // Y = B * X - Y
        slate::multiply( one, B, X, -one, Y );
        // error = norm( Y ) / y_norm
        real_t error = slate::norm( norm, Y, opts )/y_norm;
        params.error() = error;

        // Allow 3*eps; complex needs 2*sqrt(2) factor; see Higham, 2002, sec. 3.6.
        real_t eps = std::numeric_limits<real_t>::epsilon();
        params.okay() = (params.error() <= 3*eps);
    }

    if (ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            // comparison with reference routine from ScaLAPACK

            // initialize BLACS and ScaLAPACK
            blas_int ictxt, A_desc[9], B_desc[9], Bref_desc[9];
            A_alloc.create_ScaLAPACK_context( &ictxt );

            A_alloc.ScaLAPACK_descriptor( ictxt, A_desc );
            B_alloc.ScaLAPACK_descriptor( ictxt, B_desc );
            B_alloc.ScaLAPACK_descriptor( ictxt, Bref_desc );

            auto& A_data = A_alloc.A_data;
            auto& B_data = B_alloc.A_data;
            auto& Bref_data = B_alloc.Aref_data;

            if (origin != slate::Origin::ScaLAPACK) {
                A_data.resize( A_alloc.lld * A_alloc.nloc );
                B_data.resize( B_alloc.lld * B_alloc.nloc );

                // Copy SLATE matrix into ScaLAPACK matrix
                copy(A, &A_data[0], A_desc);
                copy(B, &B_data[0], B_desc);
            }

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            time = barrier_get_wtime(MPI_COMM_WORLD);
            scalapack_ptrmm(to_c_string( side ), to_c_string( uplo ), to_c_string( transA ), to_c_string( diag ),
                            m, n, alpha,
                            &A_data[0], 1, 1, A_desc,
                            &Bref_data[0], 1, 1, Bref_desc);
            time = barrier_get_wtime(MPI_COMM_WORLD) - time;

            // get differences B = B - Bref
            slate::add(-one, Bref, one, B);

            // norm(B - Bref)
            real_t B_diff_norm = slate::norm(norm, B);

            real_t error = B_diff_norm
                         / (sqrt(real_t(Am) + 2) * std::abs(alpha) * A_norm * B_orig_norm);

            params.ref_time() = time;
            params.ref_gflops() = gflop / time;
            params.error() = error;

            // Allow 3*eps; complex needs 2*sqrt(2) factor; see Higham, 2002, sec. 3.6.
            real_t eps = std::numeric_limits<real_t>::epsilon();
            params.okay() = (params.error() <= 3*eps);

            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering
        #else  // not SLATE_HAVE_SCALAPACK
            if (A.mpiRank() == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }
}

// -----------------------------------------------------------------------------
void test_trmm(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_trmm_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_trmm_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_trmm_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_trmm_work<std::complex<double>> (params, run);
            break;

        default:
            throw std::runtime_error( "unknown datatype" );
            break;
    }
}
