// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"

#include "grid_utils.hh"
#include "matrix_utils.hh"
#include "test_utils.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template<typename scalar_t>
void test_gemm_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using slate::Norm;

    // Constants
    const scalar_t zero = 0.0, one = 1.0;

    // Decode routine, setting method.
    if (params.routine == "gemmA")
        params.method_gemm() = slate::MethodGemm::GemmA;
    else if (params.routine == "gemmC")
        params.method_gemm() = slate::MethodGemm::GemmC;

    // get & mark input values
    slate::Op transA = params.transA();
    slate::Op transB = params.transB();
    scalar_t alpha = params.alpha.get<scalar_t>();
    scalar_t beta = params.beta.get<scalar_t>();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs();
    int64_t k = params.dim.k();
    int64_t lookahead = params.lookahead();
    bool ref_only = params.ref() == 'o';
    slate::Norm norm = params.norm();
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    bool trace = params.trace() == 'y';
    bool nonuniform_nb = params.nonuniform_nb() == 'y';
    int verbose = params.verbose();
    slate::Target target = params.target();
    slate::Origin origin = params.origin();
    slate::Method method_gemm = params.method_gemm();
    params.matrix.mark();
    params.matrixB.mark();
    params.matrixC.mark();

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

    #ifndef SLATE_HAVE_SCALAPACK
        // Can run ref only when we have ScaLAPACK.
        if (ref) {
            if (mpi_rank == 0)
                printf( "ScaLAPACK not available\n" );
            ref = false;
        }
    #endif

    slate::Options const opts =  {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MethodGemm, method_gemm},
    };

    // Error analysis applies in these norms.
    slate_assert(norm == Norm::One || norm == Norm::Inf || norm == Norm::Fro);

    // sizes of A and B
    int64_t Am = (transA == slate::Op::NoTrans ? m : k);
    int64_t An = (transA == slate::Op::NoTrans ? k : m);
    int64_t Bm = (transB == slate::Op::NoTrans ? k : n);
    int64_t Bn = (transB == slate::Op::NoTrans ? n : k);
    int64_t Cm = m;
    int64_t Cn = n;

    auto A_alloc = allocate_test_Matrix<scalar_t>( false, true, Am, An, params );
    auto B_alloc = allocate_test_Matrix<scalar_t>( false, true, Bm, Bn, params );
    auto C_alloc = allocate_test_Matrix<scalar_t>( ref, true, Cm, Cn, params );

    auto& A         = A_alloc.A;
    auto& B         = B_alloc.A;
    auto& C         = C_alloc.A;
    auto& Cref      = C_alloc.Aref;

    slate::generate_matrix(params.matrix, A);
    slate::generate_matrix(params.matrixB, B);
    slate::generate_matrix(params.matrixC, C);

    // if reference run is required, copy test data.
    if (ref) {
        slate::copy( C, Cref );
    }

    if (transA == slate::Op::Trans)
        A = transpose(A);
    else if (transA == slate::Op::ConjTrans)
        A = conj_transpose( A );

    if (transB == slate::Op::Trans)
        B = transpose(B);
    else if (transB == slate::Op::ConjTrans)
        B = conj_transpose( B );

    slate_assert(A.mt() == C.mt());
    slate_assert(B.nt() == C.nt());
    slate_assert(A.nt() == B.mt());

    // If reference run is required, record norms to be used in the check/ref.
    real_t A_norm=0, B_norm=0, C_orig_norm=0;
    if (ref) {
        A_norm = slate::norm(norm, A);
        B_norm = slate::norm(norm, B);
        C_orig_norm = slate::norm(norm, Cref);
    }

    // If check run, perform first half of SLATE residual check.
    TestMatrix<slate::Matrix<scalar_t>> X_alloc, Y_alloc, Z_alloc;
    if (check && ! ref) {
        // Compute Y = alpha A * (B * X) + (beta C * X).
        X_alloc = allocate_test_Matrix<scalar_t>( false, true, n, nrhs, params );
        Y_alloc = allocate_test_Matrix<scalar_t>( false, true, m, nrhs, params );
        Z_alloc = allocate_test_Matrix<scalar_t>( false, true, k, nrhs, params );

        auto& X = X_alloc.A;
        auto& Y = Y_alloc.A;
        auto& Z = Z_alloc.A;

        MatrixParams mp;
        mp.kind.set_default( "rand" );
        slate::generate_matrix( mp, X );

        // Z = B * X;
        slate::multiply( one, B, X, zero, Z, opts );
        // Y = beta * C * X
        slate::multiply( beta, C, X, zero, Y, opts );
        // Y = alpha * A * Z + Y;
        slate::multiply( alpha, A, Z, one, Y, opts );
    }

    print_matrix( "A", A, params );
    print_matrix( "B", B, params );
    print_matrix( "C", C, params );

    // compute and save timing/performance
    double gflop = blas::Gflop<scalar_t>::gemm(m, n, k);

    if (! ref_only) {

        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        double time = barrier_get_wtime(MPI_COMM_WORLD);

        //==================================================
        // Run SLATE test.
        // C = alpha A B + beta C.
        //==================================================
        slate::multiply( alpha, A, B, beta, C, opts );
        // Using traditional BLAS/LAPACK name
        // slate::gemm( alpha, A, B, beta, C, opts );

        time = barrier_get_wtime(MPI_COMM_WORLD) - time;

        if (trace) slate::trace::Trace::finish();

        if (verbose >= 2) {
            C.tileGetAllForReading( slate::HostNum, slate::LayoutConvert::None );
            print_matrix( "C_out", C, params );
        }

        // compute and save timing/performance
        params.time() = time;
        params.gflops() = gflop / time;
    }

    if (check && ! ref) {
        auto& X = X_alloc.A;
        auto& Y = Y_alloc.A;

        // SLATE residual check.
        // Check error, C*X - Y.
        real_t y_norm = slate::norm( norm, Y, opts );
        // Y = C * X - Y
        slate::multiply( one, C, X, -one, Y, opts );
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
            if (nonuniform_nb) {
                params.msg() = "skipping reference: nonuniform tile not supported with ScaLAPACK";
                return;
            }

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
                // todo: C_data not needed anymore?
                copy(C, &C_data[0], C_desc);
            }

            print_matrix( "Cref", Cref, params );

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            double time = barrier_get_wtime(MPI_COMM_WORLD);

            scalapack_pgemm(op2str(transA), op2str(transB), m, n, k, alpha,
                            &A_data[0], 1, 1, A_desc,
                            &B_data[0], 1, 1, B_desc, beta,
                            &Cref_data[0], 1, 1, Cref_desc);

            time = barrier_get_wtime(MPI_COMM_WORLD) - time;

            print_matrix( "Cref_out", Cref, params );

            // get differences C = C - Cref
            slate::add(-one, Cref, one, C);

            print_matrix( "Diff", C, params );

            // norm(C - Cref)
            real_t C_diff_norm = slate::norm(norm, C);

            real_t error = C_diff_norm
                        / (sqrt(real_t(k) + 2) * std::abs(alpha) * A_norm * B_norm
                            + 2 * std::abs(beta) * C_orig_norm);

            params.ref_time() = time;
            params.ref_gflops() = gflop / time;
            params.error() = error;

            // Allow 3*eps; complex needs 2*sqrt(2) factor; see Higham, 2002, sec. 3.6.
            real_t eps = std::numeric_limits<real_t>::epsilon();
            params.okay() = (params.error() <= 3*eps);

            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering
        #endif
    }
}

// -----------------------------------------------------------------------------
void test_gemm(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_gemm_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_gemm_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_gemm_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gemm_work<std::complex<double>> (params, run);
            break;

        default:
            throw std::runtime_error( "unknown datatype" );
            break;
    }
}
