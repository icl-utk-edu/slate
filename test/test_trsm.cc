// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"

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
void test_trsm_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using slate::Op;
    using slate::Norm;
    using blas::real;

    // Constants
    const scalar_t one = 1;

    // Decode routine, setting method.
    if (params.routine == "trsmA")
        params.method_trsm() = slate::MethodTrsm::A;
    else if (params.routine == "trsmB")
        params.method_trsm() = slate::MethodTrsm::B;

    // get & mark input values
    slate::Side side = params.side();
    slate::Uplo uplo = params.uplo();
    slate::Op transA = params.transA();
    slate::Diag diag = params.diag();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    scalar_t alpha = params.alpha.get<scalar_t>();
    int64_t lookahead = params.lookahead();
    slate::Norm norm = params.norm();
    bool check = params.check() == 'y';
    bool ref = params.ref() == 'y';
    bool trace = params.trace() == 'y';
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    slate::MethodTrsm method_trsm = params.method_trsm();
    params.matrix.mark();
    params.matrixB.mark();

    mark_params_for_test_TriangularMatrix( params );
    mark_params_for_test_Matrix( params );

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    // Suppress norm from output; it's only for checks.
    params.norm.width( 0 );

    if (! run) {
        params.matrix.kind.set_default( "rand_dominant" );
        return;
    }

    // Check for common invalid combinations
    if (is_invalid_parameters( params )) {
        return;
    }

    slate::Options const opts =  {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MethodTrsm, method_trsm},
    };

    // Error analysis applies in these norms.
    slate_assert(norm == Norm::One || norm == Norm::Inf || norm == Norm::Fro);

    // setup so B is m-by-n
    int64_t An  = (side == slate::Side::Left ? m : n);
    int64_t Am  = An;
    int64_t Bm  = m;
    int64_t Bn  = n;

    auto A_alloc = allocate_test_TriangularMatrix<scalar_t>( false, true, An, params );
    auto B_alloc = allocate_test_Matrix<scalar_t>( check || ref, true, Bm, Bn, params );

    auto& A         = A_alloc.A;
    auto& B         = B_alloc.A;
    auto& Bref      = B_alloc.Aref;

    slate::generate_matrix( params.matrix, A );
    slate::generate_matrix( params.matrixB, B );

    // Cholesky factor of A to get a well conditioned triangular matrix.
    // Even when we replace the diagonal with unit diagonal,
    // it seems to still be well conditioned.
    auto AH = slate::HermitianMatrix<scalar_t>( A );
    slate::potrf( AH, opts );

    // If reference run is required, record norms to be used in the check/ref.
    if (check || ref) {
        slate::copy( B, Bref );
    }

    print_matrix( "A", A, params );
    print_matrix( "B", B, params );

    auto opA = A;
    if (transA == Op::Trans)
        opA = transpose(A);
    else if (transA == Op::ConjTrans)
        opA = conj_transpose( A );

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    double time = barrier_get_wtime(MPI_COMM_WORLD);

    //==================================================
    // Run SLATE test.
    // Solve AX = alpha B (left) or XA = alpha B (right).
    //==================================================
    if (side == slate::Side::Left) {
        slate::triangular_solve( alpha, opA, B, opts );
    }
    else if (side == slate::Side::Right) {
        slate::triangular_solve( alpha, B, opA, opts );
    }
    else
        throw slate::Exception("unknown side");
    // Using traditional BLAS/LAPACK name
    // slate::trsm(side, alpha, A, B, opts);

    time = barrier_get_wtime(MPI_COMM_WORLD) - time;

    if (trace) slate::trace::Trace::finish();

    // Compute and save timing/performance
    double gflop = blas::Gflop<scalar_t>::trsm(side, m, n);
    params.time() = time;
    params.gflops() = gflop / time;

    print_matrix( "B_out", B, params );

    if (check) {
        //==================================================
        // Test results by checking the residual
        //
        //      || B - 1/alpha AX ||_1
        //     ------------------------ < epsilon
        //      || A ||_1 * N
        //
        //==================================================

        // get norms of the original data
        // todo: add TriangularMatrix norm
        auto AZ = static_cast< slate::TrapezoidMatrix<scalar_t> >( opA );
        real_t A_norm = slate::norm(norm, AZ);

        slate::trmm(side, one/alpha, opA, B);
        slate::add(-one, Bref, one, B);
        real_t error = slate::norm(norm, B);
        error = error / (Am * A_norm);
        params.error() = error;

        // Allow 3*eps; complex needs 2*sqrt(2) factor; see Higham, 2002, sec. 3.6.
        real_t eps = std::numeric_limits<real_t>::epsilon();
        params.okay() = (params.error() <= 3*eps);
    }

    if (ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            // comparison with reference routine from ScaLAPACK for timing only

            // initialize BLACS and ScaLAPACK
            blas_int ictxt, A_desc[9], Bref_desc[9];
            A_alloc.create_ScaLAPACK_context( &ictxt );

            A_alloc.ScaLAPACK_descriptor( ictxt, A_desc );
            B_alloc.ScaLAPACK_descriptor( ictxt, Bref_desc );

            auto& A_data = A_alloc.A_data;
            auto& Bref_data = B_alloc.Aref_data;

            if (origin != slate::Origin::ScaLAPACK) {
                A_data.resize( A_alloc.lld * A_alloc.nloc );

                // Copy SLATE matrix into ScaLAPACK matrix
                copy(A, &A_data[0], A_desc);
            }

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            time = barrier_get_wtime(MPI_COMM_WORLD);
            scalapack_ptrsm(to_c_string( side ), to_c_string( uplo ), to_c_string( transA ), to_c_string( diag ),
                            m, n, alpha,
                            &A_data[0], 1, 1, A_desc,
                            &Bref_data[0], 1, 1, Bref_desc);
            time = barrier_get_wtime(MPI_COMM_WORLD) - time;

            print_matrix( "Bref", Bref, params );

            params.ref_time() = time;
            params.ref_gflops() = gflop / time;

            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering.
        #else  // not SLATE_HAVE_SCALAPACK
            if (A.mpiRank() == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }
}

// -----------------------------------------------------------------------------
void test_trsm(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_trsm_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_trsm_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_trsm_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_trsm_work<std::complex<double>> (params, run);
            break;

        default:
            throw std::runtime_error( "unknown datatype" );
            break;
    }
}
