// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_copy.hh"
#include "print_matrix.hh"

#include "grid_utils.hh"
#include "matrix_utils.hh"
#include "test_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename matrix_type>
void test_set_work(Params& params, bool run)
{
    using scalar_t = typename matrix_type::value_type;
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;
    using slate::ceildiv;

    // get & mark input values
    slate::Uplo uplo;
    if (std::is_same< matrix_type, slate::Matrix<scalar_t> >::value)
        uplo = slate::Uplo::General;
    else
        uplo = params.uplo();
    slate::Op trans = params.trans();
    slate::Diag diag = slate::Diag::NonUnit;
    scalar_t alpha = params.alpha.get<real_t>();
    scalar_t beta = params.beta.get<real_t>();
    int64_t m = params.dim.m();
    int64_t n;
    if (std::is_same< matrix_type, slate::TriangularMatrix<scalar_t> >::value
        || std::is_same< matrix_type, slate::SymmetricMatrix<scalar_t> >::value
        || std::is_same< matrix_type, slate::HermitianMatrix<scalar_t> >::value) {
        n = m;  // square
    }
    else {
        n = params.dim.n();
    }
    bool ref_only = params.ref() == 'o';
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    bool trace = params.trace() == 'y';
    slate::Target target = params.target();
    params.matrix.mark();

    mark_params_for_test_Matrix( params );

    // mark non-standard output values
    params.time();
    params.ref_time();

    if (! run)
        return;

    // Check for common invalid combinations
    if (is_invalid_parameters( params )) {
        return;
    }

    slate::Options const opts =  {
        {slate::Option::Target, target}
    };

    auto A_alloc = allocate_test_Matrix<scalar_t>( check || ref, false, m, n, params );

    auto& Afull     = A_alloc.A;
    auto& Aref_full = A_alloc.Aref;
    auto& Aref_data = A_alloc.Aref_data;

    slate::generate_matrix( params.matrix, Afull );

    if (check || ref) {
        copy_matrix( Afull, Aref_full );
    }

    // Cast to desired matrix type.
    matrix_type A = matrix_cast< matrix_type >( Afull, uplo, diag );

    if (trans == slate::Op::Trans)
        A = transpose( A );
    else if (trans == slate::Op::ConjTrans)
        A = conj_transpose( A );

    print_matrix( "Afull", Afull, params );
    print_matrix( "A", A, params );

    if (! ref_only) {
        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        //==================================================
        // Run SLATE test.
        // Set A to alpha on off-diagonal entries,
        //           beta on the diagonal entries.
        //==================================================
        double time = barrier_get_wtime(MPI_COMM_WORLD);

        slate::set( alpha, beta, A, opts );

        time = barrier_get_wtime(MPI_COMM_WORLD) - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time;

        print_matrix( "Afull_out", Afull, params );
        print_matrix( "A_out", A, params );
    }

    if (check || ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            // comparison with reference routine from ScaLAPACK

            // initialize BLACS and ScaLAPACK
            blas_int ictxt, A_desc[9];
            A_alloc.create_ScaLAPACK_context( &ictxt );
            A_alloc.ScaLAPACK_descriptor( ictxt, A_desc );

            real_t A_norm = slate::norm( slate::Norm::One, A );

            print_matrix( "Aref_full", Aref_full, params );

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            double time = barrier_get_wtime(MPI_COMM_WORLD);

            scalapack_plaset( uplo2str(uplo), m, n, alpha, beta,
                              &Aref_data[0], 1, 1, A_desc );

            time = barrier_get_wtime(MPI_COMM_WORLD) - time;
            params.ref_time() = time;

            print_matrix( "Aref_full_out", Aref_full, params );

            // Get difference A = A - Aref.
            // Do this on full m-by-n matrix to detect if on, say,
            // a lower triangular matrix, the kernel accidentally modifies
            // the upper triangle.
            subtract_matrices( Afull, Aref_full );
            real_t A_diff_norm = slate::norm( slate::Norm::One, Aref_full );

            print_matrix( "A_diff_full", Aref_full, params );

            real_t error = A_diff_norm / (n * A_norm);
            params.error() = error;
            params.okay() = (error == 0);  // Set should be exact.

            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering
        #else  // not SLATE_HAVE_SCALAPACK
            if (A.mpiRank() == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }
}

// -----------------------------------------------------------------------------
template <typename scalar_t>
void test_set_dispatch(Params& params, bool run )
{
    std::string routine = params.routine;
    if (routine == "set") {
        test_set_work< slate::Matrix<scalar_t> >( params, run );
    }
    else if (routine == "tzset") {
        test_set_work< slate::TrapezoidMatrix<scalar_t> >( params, run );
    }
    else if (routine == "trset") {
        test_set_work< slate::TriangularMatrix<scalar_t> >( params, run );
    }
    else if (routine == "syset") {
        test_set_work< slate::SymmetricMatrix<scalar_t> >( params, run );
    }
    else if (routine == "heset") {
        test_set_work< slate::HermitianMatrix<scalar_t> >( params, run );
    }
    else {
        throw slate::Exception("unknown routine: " + routine);
    }
}

// -----------------------------------------------------------------------------
void test_set(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_set_dispatch<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_set_dispatch<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_set_dispatch<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_set_dispatch<std::complex<double>> (params, run);
            break;

        default:
            throw std::runtime_error( "unknown datatype" );
            break;
    }
}
