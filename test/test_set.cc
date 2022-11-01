// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"
#include "print_matrix.hh"
#include "grid_utils.hh"
#include "matrix_utils.hh"

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

    // Constants
    const scalar_t one = 1.0;

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
    int64_t nb = params.nb();
    int64_t p = params.grid.m();
    int64_t q = params.grid.n();
    bool ref_only = params.ref() == 'o';
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    bool trace = params.trace() == 'y';
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    params.matrix.mark();

    // mark non-standard output values
    params.time();
    params.ref_time();

    if (! run)
        return;

    slate::Options const opts =  {
        {slate::Option::Target, target}
    };

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    // Matrix A: figure out local size.
    int64_t mlocA = num_local_rows_cols(m, nb, myrow, p);
    int64_t nlocA = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldA  = blas::max(1, mlocA); // local leading dimension of A
    std::vector<scalar_t> A_data;

    slate::Matrix<scalar_t> Afull;
    if (origin != slate::Origin::ScaLAPACK) {
        // SLATE allocates CPU or GPU tiles.
        slate::Target origin_target = origin2target( origin );
        Afull = slate::Matrix<scalar_t>( m, n, nb, p, q, MPI_COMM_WORLD);
        Afull.insertLocalTiles( origin_target );
    }
    else {
        // Allocate ScaLAPACK data.
        A_data.resize( lldA*nlocA );
        // Create SLATE matrix from the ScaLAPACK layout.
        Afull = slate::Matrix<scalar_t>::fromScaLAPACK(
                    m, n, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD );
    }
    slate::generate_matrix( params.matrix, Afull );

    // Cast to desired matrix type.
    matrix_type A = matrix_cast< matrix_type >( Afull, uplo, diag );

    // if reference run is required, copy test data
    std::vector<scalar_t> Aref_data;
    slate::Matrix<scalar_t> Aref_full;
    matrix_type Aref;
    if (check || ref) {
        // For simplicity, always use ScaLAPACK format for ref matrices.
        Aref_data.resize( lldA*nlocA );
        Aref_full = slate::Matrix<scalar_t>::fromScaLAPACK(
                        m,  n, &Aref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        Aref = matrix_cast< matrix_type >( Afull, uplo, diag );
        slate::copy( Afull, Aref_full );
    }

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

            // BLACS/MPI variables
            int ictxt, p_, q_, myrow_, mycol_, info;
            int A_desc[9];
            int mpi_rank_ = 0, nprocs = 1;

            // initialize BLACS and ScaLAPACK
            Cblacs_pinfo(&mpi_rank_, &nprocs);
            slate_assert( mpi_rank_ == mpi_rank );
            slate_assert(p*q <= nprocs);
            Cblacs_get(-1, 0, &ictxt);
            Cblacs_gridinit(&ictxt, "Col", p, q);
            Cblacs_gridinfo(ictxt, &p_, &q_, &myrow_, &mycol_);
            slate_assert( p == p_ );
            slate_assert( q == q_ );
            slate_assert( myrow == myrow_ );
            slate_assert( mycol == mycol_ );

            scalapack_descinit(A_desc, m, n, nb, nb, 0, 0, ictxt, lldA, &info);
            slate_assert(info == 0);

            real_t A_norm = slate::norm( slate::Norm::One, A );

            print_matrix( "Aref_full", Aref_full, params );

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            double time = barrier_get_wtime(MPI_COMM_WORLD);

            scalapack_plaset( uplo2str(uplo), m, n, alpha, beta,
                              &Aref_data[0], 1, 1, A_desc );
            slate_assert(info == 0);

            time = barrier_get_wtime(MPI_COMM_WORLD) - time;
            params.ref_time() = time;

            print_matrix( "Aref_full_out", Aref_full, params );

            // Get difference A = A - Aref.
            // Do this on full m-by-n matrix to detect if on, say,
            // a lower triangular matrix, the kernel accidentally modifies
            // the upper triangle.
            slate::add( -one, Aref_full, one, Afull );
            real_t A_diff_norm = slate::norm( slate::Norm::One, Afull );

            print_matrix( "A_diff_full", Afull, params );

            real_t error = A_diff_norm / (n * A_norm);
            params.error() = error;
            params.okay() = (error == 0);  // Set should be exact.

            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering
        #else  // not SLATE_HAVE_SCALAPACK
            SLATE_UNUSED( one );
            if (mpi_rank == 0)
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
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

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
    }
}
