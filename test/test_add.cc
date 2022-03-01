// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

// -----------------------------------------------------------------------------
// The following routine : subtract_matrix : takes input matrices A and B,
// assumed to be of the same dimension and performs the operation B = A - B.
// This was developed for checking slate::add without using slate::add to check.
// It is a CPU-only implementation and assumes column-major.
template <typename scalar_t>
void subtract_matrix( slate::Matrix<scalar_t>& A, slate::Matrix<scalar_t>& B )
{
    int64_t mt = A.mt();
    int64_t nt = A.nt();
    #pragma omp parallel for collapse(2)
    for (int64_t j = 0; j < nt; ++j) {
        for (int64_t i = 0; i < mt; ++i) {
            if (A.tileIsLocal( i, j )) {
                A.tileGetForReading( i, j, slate::LayoutConvert::None );
                B.tileGetForWriting( i, j, slate::LayoutConvert::None );
                auto TA = A( i, j );
                auto TB = B( i, j );
                int64_t mb = TA.mb();
                int64_t nb = TA.nb();
                int64_t lda = TA.stride();
                int64_t ldb = TB.stride();
                scalar_t const* TA_data = TA.data();
                scalar_t*       TB_data = TB.data();
                for (int64_t jj = 0; jj < nb; ++jj) {
                    for (int64_t ii = 0; ii < mb; ++ii) {
                        TB_data[ ii + jj*ldb ] -= TA_data[ ii + jj*lda ];
                    }
                }
            }
        }
    }
}

#define SLATE_HAVE_SCALAPACK
//------------------------------------------------------------------------------
template<typename scalar_t>
void test_add_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;
    using slate::ceildiv;

    // get & mark input values
    //
    // if transposition operations are included in add.cc,
    // the following slate::Op should can change.
    // slate::Op transA = params.trans();
    slate::Op transA = slate::Op::NoTrans;
    scalar_t alpha = params.alpha.get<real_t>();
    scalar_t beta = params.beta.get<real_t>();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
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
    params.matrixB.mark();

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

    // Matrix B: using the fact that B must be same dimensions as A.
    int64_t mlocB, nlocB, lldB;
    mlocB = mlocA, nlocB = nlocA, lldB = lldA;

    // ScaLAPACK data if needed.
    std::vector<scalar_t> A_data, B_data;

    slate::Matrix<scalar_t> A;
    slate::Matrix<scalar_t> B;
    if (origin != slate::Origin::ScaLAPACK) {
        // SLATE allocates CPU or GPU tiles.
        slate::Target origin_target = origin2target(origin);
        A = slate::Matrix<scalar_t>(m, n, nb, p, q, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);
        B = slate::Matrix<scalar_t>(m, n, nb, p, q, MPI_COMM_WORLD);
        B.insertLocalTiles(origin_target);
    }
    else {
        // Create SLATE matrix from the ScaLAPACK layout.
        A_data.resize( lldA * nlocA );
        B_data.resize( lldB * nlocB );
        A = slate::Matrix<scalar_t>::fromScaLAPACK(
                m, n, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        B = slate::Matrix<scalar_t>::fromScaLAPACK(
                m, n, &B_data[0], lldB, nb, p, q, MPI_COMM_WORLD);
    }

    slate::generate_matrix(params.matrix, A);
    slate::generate_matrix(params.matrixB, B);

    // if reference run is required, copy test data
    std::vector<scalar_t> Aref_data;
    slate::Matrix<scalar_t> Aref;
    std::vector<scalar_t> Bref_data;
    slate::Matrix<scalar_t> Bref;
    if (check || ref) {
        // For simplicity, always use ScaLAPACK format for ref matrices.
        Aref_data.resize( lldA*nlocA );
        Aref = slate::Matrix<scalar_t>::fromScaLAPACK(
                   m,  n, &Aref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        slate::copy(A, Aref);
        //
        Bref_data.resize( lldB*nlocB );
        Bref = slate::Matrix<scalar_t>::fromScaLAPACK(
                    m,  n, &Bref_data[0], lldB, nb, p, q, MPI_COMM_WORLD);
        slate::copy(B, Bref);
    }

/*
    if (trans == slate::Op::Trans)
        A = transpose(A);
    else if (trans == slate::Op::ConjTrans)
        A = conjTranspose(A);
*/

    print_matrix("A", A, params);
    print_matrix("B", B, params);

    if (! ref_only) {
        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        //==================================================
        // Run SLATE test.
        // Add alpha A by beta B into B.
        //==================================================
        double time = barrier_get_wtime(MPI_COMM_WORLD);

        slate::add(alpha, A, beta, B);

        time = barrier_get_wtime(MPI_COMM_WORLD) - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time;
    }

    if (check || ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            // comparison with reference routine from ScaLAPACK

            // BLACS/MPI variables
            int ictxt, p_, q_, myrow_, mycol_, info;
            int A_desc[9], B_desc[9];
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
            scalapack_descinit(B_desc, m, n, nb, nb, 0, 0, ictxt, lldB, &info);
            slate_assert(info == 0);

            real_t A_norm = slate::norm(slate::Norm::One, A);
            real_t B_norm = slate::norm(slate::Norm::One, B);
            // set MKL num threads appropriately for parallel BLAS
            int omp_num_threads;
            #pragma omp parallel
            { omp_num_threads = omp_get_num_threads(); }
            int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);

            print_matrix("Aref", mlocA, nlocA, &Aref_data[0], lldA, p, q, MPI_COMM_WORLD, params);
            print_matrix("Bref", mlocB, nlocB, &Bref_data[0], lldB, p, q, MPI_COMM_WORLD, params);

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            double time = barrier_get_wtime(MPI_COMM_WORLD);

            scalapack_pgeadd(op2str(transA), m, n,
                             alpha, &Aref_data[0], 1, 1, A_desc,
                             beta, &Bref_data[0], 1, 1, B_desc, &info);

            slate_assert(info == 0);

            time = barrier_get_wtime(MPI_COMM_WORLD) - time;

            print_matrix("Aref", mlocA, nlocA, &Aref_data[0], lldA, p, q, MPI_COMM_WORLD, params);
            print_matrix("Bref", mlocB, nlocA, &Bref_data[0], lldB, p, q, MPI_COMM_WORLD, params);

            // get differences A = A - Aref
            subtract_matrix(Aref,A);

            // get differences B = B - Bref
            subtract_matrix(Bref,B);

            print_matrix("DiffA", A, params);
            print_matrix("DiffB", B, params);

            // norm(A - Aref)
            real_t A_diff_norm = slate::norm(slate::Norm::One, A);
            // norm(B - Bref)
            real_t B_diff_norm = slate::norm(slate::Norm::One, B);

            params.ref_time() = time;

            real_t errorA = A_diff_norm / (n * A_norm);
            real_t errorB = B_diff_norm / (n * B_norm);

            params.error() = errorA + errorB;

            slate_set_num_blas_threads(saved_num_threads);

            real_t tol = params.tol() * std::numeric_limits<real_t>::epsilon()/2;
            // Allow for difference
            params.okay() = (params.error() <= tol);

            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering
        #endif
    }
}

// -----------------------------------------------------------------------------
void test_add(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_add_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_add_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_add_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_add_work<std::complex<double>> (params, run);
            break;
    }
}
