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
#include "band_utils.hh"
#include "grid_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template<typename scalar_t>
void test_hbnorm_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;
    using slate::ceildiv;

    // get & mark input values
    slate::Norm norm = params.norm();
    slate::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t kd = params.kd();
    int64_t nb = params.nb();
    int p = params.grid.m();
    int q = params.grid.n();
    bool check = params.check() == 'y';
    bool ref = params.ref() == 'y';
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();

    // mark non-standard output values
    params.time();
    params.ref_time();

    if (! run)
        return;

    if (origin != slate::Origin::ScaLAPACK) {
        params.msg() = "skipping: currently only origin=scalapack is supported";
        return;
    }

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    // Matrix A: figure out local size.
    int64_t mlocA = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocA = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldA  = blas::max(1, mlocA); // local leading dimension of A
    std::vector<scalar_t> A_data(lldA*nlocA);

    // todo: fix the generation
    int64_t iseeds[4] = { myrow, mycol, 2, 3 };
    //lapack::larnv(2, iseeds, lldA*nlocA, &A_data[0]);
    for (int64_t j = 0; j < nlocA; ++j)
        lapack::larnv(2, iseeds, mlocA, &A_data[j*lldA]);

    zeroOutsideBand(uplo, &A_data[0], n, kd, nb, myrow, mycol, p, q, lldA);

    // Create SLATE matrix from the ScaLAPACK layout.
    // TODO: data origin on GPU
    auto A = HermitianBandFromScaLAPACK(
                 uplo, n, kd, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD);

    print_matrix("A", A, params);

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    double time = barrier_get_wtime(MPI_COMM_WORLD);

    //==================================================
    // Run SLATE test.
    // Compute || A ||_norm.
    //==================================================
    real_t A_norm = slate::norm(norm, A, {
        {slate::Option::Target, target}
    });

    time = barrier_get_wtime(MPI_COMM_WORLD) - time;

    if (trace) slate::trace::Trace::finish();

    // compute and save timing/performance
    params.time() = time;

    if (check || ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            // comparison with reference routine from ScaLAPACK

            // BLACS/MPI variables
            int ictxt, p_, q_, myrow_, mycol_, info;
            int A_desc[9];
            int mpi_rank_ = 0, nprocs = 1;

            // initialize BLACS and ScaLAPACK
            Cblacs_pinfo(&mpi_rank_, &nprocs);
            slate_assert( mpi_rank == mpi_rank_ );
            slate_assert(p*q <= nprocs);
            Cblacs_get(-1, 0, &ictxt);
            Cblacs_gridinit(&ictxt, "Col", p, q);
            Cblacs_gridinfo(ictxt, &p_, &q_, &myrow_, &mycol_);
            slate_assert( p == p_ );
            slate_assert( q == q_ );
            slate_assert( myrow == myrow_ );
            slate_assert( mycol == mycol_ );

            scalapack_descinit(A_desc, n, n, nb, nb, 0, 0, ictxt, lldA, &info);
            slate_assert(info == 0);

            // allocate work space
            int lcm = scalapack_ilcm(&p, &q);
            int ldw = nb*slate::ceildiv(int(slate::ceildiv(nlocA, nb)), (lcm / p));
            int lwork = 2*mlocA + nlocA + ldw;
            std::vector<real_t> worklanhe(lwork);

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            time = barrier_get_wtime(MPI_COMM_WORLD);
            real_t A_norm_ref = scalapack_planhe(
                                    norm2str(norm), uplo2str(A.uplo()),
                                    n, &A_data[0], 1, 1, A_desc, &worklanhe[0]);
            time = barrier_get_wtime(MPI_COMM_WORLD) - time;

            //A_norm_ref = lapack::lanhe(
            //    norm, A.uplo(),
            //    n, &A_data[0], lldA);

            // difference between norms
            real_t error = std::abs(A_norm - A_norm_ref) / A_norm_ref;
            if (norm == slate::Norm::One || norm == slate::Norm::Inf) {
                error /= sqrt(n);
            }
            else if (norm == slate::Norm::Fro) {
                error /= n;  // = sqrt( n*n );
            }

            if (verbose && mpi_rank == 0) {
                printf("norm %15.8e, ref %15.8e, ref - norm %5.2f, error %9.2e\n",
                       A_norm, A_norm_ref, A_norm_ref - A_norm, error);
            }

            // Allow for difference, except max norm in real should be exact.
            real_t eps = std::numeric_limits<real_t>::epsilon();
            real_t tol;
            if (norm == slate::Norm::Max && ! slate::is_complex<scalar_t>::value)
                tol = 0;
            else
                tol = 10*eps;

            params.ref_time() = time;
            params.error() = error;

            // Allow for difference
            params.okay() = (params.error() <= tol);

            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering
        #else  // not SLATE_HAVE_SCALAPACK
            SLATE_UNUSED( A_norm );
            SLATE_UNUSED( verbose );
            if (mpi_rank == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }
}

// -----------------------------------------------------------------------------
void test_hbnorm(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_hbnorm_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_hbnorm_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_hbnorm_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_hbnorm_work<std::complex<double>> (params, run);
            break;
    }
}
