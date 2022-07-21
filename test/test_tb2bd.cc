// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "blas.hh"
#include "test.hh"
#include "print_matrix.hh"
#include "grid_utils.hh"
#include "scalapack_support_routines.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_tb2bd_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t nb = params.nb();
    int64_t ku = nb;  // upper band; for now use ku == nb.
    int64_t p = params.grid.m();
    int64_t q = params.grid.n();
    bool check = params.check() == 'y';
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    int64_t info;

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();
    params.error2();
    params.error.name("S - Sref\nerror");
    params.error2.name("off-diag\nerror");

    if (! run)
        return;

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    int64_t lda = m;
    int64_t seed[] = {0, 1, 2, 3};
    int64_t min_mn = std::min(m, n);

    std::vector<scalar_t> Afull_data( lda*n );
    lapack::larnv(1, seed, Afull_data.size(), &Afull_data[0]);

    // Zero outside the upper band.
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) {
            if (j > i+ku || j < i)
                Afull_data[i + j*lda] = 0;
        }
        // Diagonal from ge2tb is real.
        Afull_data[j + j*lda] = real( Afull_data[j + j*lda] );
    }
    if (mpi_rank == 0) {
        print_matrix( "Afull_data", m, n, &Afull_data[0], lda, params );
    }

    auto Afull = slate::Matrix<scalar_t>::fromLAPACK(
        m, n, &Afull_data[0], lda, nb, p, q, MPI_COMM_WORLD);

    // Copy band of Afull, currently to rank 0.
    auto Aband = slate::TriangularBandMatrix<scalar_t>(
        slate::Uplo::Upper, slate::Diag::NonUnit, n, ku, nb,
        1, 1, MPI_COMM_WORLD);
    Aband.insertLocalTiles();
    Aband.ge2tbGather( Afull );

    if (verbose) {
        print_matrix("Aband", Aband, params);
    }

    std::vector<real_t> Sigma1(min_mn);
    if (check && mpi_rank == 0) {
        //==================================================
        // For checking results, compute SVD of original matrix A.
        //==================================================
        std::vector<scalar_t> Afull_copy = Afull_data;
        scalar_t dummy[1];  // U, VT not needed for NoVec
        info = lapack::gesvd(lapack::Job::NoVec, lapack::Job::NoVec,
                             m, n, &Afull_copy[0], lda, &Sigma1[0],
                             dummy, 1, dummy, 1);
        assert(info == 0);
    }

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    double time = barrier_get_wtime(MPI_COMM_WORLD);

    //==================================================
    // Run SLATE test.
    // Currently runs only on rank 0.
    //==================================================
    if (mpi_rank == 0) {
        slate::tb2bd(Aband);
    }

    time = barrier_get_wtime(MPI_COMM_WORLD) - time;
    params.time() = time;

    if (trace) slate::trace::Trace::finish();

    if (check) {
        //==================================================
        // Test results
        // Gather the whole matrix onto rank 0.
        //==================================================
        // Copy Aband back to Afull_data on rank 0.
        Aband.gather(&Afull_data[0], lda);

        if (mpi_rank == 0) {
            print_matrix( "Afull_data_out", m, n, &Afull_data[0], lda, params );

            // Check that updated Aband is real bidiagonal by finding max value
            // outside bidiagonal, and imaginary parts of bidiagonal.
            // Unclear why this increases modestly with n.
            real_t max_value = 0;
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = 0; i < m; ++i) {
                    auto val = Afull_data[i + j*lda];
                    if (j > i+1 || j < i)
                        max_value = std::max( std::abs(val), max_value );
                    else
                        max_value = std::max( std::abs(imag(val)), max_value );
                }
            }
            params.error2() = max_value / sqrt(n);

            // Check that the singular values of updated Aband
            // match the singular values of the original Aband.
            real_t tol = params.tol() * 0.5 * std::numeric_limits<real_t>::epsilon();
            std::vector<real_t> Sigma2(min_mn);
            std::vector<real_t> E(min_mn - 1);  // super-diagonal
            scalar_t dummy[1];  // U, VT, C not needed for NoVec

            // Copy diagonal & super-diagonal.
            int64_t D_index = 0;
            int64_t E_index = 0;
            for (int64_t i = 0; i < std::min(Aband.mt(), Aband.nt()); ++i) {
                // Copy 1 element from super-diagonal tile to E.
                if (i > 0) {
                    auto T = Aband(i-1, i);
                    E[E_index++] = real( T(T.mb()-1, 0) );
                }

                // Copy main diagonal to Sigma2.
                auto T = Aband(i, i);
                for (int64_t j = 0; j < T.nb(); ++j) {
                    Sigma2[D_index++] = real( T(j, j) );
                }

                // Copy super-diagonal to E.
                for (int64_t j = 0; j < T.nb()-1; ++j) {
                    E[E_index++] = real( T(j, j+1) );
                }
            }

            print_matrix("D", 1, min_mn,   &Sigma2[0], 1, params);
            print_matrix("E", 1, min_mn-1, &E[0],  1, params);

            info = lapack::bdsqr(lapack::Uplo::Upper, min_mn, 0, 0, 0,
                                 &Sigma2[0], &E[0], dummy, 1, dummy, 1, dummy, 1);
            assert(info == 0);

            if (verbose) {
                printf( "%% first and last 20 rows of Sigma1 and Sigma2\n" );
                printf( "%9s  %9s\n", "Sigma1", "Sigma2" );
                for (int64_t i = 0; i < std::min(m, n); ++i) {
                    if (i < 20 || i >= std::min(m, n)-20) {
                        bool okay = std::abs( Sigma1[i] - Sigma2[i] ) < tol;
                        printf( "%9.6f  %9.6f%s\n",
                                Sigma1[i], Sigma2[i], (okay ? "" : " !!") );
                    }
                    else if (i == 20) {
                        printf( "--------------------\n" );
                    }
                }
                printf( "\n" );
            }

            // Relative forward error: || Sigma - Sigma_ref || / || Sigma_ref ||.
            blas::axpy(Sigma2.size(), -1.0, &Sigma1[0], 1, &Sigma2[0], 1);
            params.error() = blas::nrm2(Sigma2.size(), &Sigma2[0], 1)
                           / blas::nrm2(Sigma1.size(), &Sigma1[0], 1);
            params.okay() = (params.error() <= tol && params.error2() <= tol);
        }
    }
}

// -----------------------------------------------------------------------------
void test_tb2bd(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_tb2bd_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_tb2bd_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_tb2bd_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_tb2bd_work<std::complex<double>> (params, run);
            break;
    }
}
