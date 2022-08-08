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
void test_hb2st_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;

    // get & mark input values
    int64_t n = params.dim.n();
    int64_t nb = params.nb();
    int64_t band = nb;  // for now use band == nb.
    int64_t p = params.grid.m();
    int64_t q = params.grid.n();
    slate::Uplo uplo = params.uplo();
    bool upper = uplo == slate::Uplo::Upper;
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
    params.error.name("L - Lref\nerror");
    params.error2.name("off-diag\nerror");

    if (! run)
        return;

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    int64_t lda = n;
    int64_t seed[] = {0, 1, 2, 3};

    std::vector<scalar_t> Afull_data( lda*n );
    lapack::larnv(1, seed, Afull_data.size(), &Afull_data[0]);

    // Zero outside the band.
    for (int64_t j = 0; j < n; ++j) {
        if (upper) {
            for (int64_t i = 0; i < n; ++i) {
                if (j > i+band || j < i)
                    Afull_data[i + j*lda] = 0;
            }
        }
        else { // lower
            for (int64_t i = 0; i < n; ++i) {
                if (j < i-band || j > i)
                    Afull_data[i + j*lda] = 0;
            }
        }
        // Diagonal from he2hb is real.
        Afull_data[j + j*lda] = real( Afull_data[j + j*lda] );
    }
    if (mpi_rank == 0) {
        print_matrix( "Afull_data", n, n, &Afull_data[0], lda, params );
    }

    auto Afull = slate::HermitianMatrix<scalar_t>::fromLAPACK(
        uplo, n, &Afull_data[0], lda, nb, p, q, MPI_COMM_WORLD);

    // Copy band of Afull, currently to rank 0.
    auto Aband = slate::HermitianBandMatrix<scalar_t>(
        uplo, n, band, nb,
        1, 1, MPI_COMM_WORLD);
    Aband.insertLocalTiles();
    Aband.he2hbGather( Afull );

    if (verbose) {
        print_matrix( "Aband", Aband, params );
    }

    // [code copied from heev.cc]
    // Matrix to store Householder vectors.
    // Could pack into a lower triangular matrix, but we store each
    // parallelogram in a 2nb-by-nb tile, with nt(nt + 1)/2 tiles.
    int64_t vm = 2*nb;
    int64_t nt = Afull.nt();
    int64_t vn = nt*(nt + 1)/2*nb;
    slate::Matrix<scalar_t> V(vm, vn, vm, nb, 1, 1, MPI_COMM_WORLD);
    V.insertLocalTiles();

    std::vector<real_t> Lambda1(n);
    if (check && mpi_rank == 0) {
        //==================================================
        // For checking results, compute eig of original matrix A.
        //==================================================
        std::vector<scalar_t> Afull_copy = Afull_data;
        info = lapack::heev(lapack::Job::NoVec, uplo, n, &Afull_copy[0], lda,
                            &Lambda1[0]);
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
        slate::hb2st(Aband, V);
    }

    time = barrier_get_wtime(MPI_COMM_WORLD) - time;
    params.time() = time;

    if (trace) slate::trace::Trace::finish();

    if (check) {
        //==================================================
        // Test results
        //==================================================
        // Copy Aband back to Afull_data on rank 0.
        Aband.gather(&Afull_data[0], lda);

        if (mpi_rank == 0) {
            print_matrix( "Afull_data_out", n, n, &Afull_data[0], lda, params );

            // Check that updated Aband is real tridiagonal by finding max value
            // outside tridiagonal, and imaginary parts of tridiagonal.
            // Unclear why this increases modestly with n.
            real_t max_value = 0;
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = 0; i < n; ++i) {
                    auto val = Afull_data[i + j*lda];
                    if (upper) {
                        if (j > i+1 || j < i)
                            max_value = std::max( std::abs(val), max_value );
                        else
                            max_value = std::max( std::abs(imag(val)), max_value );
                    }
                    else { // lower
                        if (j < i-1 || j > i)
                            max_value = std::max( std::abs(val), max_value );
                        else
                            max_value = std::max( std::abs(imag(val)), max_value );
                    }
                }
            }
            params.error2() = max_value / sqrt(n);

            // Check that the singular values of updated Aband
            // match the singular values of the original Aband.
            real_t tol = params.tol() * std::numeric_limits<real_t>::epsilon()/2;
            std::vector<real_t> Lambda2(n);
            std::vector<real_t> E(n - 1);  // super-diagonal
            scalar_t dummy[1];  // U, VT, C not needed for NoVec

            // Copy diagonal & super-diagonal.
            int64_t D_index = 0;
            int64_t E_index = 0;
            for (int64_t i = 0; i < Aband.nt(); ++i) {
                // Copy 1 element from super- or sub-diagonal tile to E.
                if (i > 0) {
                    if (upper) {
                        auto T = Aband(i-1, i);
                        E[E_index++] = real( T(T.mb()-1, 0) );
                    }
                    else {
                        auto T = Aband(i, i-1);
                        E[E_index++] = real( T(0, T.nb()-1) );
                    }
                }

                // Copy main diagonal to Lambda2.
                auto T = Aband(i, i);
                for (int64_t j = 0; j < T.nb(); ++j) {
                    Lambda2[D_index++] = real( T(j, j) );
                }

                // Copy super- or sub-diagonal to E.
                if (upper) {
                    for (int64_t j = 0; j < T.nb()-1; ++j) {
                        E[E_index++] = real( T(j, j+1) );
                    }
                }
                else {
                    for (int64_t j = 0; j < T.nb()-1; ++j) {
                        E[E_index++] = real( T(j+1, j) );
                    }
                }
            }

            print_matrix("D", 1, n,   &Lambda2[0], 1, params);
            print_matrix("E", 1, n-1, &E[0],  1, params);

            info = lapack::steqr(lapack::Job::NoVec, n, &Lambda2[0], &E[0],
                                 dummy, 1);
            assert(info == 0);

            if (verbose) {
                printf( "%% first and last 20 rows of Lambda1 and Lambda2\n" );
                printf( "%9s  %9s\n", "Lambda1", "Lambda2" );
                for (int64_t i = 0; i < n; ++i) {
                    if (i < 20 || i >= n-20) {
                        bool okay = std::abs( Lambda1[i] - Lambda2[i] ) < tol;
                        printf( "%9.6f  %9.6f%s\n",
                                Lambda1[i], Lambda2[i], (okay ? "" : " !!") );
                    }
                    else if (i == 20) {
                        printf( "--------------------\n" );
                    }
                }
                printf( "\n" );
            }

            // Relative forward error: || L - Lref || / || Lref ||.
            blas::axpy(Lambda2.size(), -1.0, &Lambda1[0], 1, &Lambda2[0], 1);
            params.error() = blas::nrm2(Lambda2.size(), &Lambda2[0], 1)
                           / blas::nrm2(Lambda1.size(), &Lambda1[0], 1);
            params.okay() = (params.error() <= tol && params.error2() <= tol);
        }
    }
}

// -----------------------------------------------------------------------------
void test_hb2st(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_hb2st_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_hb2st_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_hb2st_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_hb2st_work<std::complex<double>> (params, run);
            break;
    }
}
