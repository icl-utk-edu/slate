// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"
#include "print_matrix.hh"
#include "grid_utils.hh"
#include "matrix_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include <cuda_profiler_api.h>
//------------------------------------------------------------------------------
template <typename scalar_t>
void test_unmtr_hb2st_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;

    // Constants
    const scalar_t zero = 0;
    const scalar_t one  = 1;

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

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();
    params.ortho();

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
    if (verbose >= 2 && mpi_rank == 0) {
        print_matrix( "Afull_data", n, n, &Afull_data[0], lda );
    }

    auto Afull = slate::HermitianMatrix<scalar_t>::fromLAPACK(
        uplo, n, &Afull_data[0], lda, nb, p, q, MPI_COMM_WORLD);

    // Copy band of Afull, currently to rank 0.
    auto Aband = slate::HermitianBandMatrix<scalar_t>(
        uplo, n, band, nb,
        1, 1, MPI_COMM_WORLD);
    Aband.insertLocalTiles();
    Aband.he2hbGather( Afull );

    if (verbose >= 2) {
        print_matrix( "Aband", Aband );
    }

    //--------------------
    // [code copied from heev.cc]
    // Matrix to store Householder vectors.
    // Could pack into a lower triangular matrix, but we store each
    // parallelogram in a 2nb-by-nb tile, with nt(nt + 1)/2 tiles.
    int64_t vm = 2*nb;
    int64_t nt = Afull.nt();
    int64_t vn = nt*(nt + 1)/2*nb;
    slate::Matrix<scalar_t> V(vm, vn, vm, nb, 1, 1, MPI_COMM_WORLD);
    V.insertLocalTiles();
    //--------------------

    // Compute tridiagonal and Householder vectors V.
    if (mpi_rank == 0) {
        slate::hb2st(Aband, V);
    }
    if (verbose >= 2) {
        print_matrix( "Aband2", Aband );
        print_matrix( "V", V );
    }

    // Set Q = Identity. Use 1D column cyclic.
    slate::Matrix<scalar_t> Q(n, n, nb, 1, p*q, MPI_COMM_WORLD);
    Q.insertLocalTiles();
    set(zero, one, Q);
    if (verbose >= 2) {
        print_matrix( "Q0", Q );
    }

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    double time = barrier_get_wtime(MPI_COMM_WORLD);

    //==================================================
    // Run SLATE test.
    //==================================================
    cudaProfilerStart();
    #pragma omp parallel
    #pragma omp master
    {
        omp_set_nested(1);
        #pragma omp task
        {
            slate::unmtr_hb2st(slate::Side::Left, slate::Op::NoTrans, V, Q);
        }
    }
    cudaProfilerStop();
    time = barrier_get_wtime(MPI_COMM_WORLD) - time;

    if (trace) slate::trace::Trace::finish();

    // compute and save timing/performance
    params.time() = time;
    //params.gflops() = gflop / time;

    if (verbose >= 2) {
        print_matrix( "Q", Q );
    }

    if (check) {
        //==================================================
        // Test results
        // || I - Q^H Q || / n < tol
        // || A - Q S Q^H || / (n || A ||) < tol  // todo
        //==================================================
        slate::Matrix<scalar_t> R( n, n, nb, 1, 1, MPI_COMM_WORLD );
        R.insertLocalTiles();
        set(zero, one, R);
        if (verbose >= 2) {
            print_matrix( "R0", R );
        }

        auto QH = conj_transpose(Q);
        slate::gemm(-one, QH, Q, one, R);
        if (verbose >= 2) {
            print_matrix( "R", R );
        }

        real_t R_norm = slate::norm(slate::Norm::One, R);
        params.ortho() = R_norm / n;

        real_t tol = params.tol() * std::numeric_limits<real_t>::epsilon() / 2;
        params.okay() = (params.ortho() <= tol);
    }
}

// -----------------------------------------------------------------------------
void test_unmtr_hb2st(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_unmtr_hb2st_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_unmtr_hb2st_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_unmtr_hb2st_work< std::complex<float> > (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_unmtr_hb2st_work< std::complex<double> > (params, run);
            break;
    }
}
