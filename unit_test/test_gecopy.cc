// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Tile.hh"
#include "internal/internal.hh"
#include "slate/Tile_blas.hh"
#include "internal/Tile_lapack.hh"
#include "slate/internal/device.hh"
#include "slate/internal/util.hh"

#include "unit_test.hh"

using slate::roundup;

//------------------------------------------------------------------------------
// global variables
int mpi_rank;
int mpi_size;
int verbose;
int num_devices;

//------------------------------------------------------------------------------
void test_gecopy_dev()
{
    if (num_devices == 0) {
        test_skip("requires num_devices > 0");
    }

    double eps = std::numeric_limits<double>::epsilon();
    int m = 20;
    int n = 30;
    int lda = roundup(m, 8);
    int ldb = lda;
    double alpha = 0.5;
    double beta  = 0.3;
    double neg_one = -1.0;
    int ione = 1;
    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };

    double* Adata = new double[ lda * n ];
    slate::Tile<double> A(m, n, Adata, lda, -1, slate::TileKind::UserOwned);
    lapack::larnv( idist, iseed, lda * n, A.data() );

    double* Bdata = new double[ ldb * n ];
    slate::Tile<double> B(m, n, Bdata, ldb, -1, slate::TileKind::UserOwned);
    //gecopy( A, B );

    int device_idx;
    blas::get_device(&device_idx);
    const int batch_arrays_index = 0;
    blas::Queue queue(device_idx, batch_arrays_index);
    //test_assert(cudaStreamCreate(&stream) == cudaSuccess);

    double* dAdata;
    //test_assert(cudaMalloc((void**)&dAdata, sizeof(double) * lda * n) == cudaSuccess);
    test_assert(dAdata != nullptr);
    slate::Tile<double> dA(m, n, dAdata, lda, 0, slate::TileKind::UserOwned);
    A.copyData(&dA, queue.stream());

    double* dBdata;
    //test_assert(cudaMalloc((void**)&dBdata, sizeof(double) * ldb * n) == cudaSuccess);
    test_assert(dBdata != nullptr);
    slate::Tile<double> dB(m, n, dBdata, ldb, 0, slate::TileKind::UserOwned);
    B.copyData(&dB, queue.stream());

    const int batch_count = 1;
    double* Aarray[batch_count];
    double** dAarray;
    test_assert(cudaMalloc((void**)&dAarray, sizeof(double*) * batch_count) == cudaSuccess);
    test_assert(dAarray != nullptr);
    Aarray[0] = dA.data();
    //test_assert(cudaMemcpy(dAarray, Aarray, sizeof(double*) * batch_count,
    //                       cudaMemcpyHostToDevice ) == cudaSuccess);

    double* Barray[batch_count];
    double** dBarray;
    //test_assert(cudaMalloc((void**)&dBarray, sizeof(double*) * batch_count) == cudaSuccess);
    test_assert(dBarray != nullptr);
    Barray[0] = dB.data();
    //test_assert(cudaMemcpy(dBarray, Barray, sizeof(double*) * batch_count,
    //                       cudaMemcpyHostToDevice ) == cudaSuccess);
    slate::device::gecopy( m, n,
                           dAarray, lda,
                           dBarray, ldb,
                           batch_count, queue );

    queue.sync();

    //test_assert(cudaMemcpy( Barray, dBarray, sizeof(double*) * batch_count,
    //                        cudaMemcpyDeviceToHost ) == cudaSuccess );
    dB.copyData(&B, queue.stream());

    // compute B-B0 on CPU to check the results
    //blas::axpy( A.size(), neg_one, A.data(), ione, B.data(), ione );
    for( int j = 0; j < n; ++j ) {
        for( int i = 0; i < m; ++i ) {
            Adata[i + j*lda] = Adata[i + j*lda] -  Bdata[i + j*ldb];
        }
    }

    // check final norm result
    double result = lapack::lange(
        lapack::Norm::Fro, A.mb(), A.nb(), A.data(), A.stride() );

    if (verbose) {
        printf( "\nerror %.2e ",
                result );
    }

    cudaFree( dAdata );
    cudaFree( dBdata );
    cudaFree( dAarray );
    cudaFree( dBarray );
    delete[] Adata;
    delete[] Bdata;

    test_assert( result < 3*eps );
}

//------------------------------------------------------------------------------
/// Runs all tests. Called by unit test main().
void run_tests()
{
    if (mpi_rank == 0) {
        //-------------------- genorm_dev
        for (int i = 0; i < 10; i++)
        run_test(
            test_gecopy_dev, "gecopy_dev");
    }
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    cudaGetDeviceCount(&num_devices);

    verbose = 0;
    for (int i = 1; i < argc; ++i)
        if (argv[i] == std::string("-v"))
            verbose += 1;

    int err = unit_test_main(MPI_COMM_WORLD);  // which calls run_tests()

    MPI_Finalize();
    return err;
}
