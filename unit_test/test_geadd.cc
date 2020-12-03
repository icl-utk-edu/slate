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
void test_geadd_dev()
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
    lapack::larnv( idist, iseed, ldb * n, B.data() );

    double* B0data = new double[ ldb * n ];
    slate::Tile<double> B0(m, n, B0data, ldb, -1, slate::TileKind::UserOwned);
    gecopy( B, B0 );

    double* C0data = new double[ ldb * n ];
    slate::Tile<double> C0(m, n, C0data, ldb, -1, slate::TileKind::UserOwned);

    int device_idx;
    blas::get_device(&device_idx);
    const int batch_arrays_index = 0;
    blas::Queue queue(device_idx, batch_arrays_index);

    double* dAdata;
    dAdata = blas::device_malloc<double>(lda * n);
    test_assert(dAdata != nullptr);
    slate::Tile<double> dA(m, n, dAdata, lda, 0, slate::TileKind::UserOwned);
    A.copyData(&dA, queue.stream());

    double* dBdata;
    dBdata = blas::device_malloc<double>(ldb * n);
    test_assert(dBdata != nullptr);
    slate::Tile<double> dB(m, n, dBdata, ldb, 0, slate::TileKind::UserOwned);
    B.copyData(&dB, queue.stream());

    const int batch_count = 1;
    double* Aarray[batch_count];
    double** dAarray;
    dAarray = blas::device_malloc<double*>(batch_count);
    test_assert(dAarray != nullptr);
    Aarray[0] = dA.data();
    blas::device_memcpy<double*>((void*)dAarray, (void*)Aarray,
                        batch_count,
                        blas::MemcpyKind::HostToDevice,
                        queue);

    double* Barray[batch_count];
    double** dBarray;
    dBarray = blas::device_malloc<double*>(batch_count);
    test_assert(dBarray != nullptr);
    Barray[0] = dB.data();
    blas::device_memcpy<double*>((void*)dBarray, (void*)Barray,
                        batch_count,
                        blas::MemcpyKind::HostToDevice,
                        queue);
    slate::device::geadd( m, n,
                          alpha, dAarray, lda,
                          beta,  dBarray, ldb,
                          batch_count, queue );

    queue.sync();

    blas::device_memcpy<double*>((void*)Barray, (void*)dBarray,
                        batch_count,
                        blas::MemcpyKind::DeviceToHost,
                        queue);
    dB.copyData(&B, queue.stream());

    // compute on CPU to check the results
    for( int j = 0; j < n; ++j ) {
        for( int i = 0; i < m; ++i ) {
            C0data[i + j*ldb] = alpha * Adata[i + j*lda] + beta * B0data[i + j*ldb];
        }
    }

    //blas::axpy( B.size(), neg_one, B.data(), ione, C0.data(), ione );
    for( int j = 0; j < n; ++j ) {
        for( int i = 0; i < m; ++i ) {
            Adata[i + j*lda] = Bdata[i + j*ldb] -  C0data[i + j*ldb];
        }
    }

    //printf("\n C0 \n");
    //for( int i = 0; i < m; ++i ) {
    //    for( int j = 0; j < n; ++j ) {
    //        printf("\t %e", C0data[i + j*ldb]);
    //    }
    //    printf("\n");
    //}
    //printf("\n B \n");
    //for( int i = 0; i < m; ++i ) {
    //    for( int j = 0; j < n; ++j ) {
    //        printf("\t %e", Bdata[i + j*ldb]);
    //    }
    //    printf("\n");
    //}

    // check final norm result
    double result = lapack::lange(
        lapack::Norm::Fro, A.mb(), A.nb(), A.data(), A.stride() );

    if (verbose) {
        printf( "\nerror %.2e ",
                result );
    }

    blas::device_free(dAdata);
    blas::device_free(dBdata);
    blas::device_free(dAarray);
    blas::device_free(dBarray);
    delete[] Adata;
    delete[] Bdata;
    delete[] B0data;
    delete[] C0data;

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
            test_geadd_dev, "geadd_dev");
    }
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    num_devices = blas::get_device_count();

    verbose = 0;
    for (int i = 1; i < argc; ++i)
        if (argv[i] == std::string("-v"))
            verbose += 1;

    int err = unit_test_main(MPI_COMM_WORLD);  // which calls run_tests()

    MPI_Finalize();
    return err;
}
