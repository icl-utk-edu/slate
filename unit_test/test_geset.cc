// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
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

namespace test {

//------------------------------------------------------------------------------
// global variables
int mpi_rank;
int mpi_size;
int verbose;
int num_devices;

//------------------------------------------------------------------------------
void test_geset_dev_worker(
    int m, int n, int lda,
    double offdiag_value, double diag_value)
{
    if (num_devices == 0) {
        test_skip("requires num_devices > 0");
    }

    double eps = std::numeric_limits<double>::epsilon();
    int ldb = lda;

    double* Adata = new double[ lda * n ];
    slate::Tile<double> A( m, n, Adata, lda,
        slate::HostNum, slate::TileKind::UserOwned );

    double* Bdata = new double[ ldb * n ];
    slate::Tile<double> B( m, n, Bdata, ldb,
        slate::HostNum, slate::TileKind::UserOwned );

    int device_idx;
    blas::get_device( &device_idx );
    const int batch_arrays_index = 0;
    blas::Queue queue( device_idx, batch_arrays_index );

    double* dAdata;
    dAdata = blas::device_malloc<double>( blas::max( lda * n, 1 ) );
    test_assert( dAdata != nullptr );
    slate::Tile<double> dA( m, n, dAdata, lda,
        device_idx, slate::TileKind::UserOwned );

    slate::device::geset( m, n,
                          offdiag_value, diag_value,
                          dA.data(), lda,
                          queue );

    queue.sync();
    dA.copyData( &A, queue );

    // compute on CPU to check the results
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            if (i == j) {
                Bdata[ i + j*ldb ] = diag_value;
            }
            else {
                Bdata[ i + j*ldb ] = offdiag_value;
            }
        }
    }

    //blas::axpy( lda*n, neg_one, B.data(), ione, A.data(), ione );
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            Adata[i + j*lda] = Bdata[i + j*ldb] -  Adata[i + j*lda];
        }
    }

    // check final norm result
    double result = lapack::lange(
        lapack::Norm::Fro, A.mb(), A.nb(), A.data(), A.stride() );

    if (verbose) {
        printf( "\n(%4d, %4d, %4d, %4.2f, %4.2f ): error %.2f",
            m, n, lda, offdiag_value, diag_value, result );
    }

    blas::device_free( dAdata );
    delete[] Adata;
    delete[] Bdata;

    test_assert( result < 3*eps );
}

void test_geset_dev()
{
    // Each tuple contains (mA, nA, lda)
    std::list< std::tuple< int, int, int > > dims_list{
            // Corner cases
            {   0,   0,   0 },
            { 100,   0, 100 },
            {   0, 100,   0 },
            // Square A matrix
            {   1,   1,   1 },
            {  10,  10,  10 },
            {  20,  20,  20 },
            {  50,  50,  50 },
            { 100, 100, 100 },
            {   7,   7,   7 },
            {  11,   11, 11 },
            {  33,   33, 33 },
            // Rectangular A matrix
            {   1, 100,   1 },
            { 100,   1, 100 },
            {  20,  30,  20 },
            // lda != mA
            {  20,  30,  roundup(20, 8) }, //as it was in the original test
            {  33,  33,  50 },
        };

    // Each tuple contains (offdiag_value, diag_value)
    std::list< std::tuple< double, double > > values_list{
            {   0,   0 }, //Special case
            {   0, 0.5 },
            { 0.3,   0 },
            { 0.3, 0.5 },
        };

    for (auto dims : dims_list) {
        int mA  = std::get<0>( dims );
        int nA  = std::get<1>( dims );
        int lda = std::get<2>( dims );
        for (auto values : values_list) {
            double offdiag_value  = std::get<0>( values );
            double diag_value     = std::get<1>( values );
            test_geset_dev_worker(
                mA, nA, lda, offdiag_value, diag_value );
        }
    }
}

//------------------------------------------------------------------------------
void test_geset_batch_dev_worker(
    int m, int n, int lda,
    double offdiag_value, double diag_value,
    int batch_count)
{
    if (num_devices == 0) {
        test_skip("requires num_devices > 0");
    }

    double eps = std::numeric_limits<double>::epsilon();
    int ldb = lda;
    std::vector< slate::Tile< double > > list_A( 0 );
    std::vector< slate::Tile< double > > list_dA( 0 );

    for (int m_i = 0; m_i < batch_count; ++m_i) {
        double* tmp_data = new double[ lda * n ];
        test_assert( tmp_data != nullptr );
        list_A.push_back( slate::Tile<double>( m, n, tmp_data, lda,
            slate::HostNum, slate::TileKind::UserOwned ) );
    }

    double* Bdata = new double[ ldb * n ];
    slate::Tile<double> B( m, n, Bdata, ldb,
        slate::HostNum, slate::TileKind::UserOwned );

    int device_idx;
    blas::get_device( &device_idx );
    const int batch_arrays_index = 0;
    blas::Queue queue( device_idx, batch_arrays_index );

    for (int m_i = 0; m_i < batch_count; ++m_i) {
        double* dtmp_data;
        dtmp_data = blas::device_malloc<double>( blas::max( lda * n, 1 ) );
        test_assert( dtmp_data != nullptr );
        list_dA.push_back( slate::Tile<double>( m, n, dtmp_data, lda,
            device_idx, slate::TileKind::UserOwned ) );
    }

    double** Aarray = new double*[ batch_count ];
    double** dAarray;
    dAarray = blas::device_malloc<double*>( batch_count );
    test_assert( dAarray != nullptr );
    for (int m_i = 0; m_i < batch_count; ++m_i) {
      auto dA = list_dA[ m_i ];
      Aarray[m_i] = dA.data();
    }
    blas::device_memcpy<double*>( dAarray, Aarray,
                        batch_count,
                        blas::MemcpyKind::HostToDevice,
                        queue );

    slate::device::batch::geset( m, n,
                          offdiag_value, diag_value, dAarray, lda,
                          batch_count, queue );

    queue.sync();
    for (int m_i = 0; m_i < batch_count; ++m_i) {
        auto dA = list_dA[ m_i ];
        auto A = list_A[ m_i ];
        dA.copyData( &A, queue );
    }

    // compute on CPU to check the results
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            if (i == j) {
                Bdata[ i + j*ldb ] = diag_value;
            }
            else {
                Bdata[ i + j*ldb ] = offdiag_value;
            }
        }
    }

    for (int m_i = 0; m_i < batch_count; ++m_i) {
        auto dA = list_dA[ m_i ];
        auto A  = list_A[ m_i ];
        double *Adata = A.data();

        //blas::axpy( lda*n, neg_one, B.data(), ione, A.data(), ione );
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                Adata[i + j*lda] = Bdata[i + j*ldb] -  Adata[i + j*lda];
            }
        }

        // check final norm result
        double result = lapack::lange(
            lapack::Norm::Fro, A.mb(), A.nb(), A.data(), A.stride() );

        if (verbose) {
            // Display (m, n, lda, offdiag_value, diag_value)
            if (m_i == 0)
                printf( "\n(%4d, %4d, %4d, %4.2f, %4.2f ):",
                    m, n, lda, offdiag_value, diag_value );
            if (verbose > 1)
                printf( "\n\t[%d] error %.2e ",
                    m_i, result );
        }

        blas::device_free( dA.data() );
        delete[] A.data();

        test_assert( result < 3*eps );
    }
    blas::device_free( dAarray );
    delete[] Bdata;
    delete[] Aarray;

}

void test_geset_batch_dev()
{
    // Each tuple contains (mA, nA, lda)
    std::list< std::tuple< int, int, int > > dims_list{
            // Corner cases
            {   0,   0,   0 },
            { 100,   0, 100 },
            {   0, 100,   0 },
            // Square A matrix
            {   1,   1,   1 },
            {  10,  10,  10 },
            {  20,  20,  20 },
            {   7,   7,   7 },
            {  11,   11, 11 },
            {  33,   33, 33 },
            // Rectangular A matrix
            {   1, 100,   1 },
            { 100,   1, 100 },
            {  20,  30,  20 },
            // lda != mA
            {  20,  30,  roundup(20, 8) }, //as it was in the original test
            {  33,  33,  50 },
        };

    // Each tuple contains (offdiag_value, diag_value)
    std::list< std::tuple< double, double > > values_list{
            {   0,   0 }, //Special case
            {   0, 0.5 },
            { 0.3,   0 },
            { 0.3, 0.5 },
        };

    std::list< int > batch_count_list{ 1, 2, 3, 4, 5, 10, 20, 100 };

    for (auto dims : dims_list) {
        int mA  = std::get<0>( dims );
        int nA  = std::get<1>( dims );
        int lda = std::get<2>( dims );
        for (auto values : values_list) {
            double offdiag_value  = std::get<0>( values );
            double diag_value     = std::get<1>( values );
            for (auto batch_count : batch_count_list)
                test_geset_batch_dev_worker(
                    mA, nA, lda, offdiag_value, diag_value, batch_count );
        }
    }
}

//------------------------------------------------------------------------------
/// Runs all tests. Called by unit test main().
void run_tests()
{
    if (mpi_rank == 0) {
        //-------------------- geset_dev
        run_test(
            test_geset_dev, "geset_dev" );
        //-------------------- geset_batch_dev
        run_test(
            test_geset_batch_dev, "geset_batch_dev" );
    }
}

}  // namespace test

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    using namespace test;  // for globals mpi_rank, etc.

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
