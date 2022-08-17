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
/// Sets Aij = (mpi_rank + 1)*1000 + i + j/1000, for all i, j.
template <typename scalar_t>
void setup_data(slate::Tile<scalar_t>& A)
{
    //int m = A.mb();
    int n = A.nb();
    int lda = A.stride();
    scalar_t* Ad = A.data();
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < lda; ++i) {  // note: to lda, not just m
            Ad[ i + j*lda ] = (mpi_rank + 1)*1000 + i + j/1000.;
        }
    }
}

/// Sets Aij = (i == j ? diag_value : offdiag_value)
/// If offdiag_value and diag_value are 0, call a variant.
template <typename scalar_t>
void setup_data(slate::Tile<scalar_t>& A,
    double offdiag_value, double diag_value)
{
    if (offdiag_value == 0.0 && diag_value == 0.0 ) {
        setup_data( A );
        return;
    }

    int n = A.nb();
    int lda = A.stride();
    scalar_t* Ad = A.data();
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < lda; ++i) {  // note: to lda, not just m
            if (i == j) {
                Ad[ i + j*lda ] = diag_value;
            }
            else {
                Ad[ i + j*lda ] = offdiag_value;
            }
        }
    }
}

//------------------------------------------------------------------------------
//void test_gescale_dev() {
void test_gescale_dev_worker(int m, int n, int lda,
    double offdiag_value, double diag_value,
    double numer, double denom)
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
    dAdata = blas::device_malloc<double>( lda * n );
    test_assert( dAdata != nullptr );
    slate::Tile<double> dA( m, n, dAdata, lda,
        device_idx, slate::TileKind::UserOwned );

    setup_data( A, offdiag_value, diag_value );

    blas::device_memcpy<double>( dA.data(), A.data(),
                        lda * n,
                        blas::MemcpyKind::HostToDevice,
                        queue );

    slate::device::gescale( m, n,
                          numer, denom,
                          dA.data(), lda,
                          queue );

    queue.sync();
    dA.copyData( &A, queue );

    setup_data( B, offdiag_value, diag_value );
    // compute on CPU to check the results
    double scaling = numer / denom;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            Bdata[ i + j*ldb ] *= scaling;
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
        // Display (m, n, lda, offdiag_value, diag_value, numer, denom)
        printf( "\n(%4d, %4d, %4d, %4.2f, %4.2f, %4.2f, %4.2f): error %.2e ",
            m, n, lda, offdiag_value, diag_value, numer, denom, result );
    }

    blas::device_free( dAdata );
    delete[] Adata;
    delete[] Bdata;

    test_assert( result < 3*eps );
}

void test_gescale_dev() {
    // Each tuple contains (mA, nA, lda)
    std::list< std::tuple< int, int, int > > dims_list{
            // Square A matrix
            {   1,   1,   1},
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

    // Each tuple contains (numer, denom)
    std::list< std::tuple< double, double > > scalings_list{
        // Positive scaling only
        {   0, 1.0 },
        { 1.0, 1.0 },
        { 1.0, 2.0 },
        { 2.0, 1.0 },
        { 2.0, 3.0 },
        { 3.0, 2.0 },
        // Negative scaling
        {-1.0, 1.0 },
        { 1.0,-1.0 },
        {-2.0, 3.0 },
        { 3.0,-2.0 },
      };

    for (auto dims : dims_list) {
        int mA  = std::get<0>( dims );
        int nA  = std::get<1>( dims );
        int lda = std::get<2>( dims );
        for (auto values : values_list) {
            double offdiag_value  = std::get<0>( values );
            double diag_value     = std::get<1>( values );
            for (auto scalings : scalings_list) {
                double numer = std::get<0>( scalings );
                double denom = std::get<1>( scalings );
                test_gescale_dev_worker(
                    mA, nA, lda, offdiag_value, diag_value, numer, denom );
            }
        }
    }
}

//------------------------------------------------------------------------------
void test_gescale_batch_dev_worker(int m, int n, int lda,
    double offdiag_value, double diag_value,
    double numer, double denom, int batch_count)
{
    if (num_devices == 0) {
        test_skip("requires num_devices > 0");
    }

    double eps = std::numeric_limits<double>::epsilon();
    int ldb = lda;
    std::vector< slate::Tile< double > > list_A( 0 );
    std::vector< slate::Tile< double > > list_dA( 0 );

    // Create the A matrices on the Host
    for (int m_i = 0; m_i < batch_count; ++m_i) {
        double* tmp_data = new double[ lda * n ];
        test_assert( tmp_data != nullptr );
        list_A.push_back( slate::Tile<double>( m, n, tmp_data, lda,
            slate::HostNum, slate::TileKind::UserOwned ) );
        setup_data( list_A.back(), offdiag_value, diag_value );
    }

    // Create the B matrix on the Host
    double* Bdata = new double[ ldb * n ];
    slate::Tile<double> B( m, n, Bdata, ldb,
        slate::HostNum, slate::TileKind::UserOwned );

    // Create the queue
    int device_idx;
    blas::get_device( &device_idx );
    const int batch_arrays_index = 0;
    blas::Queue queue( device_idx, batch_arrays_index );

    // Create the dA matrices on the device
    for (int m_i = 0; m_i < batch_count; ++m_i) {
        double* dtmp_data;
        dtmp_data = blas::device_malloc<double>( lda * n );
        test_assert( dtmp_data != nullptr );
        list_dA.push_back( slate::Tile<double>( m, n, dtmp_data, lda,
            device_idx, slate::TileKind::UserOwned ) );
    }

    double** Aarray = new double*[ batch_count ];
    double** dAarray;
    dAarray = blas::device_malloc<double*>( batch_count );
    test_assert( dAarray != nullptr );
    for (int m_i = 0; m_i < batch_count; ++m_i) {
        auto A  = list_A[ m_i ];
        auto dA = list_dA[ m_i ];
        // Register the address on the device
        Aarray[m_i] = dA.data();

        // Copy the m_i'th matrix to the device
        blas::device_memcpy<double>( dA.data(), A.data(),
                            lda * n,
                            blas::MemcpyKind::HostToDevice,
                            queue );
    }
    // Transfer the batch_array to the device
    blas::device_memcpy<double*>( dAarray, Aarray,
                        batch_count,
                        blas::MemcpyKind::HostToDevice,
                        queue );

    slate::device::batch::gescale( m, n,
                          numer, denom,
                          dAarray, lda,
                          batch_count, queue );

    queue.sync();

    // Copy the dA matrices back to the Host for check purpose
    for (int m_i = 0; m_i < batch_count; ++m_i) {
        auto dA = list_dA[ m_i ];
        auto A = list_A[ m_i ];
        dA.copyData( &A, queue );
    }

    // Setup B, imitating what happens on the device
    setup_data( B, offdiag_value, diag_value );
    // compute on CPU to check the results
    double scaling = numer / denom;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            Bdata[ i + j*ldb ] *= scaling;
        }
    }

    // Check each A matrix
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
            // Display (m, n, lda, offdiag_value, diag_value, numer, denom)
            if (m_i == 0)
                printf( "\n(%4d, %4d, %4d, %4.2f, %4.2f, %4.2f, %4.2f):",
                    m, n, lda, offdiag_value, diag_value, numer, denom );
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

}

void test_gescale_batch_dev() {
    // Each tuple contains (mA, nA, lda)
    std::list< std::tuple< int, int, int > > dims_list{
            // Square A matrix
            {   1,   1,   1},
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

    // Each tuple contains (numer, denom)
    std::list< std::tuple< double, double > > scalings_list{
        // Positive scaling only
        { 2.0, 3.0 },
        { 3.0, 2.0 },
        // Negative scaling
        {-2.0, 3.0 },
        { 3.0,-2.0 },
      };

    std::list< int > batch_count_list{ 1, 2, 3, 4, 5, 10, 20, 100 };

    for (auto dims : dims_list) {
        int mA  = std::get<0>( dims );
        int nA  = std::get<1>( dims );
        int lda = std::get<2>( dims );
        for (auto values : values_list) {
            double offdiag_value  = std::get<0>( values );
            double diag_value     = std::get<1>( values );
            for (auto scalings : scalings_list) {
                double numer = std::get<0>( scalings );
                double denom = std::get<1>( scalings );
                for (auto batch_count : batch_count_list)
                    test_gescale_batch_dev_worker(
                        mA, nA, lda, offdiag_value, diag_value,
                        numer, denom, batch_count );
            }
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
            test_gescale_dev, "gescale_dev" );
        //-------------------- geset_batch_dev
        run_test(
            test_gescale_batch_dev, "gescale_batch_dev" );
    }
}

}  // namespace test

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    using namespace test;  // for globals mpi_rank, etc.

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &mpi_size );

    num_devices = blas::get_device_count();

    verbose = 0;
    for (int i = 1; i < argc; ++i)
        if (argv[i] == std::string( "-v" ))
            verbose += 1;

    int err = unit_test_main( MPI_COMM_WORLD );  // which calls run_tests()

    MPI_Finalize();
    return err;
}

