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
#include "../testsweeper/testsweeper.hh"

using slate::roundup;

namespace test {

//------------------------------------------------------------------------------
// global variables
int mpi_rank;
int mpi_size;
int verbose;
int num_devices;

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_geadd_dev_worker(
    int m, int n, int lda,
    scalar_t alpha, scalar_t beta)
{
    using real_t = blas::real_type<scalar_t>;

    if (num_devices == 0) {
        test_skip("requires num_devices > 0");
    }

    real_t eps = std::numeric_limits<real_t>::epsilon();
    int ldb = lda;
    int64_t idist = 2;
    int64_t iseed[4] = { 0, 1, 2, 3 };

    // Create A on the Host and setup with random data
    scalar_t* Adata = new scalar_t[ lda * n ];
    slate::Tile<scalar_t> A( m, n, Adata, lda,
        slate::HostNum, slate::TileKind::UserOwned );
    lapack::larnv( idist, iseed, lda * n, A.data() );

    // Create B on the Host and setup with random data
    scalar_t* Bdata = new scalar_t[ ldb * n ];
    slate::Tile<scalar_t> B( m, n, Bdata, ldb,
        slate::HostNum, slate::TileKind::UserOwned );
    lapack::larnv( idist, iseed, ldb * n, B.data() );

    // Create B0, a copy of B on the Host
    scalar_t* B0data = new scalar_t[ ldb * n ];
    slate::Tile<scalar_t> B0( m, n, B0data, ldb,
        slate::HostNum, slate::TileKind::UserOwned );
    slate::tile::gecopy( B, B0 );

    // Create C0 on the Host, used later to check the result
    scalar_t* C0data = new scalar_t[ ldb * n ];
    slate::Tile<scalar_t> C0( m, n, C0data, ldb,
        slate::HostNum, slate::TileKind::UserOwned );

    // Create the queue
    int device_idx;
    blas::get_device( &device_idx );
    const int batch_arrays_index = 0;
    blas::Queue queue( device_idx, batch_arrays_index );

    // Create A on device and copy data
    scalar_t* dAdata;
    dAdata = blas::device_malloc<scalar_t>( blas::max( lda * n, 1 ), queue );
    test_assert(dAdata != nullptr);
    slate::Tile<scalar_t> dA( m, n, dAdata, lda,
        device_idx, slate::TileKind::UserOwned );
    A.copyData( &dA, queue );

    // Create B on device and copy data
    scalar_t* dBdata;
    dBdata = blas::device_malloc<scalar_t>( blas::max( ldb * n, 1 ), queue );
    test_assert( dBdata != nullptr );
    slate::Tile<scalar_t> dB( m, n, dBdata, ldb,
        device_idx, slate::TileKind::UserOwned );
    B.copyData( &dB, queue );

    // Add: B = \alpha * A + \beta * B
    slate::device::geadd( m, n,
                          alpha, dA.data(), lda,
                          beta,  dB.data(), ldb,
                          queue );

    queue.sync();

    // Copy the result back to the Host
    dB.copyData( &B, queue );

    // compute on CPU to check the results
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            C0data[ i + j*ldb ] = alpha * Adata[ i + j*lda ]
                                  + beta * B0data[ i + j*ldb ];
        }
    }

    // Get max values in order to check the error
    real_t max_A = lapack::lange(
        lapack::Norm::Max, A.mb(), A.nb(), A.data(), A.stride() );
    real_t max_B = lapack::lange(
        lapack::Norm::Max, B0.mb(), B0.nb(), B0.data(), B0.stride() );
    // Create a vector from \alpha and \beta
    std::vector<scalar_t> cst(4);
    cst[ 0 ] = cst[ 2 ] = alpha;
    cst[ 1 ] = cst[ 3 ] = beta;
    real_t max_csts = lapack::lange( lapack::Norm::Max, 4, 1, cst.data(), 1 );

    //blas::axpy( B.size(), neg_one, B.data(), ione, C0.data(), ione );
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            Adata[ i + j*lda ] = Bdata[ i + j*ldb ] -  C0data[ i + j*ldb ];
        }
    }

    // check final norm result
    real_t result = lapack::lange(
        lapack::Norm::Max, A.mb(), A.nb(), A.data(), A.stride() );

    if (max_csts != scalar_t(0.0) && result != scalar_t(0.0))
        result /= std::max( max_A, max_B ) * max_csts;

    if (verbose) {
        printf(
            "\n(%4d, %4d, %4d, (%4.2f, %4.2f), (%4.2f, %4.2f) ): error %.2e",
            m, n, lda,
            std::real( alpha ), std::imag( alpha ),
            std::real( beta ), std::imag( beta ),
            result );

        if (verbose > 2 && result < 3*eps) {
            printf( "\n" );
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < m; ++i) {
                    printf(
                        "[%d, %d]:"
                        " (%.8e + i%.8e) - (%.8e + i%.8e) = (%.8e + i%.8e)\n",
                        i, j,
                        std::real( Bdata[ i + j*ldb ] ),
                        std::imag( Bdata[ i + j*ldb ] ),
                        std::real( C0data[ i + j*ldb ] ),
                        std::imag( C0data[ i + j*ldb ] ),
                        std::real( Adata[ i + j*lda ] ),
                        std::imag( Adata[ i + j*lda ] ) );
                }
            }
        }
    }

    blas::device_free( dAdata, queue );
    blas::device_free( dBdata, queue );
    delete[] Adata;
    delete[] Bdata;
    delete[] B0data;
    delete[] C0data;

    test_assert( result < 3*eps );
}

template <typename scalar_t>
void test_geadd_dev()
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

    // Each tuple contains (alpha, beta)
    std::list<
              std::tuple< std::complex<double>, std::complex<double> >
             > values_list{
            // All 0
            { {   0,   0 },
              {   0,   0 } },
            // 1 and 1
            { {   1,   0 },
              {   1,   0 } },
            // Offdiag 0, diag != 0
            { {   0,   0 },
              { 0.5, 0.5 } },
            // Offdiag != 0, diag 0
            { { 0.3, 0.3 },
              {   0,   0 } },
            // Real != 0, Imag 0
            { {   0, 0.3 },
              {   0, 0.5 } },
            // Real = 0, Imag != 0
            { { 0.3,   0 },
              { 0.5,   0 } },
            // Standard case
            { { 3.141592653589793, 1.414213562373095 },
              { 2.718281828459045, 1.732050807568877 } },
            // Some negative values
            { { -3.141592653589793, 1.414213562373095 },
              { 2.718281828459045, -1.732050807568877 } },
            // Huge values
            { { 3.141592653589793e8, 1.414213562373095 },
              { 2.718281828459045e7, 1.732050807568877 } },
        };

    for (auto dims : dims_list) {
        int mA  = std::get<0>( dims );
        int nA  = std::get<1>( dims );
        int lda = std::get<2>( dims );
        for (auto values : values_list) {
            std::complex<double> alpha = std::get<0>( values );
            std::complex<double> beta  = std::get<1>( values );
            test_geadd_dev_worker<scalar_t>(
                mA, nA, lda,
                testsweeper::make_scalar<scalar_t>( alpha ),
                testsweeper::make_scalar<scalar_t>( beta ) );
        }
    }
}

template <typename... scalar_t>
void run_tests_geadd_device()
{
    ( run_test<scalar_t>(
                          test_geadd_dev<scalar_t>, "geadd_dev" ),
      ... );
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_geadd_batch_dev_worker(
    int m, int n, int lda,
    scalar_t alpha, scalar_t beta,
    int batch_count)
{
    using real_t = blas::real_type<scalar_t>;

    if (num_devices == 0) {
        test_skip("requires num_devices > 0");
    }

    real_t eps = std::numeric_limits<real_t>::epsilon();
    int ldb = lda;
    std::vector< slate::Tile< scalar_t > > list_A(  0 );
    std::vector< slate::Tile< scalar_t > > list_dA( 0 );
    std::vector< slate::Tile< scalar_t > > list_B(  0 );
    std::vector< slate::Tile< scalar_t > > list_dB( 0 );
    std::vector< slate::Tile< scalar_t > > list_B0( 0 );
    std::vector< slate::Tile< scalar_t > > list_C0( 0 );
    int64_t idist = 2;
    int64_t iseed[4] = { 0, 1, 2, 3 };

    // Create a list of A on the Host and setup with random data
    for (int m_i = 0; m_i < batch_count; ++m_i) {
        scalar_t* tmp_data = new scalar_t[ lda * n ];
        test_assert( tmp_data != nullptr );
        list_A.push_back(
            slate::Tile<scalar_t>( m, n, tmp_data, lda,
                slate::HostNum, slate::TileKind::UserOwned ) );
        lapack::larnv( idist, iseed, lda * n, tmp_data );
    }

    // Create a list of B on the Host and setup with random data
    for (int m_i = 0; m_i < batch_count; ++m_i) {
        scalar_t* tmp_data = new scalar_t[ ldb * n ];
        test_assert( tmp_data != nullptr );
        list_B.push_back(
            slate::Tile<scalar_t>( m, n, tmp_data, ldb,
                slate::HostNum, slate::TileKind::UserOwned ) );
        lapack::larnv( idist, iseed, ldb * n, tmp_data );
    }

    // Create B0, a copy of B on the Host
    for (int m_i = 0; m_i < batch_count; ++m_i) {
        scalar_t* tmp_data = new scalar_t[ ldb * n ];
        test_assert( tmp_data != nullptr );
        list_B0.push_back(
            slate::Tile<scalar_t>( m, n, tmp_data, ldb,
                slate::HostNum, slate::TileKind::UserOwned ) );
        // Create tmp tiles in order to copy the content
        auto T  = list_B[ m_i ];
        auto T0 = list_B0[ m_i ];
        slate::tile::gecopy( T, T0 );
    }

    // Create C0 on the Host, used later to check the result
    for (int m_i = 0; m_i < batch_count; ++m_i) {
        scalar_t* tmp_data = new scalar_t[ ldb * n ];
        test_assert( tmp_data != nullptr );
        list_C0.push_back(
            slate::Tile<scalar_t>( m, n, tmp_data, ldb,
                slate::HostNum, slate::TileKind::UserOwned ) );
    }

    // Create the queue
    int device_idx;
    blas::get_device( &device_idx );
    const int batch_arrays_index = 0;
    blas::Queue queue( device_idx, batch_arrays_index );

    // Create list of A on device and copy data
    for (int m_i = 0; m_i < batch_count; ++m_i) {
        scalar_t* dtmp_data;
        dtmp_data = blas::device_malloc<scalar_t>( blas::max( lda * n, 1 ) );
        test_assert( dtmp_data != nullptr );
        list_dA.push_back(
            slate::Tile<scalar_t>( m, n, dtmp_data, lda,
                device_idx, slate::TileKind::UserOwned ) );
        list_A[ m_i ].copyData( &list_dA[ m_i ], queue );
    }

    // Create B on device and copy data
    for (int m_i = 0; m_i < batch_count; ++m_i) {
        scalar_t* dtmp_data;
        dtmp_data = blas::device_malloc<scalar_t>( blas::max( ldb * n, 1 ) );
        test_assert( dtmp_data != nullptr );
        list_dB.push_back(
            slate::Tile<scalar_t>( m, n, dtmp_data, ldb,
                device_idx, slate::TileKind::UserOwned ) );
        list_B[ m_i ].copyData( &list_dB[ m_i ], queue );
    }

    // Create batch array of dA
    scalar_t** Aarray = new scalar_t*[ batch_count ];
    scalar_t** dAarray;
    dAarray = blas::device_malloc<scalar_t*>( batch_count );
    test_assert( dAarray != nullptr );
    for (int m_i = 0; m_i < batch_count; ++m_i) {
        auto dA = list_dA[ m_i ];
        Aarray[ m_i ] = dA.data();
    }
    blas::device_memcpy<scalar_t*>( dAarray, Aarray,
                        batch_count,
                        blas::MemcpyKind::HostToDevice,
                        queue );

    // Create batch array of dB
    scalar_t** Barray = new scalar_t*[ batch_count ];
    scalar_t** dBarray;
    dBarray = blas::device_malloc<scalar_t*>( batch_count );
    test_assert( dAarray != nullptr );
    for (int m_i = 0; m_i < batch_count; ++m_i) {
        auto dB = list_dB[ m_i ];
        Barray[ m_i ] = dB.data();
    }
    blas::device_memcpy<scalar_t*>( dBarray, Barray,
                        batch_count,
                        blas::MemcpyKind::HostToDevice,
                        queue );

    // Add: B[k] = \alpha * A[k] + \beta * B[k]
    slate::device::batch::geadd( m, n,
                          alpha, dAarray, lda,
                          beta,  dBarray, ldb,
                          batch_count, queue );

    queue.sync();

    // Copy the result back to the Host
    for (int m_i = 0; m_i < batch_count; ++m_i) {
        auto dB = list_dB[ m_i ];
        auto B = list_B[ m_i ];
        dB.copyData( &B, queue );
    }

    // compute on CPU to check the results
    for (int m_i = 0; m_i < batch_count; ++m_i) {
        auto Adata  = list_A[  m_i ].data();
        auto B0data = list_B0[ m_i ].data();
        auto C0data = list_C0[ m_i ].data();
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                C0data[ i + j*ldb ] = alpha * Adata[ i + j*lda ]
                                      + beta * B0data[ i + j*ldb ];
            }
        }
    }

    // Create a vector from \alpha and \beta
    std::vector<scalar_t> cst(4);
    cst[ 0 ] = cst[ 2 ] = alpha;
    cst[ 1 ] = cst[ 3 ] = beta;
    real_t max_csts = lapack::lange( lapack::Norm::Max, 4, 1, cst.data(), 1 );

    for (int m_i = 0; m_i < batch_count; ++m_i) {
        auto dA = list_dA[ m_i ];
        auto A  = list_A[  m_i ];
        auto B0 = list_B0[ m_i ];
        scalar_t *Adata   = A.data(); // Overwritten
        scalar_t *Bdata   = list_B[  m_i ].data(); // Computed on device_idx
        scalar_t *C0data  = list_C0[ m_i ].data(); // Computed on the Host

        // Get max values in order to check the error
        real_t max_A = lapack::lange(
            lapack::Norm::Max, A.mb(), A.nb(), A.data(), A.stride() );
        real_t max_B = lapack::lange(
            lapack::Norm::Max, B0.mb(), B0.nb(), B0.data(), B0.stride() );

        //blas::axpy( B.size(), neg_one, B.data(), ione, C0.data(), ione );
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                Adata[i + j*lda] = Bdata[i + j*ldb] -  C0data[i + j*ldb];
            }
        }

        // check final norm result
        real_t result = lapack::lange(
            lapack::Norm::Max, A.mb(), A.nb(), A.data(), A.stride() );

        if (max_csts != scalar_t(0.0) && result != scalar_t(0.0))
            result /= std::max( max_A, max_B ) * max_csts;

        if (verbose) {
            // Display (m, n, lda, alpha, beta)
            if (m_i == 0)
                printf( "\n(%4d, %4d, %4d, (%4.2f, %4.2f), (%4.2f, %4.2f) ):",
                    m, n, lda,
                    std::real( alpha ), std::imag( alpha ),
                    std::real( beta ), std::imag( beta ) );
            if (verbose > 1)
                printf( "\n\t[%d] error %.2e ", m_i, result );

            if (verbose > 2 && result < 3*eps) {
                printf( "\n" );
                for (int j = 0; j < n; ++j) {
                    for (int i = 0; i < m; ++i) {
                        printf(
                            "[%d, %d]:"
                            " (%.8e + i%.8e) - (%.8e + i%.8e) = (%.8e + i%.8e)\n",
                            i, j,
                            std::real( Bdata[ i + j*ldb ] ),
                            std::imag( Bdata[ i + j*ldb ] ),
                            std::real( C0data[ i + j*ldb ] ),
                            std::imag( C0data[ i + j*ldb ] ),
                            std::real( Adata[ i + j*lda ] ),
                            std::imag( Adata[ i + j*lda ] ) );
                    }
                }
            }
        }

        blas::device_free( dA.data() );
        delete[] Adata;
        delete[] Bdata;
        delete[] list_B0[ m_i ].data();
        delete[] C0data;

        test_assert( result < 3*eps );
    }

    blas::device_free(dAarray);
    blas::device_free(dBarray);
    delete[] Aarray;
    delete[] Barray;
}

template <typename scalar_t>
void test_geadd_batch_dev()
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

    // Each tuple contains (alpha, beta)
    std::list<
              std::tuple< std::complex<double>, std::complex<double> >
             > values_list{
            // All 0
            { {   0,   0 },
              {   0,   0 } },
            // Offdiag 0, diag != 0
            { {   0,   0 },
              { 0.5, 0.5 } },
            // Offdiag != 0, diag 0
            { { 0.3, 0.3 },
              {   0,   0 } },
            // Real != 0, Imag 0
            { {   0, 0.3 },
              {   0, 0.5 } },
            // Real = 0, Imag != 0
            { { 0.3,   0 },
              { 0.5,   0 } },
            // Standard case
            { { 3.141592653589793, 1.414213562373095 },
              { 2.718281828459045, 1.732050807568877 } },
            // Some negative values
            { { -3.141592653589793, 1.414213562373095 },
              { 2.718281828459045, -1.732050807568877 } },
            // Huge values
            { { 3.141592653589793e8, 1.414213562373095 },
              { 2.718281828459045e7, 1.732050807568877 } },
        };

    std::list< int > batch_count_list{ 1, 2, 3, 4, 5, 10, 20, 100 };

    for (auto dims : dims_list) {
        int mA  = std::get<0>( dims );
        int nA  = std::get<1>( dims );
        int lda = std::get<2>( dims );
        for (auto values : values_list) {
            std::complex<double> alpha = std::get<0>( values );
            std::complex<double> beta  = std::get<1>( values );
            for (auto batch_count : batch_count_list)
                test_geadd_batch_dev_worker<scalar_t>(
                    mA, nA, lda,
                    testsweeper::make_scalar<scalar_t>( alpha ),
                    testsweeper::make_scalar<scalar_t>( beta ),
                    batch_count );
        }
    }
}

template <typename... scalar_t>
void run_tests_geadd_batch_device()
{
    ( run_test<scalar_t>(
                          test_geadd_batch_dev<scalar_t>, "geadd_batch_dev" ),
      ... );
}

//------------------------------------------------------------------------------
/// Runs all tests. Called by unit test main().
void run_tests()
{
    if (mpi_rank == 0) {
        //-------------------- geadd_dev
        run_tests_geadd_device<
            float, double, std::complex<float>, std::complex<double>
            >();
        //-------------------- geadd_batch_dev
        run_tests_geadd_batch_device<
            float, double, std::complex<float>, std::complex<double>
            >();
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
