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
/// Sets Aij = (mpi_rank + 1)*1000 + i + j/1000, for all i, j.
template <typename scalar_t>
void setup_data(slate::Tile<scalar_t>& A)
{
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
    scalar_t offdiag_value, scalar_t diag_value)
{
    if (std::real( offdiag_value ) == 0.0 && std::imag( offdiag_value ) == 0.0
        && std::real( diag_value ) == 0.0 && std::imag( diag_value ) == 0.0)
    {
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
template <typename scalar_t>
void test_gescale_dev_worker(
    int m, int n, int lda,
    scalar_t offdiag_value, scalar_t diag_value,
    scalar_t numer, scalar_t denom,
    blas::Queue& queue)
{
    using real_t = blas::real_type<scalar_t>;

    if (num_devices == 0) {
        test_skip("requires num_devices > 0");
    }

    real_t eps = std::numeric_limits<real_t>::epsilon();
    int ldb = lda;
    int device_idx = queue.device();

    scalar_t* Adata = new scalar_t[ lda * n ];
    slate::Tile<scalar_t> A( m, n, Adata, lda,
        slate::HostNum, slate::TileKind::UserOwned );

    scalar_t* Bdata = new scalar_t[ ldb * n ];
    slate::Tile<scalar_t> B( m, n, Bdata, ldb,
        slate::HostNum, slate::TileKind::UserOwned );

    scalar_t* dAdata;
    dAdata = blas::device_malloc<scalar_t>( blas::max( lda * n, 1 ) );
    test_assert( dAdata != nullptr );
    slate::Tile<scalar_t> dA( m, n, dAdata, lda,
        device_idx, slate::TileKind::UserOwned );

    setup_data( A, offdiag_value, diag_value );

    blas::device_memcpy<scalar_t>(
                        dA.data(), A.data(),
                        lda * n,
                        blas::MemcpyKind::HostToDevice,
                        queue );

    slate::device::gescale(
                          m, n,
                          numer, denom,
                          dA.data(), lda,
                          queue );

    queue.sync();
    dA.copyData( &A, queue );

    setup_data( B, offdiag_value, diag_value );
    // compute on CPU to check the results
    scalar_t scaling = numer / denom;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            Bdata[ i + j*ldb ] *= scaling;
        }
    }

    //blas::axpy( lda*n, neg_one, B.data(), ione, A.data(), ione );
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            Adata[ i + j*lda ] = Bdata[ i + j*ldb ] -  Adata[ i + j*lda ];
        }
    }

    // check final norm result
    real_t result = lapack::lange(
        lapack::Norm::Fro, A.mb(), A.nb(), A.data(), A.stride() );

    if (verbose) {
        // Display (m, n, lda, offdiag_value, diag_value, numer, denom)
        printf(
            "\n(%4d, %4d, %4d, (%6.2f, %6.2f), (%6.2f, %6.2f),"
            " (%6.2f, %6.2f), (%6.2f, %6.2f) ): error %.2e ",
            m, n, lda,
            std::real( offdiag_value ), std::imag( offdiag_value ),
            std::real( diag_value ), std::imag( diag_value ),
            std::real( numer ), std::imag( numer ),
            std::real( denom ), std::imag( denom ),
            result );
    }

    blas::device_free( dAdata );
    delete[] Adata;
    delete[] Bdata;

    test_assert( result < 3*eps );
}

template <typename scalar_t>
void test_gescale_dev()
{
    // Each tuple contains (mA, nA, lda)
    std::list< std::tuple< int, int, int > > dims_list{
            // Corner cases
            {   0,   0,   0 },
            { 100,   0, 100 },
            {   0, 100,   0 },
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
    std::list<
              std::tuple< std::complex<double>, std::complex<double> >
              > values_list{
            // Special case
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
        };

    // Each tuple contains (numer, denom)
    std::list<
              std::tuple< std::complex<double>, std::complex<double> >
              > scalings_list{
        // Positive scaling only
        // Scale by 0/1
        { {    0,    0 },
          {  1.0,    0 } },
        // Scale by 1/1
        { {  1.0,    0 },
          {  1.0,    0 } },
        // Scale by (1+1i) / (2+2i)
        { {  1.0,  1.0 },
          {  2.0,  2.0 } },
        // Scale by (2+2i) / (1+1i)
        { {  2.0,  2.0 },
          {  1.0,  1.0 } },
        // Scale by (2+2i) / (3+3i)
        { {  2.0,  2.0 },
          {  3.0,  3.0 } },
        // Scale by (3+3i) / (2+2i)
        { {  3.0,  3.0 },
          {  2.0,  2.0 } },
        // Negative scaling
        // Scale by (-1+1i) / (1-1i)
        { { -1.0,  1.0 },
          {  1.0, -1.0 } },
        // Scale by (1-1i) / (-1+1i)
        { {  1.0, -1.0 },
          { -1.0,  1.0 } },
        // Scale by (-2+2i) / (3-3i)
        { { -2.0,  2.0 },
          {  3.0, -3.0 } },
        // Scale by (3+3i) / (-2-2i)
        { {  3.0,  3.0 },
          { -2.0, -2.0 } },
      };

    int device_idx;
    blas::get_device( &device_idx );
    const int batch_arrays_index = 0;
    blas::Queue queue( device_idx, batch_arrays_index );

    for (auto dims : dims_list) {
        int mA  = std::get<0>( dims );
        int nA  = std::get<1>( dims );
        int lda = std::get<2>( dims );
        for (auto values : values_list) {
            std::complex<double> offdiag_value  = std::get<0>( values );
            std::complex<double> diag_value     = std::get<1>( values );
            for (auto scalings : scalings_list) {
                std::complex<double> numer = std::get<0>( scalings );
                std::complex<double> denom = std::get<1>( scalings );
                test_gescale_dev_worker<scalar_t>(
                    mA, nA, lda,
                    testsweeper::make_scalar<scalar_t>( offdiag_value ),
                    testsweeper::make_scalar<scalar_t>( diag_value ),
                    testsweeper::make_scalar<scalar_t>( numer ),
                    testsweeper::make_scalar<scalar_t>( denom ),
                    queue );
            }
        }
    }
}

template <typename... scalar_t>
void run_tests_gescale_device()
{
    ( run_test<scalar_t>(
                          test_gescale_dev<scalar_t>, "gescale_dev" ),
      ... );
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_gescale_batch_dev_worker(
    int m, int n, int lda,
    scalar_t offdiag_value, scalar_t diag_value,
    scalar_t numer, scalar_t denom, int batch_count,
    blas::Queue& queue)
{
    using real_t = blas::real_type<scalar_t>;

    if (num_devices == 0) {
        test_skip("requires num_devices > 0");
    }

    real_t eps = std::numeric_limits<real_t>::epsilon();
    int ldb = lda;
    std::vector< slate::Tile< scalar_t > > list_A( 0 );
    std::vector< slate::Tile< scalar_t > > list_dA( 0 );
    int device_idx = queue.device();

    // Create the A matrices on the Host
    for (int m_i = 0; m_i < batch_count; ++m_i) {
        scalar_t* tmp_data = new scalar_t[ lda * n ];
        test_assert( tmp_data != nullptr );
        list_A.push_back( slate::Tile<scalar_t>( m, n, tmp_data, lda,
            slate::HostNum, slate::TileKind::UserOwned ) );
        setup_data( list_A.back(), offdiag_value, diag_value );
    }

    // Create the B matrix on the Host
    scalar_t* Bdata = new scalar_t[ ldb * n ];
    slate::Tile<scalar_t> B( m, n, Bdata, ldb,
        slate::HostNum, slate::TileKind::UserOwned );

    // Create the dA matrices on the device
    for (int m_i = 0; m_i < batch_count; ++m_i) {
        scalar_t* dtmp_data;
        dtmp_data = blas::device_malloc<scalar_t>( blas::max( lda * n, 1 ) );
        test_assert( dtmp_data != nullptr );
        list_dA.push_back( slate::Tile<scalar_t>( m, n, dtmp_data, lda,
            device_idx, slate::TileKind::UserOwned ) );
    }

    scalar_t** Aarray = new scalar_t*[ batch_count ];
    scalar_t** dAarray;
    dAarray = blas::device_malloc<scalar_t*>( batch_count );
    test_assert( dAarray != nullptr );
    for (int m_i = 0; m_i < batch_count; ++m_i) {
        auto A  = list_A[ m_i ];
        auto dA = list_dA[ m_i ];
        // Register the address on the device
        Aarray[ m_i ] = dA.data();

        // Copy the m_i'th matrix to the device
        blas::device_memcpy<scalar_t>(
                            dA.data(), A.data(),
                            lda * n,
                            blas::MemcpyKind::HostToDevice,
                            queue );
    }
    // Transfer the batch_array to the device
    blas::device_memcpy<scalar_t*>(
                        dAarray, Aarray,
                        batch_count,
                        blas::MemcpyKind::HostToDevice,
                        queue );

    slate::device::batch::gescale(
                          m, n,
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
    scalar_t scaling = numer / denom;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            Bdata[ i + j*ldb ] *= scaling;
        }
    }

    // Check each A matrix
    for (int m_i = 0; m_i < batch_count; ++m_i) {
        auto dA = list_dA[ m_i ];
        auto A  = list_A[ m_i ];
        scalar_t *Adata = A.data();

        //blas::axpy( lda*n, neg_one, B.data(), ione, A.data(), ione );
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                Adata[ i + j*lda ] = Bdata[ i + j*ldb ] -  Adata[ i + j*lda ];
            }
        }

        // check final norm result
        real_t result = lapack::lange(
            lapack::Norm::Fro, A.mb(), A.nb(), A.data(), A.stride() );

        if (verbose) {
            // Display (m, n, lda, offdiag_value, diag_value, numer, denom)
            if (m_i == 0)
                printf(
                    "\n(%4d, %4d, %4d, (%6.2f, %6.2f), (%6.2f, %6.2f),"
                    " (%6.2f, %6.2f), (%6.2f, %6.2f) ):",
                    m, n, lda,
                    std::real( offdiag_value ), std::imag( offdiag_value ),
                    std::real( diag_value ), std::imag( diag_value ),
                    std::real( numer ), std::imag( numer ),
                    std::real( denom ), std::imag( denom ) );
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

template <typename scalar_t>
void test_gescale_batch_dev()
{
    // Each tuple contains (mA, nA, lda)
    std::list< std::tuple< int, int, int > > dims_list{
            // Corner cases
            {   0,   0,   0 },
            { 100,   0, 100 },
            {   0, 100,   0 },
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
    std::list<
              std::tuple< std::complex<double>, std::complex<double> >
              > values_list{
            // Special case
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
        };

    // Each tuple contains (numer, denom)
    std::list<
              std::tuple< std::complex<double>, std::complex<double> >
              > scalings_list{
        // Positive scaling only
        // Scale by (2+2i) / (3+3i)
        { {  2.0,  2.0 },
          {  3.0,  3.0 } },
        // Scale by (3+3i) / (2+2i)
        { {  3.0,  3.0 },
          {  2.0,  2.0 } },
        // Negative scaling
        // Scale by (-2+2i) / (3-3i)
        { { -2.0,  2.0 },
          {  3.0, -3.0 } },
        // Scale by (3+3i) / (-2-2i)
        { {  3.0,  3.0 },
          { -2.0, -2.0 } },
      };

    std::list< int > batch_count_list{ 1, 2, 3, 4, 5, 10, 20, 100 };

    // Create the queue
    int device_idx;
    blas::get_device( &device_idx );
    const int batch_arrays_index = 0;
    blas::Queue queue( device_idx, batch_arrays_index );

    for (auto dims : dims_list) {
        int mA  = std::get<0>( dims );
        int nA  = std::get<1>( dims );
        int lda = std::get<2>( dims );
        for (auto values : values_list) {
            std::complex<double> offdiag_value  = std::get<0>( values );
            std::complex<double> diag_value     = std::get<1>( values );
            for (auto scalings : scalings_list) {
                std::complex<double> numer = std::get<0>( scalings );
                std::complex<double> denom = std::get<1>( scalings );
                for (auto batch_count : batch_count_list)
                    test_gescale_batch_dev_worker<scalar_t>(
                        mA, nA, lda,
                        testsweeper::make_scalar<scalar_t>( offdiag_value ),
                        testsweeper::make_scalar<scalar_t>( diag_value ),
                        testsweeper::make_scalar<scalar_t>( numer ),
                        testsweeper::make_scalar<scalar_t>( denom ),
                        batch_count, queue );
            }
        }
    }
}

template <typename... scalar_t>
void run_tests_gescale_batch_device()
{
    ( run_test<scalar_t>(
                        test_gescale_batch_dev<scalar_t>, "gescale_batch_dev" ),
      ... );
}

//------------------------------------------------------------------------------
/// Runs all tests. Called by unit test main().
void run_tests()
{
    if (mpi_rank == 0) {
        //-------------------- gescale_dev
        run_tests_gescale_device<
            float, double, std::complex<float>, std::complex<double>
            >();

        //-------------------- gescale_batch_dev
        run_tests_gescale_batch_device<
            float, double, std::complex<float>, std::complex<double>
            >();
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

