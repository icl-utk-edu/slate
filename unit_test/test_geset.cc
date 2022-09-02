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

// The codes below are used to get a correct print of the types
// XXX Where should we put it, if we keep it.
template <typename scalar_t>
struct TypeName
{
    static const char* name()
    {
        return typeid(scalar_t).name();
    }
};

template <>
struct TypeName<float>{
    static const char* name()
    {
        return "float";
    }
};

template <>
struct TypeName<double>{
    static const char* name()
    {
        return "double";
    }
};

template <>
struct TypeName<std::complex<float>>{
    static const char* name()
    {
        return "std::complex<float>";
    }
};

template <>
struct TypeName<std::complex<double>>{
    static const char* name()
    {
        return "std::complex<double>";
    }
};


// The codes below come from testsweeper.hh
// XXX Should we include the header instead?

/// For real scalar types.
template <typename real_t>
struct MakeScalarTraits {
    static real_t make( real_t re, real_t im )
        { return re; }
};

/// For complex scalar types.
template <typename real_t>
struct MakeScalarTraits< std::complex<real_t> > {
    static std::complex<real_t> make( real_t re, real_t im )
        { return std::complex<real_t>( re, im ); }
};

/// Converts complex value into scalar_t,
/// discarding imaginary part if scalar_t is real.
template <typename scalar_t>
scalar_t make_scalar( std::complex<double> val )
{
    return MakeScalarTraits<scalar_t>::make( std::real(val), std::imag(val) );
}

//------------------------------------------------------------------------------
// global variables
int mpi_rank;
int mpi_size;
int verbose;
int num_devices;

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_geset_dev_worker(
    int m, int n, int lda,
    scalar_t offdiag_value, scalar_t diag_value)
{
    if (num_devices == 0) {
        test_skip("requires num_devices > 0");
    }

    double eps = std::numeric_limits<double>::epsilon();
    int ldb = lda;

    scalar_t* Adata = new scalar_t[ lda * n ];
    slate::Tile<scalar_t> A( m, n, Adata, lda,
        slate::HostNum, slate::TileKind::UserOwned );

    scalar_t* Bdata = new scalar_t[ ldb * n ];
    slate::Tile<scalar_t> B( m, n, Bdata, ldb,
        slate::HostNum, slate::TileKind::UserOwned );

    int device_idx;
    blas::get_device( &device_idx );
    const int batch_arrays_index = 0;
    blas::Queue queue( device_idx, batch_arrays_index );

    scalar_t* dAdata;
    dAdata = blas::device_malloc<scalar_t>( blas::max( lda * n, 1 ) );
    test_assert( dAdata != nullptr );
    slate::Tile<scalar_t> dA( m, n, dAdata, lda,
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
        printf(
            "\n(%4d, %4d, %4d, (%4.2f, %4.2f), (%4.2f, %4.2f) ): error %.2f",
            m, n, lda,
            std::real( offdiag_value ), std::imag( offdiag_value ),
            std::real( diag_value ), std::imag( diag_value ),
            result );
    }

    blas::device_free( dAdata );
    delete[] Adata;
    delete[] Bdata;

    test_assert( result < 3*eps );
}

template <typename scalar_t>
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
        };

    printf( "\n\t%s, ", TypeName<scalar_t>::name() );
    for (auto dims : dims_list) {
        int mA  = std::get<0>( dims );
        int nA  = std::get<1>( dims );
        int lda = std::get<2>( dims );
        for (auto values : values_list) {
            std::complex<double> offdiag_value  = std::get<0>( values );
            std::complex<double> diag_value     = std::get<1>( values );
            test_geset_dev_worker<scalar_t>(
                mA, nA, lda,
                make_scalar<scalar_t>( offdiag_value ),
                make_scalar<scalar_t>( diag_value ) );
        }
    }
}

void test_geset_device()
{
    // TODO have a "pass/fail" message for each type, instead of one overall.
    test_geset_dev<float>();
    test_geset_dev<double>();
    // Complex cases
    test_geset_dev<std::complex<float>>();
    test_geset_dev<std::complex<double>>();
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_geset_batch_dev_worker(
    int m, int n, int lda,
    scalar_t offdiag_value, scalar_t diag_value,
    int batch_count)
{
    if (num_devices == 0) {
        test_skip("requires num_devices > 0");
    }

    double eps = std::numeric_limits<double>::epsilon();
    int ldb = lda;
    std::vector< slate::Tile< scalar_t > > list_A( 0 );
    std::vector< slate::Tile< scalar_t > > list_dA( 0 );

    for (int m_i = 0; m_i < batch_count; ++m_i) {
        scalar_t* tmp_data = new scalar_t[ lda * n ];
        test_assert( tmp_data != nullptr );
        list_A.push_back( slate::Tile<scalar_t>( m, n, tmp_data, lda,
            slate::HostNum, slate::TileKind::UserOwned ) );
    }

    scalar_t* Bdata = new scalar_t[ ldb * n ];
    slate::Tile<scalar_t> B( m, n, Bdata, ldb,
        slate::HostNum, slate::TileKind::UserOwned );

    int device_idx;
    blas::get_device( &device_idx );
    const int batch_arrays_index = 0;
    blas::Queue queue( device_idx, batch_arrays_index );

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
        auto dA = list_dA[ m_i ];
        Aarray[m_i] = dA.data();
    }
    blas::device_memcpy<scalar_t*>( dAarray, Aarray,
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
        scalar_t *Adata = A.data();

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
                printf( "\n(%4d, %4d, %4d, (%4.2f, %4.2f), (%4.2f, %4.2f) ):",
                    m, n, lda,
                    std::real( offdiag_value ), std::imag( offdiag_value ),
                    std::real( diag_value ), std::imag( diag_value ) );
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

template <typename scalar_t>
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
        };

    std::list< int > batch_count_list{ 1, 2, 3, 4, 5, 10, 20, 100 };

    printf( "\n\t%s, ", TypeName<scalar_t>::name() );
    for (auto dims : dims_list) {
        int mA  = std::get<0>( dims );
        int nA  = std::get<1>( dims );
        int lda = std::get<2>( dims );
        for (auto values : values_list) {
            std::complex<double> offdiag_value  = std::get<0>( values );
            std::complex<double> diag_value     = std::get<1>( values );
            for (auto batch_count : batch_count_list)
                test_geset_batch_dev_worker<scalar_t>(
                    mA, nA, lda,
                    make_scalar<scalar_t>( offdiag_value ),
                    make_scalar<scalar_t>( diag_value ),
                    batch_count );
        }
    }
}

void test_geset_batch_device()
{
    // TODO have a "pass/fail" message for each type, instead of one overall.
    test_geset_batch_dev<float>();
    test_geset_batch_dev<double>();
    // Complex cases
    test_geset_batch_dev<std::complex<float>>();
    test_geset_batch_dev<std::complex<double>>();
}

//------------------------------------------------------------------------------
/// Runs all tests. Called by unit test main().
void run_tests()
{
    if (mpi_rank == 0) {
        // XXX Do we want to have ':' at the end for the output or move it
        // somewhere else?
        //-------------------- geset_dev
        run_test(
            test_geset_device, "geset_device:" );
        //-------------------- geset_batch_dev
        run_test(
            test_geset_batch_device, "geset_batch_device:" );
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
