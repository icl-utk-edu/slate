// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"

#include "unit_test.hh"

namespace test {

//------------------------------------------------------------------------------
// global variables
int mpi_rank;
int mpi_size;
int verbose;
int num_devices;

//------------------------------------------------------------------------------
void test_gpu_aware_mpi()
{
    bool value = slate::gpu_aware_mpi();
    if (verbose)
        printf( "\ngpu_aware_mpi = %d\n", value );

    slate::gpu_aware_mpi( true );
    test_assert( slate::gpu_aware_mpi() );

    slate::gpu_aware_mpi( false );
    test_assert( ! slate::gpu_aware_mpi() );
}

//------------------------------------------------------------------------------
/// Runs all tests. Called by unit test main().
void run_tests()
{
    if (mpi_rank == 0) {
        run_test(
            test_gpu_aware_mpi, "gpu_aware_mpi()");
    }
}

}  // namespace test

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
    using namespace test;  // for globals mpi_rank, etc.

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &mpi_size );

    num_devices = blas::get_device_count();

    verbose = 0;
    for (int i = 1; i < argc; ++i)
        if (argv[ i ] == std::string( "-v" ))
            verbose += 1;

    int err = unit_test_main( MPI_COMM_WORLD );  // which calls run_tests()

    MPI_Finalize();
    return err;
}
