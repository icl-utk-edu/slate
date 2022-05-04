// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/internal/Memory.hh"

#include "unit_test.hh"
#include "slate/Exception.hh"

using std::max;
using slate::HostNum;

namespace test {

//------------------------------------------------------------------------------
// global variables
int nb;

//------------------------------------------------------------------------------
/// Tests Memory(size) constructor. Doesn't allocate memory.
void test_Memory()
{
    slate::Memory mem(sizeof(double) * nb * nb);

    test_assert( int( mem.available( HostNum ) ) == 0 );
    test_assert( int( mem.capacity(  HostNum ) ) == 0 );

    for (int dev = 0; dev < mem.num_devices_; ++dev) {
        test_assert(int(mem.available(dev)) == 0);
        test_assert(int(mem.capacity (dev)) == 0);
    }
}

//------------------------------------------------------------------------------
/// Tests reserving host blocks.
void test_addHostBlocks()
{
    slate::Memory mem(sizeof(double) * nb * nb);

    const int cnt = 5;
    mem.addHostBlocks(cnt);
    // Memory class no longer reserves CPU blocks, it allocates on-the-fly.
    //test_assert( int( mem.available( HostNum ) ) == cnt );
    //test_assert( int( mem.capacity(  HostNum ) ) == cnt );
    test_assert( int( mem.available( HostNum ) ) == 0 );
    test_assert( int( mem.capacity(  HostNum ) ) == 0 );

    // Devices still 0.
    for (int dev = 0; dev < mem.num_devices_; ++dev) {
        test_assert(int(mem.available(dev)) == 0);
        test_assert(int(mem.capacity (dev)) == 0);
    }
}

//------------------------------------------------------------------------------
/// Tests reserving device blocks.
void test_addDeviceBlocks()
{
    slate::Memory mem(sizeof(double) * nb * nb);
    if (mem.num_devices_ == 0) {
        test_skip("no GPU devices available");
    }

    const int cnt = 5;
    for (int dev = 0; dev < mem.num_devices_; ++dev) {
        mem.addDeviceBlocks(dev, cnt);
        test_assert(int(mem.available(dev)) == cnt);
        test_assert(int(mem.capacity (dev)) == cnt);

        // Remaining devices still 0.
        for (int dev2 = dev+1; dev2 < mem.num_devices_; ++dev2) {
            test_assert(int(mem.available(dev2)) == 0);
            test_assert(int(mem.capacity (dev2)) == 0);
        }
    }

    test_assert( int( mem.available( HostNum ) ) == 0 );
    test_assert( int( mem.capacity(  HostNum ) ) == 0 );
}

//------------------------------------------------------------------------------
/// Tests allocating and freeing host blocks.
void test_alloc_host()
{
    slate::Memory mem(sizeof(double) * nb * nb);

    const int cnt = 5;
    mem.addHostBlocks(cnt);

    // Allocate 2*cnt blocks.
    // First cnt blocks come from reserve, next cnt blocks malloc'd 1-by-1.
    double* hx[ 2*cnt ];
    for (int i = 0; i < 2*cnt; ++i) {
        hx[i] = (double*) mem.alloc( HostNum, sizeof(double) * nb * nb );
        test_assert(hx[i] != nullptr);
        // Memory class no longer reserves CPU blocks, it allocates on-the-fly.
        //test_assert( int( mem.available( HostNum ) ) == max( cnt-(i+1), 0 ) );
        //test_assert( int( mem.capacity(  HostNum ) ) == max( cnt, i+1 ) );
        test_assert( int( mem.available( HostNum ) ) == 0 );
        test_assert( int( mem.capacity(  HostNum ) ) == 0 );

        // Touch memory to verify it is valid.
        for (int j = 0; j < nb*nb; ++j) {
            hx[i][j] = i*1000000 + j;
        }
    }

    // Free some.
    int some = cnt/2;
    for (int i = 0; i < some; ++i) {
        mem.free( hx[i], HostNum );
        hx[i] = nullptr;
        //test_assert( int( mem.available( HostNum ) ) == i+1 );
        //test_assert( int( mem.capacity(  HostNum ) ) == 2*cnt );
        test_assert( int( mem.available( HostNum ) ) == 0 );
        test_assert( int( mem.capacity(  HostNum ) ) == 0 );
    }

    // Re-alloc some.
    for (int i = 0; i < some; ++i) {
        hx[i] = (double*) mem.alloc( HostNum, sizeof(double) * nb * nb );
        test_assert(hx[i] != nullptr);
        //test_assert( int( mem.available( HostNum ) ) == some - ( i+1 ) );
        //test_assert( int( mem.capacity(  HostNum ) ) == 2*cnt );
        test_assert( int( mem.available( HostNum ) ) == 0 );
        test_assert( int( mem.capacity(  HostNum ) ) == 0 );
    }
}

//------------------------------------------------------------------------------
/// Tests allocating and freeing device blocks.
void test_alloc_device()
{
    slate::Memory mem(sizeof(double) * nb * nb);
    if (mem.num_devices_ == 0) {
        test_skip("no GPU devices available");
    }

    double* hx = new double[ nb*nb ];
    const int batch_arrays_index = 0;

    const int cnt = 5;
    for (int dev = 0; dev < mem.num_devices_; ++dev) {
        mem.addDeviceBlocks(dev, cnt);

        // Allocate 2*cnt blocks.
        // First cnt blocks come from reserve, next cnt blocks malloc'd 1-by-1.
        double* dx[ 2*cnt ];
        for (int i = 0; i < 2*cnt; ++i) {
            dx[i] = (double*) mem.alloc(dev, sizeof(double) * nb * nb);
            test_assert(dx[i] != nullptr);
            test_assert(int(mem.available(dev)) == max(cnt - (i+1), 0));
            test_assert(int(mem.capacity (dev)) == max(cnt, i+1));

            // Touch memory to verify it is valid.
            blas::Queue queue(dev, batch_arrays_index);
            blas::set_device(dev);
            blas::device_memcpy<double>(dx[i], hx, nb * nb,
                                        blas::MemcpyKind::HostToDevice,
                                        queue);
        }

        // Free some.
        int some = cnt/2;
        for (int i = 0; i < some; ++i) {
            mem.free(dx[i], dev);
            dx[i] = nullptr;
            test_assert(int(mem.available(dev)) == i+1);
            test_assert(int(mem.capacity (dev)) == 2*cnt);
        }

        // Re-alloc some.
        for (int i = 0; i < some; ++i) {
            dx[i] = (double*) mem.alloc(dev, sizeof(double) * nb * nb);
            test_assert(dx[i] != nullptr);
            test_assert(int(mem.available(dev)) == some - (i+1));
            test_assert(int(mem.capacity (dev)) == 2*cnt);
        }
    }

    delete[] hx;
}

//------------------------------------------------------------------------------
/// Tests clearing host blocks.
void test_clearHostBlocks()
{
    slate::Memory mem(sizeof(double) * nb * nb);

    const int cnt = 5;
    mem.addHostBlocks(cnt);
    //test_assert( int( mem.available( HostNum ) ) == cnt );
    //test_assert( int( mem.capacity(  HostNum ) ) == cnt );
    test_assert( int( mem.available( HostNum ) ) == 0 );
    test_assert( int( mem.capacity(  HostNum ) ) == 0 );

    // Allocate 2*cnt blocks.
    for (int i = 0; i < 2*cnt; ++i) {
        mem.alloc( HostNum, sizeof(double) * nb * nb );
    }

    test_assert( int( mem.available( HostNum ) ) == 0 );
    //test_assert( int( mem.capacity(  HostNum ) ) == 2*cnt );
    test_assert( int( mem.capacity(  HostNum ) ) == 0 );

    mem.clearHostBlocks();

    test_assert( int( mem.available( HostNum ) ) == 0 );
    test_assert( int( mem.capacity(  HostNum ) ) == 0 );
}

//------------------------------------------------------------------------------
/// Tests clearing device blocks.
void test_clearDeviceBlocks()
{
    slate::Memory mem(sizeof(double) * nb * nb);
    if (mem.num_devices_ == 0) {
        test_skip("no GPU devices available");
    }

    const int cnt = 5;
    for (int dev = 0; dev < mem.num_devices_; ++dev) {
        mem.addDeviceBlocks(dev, cnt);

        // Allocate 2*cnt blocks.
        for (int i = 0; i < 2*cnt; ++i) {
            mem.alloc(dev, sizeof(double) * nb * nb);
        }
    }

    for (int dev = 0; dev < mem.num_devices_; ++dev) {
        test_assert(int(mem.available(dev)) == 0);
        test_assert(int(mem.capacity (dev)) == 2*cnt);

        mem.clearDeviceBlocks(dev);

        test_assert(int(mem.available(dev)) == 0);
        test_assert(int(mem.capacity (dev)) == 0);
    }
}

//------------------------------------------------------------------------------
/// Runs all tests. Called by unit test main().
void run_tests()
{
    run_test(test_Memory,            "Memory()");
    run_test(test_addHostBlocks,     "addHostBlocks");
    run_test(test_addDeviceBlocks,   "addDeviceBlocks");
    run_test(test_alloc_host,        "alloc and free (host)");
    run_test(test_alloc_device,      "alloc and free (device)");
    run_test(test_clearHostBlocks,   "clearHostBlocks");
    run_test(test_clearDeviceBlocks, "clearDeviceBlocks");
}

}  // namespace test

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    using namespace test;  // for globals mpi_rank, etc.

    // global nb
    nb = 16;
    if (argc > 1) {
        nb = atoi(argv[1]);
    }
    printf("nb = %d\n", nb);
    return unit_test_main();  // which calls run_tests()
}
