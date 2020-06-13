//------------------------------------------------------------------------------
// Copyright (c) 2017, University of Tennessee
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the University of Tennessee nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF TENNESSEE BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//------------------------------------------------------------------------------
// This research was supported by the Exascale Computing Project (17-SC-20-SC),
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
//------------------------------------------------------------------------------
// For assistance with SLATE, email <slate-user@icl.utk.edu>.
// You can also join the "SLATE User" Google group by going to
// https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user,
// signing in with your Google credentials, and then clicking "Join group".
//------------------------------------------------------------------------------

#include "slate/internal/Memory.hh"

#include "unit_test.hh"
#include "slate/Exception.hh"

using std::max;

//------------------------------------------------------------------------------
// global variables
int nb;

//------------------------------------------------------------------------------
/// Tests Memory(size) constructor. Doesn't allocate memory.
void test_Memory()
{
    slate::Memory mem(sizeof(double) * nb * nb);

    test_assert(int(mem.available(mem.host_num_)) == 0);
    test_assert(int(mem.capacity (mem.host_num_)) == 0);

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
    //test_assert(int(mem.available(mem.host_num_)) == cnt);
    //test_assert(int(mem.capacity (mem.host_num_)) == cnt);
    test_assert(int(mem.available(mem.host_num_)) == 0);
    test_assert(int(mem.capacity (mem.host_num_)) == 0);

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

    test_assert(int(mem.available(mem.host_num_)) == 0);
    test_assert(int(mem.capacity (mem.host_num_)) == 0);
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
        hx[i] = (double*) mem.alloc(mem.host_num_, sizeof(double) * nb * nb);
        test_assert(hx[i] != nullptr);
        // Memory class no longer reserves CPU blocks, it allocates on-the-fly.
        //test_assert(int(mem.available(mem.host_num_)) == max(cnt - (i+1), 0));
        //test_assert(int(mem.capacity (mem.host_num_)) == max(cnt, i+1));
        test_assert(int(mem.available(mem.host_num_)) == 0);
        test_assert(int(mem.capacity (mem.host_num_)) == 0);

        // Touch memory to verify it is valid.
        for (int j = 0; j < nb*nb; ++j) {
            hx[i][j] = i*1000000 + j;
        }
    }

    // Free some.
    int some = cnt/2;
    for (int i = 0; i < some; ++i) {
        mem.free(hx[i], mem.host_num_);
        hx[i] = nullptr;
        //test_assert(int(mem.available(mem.host_num_)) == i+1);
        //test_assert(int(mem.capacity (mem.host_num_)) == 2*cnt);
        test_assert(int(mem.available(mem.host_num_)) == 0);
        test_assert(int(mem.capacity (mem.host_num_)) == 0);
    }

    // Re-alloc some.
    for (int i = 0; i < some; ++i) {
        hx[i] = (double*) mem.alloc(mem.host_num_, sizeof(double) * nb * nb);
        test_assert(hx[i] != nullptr);
        //test_assert(int(mem.available(mem.host_num_)) == some - (i+1));
        //test_assert(int(mem.capacity (mem.host_num_)) == 2*cnt);
        test_assert(int(mem.available(mem.host_num_)) == 0);
        test_assert(int(mem.capacity (mem.host_num_)) == 0);
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
            slate_cuda_call(
                cudaSetDevice(dev));
            slate_cuda_call(
                cudaMemcpy(dx[i], hx, sizeof(double) * nb * nb,
                           cudaMemcpyHostToDevice));
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
    //test_assert(int(mem.available(mem.host_num_)) == cnt);
    //test_assert(int(mem.capacity (mem.host_num_)) == cnt);
    test_assert(int(mem.available(mem.host_num_)) == 0);
    test_assert(int(mem.capacity (mem.host_num_)) == 0);

    // Allocate 2*cnt blocks.
    for (int i = 0; i < 2*cnt; ++i) {
        mem.alloc(mem.host_num_, sizeof(double) * nb * nb);
    }

    test_assert(int(mem.available(mem.host_num_)) == 0);
    //test_assert(int(mem.capacity (mem.host_num_)) == 2*cnt);
    test_assert(int(mem.capacity (mem.host_num_)) == 0);

    mem.clearHostBlocks();

    test_assert(int(mem.available(mem.host_num_)) == 0);
    test_assert(int(mem.capacity (mem.host_num_)) == 0);
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
    run_test(test_clearHostBlocks,   "clearHostBlacks");
    run_test(test_clearDeviceBlocks, "clearDeviceBlacks");
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // global nb
    nb = 16;
    if (argc > 1) {
        nb = atoi(argv[1]);
    }
    printf("nb = %d\n", nb);
    return unit_test_main();  // which calls run_tests()
}
