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
// Need assistance with the SLATE software? Join the "SLATE User" Google group
// by going to https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user
// and clicking "Apply to join group". Upon acceptance, email your questions and
// comments to <slate-user@icl.utk.edu>.
//------------------------------------------------------------------------------

#ifndef SLATE_MEMORY_HH
#define SLATE_MEMORY_HH

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

#include <map>
#include <stack>

#ifdef SLATE_WITH_CUDA
    #include <cuda_runtime.h>
#else
    #include "slate_NoCuda.hh"
#endif

#ifdef SLATE_WITH_OPENMP
    #include <omp.h>
#else
    #include "slate_NoOpenmp.hh"
#endif

extern "C" void trace_cpu_start();
extern "C" void trace_cpu_stop(const char *color);

namespace slate {

//------------------------------------------------------------------------------
class Memory {
public:
    Memory(size_t block_size) : block_size_(block_size) {}
    ~Memory()
    {
        // printNumFreeBlocks();
    }

    void addHostBlocks(int64_t num_blocks)
    {
        // or std::byte* (C++17)
        uint8_t *host_mem;
        host_mem = (uint8_t*)allocHostMemory(block_size_*num_blocks);

        for (int64_t i = 0; i < num_blocks; ++i)
            free_blocks_[host_num_].push(host_mem+i*block_size_);

    }
    void addDeviceBlocks(int device, int64_t num_blocks)
    {
        // or std::byte* (C++17)
        uint8_t *dev_mem;
        dev_mem = (uint8_t*)allocDeviceMemory(device, block_size_*num_blocks);

        for (int64_t i = 0; i < num_blocks; ++i)
            free_blocks_[device].push(dev_mem+i*block_size_);
    }
    void clearHostBlocks()
    {
        while (!free_blocks_[host_num_].empty())
            free_blocks_[host_num_].pop();

        while (!allocated_mem_[host_num_].empty()) {
            void *host_mem = allocated_mem_[host_num_].top();
            freeHostMemory(host_mem);
            allocated_mem_[host_num_].pop();
        } 
    }
    void clearDeviceBlocks(int device)
    {
        while (!free_blocks_[device].empty())
            free_blocks_[device].pop();

        while (!allocated_mem_[device].empty()) {
            void *dev_mem = allocated_mem_[device].top();
            freeDeviceMemory(device, dev_mem);
            allocated_mem_[device].pop();
        }
    }

    void* alloc(int device_num)
    {
        void *block;
        #pragma omp critical(slate_memory)
        {
            if (free_blocks_[device_num].size() > 0) {
                block = free_blocks_[device_num].top();
                free_blocks_[device_num].pop();
            }
            else {
                block = allocBlock(device_num);
            }
        }
        return block;
    }
    void free(void *block, int device_num)
    {
        #pragma omp critical(slate_memory)
        {
            free_blocks_[device_num].push(block);
        }
    }

private:
    void* allocBlock(int device)
    {
        void *block;
        if (device == host_num_)
            block = allocHostMemory(block_size_);
        else
            block = allocDeviceMemory(device, block_size_);

        allocated_mem_[device].push(block);
        return block;
    }
    void* allocHostMemory(size_t size)
    {
        void *host_mem;
        // cudaError_t error = cudaMallocHost(&host_mem, size);
        // assert(error == cudaSuccess);
        host_mem = malloc(size);
        assert(host_mem != nullptr);
        return host_mem;
    }
    void* allocDeviceMemory(int device, size_t size)
    {
        cudaError_t error;
        error = cudaSetDevice(device);
        assert(error == cudaSuccess);

        void *dev_mem;
        error = cudaMalloc(&dev_mem, size);
        assert(error == cudaSuccess);

        return dev_mem;
    }
    void freeHostMemory(void *host_mem)
    {
        std::free(host_mem);
        // cudaError_t error = cudaFreeHost(host_mem);
        // assert(error == cudaSuccess);
    }
    void freeDeviceMemory(int device, void *dev_mem)
    {
        cudaError_t error;
        error = cudaSetDevice(device);
        assert(error == cudaSuccess);

        error = cudaFree(dev_mem);
        assert(error == cudaSuccess);
    }

    void printNumFreeBlocks()
    {
        printf("\n");
        for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
            printf("\tdevice: %d\tfree blocks: %d\n", 
                   it->first, it->second.size());
        }
    }

    static int host_num_;

    size_t block_size_;
    std::map<int, std::stack<void*>> free_blocks_;
    std::map<int, std::stack<void*>> allocated_mem_;
};

} // namespace slate

#endif // SLATE_MEMORY_HH
