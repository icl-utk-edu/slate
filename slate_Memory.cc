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

#include "slate_Debug.hh"
#include "slate_Memory.hh"

namespace slate {

int Memory::host_num_;
int Memory::num_devices_;
Memory::StaticConstructor Memory::static_constructor_;

//------------------------------------------------------------------------------
/// Construct saves block size, but does not allocate any memory.
Memory::Memory(size_t block_size):
    block_size_(block_size)
{
    // touch maps to create entries;
    // this allows available() and capacity() to be const by using at()
    free_blocks_[host_num_];
    capacity_[host_num_] = 0;
    for (int device = 0; device < num_devices_; ++device) {
        free_blocks_[device];
        capacity_[device] = 0;
    }
}

//------------------------------------------------------------------------------
/// \brief
/// Destructor frees all allocations on host and devices.
Memory::~Memory()
{
    //Debug::printNumFreeMemBlocks(*this);

    clearHostBlocks();
    for (int device = 0; device < num_devices_; ++device)
        clearDeviceBlocks(device);
}

//------------------------------------------------------------------------------
/// \brief
/// Allocates num_blocks in host memory
/// and adds them to the pool of free blocks.
///
// todo: merge with addDeviceBlocks by recognizing host_num_?
void Memory::addHostBlocks(int64_t num_blocks)
{
    // or std::byte* (C++17)
    uint8_t* host_mem;
    host_mem = (uint8_t*) allocHostMemory(block_size_*num_blocks);
    capacity_[host_num_] += num_blocks;

    for (int64_t i = 0; i < num_blocks; ++i)
        free_blocks_[host_num_].push(host_mem + i*block_size_);
}

//------------------------------------------------------------------------------
/// \brief
/// Allocates num_blocks in given device's memory
/// and adds them to the pool of free blocks.
///
void Memory::addDeviceBlocks(int device, int64_t num_blocks)
{
    // or std::byte* (C++17)
    uint8_t* dev_mem;
    dev_mem = (uint8_t*) allocDeviceMemory(device, block_size_*num_blocks);
    capacity_[device] += num_blocks;

    for (int64_t i = 0; i < num_blocks; ++i)
        free_blocks_[device].push(dev_mem + i*block_size_);
}

//------------------------------------------------------------------------------
/// \brief
/// Empties the pool of free blocks of host memory and frees the allocations.
///
// todo: merge with clearDeviceBlocks by recognizing host_num_?
void Memory::clearHostBlocks()
{
    #ifdef DEBUG
        if (free_blocks_[host_num_].size() < capacity_[host_num_]) {
            std::cerr << "rank " << g_mpi_rank << " "
                      << " memory leak: freed "
                      << free_blocks_[host_num_].size()
                      << " of " << capacity_[host_num_]
                      << " blocks on host\n";
        }
        else if (free_blocks_[host_num_].size() > capacity_[host_num_]) {
            std::cerr << "rank " << g_mpi_rank << " "
                      << " freed too many: " << free_blocks_[host_num_].size()
                      << " of " << capacity_[host_num_]
                      << " blocks on host\n";
        }
    #endif

    while (! free_blocks_[host_num_].empty())
        free_blocks_[host_num_].pop();

    while (! allocated_mem_[host_num_].empty()) {
        void* host_mem = allocated_mem_[host_num_].top();
        freeHostMemory(host_mem);
        allocated_mem_[host_num_].pop();
    }
    capacity_[host_num_] = 0;
}

//------------------------------------------------------------------------------
/// \brief
/// Empties the pool of free blocks of given device's memory and frees the
/// allocations.
///
void Memory::clearDeviceBlocks(int device)
{
    #ifdef DEBUG
        if (free_blocks_[device].size() < capacity_[device]) {
            std::cerr << "rank " << g_mpi_rank << " "
                      << "memory leak: freed " << free_blocks_[device].size()
                      << " of " << capacity_[device]
                      << " blocks on device " << device << "\n";
        }
        else if (free_blocks_[device].size() > capacity_[device]) {
            std::cerr << "rank " << g_mpi_rank << " "
                      << "freed too many: " << free_blocks_[device].size()
                      << " of " << capacity_[device]
                      << " blocks on device " << device << "\n";
        }
    #endif

    while (! free_blocks_[device].empty())
        free_blocks_[device].pop();

    while (! allocated_mem_[device].empty()) {
        void* dev_mem = allocated_mem_[device].top();
        freeDeviceMemory(device, dev_mem);
        allocated_mem_[device].pop();
    }
    capacity_[device] = 0;
}

//------------------------------------------------------------------------------
/// \brief
/// @return single block of memory on the given device, which can be host,
/// either from free blocks or by allocating a new block.
///
void* Memory::alloc(int device)
{
    void* block;
    #pragma omp critical(slate_memory)
    {
        if (free_blocks_[device].size() > 0) {
            block = free_blocks_[device].top();
            free_blocks_[device].pop();
        }
        else
            block = allocBlock(device);
    }
    return block;
}

//------------------------------------------------------------------------------
/// \brief
/// Puts a single block of memory back into the pool of free blocks
/// for the given device, which can be host.
///
void Memory::free(void* block, int device)
{
    #pragma omp critical(slate_memory)
    {
        free_blocks_[device].push(block);
    }
}

//------------------------------------------------------------------------------
/// \brief
/// Allocates a single block of memory on the given device, which can be host.
///
void* Memory::allocBlock(int device)
{
    void* block;
    if (device == host_num_)
        block = allocHostMemory(block_size_);
    else
        block = allocDeviceMemory(device, block_size_);

    allocated_mem_[device].push(block);
    capacity_[device] += 1;
    return block;
}

//------------------------------------------------------------------------------
/// \brief
/// Allocates host memory of given size.
///
void* Memory::allocHostMemory(size_t size)
{
    void* host_mem;
    // cudaError_t error = cudaMallocHost(&host_mem, size);
    // assert(error == cudaSuccess);
    host_mem = malloc(size);
    assert(host_mem != nullptr);

    return host_mem;
}

//------------------------------------------------------------------------------
/// \brief
/// Allocates GPU device memory of given size.
///
void* Memory::allocDeviceMemory(int device, size_t size)
{
    cudaError_t error;
    error = cudaSetDevice(device);
    assert(error == cudaSuccess);

    double* dev_mem;
    error = cudaMalloc((void**)&dev_mem, size);
    assert(error == cudaSuccess);

    return dev_mem;
}

//------------------------------------------------------------------------------
/// \brief
/// Frees host memory.
///
void Memory::freeHostMemory(void* host_mem)
{
    std::free(host_mem);
    // cudaError_t error = cudaFreeHost(host_mem);
    // assert(error == cudaSuccess);
}

//------------------------------------------------------------------------------
/// \brief
/// Frees GPU device memory.
///
void Memory::freeDeviceMemory(int device, void* dev_mem)
{
    cudaError_t error;
    error = cudaSetDevice(device);
    assert(error == cudaSuccess);

    error = cudaFree(dev_mem);
    assert(error == cudaSuccess);
}

} // namespace slate
