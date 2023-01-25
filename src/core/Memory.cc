// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "auxiliary/Debug.hh"
#include "slate/internal/Memory.hh"

namespace slate {

int Memory::num_devices_;
Memory::StaticConstructor Memory::static_constructor_;

//------------------------------------------------------------------------------
/// Construct saves block size, but does not allocate any memory.
Memory::Memory(size_t block_size):
    block_size_(block_size)
{
    // touch maps to create entries;
    // this allows available() and capacity() to be const by using at()
    free_blocks_[ HostNum ];
    capacity_[ HostNum ] = 0;
    for (int device = 0; device < num_devices_; ++device) {
        free_blocks_[device];
        capacity_[device] = 0;
    }
}

//------------------------------------------------------------------------------
/// Destructor frees all allocations on host and devices.
Memory::~Memory()
{
    // This is just a check that an explicit clear was called before
    // the destructor happens.  For device/accelerator's, the queue is
    // needed to release memory (and can't be passed in here).  So to
    // release the memory, an explicit clear must called using the
    // queue parameter ( Memory::clearDeviceBlocks(device, *queue) ).
    assert(capacity_[ HostNum ] == 0);
    for (int device = 0; device < num_devices_; ++device) {
        assert(capacity_[ device ] == 0);
    }
    // Debug::printNumFreeMemBlocks(*this);
}

//------------------------------------------------------------------------------
/// Allocates num_blocks in host memory
/// and adds them to the pool of free blocks.
///
// todo: merge with addDeviceBlocks by recognizing HostNum?
void Memory::addHostBlocks(int64_t num_blocks)
{
/*
    // or std::byte* (C++17)
    uint8_t* host_mem;
    host_mem = (uint8_t*) allocHostMemory(block_size_*num_blocks);
    capacity_[ HostNum ] += num_blocks;

    for (int64_t i = 0; i < num_blocks; ++i)
        free_blocks_[ HostNum ].push(host_mem + i*block_size_);
*/
}

//------------------------------------------------------------------------------
/// Allocates num_blocks in given device's memory
/// and adds them to the pool of free blocks.
///
void Memory::addDeviceBlocks(int device, int64_t num_blocks, blas::Queue *queue)
{
    // or std::byte* (C++17)
    uint8_t* dev_mem;
    dev_mem = (uint8_t*) allocDeviceMemory(device, block_size_*num_blocks, queue);
    capacity_[device] += num_blocks;

    for (int64_t i = 0; i < num_blocks; ++i)
        free_blocks_[device].push(dev_mem + i*block_size_);
}

//------------------------------------------------------------------------------
/// Empties the pool of free blocks of host memory and frees the allocations.
///
// todo: merge with clearDeviceBlocks by recognizing HostNum?
void Memory::clearHostBlocks()
{
/*
    Debug::checkHostMemoryLeaks(*this);

    while (! free_blocks_[ HostNum ].empty())
        free_blocks_[ HostNum ].pop();

    while (! allocated_mem_[ HostNum ].empty()) {
        void* host_mem = allocated_mem_[ HostNum ].top();
        freeHostMemory( host_mem );
        allocated_mem_[ HostNum ].pop();
    }
    capacity_[ HostNum ] = 0;
*/
}

//------------------------------------------------------------------------------
/// Empties the pool of free blocks of given device's memory and frees the
/// allocations.
///
void Memory::clearDeviceBlocks(int device, blas::Queue *queue)
{
    Debug::checkDeviceMemoryLeaks(*this, device);

    while (! free_blocks_[device].empty())
        free_blocks_[device].pop();

    while (! allocated_mem_[device].empty()) {
        void* dev_mem = allocated_mem_[device].top();
        freeDeviceMemory(device, dev_mem, queue);
        allocated_mem_[device].pop();
    }
    capacity_[device] = 0;
}

//------------------------------------------------------------------------------
/// @return single block of memory on the given device, which can be host,
/// either from free blocks or by allocating a new block.
///
void* Memory::alloc(int device, size_t size, blas::Queue* queue)
{
    void* block;

    if (device == HostNum) {
        //block = malloc(size);
        block = new char[size];
    }
    else {
        // this block for device only
        #pragma omp critical(slate_memory)
        {
            if (free_blocks_[device].size() > 0) {
                block = free_blocks_[device].top();
                free_blocks_[device].pop();
            }
            else {
                block = allocBlock(device, queue);
            }
        }
    }
    return block;
}

//------------------------------------------------------------------------------
/// Puts a single block of memory back into the pool of free blocks
/// for the given device, which can be host.
///
void Memory::free(void* block, int device)
{
    if (device == HostNum) {
        //std::free(block);
        delete[] (char*)block;
    }
    else {
        #pragma omp critical(slate_memory)
        {
            free_blocks_[device].push(block);
        }
    }
}

//------------------------------------------------------------------------------
/// Allocates a single block of memory on the given device, which can be host.
///
void* Memory::allocBlock(int device, blas::Queue *queue)
{
    void* block;
    if (device == HostNum)
        block = allocHostMemory(block_size_);
    else
        block = allocDeviceMemory(device, block_size_, queue);

    capacity_[device] += 1;
    return block;
}

//------------------------------------------------------------------------------
/// Allocates host memory of given size.
///
void* Memory::allocHostMemory(size_t size)
{
    void* host_mem;
    host_mem = malloc(size);
    assert(host_mem != nullptr);
    allocated_mem_[ HostNum ].push( host_mem );

    return host_mem;
}

//------------------------------------------------------------------------------
/// Allocates GPU device memory of given size.
///
void* Memory::allocDeviceMemory(int device, size_t size, blas::Queue *queue)
{
    void* dev_mem = blas::device_malloc<char>(size, *queue);
    allocated_mem_[device].push(dev_mem);

    return dev_mem;
}

//------------------------------------------------------------------------------
/// Frees host memory.
///
void Memory::freeHostMemory(void* host_mem)
{
    std::free(host_mem);
}

//------------------------------------------------------------------------------
/// Frees GPU device memory.
///
void Memory::freeDeviceMemory(int device, void* dev_mem, blas::Queue *queue)
{
    blas::device_free(dev_mem, *queue);
}

} // namespace slate
