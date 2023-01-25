// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_MEMORY_HH
#define SLATE_MEMORY_HH

#include <cstdlib>
#include <cassert>
#include <cstring>
#include <iostream>
#include <iomanip>

#include <map>
#include <stack>

#include "blas.hh"

#include "slate/enums.hh"
#include "slate/internal/openmp.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Allocates workspace blocks for host and GPU devices.
/// Currently assumes a fixed-size block of block_size bytes,
/// e.g., block_size = sizeof(scalar_t) * mb * nb.
class Memory {
public:
    friend class Debug;

    static struct StaticConstructor {
        StaticConstructor()
        {
            num_devices_ = blas::get_device_count();
        }
    } static_constructor_;

    Memory(size_t block_size);
    ~Memory();

    // todo: change add* to reserve*?
    void addHostBlocks(int64_t num_blocks);
    void addDeviceBlocks(int device, int64_t num_blocks, blas::Queue *queue);

    void clearHostBlocks();
    void clearDeviceBlocks(int device, blas::Queue *queue);

    void* alloc(int device, size_t size, blas::Queue *queue);
    void free(void* block, int device);

    /// @return number of available free blocks in device's memory pool,
    /// which can be host.
    size_t available(int device) const
    {
        return free_blocks_.at(device).size();
    }

    /// @return total number of blocks in device's memory pool,
    /// which can be host.
    size_t capacity(int device) const
    {
        return capacity_.at(device);
    }

    /// @return total number of allocated blocks from device's memory pool,
    /// which can be host.
    size_t allocated(int device) const
    {
        return capacity(device) - available(device);
    }

    // ----------------------------------------
    // public static variables
    static int num_devices_;

private:
    void* allocBlock(int device, blas::Queue *queue);

    void* allocHostMemory(size_t size);
    void* allocDeviceMemory(int device, size_t size, blas::Queue *queue);

    void freeHostMemory(void* host_mem);
    void freeDeviceMemory(int device, void* dev_mem, blas::Queue *queue);

    // ----------------------------------------
    // member variables
    size_t block_size_;

    // map device number to stack of blocks
    std::map< int, std::stack<void*> > free_blocks_;
    std::map< int, std::stack<void*> > allocated_mem_;
    std::map< int, size_t > capacity_;
};

} // namespace slate

#endif // SLATE_MEMORY_HH
