
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
        allocated_mem_[host_num_].push(host_mem);
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
        allocated_mem_[device].push(dev_mem);
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
