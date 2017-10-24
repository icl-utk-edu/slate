
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
    Memory(size_t block_size, int64_t max_blocks)
        : block_size_(block_size)
    {
        int host_num = omp_get_initial_device();
        for (int64_t i = 0; i < max_blocks; ++i) {
            void *block = allocate_block(host_num);
            free_blocks_[host_num].push(block);
        }

        int num_devices = omp_get_num_devices();
        for (int device = 0; device < num_devices; ++device) {
            for (int64_t i = 0; i < max_blocks; ++i) {
                void *block = allocate_block(device);
                free_blocks_[device].push(block);
            }
        }
    }
    ~Memory()
    {
        // print_num_free_blocks();
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
                block = allocate_block(device_num);
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
    void* allocate_block(int device)
    {
        static int host_num = omp_get_initial_device();

        void *block;
        if (device == host_num)
            block = allocate_host_block();
        else
            block = allocate_device_block(device);

        return block;
    }
    void* allocate_host_block()
    {
        void *block;
        cudaError_t error = cudaMallocHost(&block, block_size_);
        assert(error == cudaSuccess);
        // block = malloc(block_size_);
        // assert(block != nullptr);
        return block;
    }
    void* allocate_device_block(int device)
    {
        cudaError_t error;
        error = cudaSetDevice(device);
        assert(error == cudaSuccess);

        void *block;
        error = cudaMalloc(&block, block_size_);
        assert(error == cudaSuccess);
        return block;
    }

    void print_num_free_blocks()
    {
        printf("\n");
        for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
            printf("\tdevice: %d\tfree blocks: %d\n", 
                   it->first, it->second.size());
        }
    }

    size_t block_size_;
    std::map<int, std::stack<void*>> free_blocks_;
};

} // namespace slate

#endif // SLATE_MEMORY_HH
