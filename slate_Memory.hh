
#ifndef SLATE_MEMORY_HH
#define SLATE_MEMORY_HH

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

#include <list>
#include <map>

#include <omp.h>
#include <cuda_runtime.h>

extern "C" void trace_cpu_start();
extern "C" void trace_cpu_stop(const char *color);

namespace slate {

//------------------------------------------------------------------------------
class Memory {
public:
    Memory(size_t block_size, int64_t num_blocks)
        : block_size_(block_size), num_blocks_(num_blocks)
    {
        printf("Memory allocator initialized\n"); fflush(stdout);

        int num_devices;
        cudaError_t error = cudaGetDeviceCount(&num_devices);
        assert(error == cudaSuccess);

        for (int device = 0; device < num_devices; ++device) {
            cudaError_t error = cudaSetDevice(device);
            assert(error == cudaSuccess);

            for (int64_t i = 0; i < num_blocks_; ++i) {
                void *block;
                cudaError_t error = cudaMalloc(&block, block_size);
                assert(error == cudaSuccess);
                free_blocks_[device].push_back(block);
            }
        }
        for (int64_t i = 0; i < num_blocks_; ++i) {
            void *block;
            cudaError_t error = cudaMallocHost(&block, block_size);
            assert(error == cudaSuccess);
            free_blocks_host_.push_back(block);
        }
    }
    ~Memory() {

    }

    void* alloc()
    {
        int device;
        cudaError_t error = cudaGetDevice(&device);
        assert(error == cudaSuccess);

        omp_set_lock(blocks_lock_);
        void *block = free_blocks_[device].front();
        free_blocks_[device].pop_front();
        omp_unset_lock(blocks_lock_);
        return block;
    }
    void* alloc_host()
    {
        omp_set_lock(blocks_lock_);
        void *block = free_blocks_host_.front();
        free_blocks_host_.pop_front();
        omp_unset_lock(blocks_lock_);
        return block;
    }
    void free(void* block)
    {
        int device;
        cudaError_t error = cudaGetDevice(&device);
        assert(error == cudaSuccess);

        omp_set_lock(blocks_lock_);
        free_blocks_[device].push_back(block);
        omp_unset_lock(blocks_lock_);
    }
    void free_host(void* block)
    {
        omp_set_lock(blocks_lock_);
        free_blocks_host_.push_back(block);
        omp_unset_lock(blocks_lock_);
    }

private:
    size_t block_size_;
    int64_t num_blocks_;

    static const int MaxDevices = 4;
    std::list<void*> free_blocks_[MaxDevices];
    std::list<void*> free_blocks_host_;

    omp_lock_t *blocks_lock_ = new omp_lock_t();
};

} // namespace slate

#endif // SLATE_MEMORY_HH
