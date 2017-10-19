
#ifndef SLATE_MEMORY_HH
#define SLATE_MEMORY_HH

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

#include <list>
#include <map>

#include <omp.h>
#ifdef SLATE_WITH_CUDA
    #include <cuda_runtime.h>
#else
    #include "slate_NoCuda.hh"
#endif

extern "C" void trace_cpu_start();
extern "C" void trace_cpu_stop(const char *color);

namespace slate {

//------------------------------------------------------------------------------
class Memory {
public:
    Memory(size_t block_size, int64_t max_blocks)
        : block_size_(block_size), max_blocks_(max_blocks)
    {
        printf("Memory allocator initializing...\n"); fflush(stdout);

        cudaError_t error = cudaGetDeviceCount(&num_devices_);
        assert(error == cudaSuccess);

        for (int device = 0; device < num_devices_; ++device) {
            cudaError_t error = cudaSetDevice(device);
            assert(error == cudaSuccess);

            for (int64_t i = 0; i < max_blocks_; ++i) {
                void *block;
                cudaError_t error = cudaMalloc(&block, block_size);
                assert(error == cudaSuccess);
                free_blocks_[device].push_back(block);
            }
            num_allocated_[device] = 0;
            max_allocated_[device] = 0;

            printf("Device %d allocator initialized!\n", device);
            fflush(stdout);
        }
        for (int64_t i = 0; i < max_blocks_*std::max(num_devices_, 1); ++i) {
            void *block;
            // cudaError_t error = cudaMallocHost(&block, block_size);
            // assert(error == cudaSuccess);
            block = malloc(block_size);
            assert(block != nullptr);
            free_blocks_host_.push_back(block);
        }
        num_allocated_host_ = 0;
        max_allocated_host_ = 0;

        printf("Host allocator initialized!\n"); fflush(stdout);
    }
    ~Memory()
    {
        printf("\n");
        for (int device = 0; device < num_devices_; ++device)
            printf("\t%d\tleaked\t%d\tmax\n", num_allocated_[device],
                                              max_allocated_[device]);
            printf("\t%d\tleaked\t%d\tmax\n", num_allocated_host_,
                                              max_allocated_host_);
    }

    void* alloc()
    {
        int device;
        cudaError_t error = cudaGetDevice(&device);
        assert(error == cudaSuccess);

        omp_set_lock(blocks_lock_);
        void *block = free_blocks_[device].front();
        free_blocks_[device].pop_front();

        ++num_allocated_[device];
        assert(num_allocated_[device] <= max_blocks_);
        if (num_allocated_[device] > max_allocated_[device])
            max_allocated_[device] = num_allocated_[device];
        omp_unset_lock(blocks_lock_);
        return block;
    }
    void* alloc_host()
    {
        omp_set_lock(blocks_lock_);
        void *block = free_blocks_host_.front();
        free_blocks_host_.pop_front();

        ++num_allocated_host_;
        assert(num_allocated_host_ <= max_blocks_*std::max(num_devices_, 1));
        if (num_allocated_host_ > max_allocated_host_)
            max_allocated_host_ = num_allocated_host_;
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
        --num_allocated_[device];
        omp_unset_lock(blocks_lock_);
    }
    void free_host(void* block)
    {
        omp_set_lock(blocks_lock_);
        free_blocks_host_.push_back(block);
        --num_allocated_host_;
        omp_unset_lock(blocks_lock_);
    }

private:
    size_t block_size_;
    int64_t max_blocks_;
    static const int MaxDevices = 4;
    int num_devices_;

    int64_t num_allocated_[MaxDevices];
    int64_t max_allocated_[MaxDevices];

    int64_t num_allocated_host_;
    int64_t max_allocated_host_;

    std::list<void*> free_blocks_[MaxDevices];
    std::list<void*> free_blocks_host_;

    omp_lock_t *blocks_lock_ = new omp_lock_t();
};

} // namespace slate

#endif // SLATE_MEMORY_HH
