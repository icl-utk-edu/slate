
#ifndef SLATE_NO_CUDA_HH
#define SLATE_NO_CUDA_HH

#include <cassert>

typedef int cudaError_t;
typedef int cudaStream_t;
typedef int cudaMemcpyKind;

enum {
    cudaMemcpyDeviceToHost,
    cudaMemcpyHostToDevice,
    cudaStreamNonBlocking, 
    cudaSuccess
};

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
cudaError_t cudaGetDevice(int *device)
{
    assert(0);
}

//------------------------------------------------------------------------------
cudaError_t cudaGetDeviceCount(int *count)
{
    *count = 0;
    return cudaSuccess;
}

//------------------------------------------------------------------------------
cudaError_t cudaMalloc(void **devPtr, size_t size)
{
    assert(0);
}

//------------------------------------------------------------------------------
cudaError_t cudaMallocHost(void **ptr, size_t size)
{
    assert(0);
}

//------------------------------------------------------------------------------
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream = 0)
{
    assert(0);
}

//------------------------------------------------------------------------------
cudaError_t cudaSetDevice(int device)
{
    assert(0);
}

//------------------------------------------------------------------------------
cudaError_t cudaStreamCreate(cudaStream_t *pStream)
{
    assert(0);
}

//------------------------------------------------------------------------------
cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    assert(0);
}

#ifdef __cplusplus
}
#endif

#endif // SLATE_NO_CUDA_HH
