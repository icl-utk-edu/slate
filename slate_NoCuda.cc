
#include "slate_NoCuda.hh"

#include <cassert>

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t cudaFree(void *devPtr)
{
    assert(0);
}

cudaError_t cudaFreeHost(void *devPtr)
{
    assert(0);
}

cudaError_t cudaGetDevice(int *device)
{
    assert(0);
}

cudaError_t cudaGetDeviceCount(int *count)
{
    *count = 0;
    return cudaSuccess;
}

cudaError_t cudaMalloc(void **devPtr, size_t size)
{
    assert(0);
}

cudaError_t cudaMallocHost(void **ptr, size_t size)
{
    assert(0);
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream)
{
    assert(0);
}

cudaError_t cudaSetDevice(int device)
{
    assert(0);
}

cudaError_t cudaStreamCreate(cudaStream_t *pStream)
{
    assert(0);
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    assert(0);
}

#ifdef __cplusplus
}
#endif
