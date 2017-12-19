
#ifndef SLATE_NO_CUDA_HH
#define SLATE_NO_CUDA_HH

#include <cstdlib>

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

cudaError_t cudaFree(void *devPtr);
cudaError_t cudaFreeHost(void *ptr);

cudaError_t cudaGetDevice(int *device);
cudaError_t cudaGetDeviceCount(int *count);

cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaMallocHost(void **ptr, size_t size);

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream = 0);

cudaError_t cudaSetDevice(int device);

cudaError_t cudaStreamCreate(cudaStream_t *pStream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // SLATE_NO_CUDA_HH
