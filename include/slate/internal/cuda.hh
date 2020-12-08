// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

//------------------------------------------------------------------------------
/// @file
///
#ifndef SLATE_CUDA_HH
#define SLATE_CUDA_HH

#include "slate/Exception.hh"

#include "blas.hh"

#if (!defined(SLATE_NO_CUDA)) || defined(__NVCC__)
    #include <cuda_runtime.h>
    #include <cuComplex.h>
#else

#include <cstdlib>
#include <complex>

typedef int cudaError_t;
typedef void* cudaStream_t;
typedef int cudaMemcpyKind;
typedef std::complex<float> cuComplex;
typedef cuComplex cuFloatComplex;
typedef std::complex<double> cuDoubleComplex;

enum {
    cudaSuccess,
    cudaStreamNonBlocking,
    cudaMemcpyHostToHost,
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
    cudaMemcpyDeviceToDevice,
    cudaMemcpyDefault
};

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t cudaFree(void* devPtr);
cudaError_t cudaFreeHost(void* ptr);

cudaError_t cudaGetDevice(int* device);
cudaError_t cudaGetDeviceCount(int* count);

cudaError_t cudaMalloc(void** devPtr, size_t size);

cudaError_t cudaMallocHost(void** ptr, size_t size);

cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch,
                              const void* src, size_t spitch,
                              size_t width, size_t height,
                              cudaMemcpyKind kind, cudaStream_t stream = 0);

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream = 0);

cudaError_t cudaMemcpy(void* dst, const void*  src,
                       size_t count, cudaMemcpyKind kind);

cudaError_t cudaSetDevice(int device);

cudaError_t cudaStreamCreate(cudaStream_t* pStream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamDestroy(cudaStream_t pStream);

const char* cudaGetErrorName(cudaError_t error);
const char* cudaGetErrorString(cudaError_t error);

#ifdef __cplusplus
}
#endif

#endif // SLATE_NO_CUDA

namespace slate {

//------------------------------------------------------------------------------
/// Exception class for slate_cuda_call().
class CudaException : public Exception {
public:
    CudaException(const char* call,
                  cudaError_t code,
                  const char* func,
                  const char* file,
                  int line)
        : Exception()
    {
        const char* name = cudaGetErrorName(code);
        const char* string = cudaGetErrorString(code);

        what(std::string("SLATE CUDA ERROR: ")
             + call + " failed: " + string
             + " (" + name + "=" + std::to_string(code) + ")",
             func, file, line);
    }
};

/// Throws a CudaException if the CUDA call fails.
/// Example:
///
///     try {
///         slate_cuda_call( cudaSetDevice( device ) );
///     }
///     catch (CudaException& e) {
///         ...
///     }
///
#define slate_cuda_call(call) \
    do { \
        cudaError_t slate_cuda_call_ = call; \
        if (slate_cuda_call_ != cudaSuccess) \
            throw slate::CudaException( \
                #call, slate_cuda_call_, __func__, __FILE__, __LINE__); \
    } while(0)

} // namespace slate

//------------------------------------------------------------------------------
// Extend BLAS real_type to cover cuComplex
namespace blas {

template<>
struct real_type_traits<cuFloatComplex> {
    using real_t = float;
};

template<>
struct real_type_traits<cuDoubleComplex> {
    using real_t = double;
};

} // namespace blas

#endif // SLATE_CUDA_HH
