// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

//------------------------------------------------------------------------------
/// @file
///
#ifndef SLATE_CUDA_HH
#define SLATE_CUDA_HH

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
#endif // SLATE_CUDA_HH
