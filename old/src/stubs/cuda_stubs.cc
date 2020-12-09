// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

//------------------------------------------------------------------------------
/// @file
///
#include "slate/internal/cuda.hh"

#include <cassert>

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t cudaFree(void* devPtr)
{
    assert(0);
}

cudaError_t cudaFreeHost(void* devPtr)
{
    assert(0);
}

cudaError_t cudaGetDevice(int* device)
{
    assert(0);
}

cudaError_t cudaGetDeviceCount(int* count)
{
    *count = 0;
    return cudaSuccess;
}

cudaError_t cudaMalloc(void** devPtr, size_t size)
{
    assert(0);
}

cudaError_t cudaMallocHost(void** ptr, size_t size)
{
    assert(0);
}

cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch,
                              const void* src, size_t spitch,
                              size_t width, size_t height,
                              cudaMemcpyKind kind, cudaStream_t stream)
{
    assert(0);
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream)
{
    assert(0);
}

cudaError_t cudaMemcpy(void* dst, const void* src,
                       size_t count, cudaMemcpyKind kind)
{
    assert(0);
}

cudaError_t cudaSetDevice(int device)
{
    assert(0);
}

cudaError_t cudaStreamCreate(cudaStream_t* pStream)
{
    assert(0);
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    assert(0);
}

cudaError_t cudaStreamDestroy(cudaStream_t pStream)
{
    assert(0);
}

const char* cudaGetErrorName(cudaError_t error)
{
    assert(0);
}

const char* cudaGetErrorString(cudaError_t error)
{
    assert(0);
}

#ifdef __cplusplus
}
#endif
