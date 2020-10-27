// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/internal/device.hh"
#include "internal/internal_batch.hh"
#include "internal/internal.hh"
#include "slate/internal/util.hh"
#include "internal/Tile_lapack.hh"
#include "slate/types.hh"

#include <vector>

namespace slate {

//------------------------------------------------------------------------------
// On macOS, nvcc using clang++ generates a different C++ name mangling
// (std::__1::complex) than g++ for std::complex. This solution is to use
// cu*Complex in .cu files, and cast from std::complex here.
namespace device {

template <>
void transpose(
    int64_t n,
    std::complex<float>* A, int64_t lda,
    cudaStream_t stream)
{
#if !defined(SLATE_NO_CUDA) || defined(__NVCC__)
    transpose(n, (cuFloatComplex*) A, lda, stream);
#endif
}

template <>
void transpose(
    int64_t n,
    std::complex<double>* A, int64_t lda,
    cudaStream_t stream)
{
#if !defined(SLATE_NO_CUDA) || defined(__NVCC__)
    transpose(n, (cuDoubleComplex*) A, lda, stream);
#endif
}

//--------------------
template <>
void transpose_batch(
    int64_t n,
    std::complex<float>** Aarray, int64_t lda,
    int64_t batch_count,
    cudaStream_t stream)
{
#if !defined(SLATE_NO_CUDA) || defined(__NVCC__)
    transpose_batch(n, (cuFloatComplex**) Aarray, lda, batch_count, stream);
#endif
}

template <>
void transpose_batch(
    int64_t n,
    std::complex<double>** Aarray, int64_t lda,
    int64_t batch_count,
    cudaStream_t stream)
{
#if !defined(SLATE_NO_CUDA) || defined(__NVCC__)
    transpose_batch(n, (cuDoubleComplex**) Aarray, lda, batch_count, stream);
#endif
}

//--------------------
template <>
void transpose(
    int64_t m, int64_t n,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* AT, int64_t ldat,
    cudaStream_t stream)
{
#if !defined(SLATE_NO_CUDA) || defined(__NVCC__)
    transpose(m, n, (cuFloatComplex*) A, lda, (cuFloatComplex*) AT, ldat, stream);
#endif
}

template <>
void transpose(
    int64_t m, int64_t n,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* AT, int64_t ldat,
    cudaStream_t stream)
{
#if !defined(SLATE_NO_CUDA) || defined(__NVCC__)
    transpose(m, n, (cuDoubleComplex*) A, lda, (cuDoubleComplex*) AT, ldat, stream);
#endif
}

//--------------------
template <>
void transpose_batch(
    int64_t m, int64_t n,
    std::complex<float>** Aarray, int64_t lda,
    std::complex<float>** ATarray, int64_t ldat,
    int64_t batch_count,
    cudaStream_t stream)
{
#if !defined(SLATE_NO_CUDA) || defined(__NVCC__)
    transpose_batch(m, n, (cuFloatComplex**) Aarray, lda, (cuFloatComplex**) ATarray, ldat, batch_count, stream);
#endif
}

template <>
void transpose_batch(
    int64_t m, int64_t n,
    std::complex<double>** Aarray, int64_t lda,
    std::complex<double>** ATarray, int64_t ldat,
    int64_t batch_count,
    cudaStream_t stream)
{
#if !defined(SLATE_NO_CUDA) || defined(__NVCC__)
    transpose_batch(m, n, (cuDoubleComplex**) Aarray, lda, (cuDoubleComplex**) ATarray, ldat, batch_count, stream);
#endif
}


//------------------------------------------------------------------------------
#if defined(SLATE_NO_CUDA)
// Specializations to allow compilation without CUDA.
template <>
void transpose(
    int64_t n,
    float* A, int64_t lda,
    cudaStream_t stream)
{
}

template <>
void transpose(
    int64_t n,
    double* A, int64_t lda,
    cudaStream_t stream)
{
}

//--------------------
template <>
void transpose_batch(
    int64_t n,
    float** Aarray, int64_t lda,
    int64_t batch_count,
    cudaStream_t stream)
{
}

template <>
void transpose_batch(
    int64_t n,
    double** Aarray, int64_t lda,
    int64_t batch_count,
    cudaStream_t stream)
{
}

//--------------------
template <>
void transpose(
    int64_t m, int64_t n,
    float* A, int64_t lda,
    float* AT, int64_t ldat,
    cudaStream_t stream)
{
}

template <>
void transpose(
    int64_t m, int64_t n,
    double* A, int64_t lda,
    double* AT, int64_t ldat,
    cudaStream_t stream)
{
}

//--------------------
template <>
void transpose_batch(
    int64_t m, int64_t n,
    float** Aarray, int64_t lda,
    float** ATarray, int64_t ldat,
    int64_t batch_count,
    cudaStream_t stream)
{
}

template <>
void transpose_batch(
    int64_t m, int64_t n,
    double** Aarray, int64_t lda,
    double** ATarray, int64_t ldat,
    int64_t batch_count,
    cudaStream_t stream)
{
}
#endif // not SLATE_NO_CUDA

} // namespace device
} // namespace slate
