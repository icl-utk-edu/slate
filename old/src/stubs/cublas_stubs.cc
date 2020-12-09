// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

//------------------------------------------------------------------------------
/// @file
///
#include "slate/internal/cublas.hh"

#include <cassert>

#ifdef __cplusplus
extern "C" {
#endif

cublasStatus_t cublasCreate(cublasHandle_t* handle)
{
    assert(0);
}

cublasStatus_t cublasDestroy(cublasHandle_t handle)
{
    assert(0);
}

cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId)
{
    assert(0);
}

cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t* streamId)
{
    assert(0);
}

cublasStatus_t cublasDasum(cublasHandle_t handle, int n, const double* x,
                           int incx, double* result)
{
    assert(0);
}

cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize,
                               const void* A, int lda, void* B, int ldb)
{
    assert(0);
}

cublasStatus_t cublasSgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha, const float* Aarray[], int lda,
                        const float* Barray[], int ldb,
    const float* beta,        float* Carray[], int ldc,
    int batchCount)
{
    assert(0);
}

cublasStatus_t cublasDgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double* alpha, const double* Aarray[], int lda,
                         const double* Barray[], int ldb,
    const double* beta,        double* Carray[], int ldc,
    int batchCount)
{
    assert(0);
}

cublasStatus_t cublasCgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const cuComplex* alpha, const cuComplex* Aarray[], int lda,
                            const cuComplex* Barray[], int ldb,
    const cuComplex* beta,        cuComplex* Carray[], int ldc,
    int batchCount)
{
    assert(0);
}

cublasStatus_t cublasZgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const cuDoubleComplex* alpha, const cuDoubleComplex* Aarray[], int lda,
                                  const cuDoubleComplex* Barray[], int ldb,
    const cuDoubleComplex* beta,        cuDoubleComplex* Carray[], int ldc,
    int batchCount)
{
    assert(0);
}

cublasStatus_t cublasCherk(
    cublasHandle_t handle,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int n, int k,
    const float *alpha, const cuComplex *A, int lda,
    const float *beta,        cuComplex *C, int ldc)
{
    assert(0);
}

cublasStatus_t cublasZherk(
    cublasHandle_t handle,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int n, int k,
    const double *alpha, const cuDoubleComplex *A, int lda,
    const double *beta,        cuDoubleComplex *C, int ldc)
{
    assert(0);
}


cublasStatus_t cublasCher2k(
    cublasHandle_t handle,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int n, int k,
    const cuComplex *alpha, const cuComplex *A, int lda,
                            const cuComplex *B, int ldb,
    const float *beta,            cuComplex *C, int ldc)
{
    assert(0);
}

cublasStatus_t cublasZher2k(
    cublasHandle_t handle,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int n, int k,
    const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
                                  const cuDoubleComplex *B, int ldb,
    const double *beta,                 cuDoubleComplex *C, int ldc)
{
    assert(0);
}

cublasStatus_t cublasSsyrk(
    cublasHandle_t handle,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int n, int k,
    const float *alpha, const float *A, int lda,
    const float *beta,        float *C, int ldc)
{
    assert(0);
}

cublasStatus_t cublasDsyrk(
    cublasHandle_t handle,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int n, int k,
    const double *alpha, const double *A, int lda,
    const double *beta,        double *C, int ldc)
{
    assert(0);
}

cublasStatus_t cublasSsyr2k(
    cublasHandle_t handle,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int n, int k,
    const float *alpha, const float *A, int lda,
                        const float *B, int ldb,
    const float *beta,        float *C, int ldc)
{
    assert(0);
}

cublasStatus_t cublasDsyr2k(
    cublasHandle_t handle,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int n, int k,
    const double *alpha, const double *A, int lda,
                         const double *B, int ldb,
    const double *beta,        double *C, int ldc)
{
    assert(0);
}

cublasStatus_t cublasStrsmBatched(
    cublasHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    cublasDiagType_t  diag,
    int m, int n,
    const float *alpha, const float *Aarray[], int lda,
                              float *Barray[], int ldb,
    int batchCount)
{
    assert(0);
}

cublasStatus_t cublasDtrsmBatched(
    cublasHandle_t handle,
    cublasSideMode_t  side,
    cublasFillMode_t  uplo,
    cublasOperation_t trans,
    cublasDiagType_t  diag,
    int m, int n,
    const double *alpha, const double *Aarray[], int lda,
                               double *Barray[], int ldb,
    int batchCount)
{
    assert(0);
}

cublasStatus_t cublasCtrsmBatched(
    cublasHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    cublasDiagType_t diag,
    int m, int n,
    const cuComplex *alpha, const cuComplex *Aarray[], int lda,
                                  cuComplex *Barray[], int ldb,
    int batchCount)
{
    assert(0);
}

cublasStatus_t cublasZtrsmBatched(
    cublasHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    cublasDiagType_t  diag,
    int m, int n,
    const cuDoubleComplex *alpha, const cuDoubleComplex *Aarray[], int lda,
                                        cuDoubleComplex *Barray[], int ldb,
    int batchCount)
{
    assert(0);
}


cublasStatus_t cublasSswap(
    cublasHandle_t handle,
    int n,
    float* x, int incx,
    float* y, int incy)
{
    assert(0);
}

cublasStatus_t cublasDswap(
    cublasHandle_t handle,
    int n,
    double* x, int incx,
    double* y, int incy)
{
    assert(0);
}

cublasStatus_t cublasCswap(
    cublasHandle_t handle,
    int n,
    cuComplex* x, int incx,
    cuComplex* y, int incy)
{
    assert(0);
}

cublasStatus_t cublasZswap(
    cublasHandle_t handle,
    int n,
    cuDoubleComplex* x, int incx,
    cuDoubleComplex* y, int incy)
{
    assert(0);
}

cublasStatus_t cublasGetMatrix(
    int rows, int cols, int elemSize,
    const void* A, int lda, void* B, int ldb)
{
    assert(0);
}

#ifdef __cplusplus
}
#endif
