//------------------------------------------------------------------------------
// Copyright (c) 2017, University of Tennessee
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the University of Tennessee nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF TENNESSEE BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//------------------------------------------------------------------------------
// This research was supported by the Exascale Computing Project (17-SC-20-SC),
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
//------------------------------------------------------------------------------
// For assistance with SLATE, email <slate-user@icl.utk.edu>.
// You can also join the "SLATE User" Google group by going to
// https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user,
// signing in with your Google credentials, and then clicking "Join group".
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
/// @file
///
#ifndef SLATE_CUBLAS_HH
#define SLATE_CUBLAS_HH

#ifndef SLATE_NO_CUDA
    #include <cublas_v2.h>
#else

#include "slate/internal/cuda.hh"

#include <complex>

typedef void* cublasHandle_t;

typedef int cublasOperation_t;
typedef int cublasFillMode_t;
typedef int cublasSideMode_t;
typedef int cublasDiagType_t;

enum {
    CUBLAS_OP_C,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
};

enum {
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_FILL_MODE_UPPER,
};

enum {
    CUBLAS_SIDE_LEFT,
    CUBLAS_SIDE_RIGHT,
};

enum {
    CUBLAS_DIAG_NON_UNIT,
    CUBLAS_DIAG_UNIT,
};

enum cublasStatus_t {
    CUBLAS_STATUS_SUCCESS,
    CUBLAS_STATUS_NOT_INITIALIZED,
    CUBLAS_STATUS_ALLOC_FAILED,
    CUBLAS_STATUS_INVALID_VALUE,
    CUBLAS_STATUS_ARCH_MISMATCH,
    CUBLAS_STATUS_MAPPING_ERROR,
    CUBLAS_STATUS_EXECUTION_FAILED,
    CUBLAS_STATUS_INTERNAL_ERROR,
    CUBLAS_STATUS_NOT_SUPPORTED,
    CUBLAS_STATUS_LICENSE_ERROR,
};

#ifdef __cplusplus
extern "C" {
#endif

cublasStatus_t cublasCreate(cublasHandle_t* handle);
cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId);
cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t* streamId);
cublasStatus_t cublasDasum(
    cublasHandle_t handle, int n, const double* x, int incx, double* result);
cublasStatus_t cublasDestroy(cublasHandle_t handle);

cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize,
                               const void* A, int lda, void* B, int ldb);

cublasStatus_t cublasSgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha, const float* Aarray[], int lda,
                        const float* Barray[], int ldb,
    const float* beta,        float* Carray[], int ldc,
    int batchCount);

cublasStatus_t cublasDgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double* alpha, const double* Aarray[], int lda,
                         const double* Barray[], int ldb,
    const double* beta,        double* Carray[], int ldc,
    int batchCount);

cublasStatus_t cublasCgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const cuComplex* alpha, const cuComplex* Aarray[], int lda,
                            const cuComplex* Barray[], int ldb,
    const cuComplex* beta,        cuComplex* Carray[], int ldc,
    int batchCount);

cublasStatus_t cublasZgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const cuDoubleComplex* alpha, const cuDoubleComplex* Aarray[], int lda,
                                  const cuDoubleComplex* Barray[], int ldb,
    const cuDoubleComplex* beta,        cuDoubleComplex* Carray[], int ldc,
    int batchCount);

cublasStatus_t cublasCherk(
    cublasHandle_t handle,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int n, int k,
    const float *alpha, const cuComplex *A, int lda,
    const float *beta,        cuComplex *C, int ldc);

cublasStatus_t cublasZherk(
    cublasHandle_t handle,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int n, int k,
    const double *alpha, const cuDoubleComplex *A, int lda,
    const double *beta,        cuDoubleComplex *C, int ldc);

cublasStatus_t cublasCher2k(
    cublasHandle_t handle,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int n, int k,
    const cuComplex *alpha, const cuComplex *A, int lda,
                            const cuComplex *B, int ldb,
    const float *beta,            cuComplex *C, int ldc);

cublasStatus_t cublasZher2k(
    cublasHandle_t handle,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int n, int k,
    const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
                                  const cuDoubleComplex *B, int ldb,
    const double *beta,                 cuDoubleComplex *C, int ldc);

cublasStatus_t cublasSsyrk(
    cublasHandle_t handle,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int n, int k,
    const float *alpha, const float *A, int lda,
    const float *beta,        float *C, int ldc);

cublasStatus_t cublasDsyrk(
    cublasHandle_t handle,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int n, int k,
    const double *alpha, const double *A, int lda,
    const double *beta,        double *C, int ldc);

/* SYR2K */
cublasStatus_t cublasSsyr2k(
    cublasHandle_t handle,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int n, int k,
    const float *alpha, const float *A, int lda,
                        const float *B, int ldb,
    const float *beta,        float *C, int ldc);

cublasStatus_t cublasDsyr2k(
    cublasHandle_t handle,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int n, int k,
    const double *alpha, const double *A, int lda,
                         const double *B, int ldb,
    const double *beta,        double *C, int ldc);

cublasStatus_t cublasStrsmBatched(
    cublasHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    cublasDiagType_t  diag,
    int m, int n,
    const float *alpha, const float *Aarray[], int lda,
                              float *Barray[], int ldb,
    int batchCount);

cublasStatus_t cublasDtrsmBatched(
    cublasHandle_t handle,
    cublasSideMode_t  side,
    cublasFillMode_t  uplo,
    cublasOperation_t trans,
    cublasDiagType_t  diag,
    int m, int n,
    const double *alpha, const double *Aarray[], int lda,
                               double *Barray[], int ldb,
    int batchCount);

cublasStatus_t cublasCtrsmBatched(
    cublasHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    cublasDiagType_t diag,
    int m, int n,
    const cuComplex *alpha, const cuComplex *Aarray[], int lda,
                                  cuComplex *Barray[], int ldb,
    int batchCount);

cublasStatus_t cublasZtrsmBatched(
    cublasHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    cublasDiagType_t  diag,
    int m, int n,
    const cuDoubleComplex *alpha, const cuDoubleComplex *Aarray[], int lda,
                                        cuDoubleComplex *Barray[], int ldb,
    int batchCount);

cublasStatus_t cublasSswap(
    cublasHandle_t handle,
    int n,
    float* x, int incx,
    float* y, int incy);

cublasStatus_t cublasDswap(
    cublasHandle_t handle,
    int n,
    double* x, int incx,
    double* y, int incy);

cublasStatus_t cublasCswap(
    cublasHandle_t handle,
    int n,
    cuComplex *x, int incx,
    cuComplex *y, int incy);

cublasStatus_t cublasZswap(
    cublasHandle_t handle,
    int n,
    cuDoubleComplex *x, int incx,
    cuDoubleComplex *y, int incy);

cublasStatus_t cublasGetMatrix(
    int rows, int cols, int elemSize,
    const void* A, int lda, void* B, int ldb);

#ifdef __cplusplus
}
#endif

#endif // SLATE_NO_CUDA

#endif // SLATE_CUBLAS_HH
