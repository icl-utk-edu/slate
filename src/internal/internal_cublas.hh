// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

//------------------------------------------------------------------------------
/// @file
/// Provides simple precision-independent wrappers around MKL and cuBLAS batch
/// routines. Eventually to be replaced by blaspp batch routines.
#ifndef SLATE_INTERNAL_CUBLAS_HH
#define SLATE_INTERNAL_CUBLAS_HH

#include "slate/internal/cuda.hh"
#include "slate/internal/cublas.hh"

#include <complex>

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

/* HERK */
inline cublasStatus_t cublasHerk (cublasHandle_t handle,
                                  cublasFillMode_t uplo,
                                  cublasOperation_t trans,
                                  int n,
                                  int k,
                                  const float *alpha,  /* host or device pointer */
                                  const float *A,
                                  int lda,
                                  const float *beta,   /* host or device pointer */
                                  float *C,
                                  int ldc)
{
    return cublasSsyrk (handle, uplo, trans,
                        n, k,
                        alpha, A, lda,
                        beta,  C, ldc);
}

inline cublasStatus_t cublasHerk (cublasHandle_t handle,
                                  cublasFillMode_t uplo,
                                  cublasOperation_t trans,
                                  int n,
                                  int k,
                                  const double *alpha,  /* host or device pointer */
                                  const double *A,
                                  int lda,
                                  const double *beta,  /* host or device pointer */
                                  double *C,
                                  int ldc)
{
    return cublasDsyrk (handle, uplo, trans,
                        n, k,
                        alpha, A, lda,
                        beta,  C, ldc);
}

inline cublasStatus_t cublasHerk (cublasHandle_t handle,
                                  cublasFillMode_t uplo,
                                  cublasOperation_t trans,
                                  int n,
                                  int k,
                                  const float *alpha,  /* host or device pointer */
                                  const std::complex<float> *A,
                                  int lda,
                                  const float *beta,   /* host or device pointer */
                                  std::complex<float> *C,
                                  int ldc)
{
    return cublasCherk (handle, uplo, trans,
                        n, k,
                        alpha, (cuComplex*) A, lda,
                        beta,  (cuComplex*) C, ldc);
}

inline cublasStatus_t cublasHerk (cublasHandle_t handle,
                                  cublasFillMode_t uplo,
                                  cublasOperation_t trans,
                                  int n,
                                  int k,
                                  const double *alpha,  /* host or device pointer */
                                  const std::complex<double> *A,
                                  int lda,
                                  const double *beta,  /* host or device pointer */
                                  std::complex<double> *C,
                                  int ldc)
{
    return cublasZherk (handle, uplo, trans,
                        n, k,
                        alpha, (cuDoubleComplex*)A, lda,
                        beta,  (cuDoubleComplex*)C, ldc);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

/* HER2K */
inline
cublasStatus_t cublasHer2k ( cublasHandle_t handle,
                             cublasFillMode_t uplo,
                             cublasOperation_t trans,
                             int n,
                             int k,
                             const float *alpha, /* host or device pointer */
                             const float *A,
                             int lda,
                             const float *B,
                             int ldb,
                             const float *beta, /* host or device pointer */
                             float *C,
                             int ldc)
{
    return cublasSsyr2k(handle, uplo, trans,
                        n, k,
                        alpha, A, lda,
                               B, ldb,
                        beta,  C, ldc);
}

inline
cublasStatus_t cublasHer2k ( cublasHandle_t handle,
                             cublasFillMode_t uplo,
                             cublasOperation_t trans,
                             int n,
                             int k,
                             const double *alpha, /* host or device pointer */
                             const double *A,
                             int lda,
                             const double *B,
                             int ldb,
                             const double *beta, /* host or device pointer */
                             double *C,
                             int ldc)
{
    return cublasDsyr2k(handle, uplo, trans,
                        n, k,
                        alpha, A, lda,
                               B, ldb,
                        beta,  C, ldc);
}

inline
cublasStatus_t cublasHer2k ( cublasHandle_t handle,
                             cublasFillMode_t uplo,
                             cublasOperation_t trans,
                             int n,
                             int k,
                             const std::complex<float> *alpha, /* host or device pointer */
                             const std::complex<float> *A,
                             int lda,
                             const std::complex<float> *B,
                             int ldb,
                             const float *beta,   /* host or device pointer */
                             std::complex<float> *C,
                             int ldc)
{
    return cublasCher2k (handle, uplo, trans,
                         n, k,
                         (cuComplex*) alpha,
                                (cuComplex*) A, lda,
                                (cuComplex*) B, ldb,
                         beta,  (cuComplex*) C, ldc);
}

inline
cublasStatus_t cublasHer2k ( cublasHandle_t handle,
                             cublasFillMode_t uplo,
                             cublasOperation_t trans,
                             int n,
                             int k,
                             const std::complex<double> *alpha, /* host or device pointer */
                             const std::complex<double> *A,
                             int lda,
                             const std::complex<double> *B,
                             int ldb,
                             const double *beta, /* host or device pointer */
                             std::complex<double> *C,
                             int ldc)
{
    return cublasZher2k (handle, uplo, trans,
                         n, k,
                         (cuDoubleComplex*)alpha,
                                (cuDoubleComplex*) A, lda,
                                (cuDoubleComplex*) B, ldb,
                         beta,  (cuDoubleComplex*) C, ldc);
}


} // namespace slate
} // namespace internal

#endif // SLATE_INTERNAL_CUBLAS_HH
