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
