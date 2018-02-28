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
// Need assistance with the SLATE software? Join the "SLATE User" Google group
// by going to https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user
// and clicking "Apply to join group". Upon acceptance, email your questions and
// comments to <slate-user@icl.utk.edu>.
//------------------------------------------------------------------------------

///-----------------------------------------------------------------------------
/// \file
/// Provides simple precision-independent wrappers around MKL and cuBLAS batch
/// routines. Eventually to be replaced by blaspp batch routines.
#ifndef SLATE_INTERNAL_BATCH_HH
#define SLATE_INTERNAL_BATCH_HH

#ifdef SLATE_WITH_MKL
    #include <mkl_cblas.h>
#else
    #include <cblas.h>
#endif

#include <cublas_v2.h>

#include <complex>

namespace slate {
namespace internal {

#ifdef SLATE_WITH_MKL
///-----------------------------------------------------------------------------
inline void cblas_gemm_batch(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE *transA_array,
    const CBLAS_TRANSPOSE *transB_array,
    const int *m_array,
    const int *n_array,
    const int *k_array,
    const float *alpha_array,
    const float **A_array,
    const int *lda_array,
    const float **B_array,
    const int *ldb_array,
    const float *beta_array,
    float **C_array,
    const int *ldc_array,
    const int group_count,
    const int *group_size)
{
    cblas_sgemm_batch( layout, transA_array, transB_array,
                       m_array, n_array, k_array,
                       alpha_array, A_array, lda_array,
                                    B_array, ldb_array,
                       beta_array,  C_array, ldc_array,
                       group_count, group_size );
}

///-----------------------------------------------------------------------------
inline void cblas_gemm_batch(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE *transA_array,
    const CBLAS_TRANSPOSE *transB_array,
    const int *m_array,
    const int *n_array,
    const int *k_array,
    const double *alpha_array,
    const double **A_array,
    const int *lda_array,
    const double **B_array,
    const int *ldb_array,
    const double *beta_array,
    double **C_array,
    const int *ldc_array,
    const int group_count,
    const int *group_size)
{
    cblas_dgemm_batch( layout, transA_array, transB_array,
                       m_array, n_array, k_array,
                       alpha_array, A_array, lda_array,
                                    B_array, ldb_array,
                       beta_array,  C_array, ldc_array,
                       group_count, group_size );
}

///-----------------------------------------------------------------------------
inline void cblas_gemm_batch(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE *transA_array,
    const CBLAS_TRANSPOSE *transB_array,
    const int *m_array,
    const int *n_array,
    const int *k_array,
    const std::complex<float> *alpha_array,
    const std::complex<float> **A_array,
    const int *lda_array,
    const std::complex<float> **B_array,
    const int *ldb_array,
    const std::complex<float> *beta_array,
    std::complex<float> **C_array,
    const int *ldc_array,
    const int group_count,
    const int *group_size)
{
    cblas_cgemm_batch( layout, transA_array, transB_array,
                       m_array, n_array, k_array,
                       alpha_array, (const void**) A_array, lda_array,
                                    (const void**) B_array, ldb_array,
                       beta_array,  (void**)       C_array, ldc_array,
                       group_count, group_size );
}

///-----------------------------------------------------------------------------
inline void cblas_gemm_batch(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE *transA_array,
    const CBLAS_TRANSPOSE *transB_array,
    const int *m_array,
    const int *n_array,
    const int *k_array,
    const std::complex<double> *alpha_array,
    const std::complex<double> **A_array,
    const int *lda_array,
    const std::complex<double> **B_array,
    const int *ldb_array,
    const std::complex<double> *beta_array,
    std::complex<double> **C_array,
    const int *ldc_array,
    const int group_count,
    const int *group_size)
{
    cblas_zgemm_batch( layout, transA_array, transB_array,
                       m_array, n_array, k_array,
                       alpha_array, (const void**) A_array, lda_array,
                                    (const void**) B_array, ldb_array,
                       beta_array,  (void**)       C_array, ldc_array,
                       group_count, group_size );
}
#endif // SLATE_WITH_MKL

///-----------------------------------------------------------------------------
inline cublasStatus_t cublasGemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha,  /* host or device pointer */
    const float *Aarray[],
    int lda,
    const float *Barray[],
    int ldb,
    const float *beta,   /* host or device pointer */
    float *Carray[],
    int ldc,
    int batchCount)
{
    return cublasSgemmBatched( handle, transa, transb, m, n, k,
                               alpha, Aarray, lda,
                                      Barray, ldb,
                               beta,  Carray, ldc,
                               batchCount );
}

///-----------------------------------------------------------------------------
inline cublasStatus_t cublasGemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const double *alpha,  /* host or device pointer */
    const double *Aarray[],
    int lda,
    const double *Barray[],
    int ldb,
    const double *beta,   /* host or device pointer */
    double *Carray[],
    int ldc,
    int batchCount)
{
    return cublasDgemmBatched( handle, transa, transb, m, n, k,
                               alpha, Aarray, lda,
                                      Barray, ldb,
                               beta,  Carray, ldc,
                               batchCount );
}

///-----------------------------------------------------------------------------
inline cublasStatus_t cublasGemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const std::complex<float> *alpha,  /* host or device pointer */
    const std::complex<float> *Aarray[],
    int lda,
    const std::complex<float> *Barray[],
    int ldb,
    const std::complex<float> *beta,   /* host or device pointer */
    std::complex<float> *Carray[],
    int ldc,
    int batchCount)
{
    return cublasCgemmBatched( handle, transa, transb, m, n, k,
                               (cuComplex*)  alpha,
                               (const cuComplex**) Aarray, lda,
                               (const cuComplex**) Barray, ldb,
                               (cuComplex*)  beta,
                               (cuComplex**) Carray, ldc,
                               batchCount );
}

///-----------------------------------------------------------------------------
inline cublasStatus_t cublasGemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const std::complex<double> *alpha,  /* host or device pointer */
    const std::complex<double> *Aarray[],
    int lda,
    const std::complex<double> *Barray[],
    int ldb,
    const std::complex<double> *beta,   /* host or device pointer */
    std::complex<double> *Carray[],
    int ldc,
    int batchCount)
{
    return cublasZgemmBatched( handle, transa, transb, m, n, k,
                               (cuDoubleComplex*)  alpha,
                               (const cuDoubleComplex**) Aarray, lda,
                               (const cuDoubleComplex**) Barray, ldb,
                               (cuDoubleComplex*)  beta,
                               (cuDoubleComplex**) Carray, ldc,
                               batchCount );
}

} // namespace slate
} // namespace internal

#endif        //  #ifndef SLATE_INTERNAL_BATCH_HH
