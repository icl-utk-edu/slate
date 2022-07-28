// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

//------------------------------------------------------------------------------
/// @file
/// Provides simple precision-independent wrappers around MKL batch
/// routines. Eventually to be replaced by BLAS++ batch routines.
#ifndef SLATE_INTERNAL_BATCH_HH
#define SLATE_INTERNAL_BATCH_HH

#include "slate/Exception.hh"

#include <blas.hh>

#ifdef BLAS_HAVE_MKL
    #include <mkl_cblas.h>
#endif

#include <complex>
#include <set>

namespace slate {
namespace internal {

#ifdef BLAS_HAVE_MKL

//------------------------------------------------------------------------------
inline CBLAS_TRANSPOSE cblas_trans_const(Op op)
{
    switch (op) {
        case Op::NoTrans:   return CblasNoTrans;
        case Op::Trans:     return CblasTrans;
        case Op::ConjTrans: return CblasConjTrans;
        default: slate_error("unknown op");
    }
}

//------------------------------------------------------------------------------
inline void cblas_gemm_batch(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE* transA_array,
    const CBLAS_TRANSPOSE* transB_array,
    const int* m_array,
    const int* n_array,
    const int* k_array,
    const float* alpha_array,
    const float** A_array,
    const int* lda_array,
    const float** B_array,
    const int* ldb_array,
    const float* beta_array,
    float** C_array,
    const int* ldc_array,
    const int group_count,
    const int* group_size)
{
    cblas_sgemm_batch(layout, transA_array, transB_array,
                      m_array, n_array, k_array,
                      alpha_array, A_array, lda_array,
                                   B_array, ldb_array,
                      beta_array,  C_array, ldc_array,
                      group_count, group_size);
}

//------------------------------------------------------------------------------
inline void cblas_gemm_batch(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE* transA_array,
    const CBLAS_TRANSPOSE* transB_array,
    const int* m_array,
    const int* n_array,
    const int* k_array,
    const double* alpha_array,
    const double** A_array,
    const int* lda_array,
    const double** B_array,
    const int* ldb_array,
    const double* beta_array,
    double** C_array,
    const int* ldc_array,
    const int group_count,
    const int* group_size)
{
    cblas_dgemm_batch(layout, transA_array, transB_array,
                      m_array, n_array, k_array,
                      alpha_array, A_array, lda_array,
                                   B_array, ldb_array,
                      beta_array,  C_array, ldc_array,
                      group_count, group_size);
}

//------------------------------------------------------------------------------
inline void cblas_gemm_batch(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE* transA_array,
    const CBLAS_TRANSPOSE* transB_array,
    const int* m_array,
    const int* n_array,
    const int* k_array,
    const std::complex<float>* alpha_array,
    const std::complex<float>** A_array,
    const int* lda_array,
    const std::complex<float>** B_array,
    const int* ldb_array,
    const std::complex<float>* beta_array,
    std::complex<float>** C_array,
    const int* ldc_array,
    const int group_count,
    const int* group_size)
{
    cblas_cgemm_batch(layout, transA_array, transB_array,
                      m_array, n_array, k_array,
                      alpha_array, (const void**) A_array, lda_array,
                                   (const void**) B_array, ldb_array,
                      beta_array,  (void**)       C_array, ldc_array,
                      group_count, group_size);
}

//------------------------------------------------------------------------------
inline void cblas_gemm_batch(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE* transA_array,
    const CBLAS_TRANSPOSE* transB_array,
    const int* m_array,
    const int* n_array,
    const int* k_array,
    const std::complex<double>* alpha_array,
    const std::complex<double>** A_array,
    const int* lda_array,
    const std::complex<double>** B_array,
    const int* ldb_array,
    const std::complex<double>* beta_array,
    std::complex<double>** C_array,
    const int* ldc_array,
    const int group_count,
    const int* group_size)
{
    cblas_zgemm_batch(layout, transA_array, transB_array,
                      m_array, n_array, k_array,
                      alpha_array, (const void**) A_array, lda_array,
                                   (const void**) B_array, ldb_array,
                      beta_array,  (void**)       C_array, ldc_array,
                      group_count, group_size);
}
#endif // BLAS_HAVE_MKL

} // namespace slate
} // namespace internal

#endif // SLATE_INTERNAL_BATCH_HH
