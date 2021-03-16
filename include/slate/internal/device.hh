// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_DEVICE_HH
#define SLATE_DEVICE_HH

#include "slate/enums.hh"

#ifndef SLATE_NO_CUDA
#include <cuComplex.h>

//------------------------------------------------------------------------------
// Extend BLAS real_type to cover cuComplex
// todo: should we move it to BLAS++ or have a version for rocBLAS
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

#endif // #ifndef SLATE_NO_CUDA

#ifndef SLATE_NO_HIP
#include <hip/hip_complex.h>

namespace blas {

template<>
struct real_type_traits<hipFloatComplex> {
    using real_t = float;
};

template<>
struct real_type_traits<hipDoubleComplex> {
    using real_t = double;
};

} // namespace blas

#endif // #ifndef SLATE_NO_HIP

namespace slate {

/// @namespace slate::device
/// GPU device implementations of kernels.
namespace device {

//------------------------------------------------------------------------------
template <typename src_scalar_t, typename dst_scalar_t>
void gecopy(
    int64_t m, int64_t n,
    src_scalar_t** Aarray, int64_t lda,
    dst_scalar_t** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue& queue);

//------------------------------------------------------------------------------
template <typename src_scalar_t, typename dst_scalar_t>
void tzcopy(
    Uplo uplo,
    int64_t m, int64_t n,
    src_scalar_t** Aarray, int64_t lda,
    dst_scalar_t** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue& queue);

//------------------------------------------------------------------------------
template <typename scalar_t>
void geadd(
    int64_t m, int64_t n,
    scalar_t alpha, scalar_t** Aarray, int64_t lda,
    scalar_t beta, scalar_t** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue& queue);

//------------------------------------------------------------------------------
template <typename scalar_t>
void geset(
    int64_t m, int64_t n,
    scalar_t alpha, scalar_t beta, scalar_t** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

//------------------------------------------------------------------------------
template <typename scalar_t>
void genorm(
    Norm norm, NormScope scope,
    int64_t m, int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* values, int64_t ldv,
    int64_t batch_count, blas::Queue& queue);

//------------------------------------------------------------------------------
template <typename scalar_t>
void henorm(
    Norm norm, Uplo uplo,
    int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* values, int64_t ldv,
    int64_t batch_count, blas::Queue& queue);

//------------------------------------------------------------------------------
template <typename scalar_t>
void synorm(
    Norm norm, Uplo uplo,
    int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* values, int64_t ldv,
    int64_t batch_count, blas::Queue& queue);

//------------------------------------------------------------------------------
template <typename scalar_t>
void synormOffdiag(
    Norm norm,
    int64_t m, int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* values, int64_t ldv,
    int64_t batch_count, blas::Queue& queue);

//------------------------------------------------------------------------------
template <typename scalar_t>
void trnorm(
    Norm norm, Uplo uplo, Diag diag,
    int64_t m, int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* values, int64_t ldv,
    int64_t batch_count, blas::Queue& queue);

//------------------------------------------------------------------------------
template <typename scalar_t>
void transpose(
    int64_t n,
    scalar_t* A, int64_t lda, blas::Queue& queue);

//------------------------------------------------------------------------------
template <typename scalar_t>
void transpose_batch(
    int64_t n,
    scalar_t** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

//------------------------------------------------------------------------------
template <typename scalar_t>
void transpose(
    int64_t m, int64_t n,
    scalar_t* dA,  int64_t lda,
    scalar_t* dAT, int64_t ldat, blas::Queue& queue);

//------------------------------------------------------------------------------
template <typename scalar_t>
void transpose_batch(
    int64_t m, int64_t n,
    scalar_t** dA_array,  int64_t lda,
    scalar_t** dAT_array, int64_t ldat,
    int64_t batch_count, blas::Queue& queue);

} // namespace device
} // namespace slate

#endif // SLATE_DEVICE_HH
