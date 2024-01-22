// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_DEVICE_HH
#define SLATE_DEVICE_HH

#include "slate/enums.hh"

//------------------------------------------------------------------------------
// Extend BLAS real_type to cover cuComplex and hipComplex.
// todo: should we move it to BLAS++?
//
#if defined( BLAS_HAVE_CUBLAS )
    #include <cuComplex.h>

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

#elif defined( BLAS_HAVE_ROCBLAS )
    // Default to AMD platform on ROCm.
    #if ! defined(__HIP_PLATFORM_NVIDIA__) && ! defined(__HIP_PLATFORM_NVCC__) \
        && ! defined(__HIP_PLATFORM_AMD__) && ! defined(__HIP_PLATFORM_HCC__)
        #define __HIP_PLATFORM_AMD__
        #define SLATE_HIP_PLATFORM_AMD
    #endif

    #include <hip/hip_complex.h>

    // If we defined __HIP_PLATFORM_AMD__, undef it.
    #ifdef SLATE_HIP_PLATFORM_AMD
        #undef __HIP_PLATFORM_AMD__
        #undef SLATE_HIP_PLATFORM_AMD
    #endif

    namespace blas {

    template<>
    struct real_type_traits<rocblas_float_complex> {
        using real_t = float;
    };

    template<>
    struct real_type_traits<rocblas_double_complex> {
        using real_t = double;
    };

    template<>
    struct real_type_traits<hipFloatComplex> {
        using real_t = float;
    };

    template<>
    struct real_type_traits<hipDoubleComplex> {
        using real_t = double;
    };

    } // namespace blas

#elif defined( BLAS_HAVE_SYCL )
    #include <sycl/sycl.hpp>
    namespace blas {

    template <typename T>
    struct blas::real_type_traits< sycl::vec<T, 2> > {
        using real_t = T;
    };

    } // namespace blas

#endif // #defined( BLAS_HAVE_{CUBLAS,ROCBLAS,SYCL} )

namespace slate {

/// @namespace slate::device
/// GPU device implementations of kernels.
namespace device {

// Use when SYCL and oneMKL are used
#if defined( BLAS_HAVE_SYCL )
    // todo: make this build automatically
    // Manually uncomment to compile OMP target-offload kernels
    // #define SLATE_HAVE_OMPTARGET
#endif

// Simplify checking for GPU device support (CUDA / ROCm / SYCL).
#if defined( BLAS_HAVE_CUBLAS ) \
    || defined( BLAS_HAVE_ROCBLAS ) \
    || defined( BLAS_HAVE_SYCL )
    #define SLATE_HAVE_DEVICE
#endif

//------------------------------------------------------------------------------
template <typename src_scalar_t, typename dst_scalar_t>
void gecopy(
    int64_t m, int64_t n,
    src_scalar_t const* const* Aarray, int64_t lda,
    dst_scalar_t** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue& queue);

//------------------------------------------------------------------------------
template <typename src_scalar_t, typename dst_scalar_t>
void tzcopy(
    Uplo uplo,
    int64_t m, int64_t n,
    src_scalar_t const* const* Aarray, int64_t lda,
    dst_scalar_t** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue& queue);

//------------------------------------------------------------------------------
template <typename scalar_t>
void geadd(
    int64_t m, int64_t n,
    scalar_t const& alpha, scalar_t* A, int64_t lda,
    scalar_t const& beta, scalar_t* B, int64_t ldb,
    blas::Queue& queue);

//------------------------------------------------------------------------------
template <typename scalar_t>
void tzadd(
     Uplo uplo,
     int64_t m, int64_t n,
     scalar_t const& alpha, scalar_t** Aarray, int64_t lda,
     scalar_t const& beta, scalar_t** Barray, int64_t ldb,
     int64_t batch_count, blas::Queue& queue);

//------------------------------------------------------------------------------
template <typename scalar_t, typename scalar_t2>
void gescale(
    int64_t m, int64_t n,
    scalar_t2 numer, scalar_t2 denom,
    scalar_t* A, int64_t lda,
    blas::Queue& queue);

//------------------------------------------------------------------------------
template <typename scalar_t, typename scalar_t2>
void gescale_row_col_batch(
    Equed equed, int64_t m, int64_t n,
    scalar_t2 const* const* Rarray,
    scalar_t2 const* const* Carray,
    scalar_t** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

//------------------------------------------------------------------------------
template <typename scalar_t>
void geset(
    int64_t m, int64_t n,
    scalar_t const& offdiag_value, scalar_t const& diag_value,
    scalar_t* A, int64_t lda,
    blas::Queue& queue);

//------------------------------------------------------------------------------
template <typename scalar_t>
void tzset(
    Uplo uplo,
    int64_t m, int64_t n,
    scalar_t const& offdiag_value, scalar_t const& diag_value,
    scalar_t* A, int64_t lda,
    blas::Queue& queue );

namespace batch {

//------------------------------------------------------------------------------
template <typename scalar_t, typename scalar_t2>
void gescale(
    int64_t m, int64_t n,
    scalar_t2 numer, scalar_t2 denom,
    scalar_t** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

//------------------------------------------------------------------------------
template <typename scalar_t>
void tzscale(
    Uplo uplo,
    int64_t m, int64_t n,
    blas::real_type<scalar_t> numer, blas::real_type<scalar_t> denom,
    scalar_t** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

//------------------------------------------------------------------------------
template <typename scalar_t>
void geadd(
    int64_t m, int64_t n,
    scalar_t const& alpha, scalar_t** Aarray, int64_t lda,
    scalar_t const& beta, scalar_t** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue& queue);


//------------------------------------------------------------------------------
template <typename scalar_t>
void geset(
    int64_t m, int64_t n,
    scalar_t const& offdiag_value, scalar_t const& diag_value,
    scalar_t** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

//------------------------------------------------------------------------------
template <typename scalar_t>
void tzset(
    Uplo uplo,
    int64_t m, int64_t n,
    scalar_t const& offdiag_value, scalar_t const& diag_value,
    scalar_t** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue );

} // namespace batch

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
// In-place, square.
template <typename scalar_t>
void transpose(
    bool is_conj,
    int64_t n,
    scalar_t* A, int64_t lda, blas::Queue& queue);

template <typename scalar_t>
void transpose_batch(
    bool is_conj,
    int64_t n,
    scalar_t** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

//------------------------------------------------------------------------------
// Out-of-place.
template <typename scalar_t>
void transpose(
    bool is_conj,
    int64_t m, int64_t n,
    scalar_t* dA,  int64_t lda,
    scalar_t* dAT, int64_t ldat, blas::Queue& queue);

template <typename scalar_t>
void transpose_batch(
    bool is_conj,
    int64_t m, int64_t n,
    scalar_t** dA_array,  int64_t lda,
    scalar_t** dAT_array, int64_t ldat,
    int64_t batch_count, blas::Queue& queue);

} // namespace device
} // namespace slate

#endif // SLATE_DEVICE_HH
