// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/internal/device.hh"

namespace slate {

namespace device {

//------------------------------------------------------------------------------
#if ! defined( SLATE_HAVE_DEVICE )

// Stubs to allow compilation without CUDA or HIP.
template <>
void gescale_row_col_batch(
    Equed equed, int64_t m, int64_t n,
    float const* const* Rarray,
    float const* const* Carray,
    float** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue)
{
}

template <>
void gescale_row_col_batch(
    Equed equed, int64_t m, int64_t n,
    double const* const* Rarray,
    double const* const* Carray,
    double** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue)
{
}

#endif // not SLATE_HAVE_DEVICE

//------------------------------------------------------------------------------
// Wrappers to map std::complex to cuComplex and hipComplex.

#if defined( BLAS_HAVE_CUBLAS )
    typedef cuFloatComplex devFloatComplex;
    typedef cuDoubleComplex devDoubleComplex;
#elif defined( BLAS_HAVE_ROCBLAS )
    typedef hipFloatComplex devFloatComplex;
    typedef hipDoubleComplex devDoubleComplex;
#endif

//----------------------------------------
// real R, C
template <>
void gescale_row_col_batch(
    Equed equed, int64_t m, int64_t n,
    float const* const* Rarray,
    float const* const* Carray,
    std::complex<float>** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue)
{
#if defined( SLATE_HAVE_DEVICE )
    gescale_row_col_batch(
        equed, m, n,
        Rarray,
        Carray,
        (devFloatComplex**) Aarray, lda,
        batch_count, queue);
#endif
}

template <>
void gescale_row_col_batch(
    Equed equed, int64_t m, int64_t n,
    double const* const* Rarray,
    double const* const* Carray,
    std::complex<double>** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue)
{
#if defined( SLATE_HAVE_DEVICE )
    gescale_row_col_batch(
        equed, m, n,
        Rarray,
        Carray,
        (devDoubleComplex**) Aarray, lda,
        batch_count, queue);
#endif
}

//----------------------------------------
// complex R, C
template <>
void gescale_row_col_batch(
    Equed equed, int64_t m, int64_t n,
    std::complex<float> const* const* Rarray,
    std::complex<float> const* const* Carray,
    std::complex<float>** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue)
{
#if defined( SLATE_HAVE_DEVICE )
    gescale_row_col_batch(
        equed, m, n,
        (devFloatComplex**) Rarray,
        (devFloatComplex**) Carray,
        (devFloatComplex**) Aarray, lda,
        batch_count, queue);
#endif
}

template <>
void gescale_row_col_batch(
    Equed equed, int64_t m, int64_t n,
    std::complex<double> const* const* Rarray,
    std::complex<double> const* const* Carray,
    std::complex<double>** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue)
{
#if defined( SLATE_HAVE_DEVICE )
    gescale_row_col_batch(
        equed, m, n,
        (devDoubleComplex**) Rarray,
        (devDoubleComplex**) Carray,
        (devDoubleComplex**) Aarray, lda,
        batch_count, queue);
#endif
}

} // namespace device

} // namespace slate
