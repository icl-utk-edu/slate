// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Exception.hh"
#include "slate/internal/device.hh"

#include <cstdio>
#include <cmath>
#include <complex>

#include "device_util.hh"

namespace slate {
namespace device {

//------------------------------------------------------------------------------
/// Element-wise trapezoidal tile set.
/// Sets upper or lower part of Aarray[k] to
/// diag_value on the diagonal and offdiag_value on the off-diagonals.
///
/// @param[in] uplo
///     Whether each Aarray[k] is upper or lower trapezoidal.
///
/// @param[in] m
///     Number of rows of A. m >= 0.
///
/// @param[in] n
///     Number of columns of A. n >= 0.
///
/// @param[in] offdiag_value
///     Constant to set offdiagonal entries to.
///
/// @param[in] diag_value
///     Constant to set diagonal entries to.
///
/// @param[out] A
///     An m-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] lda
///     Leading dimension of A. lda >= m.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t>
void tzset(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    scalar_t const& offdiag_value, scalar_t const& diag_value,
    scalar_t* A, int64_t lda,
    blas::Queue& queue )
{
    queue.sync(); // sync queue before switching to openmp device execution
    // Use omp target offload
    #pragma omp target is_device_ptr(A) device(queue.device())
    // distribute rows (i) to threads
    #pragma omp teams distribute parallel for schedule(static, 1)
    for (int64_t i = 0; i < m; ++i) {
        scalar_t* rowA = &A[ i ];
        // lower or upper
        if (uplo == lapack::Uplo::Lower) {
            for (int64_t j = 0; j <= i && j < n; ++j) { // lower
                rowA[ j*lda ] = i == j ? diag_value : offdiag_value;
            }
        }
        else {
            for (int64_t j = n-1; j >= i; --j) { // upper
                rowA[ j*lda ] = i == j ? diag_value : offdiag_value;
            }
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void tzset(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    float const& offdiag_value, float const& diag_value,
    float* A, int64_t lda,
    blas::Queue& queue );

template
void tzset(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    double const& offdiag_value, double const& diag_value,
    double* A, int64_t lda,
    blas::Queue& queue );

template
void tzset(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<float> const& offdiag_value, std::complex<float> const& diag_value,
    std::complex<float>* A, int64_t lda,
    blas::Queue& queue );

template
void tzset(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<double> const& offdiag_value, std::complex<double> const& diag_value,
    std::complex<double>* A, int64_t lda,
    blas::Queue& queue );

//==============================================================================
namespace batch {

//------------------------------------------------------------------------------
/// Batched routine for element-wise trapezoidal tile set.
/// Sets upper or lower part of Aarray[k] to
/// diag_value on the diagonal and offdiag_value on the off-diagonals.
///
/// @param[in] m
///     Number of rows of each tile. m >= 0.
///
/// @param[in] n
///     Number of columns of each tile. n >= 0.
///
/// @param[in] offdiag_value
///     Constant to set offdiagonal entries to.
///
/// @param[in] diag_value
///     Constant to set diagonal entries to.
///
/// @param[out] Aarray
///     Array in GPU memory of dimension batch_count, containing
///     pointers to tiles, where each Aarray[k] is an m-by-n matrix
///     stored in an lda-by-n array in GPU memory.
///
/// @param[in] lda
///     Leading dimension of each tile in Aarray. lda >= m.
///
/// @param[in] batch_count
///     Size of Aarray. batch_count >= 0.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t>
void tzset(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    scalar_t const& offdiag_value, scalar_t const& diag_value,
    scalar_t** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue )
{
    // quick return
    if (batch_count == 0)
        return;

    queue.sync(); // sync queue before switching to openmp device execution
    // Use omp target offload
    #pragma omp target is_device_ptr(Aarray) device(queue.device())
    #pragma omp teams distribute
    for (int64_t k = 0; k < batch_count; ++k) {
        scalar_t* A = Aarray[ k ];
        // distribute rows (i) to threads
        #pragma omp parallel for schedule(static, 1)
        for (int64_t i = 0; i < m; ++i) {
            scalar_t* rowA = &A[ i ];
            // lower or upper
            if (uplo == lapack::Uplo::Lower) {
                for (int64_t j = 0; j <= i && j < n; ++j) { // lower
                    rowA[ j*lda ] = i == j ? diag_value : offdiag_value;
                }
            }
            else {
                for (int64_t j = n-1; j >= i; --j) { // upper
                    rowA[ j*lda ] = i == j ? diag_value : offdiag_value;
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void tzset(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    float const& offdiag_value, float const& diag_value,
    float** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue );

template
void tzset(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    double const& offdiag_value, double const& diag_value,
    double** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue );

template
void tzset(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<float> const& offdiag_value, std::complex<float> const& diag_value,
    std::complex<float>** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue );

template
void tzset(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<double> const& offdiag_value, std::complex<double> const& diag_value,
    std::complex<double>** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue );

} // namespace batch
} // namespace device
} // namespace slate
