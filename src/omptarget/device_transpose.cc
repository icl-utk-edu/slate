// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Exception.hh"
#include "slate/internal/device.hh"

#include <cstdio>

namespace slate {
namespace device {

//------------------------------------------------------------------------------
/// Device routine handles batches of square matrices.

///
/// The routine loads blocks of data into small ib x ib local storage
/// and then writes the blocks back transposed into the correct
/// location transposed.
///
template <typename scalar_t>
void transpose_sqr_batch_func(
    bool is_conj,
    int n,
    scalar_t** Aarray, int64_t lda,
    int batch_count, blas::Queue& queue)
{
    using blas::conj;
    static const int ib = 16;
    queue.sync(); // sync queue before switching to openmp device execution
    // i, j are row & column indices of top-left corner of each local block.
    #pragma omp target is_device_ptr(Aarray) device(queue.device())
    // Distribute the blocks to omp teams
    #pragma omp teams distribute collapse(2)
    for (int k = 0; k < batch_count; ++k) {
        for (int i = 0; i < n; i += ib) {
            for (int j = 0; j <= i; j += ib) {
                scalar_t *A = Aarray[k];
                // allocate local blocks for storage
                // +1 to avoid memory bank conflicts.
                scalar_t sA1[ ib ][ ib+1 ], sA2[ ib ][ ib+1 ];
                // todo: make sA1, sA2 team-local, shared by threads (OpenMP > 5)
                // #pragma omp private(sA1, sA2)
                // #pragma omp allocate (sA1,sA2) allocator(omp_pteam_mem_alloc)
                // ii, jj are row & column offsets within each block.
                int max_ii = ( i+ib < n ? std::min(ib, n) : n-i );
                int max_jj = ( j+ib < n ? std::min(ib, n) : n-j );
                if (i == j) { // diagonal blocks
                    // Load ibxib block from A(i,j) into sA1
                    #pragma omp parallel for simd collapse(2)
                    for (int ii = 0; ii < max_ii; ++ii)
                        for (int jj = 0; jj < max_jj; ++jj)
                            sA1[jj][ii] = A[i+ii + (j+jj)*lda];
                    // Save transposed block, A(i, j) = trans(sA1).
                    #pragma omp parallel for simd collapse(2)
                    for (int ii = 0; ii < max_ii; ++ii)
                        for (int jj = 0; jj < max_jj; ++jj)
                            A[i+ii + (j+jj)*lda] =
                                (is_conj) ? conj(sA1[ii][jj]) : sA1[ii][jj];
                }
                else { // off-diagonal block
                    // Load blocks A(i, j) and A(j, i) into shared memory sA1 and sA2.
                    #pragma omp parallel for simd collapse(2)
                    for (int ii = 0; ii < max_ii; ++ii)
                        for (int jj = 0; jj < max_jj; ++jj)
                            sA1[ii][jj] = A[i+ii + (j+jj)*lda]; // sA1(i,j)=A(i,j)
                    #pragma omp parallel for simd collapse(2)
                    for (int ii = 0; ii < max_ii; ++ii)
                        for (int jj = 0; jj < max_jj; ++jj)
                            sA2[jj][ii] = A[j+jj + (i+ii)*lda]; // sA2(j,i)=A(j,i)
                    // Save transposed blocks, A(i, j) = trans(sA2), A(j, i) = trans(sA1).
                    #pragma omp parallel for simd collapse(2)
                    for (int ii = 0; ii < max_ii; ++ii)
                        for (int jj = 0; jj < max_jj; ++jj)
                            // A(i,j)=trans(sA2)=sA2(i,j)
                            A[i+ii + (j+jj)*lda] =
                                (is_conj) ? conj(sA2[jj][ii]) : sA2[jj][ii];
                    #pragma omp parallel for simd collapse(2)
                    for (int ii = 0; ii < max_ii; ++ii)
                        for (int jj = 0; jj < max_jj; ++jj)
                            // A(j,i)=trans(sA1)=sA1(j,i)
                            A[j+jj + (i+ii)*lda] =
                                (is_conj) ? conj(sA1[ii][jj]) : sA1[ii][jj];
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Device routine handles single square matrix

/// The routine loads blocks of data into small ib x ib local storage
/// and then writes the blocks back transposed into the correct
/// location transposed.
///
template <typename scalar_t>
void transpose_sqr_func(
    bool is_conj,
    int n,
    scalar_t* A, int64_t lda,
    blas::Queue& queue)
{
    using blas::conj;
    // printf("%s:%d sqr queue.device() %d\n", __FILE__, __LINE__, queue.device());

    static const int ib = 16;
    queue.sync(); // sync queue before switching to openmp device execution
    // i, j are row & column indices of top-left corner of each local block.
    #pragma omp target is_device_ptr(A) device(queue.device())
    // Distribute the blocks to omp teams
    #pragma omp teams distribute collapse(2)
    for (int i = 0; i < n; i += ib) {
        for (int j = 0; j <= i; j += ib) {
            // allocate local blocks for storage
            // +1 to avoid memory bank conflicts.
            scalar_t sA1[ ib ][ ib+1 ], sA2[ ib ][ ib+1 ];
            // todo: make sA1, sA2 team-local, shared by threads (OpenMP > 5)
            // #pragma omp private(sA1, sA2)
            // #pragma omp allocate (sA1,sA2) allocator(omp_pteam_mem_alloc)
            // ii, jj are row & column offsets within each block.
            int max_ii = ( i+ib < n ? std::min(ib, n) : n-i );
            int max_jj = ( j+ib < n ? std::min(ib, n) : n-j );
            if (i == j) { // diagonal blocks
                // Load ibxib block from A(i,j) into sA1
                #pragma omp parallel for simd collapse(2)
                for (int ii = 0; ii < max_ii; ++ii)
                    for (int jj = 0; jj < max_jj; ++jj)
                        sA1[jj][ii] = A[i+ii + (j+jj)*lda];
                // Save transposed block, A(i, j) = trans(sA1).
                #pragma omp parallel for simd collapse(2)
                for (int ii = 0; ii < max_ii; ++ii)
                    for (int jj = 0; jj < max_jj; ++jj)
                        A[i+ii + (j+jj)*lda] =
                            (is_conj) ? conj(sA1[ii][jj]) : sA1[ii][jj];
            }
            else { // off-diagonal block
                // Load blocks A(i, j) and A(j, i) into shared memory sA1 and sA2.
                #pragma omp parallel for simd collapse(2)
                for (int ii = 0; ii < max_ii; ++ii)
                    for (int jj = 0; jj < max_jj; ++jj)
                        sA1[ii][jj] = A[i+ii + (j+jj)*lda]; // sA1(i,j)=A(i,j)
                #pragma omp parallel for simd collapse(2)
                for (int ii = 0; ii < max_ii; ++ii)
                    for (int jj = 0; jj < max_jj; ++jj)
                        sA2[jj][ii] = A[j+jj + (i+ii)*lda]; // sA2(j,i)=A(j,i)
                // Save transposed blocks, A(i, j) = trans(sA2), A(j, i) = trans(sA1).
                #pragma omp parallel for simd collapse(2)
                for (int ii = 0; ii < max_ii; ++ii)
                    for (int jj = 0; jj < max_jj; ++jj)
                        // A(i,j)=trans(sA2)=sA2(i,j)
                        A[i+ii + (j+jj)*lda] =
                            (is_conj) ? conj(sA2[jj][ii]) : sA2[jj][ii];
                #pragma omp parallel for simd collapse(2)
                for (int ii = 0; ii < max_ii; ++ii)
                    for (int jj = 0; jj < max_jj; ++jj)
                        // A(j,i)=trans(sA1)=sA1(j,i)
                        A[j+jj + (i+ii)*lda] =
                            (is_conj) ? conj(sA1[ii][jj]) : sA1[ii][jj];
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Device routine handles batches of rectangular matrices.
///
/// The routine loads blocks of data into small NX x NB local storage
/// and then writes the blocks back transposed into the correct
/// location transposed.
///
template <typename scalar_t, int NX>
void transpose_rect_batch_func(
    bool is_conj,
    int m, int n,
    scalar_t** dAarray, int64_t lda,
    scalar_t** dATarray, int64_t ldat,
    int batch_count, blas::Queue& queue)
{
    using blas::conj;
    static const int NB = 32;
    queue.sync(); // sync queue before switching to openmp device execution
    // i, j are row & column indices of top-left corner of each local block.
    #pragma omp target is_device_ptr(dAarray, dATarray) device(queue.device())
    #pragma omp teams distribute collapse(2)
    for (int k = 0; k < batch_count; ++k) {
        for (int i = 0; i < m; i += NX) {
            for (int j = 0; j < n; j += NB) {
                scalar_t *dA = dAarray[k];
                scalar_t *dAT = dATarray[k];
                // todo: make sA team-local, shared by threads
                // #pragma omp private(sA)
                // todo: allocators supported after OpenMP 5.0
                // #if (_OPENMP > 201810)
                // #pragma omp allocate (sA) allocator(omp_pteam_mem_alloc)
                // #endif
                scalar_t sA[ NX ][ NB+1 ];
                int max_NX = ( i+NX < m ? std::min(NX, m) : m-i );
                int max_NB = ( j+NB < n ? std::min(NB, n) : n-j );
                // Load NXxNB block from A(i,j) so that sA(i,j) = dA(i,j)
                #pragma omp parallel for simd collapse(2)
                for (int ii = 0; ii < max_NX; ++ii)
                    for (int jj = 0; jj < max_NB; ++jj)
                        sA[ii][jj] = dA[i+ii + (j+jj)*lda];
                // Save transposed block, dAT(j, i) = sA(i,j).
                #pragma omp parallel for simd collapse(2)
                for (int ii = 0; ii < max_NX; ++ii)
                    for (int jj = 0; jj < max_NB; ++jj)
                        dAT[j+jj + (i+ii)*ldat] =
                            (is_conj) ? conj(sA[ii][jj]) : sA[ii][jj];
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Device routine handles single rectangular matrox
///
/// The routine loads blocks of data into small NX x NB local storage
/// and then writes the blocks back transposed into the correct
/// location transposed.
///
template <typename scalar_t, int NX>
void transpose_rect_func(
    bool is_conj,
    int m, int n,
    scalar_t* dA, int64_t lda,
    scalar_t* dAT, int64_t ldat,
    blas::Queue& queue)
{
    using blas::conj;
    static const int NB = 32;
    queue.sync(); // sync queue before switching to openmp device execution
    // i, j are row & column indices of top-left corner of each local block.
    #pragma omp target is_device_ptr(dA, dAT) device(queue.device())
    #pragma omp teams distribute collapse(2)
    for (int i = 0; i < m; i += NX) {
        for (int j = 0; j < n; j += NB) {
            // todo: make sA team-local, shared by threads
            // #pragma omp private(sA)
            // todo: allocators supported after OpenMP 5.0
            // #if (_OPENMP > 201810)
            // #pragma omp allocate (sA) allocator(omp_pteam_mem_alloc)
            // #endif
            scalar_t sA[ NX ][ NB+1 ];
            int max_NX = ( i+NX < m ? std::min(NX, m) : m-i );
            int max_NB = ( j+NB < n ? std::min(NB, n) : n-j );
            // Load NXxNB block from A(i,j) so that sA(i,j) = dA(i,j)
            #pragma omp parallel for simd collapse(2)
            for (int ii = 0; ii < max_NX; ++ii)
                for (int jj = 0; jj < max_NB; ++jj)
                    sA[ii][jj] = dA[i+ii + (j+jj)*lda];
            // Save transposed block, dAT(j, i) = sA(i,j).
            #pragma omp parallel for simd collapse(2)
            for (int ii = 0; ii < max_NX; ++ii)
                for (int jj = 0; jj < max_NB; ++jj)
                    dAT[j+jj + (i+ii)*ldat] =
                        (is_conj) ? conj(sA[ii][jj]) : sA[ii][jj];
        }
    }
}

//------------------------------------------------------------------------------
/// Physically transpose a square matrix in place.
///
/// @param[in] n
///     Number of rows and columns of each tile. n >= 0.
///
/// @param[in,out] A
///     A square n-by-n matrix stored in an lda-by-n array in GPU memory.
///     On output, A is transposed.
///
/// @param[in] lda
///     Leading dimension of A. lda >= n.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t>
void transpose(
    bool is_conj,
    int64_t n,
    scalar_t* A, int64_t lda,
    blas::Queue& queue)
{
    if (n <= 1)
        return;
    assert(lda >= n);

    transpose_sqr_func(is_conj, n, A, lda, queue);
}

//------------------------------------------------------------------------------
/// Physically transpose a batch of square matrices in place.
///
/// @param[in] n
///     Number of rows and columns of each tile. n >= 0.
///
/// @param[in,out] Aarray
///     Array in GPU memory of dimension batch_count, containing pointers to
///     matrices, where each Aarray[k] is a square n-by-n matrix stored in an
///     lda-by-n array in GPU memory.
///     On output, each Aarray[k] is transposed.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= n.
///
/// @param[in] batch_count
///     Size of Aarray. batch_count >= 0.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t>
void transpose_batch(
    bool is_conj,
    int64_t n,
    scalar_t** Aarray, int64_t lda,
    int64_t batch_count,
    blas::Queue& queue)
{
    if (batch_count < 0 || n <= 1)
        return;
    assert(lda >= n);

    transpose_sqr_batch_func(
        is_conj, n, Aarray, lda, batch_count, queue);
}

//------------------------------------------------------------------------------
/// Physically transpose a rectangular matrix out-of-place.
///
/// @param[in] m
///     Number of columns of tile. m >= 0.
///
/// @param[in] n
///     Number of rows of tile. n >= 0.
///
/// @param[in] dA
///     A rectangular m-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] lda
///     Leading dimension of dA. lda >= m.
///
/// @param[out] dAT
///     A rectangular m-by-n matrix stored in an ldat-by-m array in GPU memory.
///     On output, dAT is the transpose of dA.
///
/// @param[in] ldat
///     Leading dimension of dAT. ldat >= n.
///
/// @param[in] batch_count
///     Size of Aarray. batch_count >= 0.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t, int NX>
void transpose(
    bool is_conj,
    int64_t m, int64_t n,
    scalar_t* dA,  int64_t lda,
    scalar_t* dAT, int64_t ldat,
    blas::Queue& queue)
{
    if ((m <= 0) || (n <= 0))
        return;
    assert(lda >= m);
    assert(ldat >= n);

    transpose_rect_func<scalar_t, NX>(
        is_conj, m, n, dA, lda, dAT, ldat, queue );
}

//------------------------------------------------------------------------------
/// Physically transpose a batch of rectangular matrices out-of-place.
///
/// @param[in] m
///     Number of columns of each tile. m >= 0.
///
/// @param[in] n
///     Number of rows of each tile. n >= 0.
///
/// @param[in] dA_array
///     Array in GPU memory of dimension batch_count, containing pointers to
///     matrices, where each dA_array[k] is a rectangular m-by-n matrix stored in an
///     lda-by-n array in GPU memory.
///
/// @param[in] lda
///     Leading dimension of each dA_array[k] tile. lda >= m.
///
/// @param[out] dAT_array
///     Array in GPU memory of dimension batch_count, containing pointers to
///     matrices, where each dAT_array[k] is a rectangular m-by-n matrix
///     stored in an ldat-by-m array in GPU memory.
///     On output, each dAT_array[k] is the transpose of dA_array[k].
///
/// @param[in] lda
///     Leading dimension of each dAT_array[k] tile. ldat >= n.
///
/// @param[in] batch_count
///     Size of Aarray. batch_count >= 0.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t, int NX>
void transpose_batch(
    bool is_conj,
    int64_t m, int64_t n,
    scalar_t **dA_array,  int64_t lda,
    scalar_t **dAT_array, int64_t ldat,
    int64_t batch_count,
    blas::Queue& queue)
{
    if ((m <= 0) || (n <= 0))
        return;
    assert(lda >= m);
    assert(ldat >= n);

    transpose_rect_batch_func<scalar_t, NX>(
        is_conj, m, n, dA_array, lda, dAT_array, ldat, batch_count, queue );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// Square matrix

template
void transpose(
    bool is_conj,
    int64_t n,
    float* A, int64_t lda,
    blas::Queue& queue);

template
void transpose(
    bool is_conj,
    int64_t n,
    double* A, int64_t lda,
    blas::Queue& queue);

template
void transpose(
    bool is_conj,
    int64_t n,
    std::complex<float>* A, int64_t lda,
    blas::Queue& queue);

template
void transpose(
    bool is_conj,
    int64_t n,
    std::complex<double>* A, int64_t lda,
    blas::Queue& queue);

// ----------------------------------------
// Explicit instantiations.
// Batch of square matrices

template
void transpose_batch(
    bool is_conj,
    int64_t n,
    float** Aarray, int64_t lda,
    int64_t batch_count,
    blas::Queue& queue);

template
void transpose_batch(
    bool is_conj,
    int64_t n,
    double** Aarray, int64_t lda,
    int64_t batch_count,
    blas::Queue& queue);

template
void transpose_batch(
    bool is_conj,
    int64_t n,
    std::complex<float>** Aarray, int64_t lda,
    int64_t batch_count,
    blas::Queue& queue);

template
void transpose_batch(
    bool is_conj,
    int64_t n,
    std::complex<double>** Aarray, int64_t lda,
    int64_t batch_count,
    blas::Queue& queue);


// ----------------------------------------
// Explicit instantiations.
// Rectangular matrix

template<>
void transpose(
    bool is_conj,
    int64_t m, int64_t n,
    float* dA,  int64_t lda,
    float* dAT, int64_t ldat,
    blas::Queue& queue)
{
    transpose<float,32>(
        is_conj,
        m, n,
        dA,  lda,
        dAT, ldat,
        queue);
}

template<>
void transpose(
    bool is_conj,
    int64_t m, int64_t n,
    double* dA,  int64_t lda,
    double* dAT, int64_t ldat,
    blas::Queue& queue)
{
    transpose<double,32>(
        is_conj,
        m, n,
        dA,  lda,
        dAT, ldat,
        queue);
}

template<>
void transpose(
    bool is_conj,
    int64_t m, int64_t n,
    std::complex<float>* dA,  int64_t lda,
    std::complex<float>* dAT, int64_t ldat,
    blas::Queue& queue)
{
    transpose<std::complex<float>,32>(
        is_conj,
        m, n,
        dA,  lda,
        dAT, ldat,
        queue);
}

template<>
void transpose(
    bool is_conj,
    int64_t m, int64_t n,
    std::complex<double>* dA,  int64_t lda,
    std::complex<double>* dAT, int64_t ldat,
    blas::Queue& queue)
{
    transpose<std::complex<double>,16>(
        is_conj,
        m, n,
        dA,  lda,
        dAT, ldat,
        queue);
}

// ----------------------------------------
// Explicit instantiations.
// Batch of rectangular matrices

template<>
void transpose_batch(
    bool is_conj,
    int64_t m, int64_t n,
    float **dA_array,  int64_t lda,
    float **dAT_array, int64_t ldat,
    int64_t batch_count,
    blas::Queue& queue)
{
    transpose_batch<float,32>(
        is_conj,
        m, n,
        dA_array,  lda,
        dAT_array, ldat,
        batch_count,
        queue);
}

template<>
void transpose_batch(
    bool is_conj,
    int64_t m, int64_t n,
    double **dA_array,  int64_t lda,
    double **dAT_array, int64_t ldat,
    int64_t batch_count,
    blas::Queue& queue)
{
    transpose_batch<double,32>(
        is_conj,
        m, n,
        dA_array,  lda,
        dAT_array, ldat,
        batch_count,
        queue);
}

template<>
void transpose_batch(
    bool is_conj,
    int64_t m, int64_t n,
    std::complex<float> **dA_array,  int64_t lda,
    std::complex<float> **dAT_array, int64_t ldat,
    int64_t batch_count,
    blas::Queue& queue)
{
    transpose_batch<std::complex<float>,32>(
        is_conj,
        m, n,
        dA_array,  lda,
        dAT_array, ldat,
        batch_count,
        queue);
}

template<>
void transpose_batch(
    bool is_conj,
    int64_t m, int64_t n,
    std::complex<double> **dA_array,  int64_t lda,
    std::complex<double> **dAT_array, int64_t ldat,
    int64_t batch_count,
    blas::Queue& queue)
{
    transpose_batch<std::complex<double>,16>(
        is_conj,
        m, n,
        dA_array,  lda,
        dAT_array, ldat,
        batch_count,
        queue);
}

} // namespace device
} // namespace slate
