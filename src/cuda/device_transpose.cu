// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Exception.hh"
#include "slate/internal/device.hh"

#include "device_util.cuh"

#include <cstdio>

namespace slate {
namespace device {

/// internal blocking
/// 16 x 16 thread block = 256 threads
/// 32 x 32 thread block = 1024 threads
static const int ib = 16;

//------------------------------------------------------------------------------
/// Device routine handles one matrix.
/// Thread block grid:
/// x = batch index (ignored here; see batch kernel),
/// y = block row index,
/// z = block col index.
/// Each thread block is ib-by-ib threads and does one ib-by-ib block of an
/// n-by-n matrix.
///
/// Let nt = ceildiv( n, ib ) be the number of blocks for one n-by-n matrix.
/// An even number of blocks uses an (nt + 1) by (nt/2) grid.
/// Example: for nt = 4 blocks, y by z = 5 by 2 grid:
///   [ A00  A01 ]
///   [----. A11 ]                  [ A10  .  |  .   .  ]
///   [ A10 '----]                  [ A20 A21 |  .   .  ]
///   [ A20  A21 ] covers matrix as [ A30 A31 | A00  .  ]
///   [ A30  A31 ]                  [ A40 A41 | A01 A11 ]
///   [ A40  A41 ]
///
/// An odd number of blocks uses an (nt) by (nt + 1)/2 grid.
/// Example: for nt = 5 blocks, y by z = 5 by 3 grid:
///   [ A00 | A01   A02 ]
///   [     '----.      ]                  [ A00  .   .  |  .   .  ]
///   [ A10   A11 | A12 ]                  [ A10 A11  .  |  .   .  ]
///   [           '-----] covers matrix as [ A20 A21 A22 |  .   .  ]
///   [ A20   A21   A22 ]                  [ A30 A31 A32 | A01  .  ]
///   [ A30   A31   A32 ]                  [ A40 A41 A42 | A02 A12 ]
///   [ A40   A41   A42 ]
///
template <typename scalar_t>
__device__ void transpose_func(
    bool is_conj,
    int n,
    scalar_t* A, int64_t lda)
{
    // +1 to avoid memory bank conflicts.
    __shared__ scalar_t sA1[ ib ][ ib+1 ];
    __shared__ scalar_t sA2[ ib ][ ib+1 ];

    // i, j are row & column indices of top-left corner of each block.
    // ii, jj are row & column offsets within each block.
    int ii = threadIdx.x;
    int jj = threadIdx.y;

    int i, j;
    if (gridDim.y - 1 == gridDim.z*2) {
        // Even number of blocks.
        //assert( ceildiv(n, ib) % 2 == 0 );
        bool lower = (blockIdx.y > blockIdx.z);
        i = (lower ? (blockIdx.y - 1) : (blockIdx.z + gridDim.z));
        j = (lower ? (blockIdx.z    ) : (blockIdx.y + gridDim.z));
    }
    else {
        // Odd number of blocks.
        //assert( ceildiv(n, ib) % 2 == 1 );
        bool lower = (blockIdx.y >= blockIdx.z);
        i = (lower ? blockIdx.y : (blockIdx.z + gridDim.z - 1));
        j = (lower ? blockIdx.z : (blockIdx.y + gridDim.z    ));
    }
    i *= ib;
    j *= ib;

    scalar_t* A1 = A + i + ii + (j + jj)*lda;  // A(i, j)
    if (i == j) { // diagonal block
        // Load block A(i, j) into shared memory sA1.
        if (i + ii < n  &&  j + jj < n) {
            sA1[jj][ii] = *A1;
        }
        __syncthreads();

        // Save transposed block, A(i, j) = trans(sA1).
        if (i + ii < n  &&  j + jj < n) {
            if (is_conj)
                *A1 = conj(sA1[ii][jj]);
            else
                *A1 = sA1[ii][jj];
        }
    }
    else { // off-diagonal block
        scalar_t* A2 = A + j + ii + (i + jj)*lda;  // A(j, i)
        // Load blocks A(i, j) and A(j, i) into shared memory sA1 and sA2.
        if (i + ii < n  &&  j + jj < n) {
            sA1[jj][ii] = *A1;
        }
        if (j + ii < n  &&  i + jj < n) {
            sA2[jj][ii] = *A2;
        }
        __syncthreads();

        // Save transposed blocks, A(i, j) = trans(sA2), A(j, i) = trans(sA1).
        if (i + ii < n && j + jj < n) {
            if (is_conj)
                *A1 = conj(sA2[ii][jj]);
            else
                *A1 = sA2[ii][jj];
        }
        if (j + ii < n && i + jj < n) {
            if (is_conj)
                *A2 = conj(sA1[ii][jj]);
            else
                *A2 = sA1[ii][jj];
        }
    }
}

//------------------------------------------------------------------------------
static const int NB = 32;  ///< block size for transpose_func
static const int NY = 8;   ///< y dim of thread block size for transpose_func
// static const int NX = 32; handled as template parameter, look below


/// tile M-by-N matrix with ceil(M/NB) by ceil(N/NB) tiles sized NB-by-NB.
/// uses NX-by-NY threads, where NB/NX, NB/NY, NX/NY evenly.
/// subtile each NB-by-NB tile with (NB/NX) subtiles sized NX-by-NB
/// for each subtile
///     load NX-by-NB subtile transposed from A into sA, as (NB/NY) blocks sized NX-by-NY
///     save NB-by-NX subtile from sA into AT,   as (NB/NX)*(NX/NY) blocks sized NX-by-NY
///     A  += NX
///     AT += NX*ldat
///
/// e.g., with NB=32, NX=32, NY=8 ([sdc] precisions)
///     load 32x32 subtile as 4   blocks of 32x8 columns: (A11  A12  A13  A14 )
///     save 32x32 subtile as 1*4 blocks of 32x8 columns: (AT11 AT12 AT13 AT14)
///
/// e.g., with NB=32, NX=16, NY=8 (z precision)
///     load 16x32 subtile as 4   blocks of 16x8 columns: (A11  A12  A13  A14)
///     save 32x16 subtile as 2*2 blocks of 16x8 columns: (AT11 AT12)
///                                                       (AT21 AT22)
///
template <typename scalar_t, int NX>
__device__ void transpose_func(
    bool is_conj,
    int m, int n,
    const scalar_t *A,  int64_t lda,
          scalar_t *AT, int64_t ldat)
{
    __shared__ scalar_t sA[NB][NX+1];

    int tx  = threadIdx.x;
    int ty  = threadIdx.y;
    int iby = blockIdx.y*NB;
    int ibz = blockIdx.z*NB;
    int i, j;

    A  += iby + tx + (ibz + ty)*lda;
    AT += ibz + tx + (iby + ty)*ldat;

    #pragma unroll
    for (int tile=0; tile < NB/NX; ++tile) {
        // load NX-by-NB subtile transposed from A into sA
        i = iby + tx + tile*NX;
        j = ibz + ty;
        if (i < m) {
            if (is_conj) {
                #pragma unroll
                for (int j2=0; j2 < NB; j2 += NY) {
                    if (j + j2 < n) {
                        sA[ty + j2][tx] = conj(A[j2*lda]);
                    }
                }
            }
            else {
                #pragma unroll
                for (int j2=0; j2 < NB; j2 += NY) {
                    if (j + j2 < n) {
                        sA[ty + j2][tx] = A[j2*lda];
                    }
                }
            }
        }
        __syncthreads();

        // save NB-by-NX subtile from sA into AT
        i = ibz + tx;
        j = iby + ty + tile*NX;
        #pragma unroll
        for (int i2=0; i2 < NB; i2 += NX) {
            if (i + i2 < n) {
                #pragma unroll
                for (int j2=0; j2 < NX; j2 += NY) {
                    if (j + j2 < m) {
                        AT[i2 + j2*ldat] = sA[tx + i2][ty + j2];
                    }
                }
            }
        }
        __syncthreads();

        // move to next subtile
        A  += NX;
        AT += NX*ldat;
    }
}

//------------------------------------------------------------------------------
/// in-place transpose of a square buffer
template <typename scalar_t>
__global__ void transpose_kernel(
    bool is_conj,
    int n,
    scalar_t* A, int64_t lda)
{
    transpose_func(is_conj, n, A, lda);
}

//------------------------------------------------------------------------------
/// in-place transpose of array of square buffers
template <typename scalar_t>
__global__ void transpose_batch_kernel(
    bool is_conj,
    int n,
    scalar_t** Aarray, int64_t lda)
{
    transpose_func(is_conj, n, Aarray[blockIdx.x], lda);
}

//------------------------------------------------------------------------------
/// out-of-place transpose of a rectangular buffer
/// transopses A onto AT
///
template <typename scalar_t, int NX>
__global__ void transpose_kernel(
    bool is_conj,
    int m, int n,
    const scalar_t *A,  int64_t lda,
          scalar_t *AT, int64_t ldat)
{
    transpose_func<scalar_t, NX>( is_conj, m, n, A, lda, AT, ldat );
}

//------------------------------------------------------------------------------
/// out-of-place transpose of an array of rectangular buffers
/// transopses dA_array onto dAT_array
///
template <typename scalar_t, int NX>
__global__ void transpose_batch_kernel(
    bool is_conj,
    int m, int n,
    scalar_t **dA_array,  int64_t lda,
    scalar_t **dAT_array, int64_t ldat)
{
    transpose_func<scalar_t, NX>(
        is_conj, m, n, dA_array[blockIdx.x], lda, dAT_array[blockIdx.x], ldat );
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

    cudaSetDevice( queue.device() );

    int nt = ceildiv( n, int64_t(ib) );
    assert(nt <= 65535);                // CUDA limitation

    // Need 1/2 * (nt + 1) * nt to cover lower triangle and diagonal of matrix.
    // Block assignment differs depending on whether nt is odd or even.
    dim3 blocks;
    if (nt % 2 == 0) {
        // even blocks
        blocks = { 1, uint(nt + 1), uint(nt/2) };
    }
    else {
        // odd blocks
        blocks = { 1, uint(nt), uint((nt + 1)/2) };
    }
    dim3 threads( ib, ib );

    transpose_kernel<<< blocks, threads, 0, queue.stream() >>>
        (is_conj, n, A, lda);

    cudaError_t error = cudaGetLastError();
    slate_assert(error == cudaSuccess);
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

    cudaSetDevice( queue.device() );

    int nt = ceildiv( n, int64_t(ib) );
    assert(nt <= 65535);                // CUDA limitation
    assert(batch_count <= 2147483647);  // CUDA limitation, 2^31 - 1

    // Need 1/2 * (nt + 1) * nt to cover lower triangle and diagonal of matrix.
    // Block assignment differs depending on whether nt is odd or even.
    dim3 blocks;
    if (nt % 2 == 0) {
        // even blocks
        blocks = { uint(batch_count), uint(nt + 1), uint(nt/2) };
    }
    else {
        // odd blocks
        blocks = { uint(batch_count), uint(nt), uint((nt + 1)/2) };
    }
    dim3 threads( ib, ib );

    transpose_batch_kernel<<< blocks, threads, 0, queue.stream() >>>
        ( is_conj, n, Aarray, lda );

    cudaError_t error = cudaGetLastError();
    slate_assert(error == cudaSuccess);
}

//------------------------------------------------------------------------------
/// Look up NX based on data type.
/// float, double, complex-float use NX = 32.
template <typename scalar_t>
struct nx_traits
{
    static const int NX = 32;
};

template <>
struct nx_traits< cuDoubleComplex >
{
    static const int NX = 16;
};

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
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t>
void transpose(
    bool is_conj,
    int64_t m, int64_t n,
    scalar_t* dA,  int64_t lda,
    scalar_t* dAT, int64_t ldat,
    blas::Queue& queue)
{
    const int NX = nx_traits<scalar_t>::NX;

    if ((m <= 0) || (n <= 0))
        return;
    assert(lda >= m);
    assert(ldat >= n);

    cudaSetDevice( queue.device() );

    int mt = ceildiv( m, int64_t(NB) );
    assert(mt <= 65535);                // CUDA limitation
    int nt = ceildiv( n, int64_t(NB) );
    assert(nt <= 65535);                // CUDA limitation

    dim3 grid( 1, mt, nt );
    dim3 threads( NX, NY );
    transpose_kernel<scalar_t, NX><<< grid, threads, 0, queue.stream() >>>
        ( is_conj, m, n, dA, lda, dAT, ldat );

    cudaError_t error = cudaGetLastError();
    slate_assert(error == cudaSuccess);
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
///     matrices, where each dAT_array[k] is a rectangular m-by-n matrix stored in an
///     ldat-by-m array in GPU memory.
///     On output, each dAT_array[k] is the transpose of dA_array[k].
///
/// @param[in] ldat
///     Leading dimension of each dAT_array[k] tile. ldat >= n.
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
    int64_t m, int64_t n,
    scalar_t **dA_array,  int64_t lda,
    scalar_t **dAT_array, int64_t ldat,
    int64_t batch_count,
    blas::Queue& queue)
{
    const int NX = nx_traits<scalar_t>::NX;

    if ((m <= 0) || (n <= 0))
        return;
    assert(lda >= m);
    assert(ldat >= n);

    cudaSetDevice( queue.device() );

    int mt = ceildiv( m, int64_t(NB) );
    assert(mt <= 65535);                // CUDA limitation
    int nt = ceildiv( n, int64_t(NB) );
    assert(nt <= 65535);                // CUDA limitation
    assert(batch_count <= 2147483647);  // CUDA limitation, 2^31 - 1

    dim3 grid( uint(batch_count), mt, nt );
    dim3 threads( NX, NY, 1 );
    transpose_batch_kernel<scalar_t, NX><<< grid, threads, 0, queue.stream() >>>
        ( is_conj, m, n, dA_array, lda, dAT_array, ldat );

    cudaError_t error = cudaGetLastError();
    slate_assert(error == cudaSuccess);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
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

//----- rectangular, out-of-place
template
void transpose(
    bool is_conj,
    int64_t m, int64_t n,
    float* A, int64_t lda,
    float* B, int64_t ldb,
    blas::Queue& queue);

template
void transpose(
    bool is_conj,
    int64_t m, int64_t n,
    double* A, int64_t lda,
    double* B, int64_t ldb,
    blas::Queue& queue);

//------------------------------------------------------------------------------
// Specializations to cast std::complex => cuComplex.
template <>
void transpose(
    bool is_conj,
    int64_t n,
    std::complex<float>* A, int64_t lda,
    blas::Queue& queue)
{
    transpose( is_conj, n,
               (cuFloatComplex*) A, lda, queue );
}

template <>
void transpose(
    bool is_conj,
    int64_t n,
    std::complex<double>* A, int64_t lda,
    blas::Queue& queue)
{
    transpose( is_conj, n,
               (cuDoubleComplex*) A, lda, queue );
}

template <>
void transpose(
    bool is_conj,
    int64_t m, int64_t n,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb,
    blas::Queue& queue)
{
    transpose( is_conj, m, n,
               (cuFloatComplex*) A, lda,
               (cuFloatComplex*) B, ldb,
               queue );
}

template <>
void transpose(
    bool is_conj,
    int64_t m, int64_t n,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb,
    blas::Queue& queue)
{
    transpose( is_conj, m, n,
               (cuDoubleComplex*) A, lda,
               (cuDoubleComplex*) B, ldb,
               queue );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
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

//----- rectangular, out-of-place
template
void transpose_batch(
    bool is_conj,
    int64_t m, int64_t n,
    float** Aarray, int64_t lda,
    float** Barray, int64_t ldb,
    int64_t batch_count,
    blas::Queue& queue);

template
void transpose_batch(
    bool is_conj,
    int64_t m, int64_t n,
    double** Aarray, int64_t lda,
    double** Barray, int64_t ldb,
    int64_t batch_count,
    blas::Queue& queue);

//------------------------------------------------------------------------------
// Specializations to cast std::complex => cuComplex.
template <>
void transpose_batch(
    bool is_conj,
    int64_t n,
    std::complex<float>** Aarray, int64_t lda,
    int64_t batch_count,
    blas::Queue& queue)
{
    transpose_batch(
        is_conj, n,
        (cuFloatComplex**) Aarray, lda,
        batch_count, queue );
}

template <>
void transpose_batch(
    bool is_conj,
    int64_t n,
    std::complex<double>** Aarray, int64_t lda,
    int64_t batch_count,
    blas::Queue& queue)
{
    transpose_batch(
        is_conj, n,
        (cuDoubleComplex**) Aarray, lda,
        batch_count, queue );
}

//----- rectangular, out-of-place
template <>
void transpose_batch(
    bool is_conj,
    int64_t m, int64_t n,
    std::complex<float>** Aarray, int64_t lda,
    std::complex<float>** Barray, int64_t ldb,
    int64_t batch_count,
    blas::Queue& queue)
{
    transpose_batch(
        is_conj, m, n,
        (cuFloatComplex**) Aarray, lda,
        (cuFloatComplex**) Barray, ldb,
        batch_count, queue );
}

template <>
void transpose_batch(
    bool is_conj,
    int64_t m, int64_t n,
    std::complex<double>** Aarray, int64_t lda,
    std::complex<double>** Barray, int64_t ldb,
    int64_t batch_count,
    blas::Queue& queue)
{
    transpose_batch(
        is_conj, m, n,
        (cuDoubleComplex**) Aarray, lda,
        (cuDoubleComplex**) Barray, ldb,
        batch_count, queue );
}

} // namespace device
} // namespace slate
