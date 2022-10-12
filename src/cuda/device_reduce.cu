// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Exception.hh"
#include "slate/internal/device.hh"

#include "device_util.cuh"

#include <cstdio>

namespace slate {
namespace device {

//------------------------------------------------------------------------------
/// Kernel implementing element-wise matrix reduction.
/// Each thread block deals with one tile.
/// Each thread deals with one row.
/// Launched by reduce().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///
/// @param[in] mt
///     Number of tiles. mt > 1.
///
/// @param[in] alpha
///     The scalar alpha.
///
/// @param[in] Aarray
///     Array of tiles of dimension gridDim.x,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile in Aarray. lda >= m.
///
template <typename scalar_t>
__global__ void reduce_kernel(
    int64_t m, int64_t n, int64_t mt,
    scalar_t alpha, scalar_t** Aarray, int64_t lda)
{
    int64_t row = threadIdx.x;
    int64_t col = blockIdx.y;
    int64_t ib  = blockDim.x;
    int64_t jb = gridDim.y;
    int64_t nt  = gridDim.x;	
    scalar_t sum;

    // The first row of tiles.
    // All other rows will be sumed up to this row.
    scalar_t* A0 = Aarray[ blockIdx.x ];


    // i, j loops sub-tile the tiles,
    for (int64_t i = row; i < m; i += ib) {
        for (int64_t j = col; j < n; j += jb) {
            // vector read into registers.
	    // initial value of sum
            sum = alpha * A0[ j*lda + i ];

            // Loop over block rows.
            for (int64_t ii = 1; ii < mt; ++ii) {
                // vector read and add
	        scalar_t* Aii = Aarray[ ii*nt + blockIdx.x ];
                sum += alpha * Aii[ j*lda + i ];
	    }	

            // vector write
            A0[ j*lda + i ] = sum;
        }	    
    }	    
}

//------------------------------------------------------------------------------
/// Batched routine for element-wise tile addition.
/// Sets
/// \[
///     Aarray[0] = \alpha Aarray[k]. k = 1:mt-1
/// \]
///
/// @param[in] m
///     Number of rows of each tile. m >= 0.
///
/// @param[in] n
///     Number of columns of each tile. n >= 0.
///
/// @param[in] alpha
///     The scalar alpha.
///
/// @param[in] Aarray
///     Array in GPU memory of dimension batch_count, containing pointers to tiles,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] lda
///     Leading dimension of each tile in A. lda >= m.
///
/// @param[in] batch_count
///     Size of Aarray and Barray. batch_count >= 0.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t>
void reduce(
    int64_t m, int64_t n, int64_t mt,
    scalar_t alpha, scalar_t** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue)
{
    // quick return
    if (batch_count == 0)
        return;

    int64_t nt = ceildiv( batch_count, mt );
    int64_t jb = 32;

    // Max threads/block=1024 for current CUDA compute capability (<= 7.5)
    cudaSetDevice( queue.device() );

    int64_t nthreads = std::min( int64_t( 1024 ), m );
    dim3 threads( nthreads );
    dim3 blocks( nt, jb );

    reduce_kernel<<< blocks, threads, 0, queue.stream() >>>(
        m, n, mt,
    	alpha, Aarray, lda);

    cudaError_t error = cudaGetLastError();
    slate_assert(error == cudaSuccess);

}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void reduce(
    int64_t m, int64_t n, int64_t mt,
    float alpha, float** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue);

template
void reduce(
    int64_t m, int64_t n, int64_t mt,
    double alpha, double** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue);

template
void reduce(
    int64_t m, int64_t n, int64_t mt,
    cuFloatComplex alpha, cuFloatComplex** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue);

template
void reduce(
    int64_t m, int64_t n, int64_t mt,
    cuDoubleComplex alpha, cuDoubleComplex** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue &queue);

} // namespace device
} // namespace slate
