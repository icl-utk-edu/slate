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

#include "slate_device.hh"

#include <cstdio>

namespace slate {
namespace device {

//------------------------------------------------------------------------------
//  max that propogates nan consistently:
//  max_nan( 1,   nan ) = nan
//  max_nan( nan, 1   ) = nan
//
//  For x=nan, y=1:
//  nan < y is false, yields x (nan)
//
//  For x=1, y=nan:
//  x < nan    is false, would yield x, but
//  isnan(nan) is true, yields y (nan)
//
template< typename T >
__host__ __device__
inline T max_nan( T x, T y )
{
    return (isnan(y) || (x) < (y) ? (y) : (x));
}

//------------------------------------------------------------------------------
__global__ void genormMaxKernel(int64_t m, int64_t n,
                                float** a, int64_t lda,
                                float* max)
{}

__global__ void genormMaxKernel(int64_t m, int64_t n,
                                double** a, int64_t lda,
                                double* max);

__global__ void genormMaxKernel(int64_t m, int64_t n,
                                std::complex<float>** a, int64_t lda,
                                float* max)
{}

__global__ void genormMaxKernel(int64_t m, int64_t n,
                                std::complex<double>** a, int64_t lda,
                                double* max)
{}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
void genormMax(
    int64_t m, int64_t n,
    scalar_t** a, int64_t lda,
    real_type<scalar_t>* max,
    int64_t batch_count,
    cudaStream_t stream)
{
    dim3 dimBlock(n);
    dim3 dimGrid(batch_count);
    genormMaxKernel<<<dimGrid, dimBlock, sizeof(double)*n, stream>>>
        (m, n, a, lda, max);
}

//----------------------------------------
// instantiations
template
void genormMax(
    int64_t m, int64_t n,
    float** a, int64_t lda,
    float* max,
    int64_t batch_count,
    cudaStream_t stream);

template
void genormMax(
    int64_t m, int64_t n,
    double** a, int64_t lda,
    double* max,
    int64_t batch_count,
    cudaStream_t stream);

template
void genormMax(
    int64_t m, int64_t n,
    std::complex<float>** a, int64_t lda,
    float* max,
    int64_t batch_count,
    cudaStream_t stream);

template
void genormMax(
    int64_t m, int64_t n,
    std::complex<double>** a, int64_t lda,
    double* max,
    int64_t batch_count,
    cudaStream_t stream);

///-----------------------------------------------------------------------------
/// \brief
///
__global__ void genormMaxKernel(int64_t m, int64_t n,
                                double** tiles, int64_t lda,
                                double* tiles_maxima)
{
    double* tile = tiles[blockIdx.x];
    double* column = &tile[lda*threadIdx.x];
    double tile_max;

    extern __shared__ double col_max[];

    double max = column[0];
    for (int64_t i = 1; i < m; ++i)
        max = max_nan(max, column[i]);

    col_max[threadIdx.x] = max;
    __syncthreads();

    if (threadIdx.x == 0) {
        tile_max = col_max[0];
        for (int64_t j = 1; j < n; ++j)
            tile_max = max_nan(tile_max, col_max[j]);

        tiles_maxima[blockIdx.x] = tile_max;
    }
}

} // namespace device
} // namespace slate
