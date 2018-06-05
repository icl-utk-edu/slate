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
#include <cuComplex.h>

namespace slate {
namespace device {

//------------------------------------------------------------------------------
/// max that propogates nan consistently:
///     max_nan( 1,   nan ) = nan
///     max_nan( nan, 1   ) = nan
///
/// For x=nan, y=1:
/// nan < y is false, yields x (nan)
///
/// For x=1, y=nan:
/// x < nan    is false, would yield x, but
/// isnan(nan) is true, yields y (nan)
///
template <typename T>
__host__ __device__
inline T max_nan(T x, T y)
{
    return (isnan(y) || (x) < (y) ? (y) : (x));
}

//------------------------------------------------------------------------------
/// Overloaded versions of absolute value on device.
__host__ __device__
inline float abs(float x)
{
    return fabsf(x);
}

__host__ __device__
inline double abs(double x)
{
    return fabs(x);
}

__host__ __device__
inline float abs(std::complex<float> x)
{
    return cuCabsf(*((cuFloatComplex*)&x));
}

__host__ __device__
inline double abs(std::complex<double> x)
{
    return cuCabs(*((cuDoubleComplex*)&x));
}

//------------------------------------------------------------------------------
/// Returns the largest absolute value of elements for each tile in tiles.
/// Each thread block deals with one tile.
/// Each thread deals with one column, followed by a reduction.
/// Kernel assumes non-trivial tiles (m, n >= 1).
/// Launched by genormMax().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///     Also the number of threads per block, hence,
///     n <= 1024 for current CUDA architectures (2.x to 6.x).
///
/// @param[in] tiles
///     Array of tiles of dimension blockDim.x,
///     where each tiles[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= m.
///
/// @param[out] tiles_maxima
///     Array of dimension blockDim.x.
///     On exit, tiles_maxima[k] = max_{i, j}( abs( A^(k)_(i, j) )).
///
template <typename scalar_t>
__global__ void genormMaxKernel(
    int64_t m, int64_t n,
    scalar_t const* const* tiles, int64_t lda,
    blas::real_type<scalar_t>* tiles_maxima)
{
    using real_type = blas::real_type<scalar_t>;
    scalar_t const* tile = tiles[blockIdx.x];
    scalar_t const* column = &tile[lda*threadIdx.x];
    real_type tile_max;

    // Each thread finds max of one column.
    extern __shared__ char dynamic_data[];
    real_type* col_max = (real_type*) dynamic_data;

    real_type max = abs(column[0]);
    for (int64_t i = 1; i < m; ++i)
        max = max_nan(max, abs(column[i]));

    col_max[threadIdx.x] = max;
    __syncthreads();

    // Reduction to find max of tile.
    // todo: parallel reduction.
    if (threadIdx.x == 0) {
        tile_max = col_max[0];
        for (int64_t j = 1; j < n; ++j)
            tile_max = max_nan(tile_max, col_max[j]);

        tiles_maxima[blockIdx.x] = tile_max;
    }
}

//------------------------------------------------------------------------------
/// Batched routine that returns the largest absolute value of elements for
/// each tile in Aarray. Sets
///     tiles_maxima[k] = max_{i, j}( abs( A^(k)_(i, j) )),
/// for each tile A^(k), where
/// A^(k) = Aarray[k],
/// k = 0, ..., blockDim.x-1,
/// i = 0, ..., m-1,
/// j = 0, ..., n-1.
///
/// @param[in] m
///     Number of rows of each tile. m >= 0.
///
/// @param[in] n
///     Number of columns of each tile. n >= 0.
///     Currently, n <= 1024 due to CUDA implementation.
///
/// @param[in] Aarray
///     Array of dimension batch_count, containing pointers to tiles,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= m.
///
/// @param[out] tiles_maxima
///     Array of dimension batch_count.
///     On exit, tiles_maxima[k] = max_{i, j}( abs( A^(k)_(i, j) )).
///
/// @param[in] batch_count
///     Size of Aarray. batch_count >= 0.
///
template <typename scalar_t>
void genormMax(
    int64_t m, int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* tiles_maxima,
    int64_t batch_count,
    cudaStream_t stream)
{
    assert(n <= 1024);
    using real_type = blas::real_type<scalar_t>;
    if (batch_count == 0) {
        // nothing to do
    }
    else if (m == 0 || n == 0) {
        cudaMemset(tiles_maxima, 0, sizeof(real_type)*batch_count);
    }
    else {
        // Max 1024 threads * 16 bytes = 16 KiB shared memory in double-complex.
        dim3 dimBlock(blas::max(1, n));
        dim3 dimGrid(batch_count);
        genormMaxKernel<<<dimGrid, dimBlock, sizeof(scalar_t)*n, stream>>>
            (m, n, Aarray, lda, tiles_maxima);
    }
}

//----------------------------------------
// instantiations
template
void genormMax(
    int64_t m, int64_t n,
    float const* const* Aarray, int64_t lda,
    float* tiles_maxima,
    int64_t batch_count,
    cudaStream_t stream);

template
void genormMax(
    int64_t m, int64_t n,
    double const* const* Aarray, int64_t lda,
    double* tiles_maxima,
    int64_t batch_count,
    cudaStream_t stream);

template
void genormMax(
    int64_t m, int64_t n,
    std::complex<float> const* const* Aarray, int64_t lda,
    float* tiles_maxima,
    int64_t batch_count,
    cudaStream_t stream);

template
void genormMax(
    int64_t m, int64_t n,
    std::complex<double> const* const* Aarray, int64_t lda,
    double* tiles_maxima,
    int64_t batch_count,
    cudaStream_t stream);

} // namespace device
} // namespace slate
