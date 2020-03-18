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
// For assistance with SLATE, email <slate-user@icl.utk.edu>.
// You can also join the "SLATE User" Google group by going to
// https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user,
// signing in with your Google credentials, and then clicking "Join group".
//------------------------------------------------------------------------------

#include "slate/internal/device.hh"
#include "device_util.cuh"

#include <cstdio>
#include <cuComplex.h>

namespace slate {
namespace device {

//------------------------------------------------------------------------------
/// Kernel implementing element-wise tile addition.
/// Each thread block deals with one tile.
/// Each thread deals with one row.
/// Launched by geadd().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///
/// @param[in] Atiles
///     Array of tiles of dimension gridDim.x,
///     where each Atiles[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile in Atiles. lda >= m.
///
/// @param[in] Btiles
///     Array of tiles of dimension gridDim.x,
///     where each Btiles[k] is an m-by-n matrix stored in an ldb-by-n array.
///
/// @param[in] ldb
///     Leading dimension of each tile in Btiles. ldb >= m.
///
template <typename scalar_t>
__global__ void geaddKernel(
    int64_t m, int64_t n,
    scalar_t alpha, scalar_t** tilesA, int64_t lda,
    scalar_t beta, scalar_t** tilesB, int64_t ldb)
{
    scalar_t* tileA = tilesA[blockIdx.x];
    scalar_t* tileB = tilesB[blockIdx.x];

    // thread per row, if more rows than threads, loop by blockDim.x
    for (int64_t ridx = threadIdx.x; ridx < m; ridx += blockDim.x) {
        // todo: should the increment be ridx += 1024?
        scalar_t* rowA = &tileA[ridx];
        scalar_t* rowB = &tileB[ridx];

        for (int64_t j = 0; j < n; ++j)
            rowB[j*ldb] = axpby(alpha, rowA[j*lda], beta, rowB[j*ldb]);
    }
}

//------------------------------------------------------------------------------
/// Batched routine for element-wise tile addition.
///
/// @param[in] m
///     Number of rows of each tile. m >= 0.
///
/// @param[in] n
///     Number of columns of each tile. n >= 0.
///
/// @param[in] Aarray
///     Array in GPU memory of dimension batch_count, containing pointers to tiles,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] lda
///     Leading dimension of each tile in A. lda >= m.
///
/// @param[out] Barray
///     Array in GPU memory of dimension batch_count, containing pointers to tiles,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] ldb
///     Leading dimension of each tile in B. ldb >= m.
///
/// @param[in] batch_count
///     Size of Aarray and Barray. batch_count >= 0.
///
/// @param[in] stream
///     CUDA stream to execute in.
///
template <typename scalar_t>
void geadd(
    int64_t m, int64_t n,
    scalar_t alpha, scalar_t** Aarray, int64_t lda,
    scalar_t beta, scalar_t** Barray, int64_t ldb,
    int64_t batch_count, cudaStream_t stream)
{
    // quick return
    if (batch_count == 0)
        return;

    // Max threads/block=1024 for current CUDA compute capability (<=7.5)
    int64_t nthreads = std::min((int64_t)1024 , m);

    geaddKernel<<<batch_count, nthreads, 0, stream>>>(
        m, n,
        alpha, Aarray, lda,
        beta, Barray, ldb);

    // check that launch succeeded (could still have async errors)
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::exception();
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void geadd(
    int64_t m, int64_t n,
    float alpha, float** Aarray, int64_t lda,
    float beta, float** Barray, int64_t ldb,
    int64_t batch_count, cudaStream_t stream);

template
void geadd(
    int64_t m, int64_t n,
    double alpha, double** Aarray, int64_t lda,
    double beta, double** Barray, int64_t ldb,
    int64_t batch_count, cudaStream_t stream);

template
void geadd(
    int64_t m, int64_t n,
    cuFloatComplex alpha, cuFloatComplex** Aarray, int64_t lda,
    cuFloatComplex beta, cuFloatComplex** Barray, int64_t ldb,
    int64_t batch_count, cudaStream_t stream);

template
void geadd(
    int64_t m, int64_t n,
    cuDoubleComplex alpha, cuDoubleComplex** Aarray, int64_t lda,
    cuDoubleComplex beta, cuDoubleComplex** Barray, int64_t ldb,
    int64_t batch_count, cudaStream_t stream);

} // namespace device
} // namespace slate
