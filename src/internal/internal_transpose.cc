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
#include "internal/internal_batch.hh"
#include "internal/internal.hh"
#include "slate/internal/util.hh"
#include "internal/Tile_lapack.hh"
#include "slate/types.hh"

#include <vector>

namespace slate {

//------------------------------------------------------------------------------
// On macOS, nvcc using clang++ generates a different C++ name mangling
// (std::__1::complex) than g++ for std::complex. This solution is to use
// cu*Complex in .cu files, and cast from std::complex here.
namespace device {

template <>
void transpose(
    int64_t n,
    std::complex<float>* A, int64_t lda,
    cudaStream_t stream)
{
#if !defined(SLATE_NO_CUDA) || defined(__NVCC__)
    transpose(n, (cuFloatComplex*) A, lda, stream);
#endif
}

template <>
void transpose(
    int64_t n,
    std::complex<double>* A, int64_t lda,
    cudaStream_t stream)
{
#if !defined(SLATE_NO_CUDA) || defined(__NVCC__)
    transpose(n, (cuDoubleComplex*) A, lda, stream);
#endif
}

//--------------------
template <>
void transpose_batch(
    int64_t n,
    std::complex<float>** Aarray, int64_t lda,
    int64_t batch_count,
    cudaStream_t stream)
{
#if !defined(SLATE_NO_CUDA) || defined(__NVCC__)
    transpose_batch(n, (cuFloatComplex**) Aarray, lda, batch_count, stream);
#endif
}

template <>
void transpose_batch(
    int64_t n,
    std::complex<double>** Aarray, int64_t lda,
    int64_t batch_count,
    cudaStream_t stream)
{
#if !defined(SLATE_NO_CUDA) || defined(__NVCC__)
    transpose_batch(n, (cuDoubleComplex**) Aarray, lda, batch_count, stream);
#endif
}

//--------------------
template <>
void transpose(
    int64_t m, int64_t n,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* AT, int64_t ldat,
    cudaStream_t stream)
{
#if !defined(SLATE_NO_CUDA) || defined(__NVCC__)
    transpose(m, n, (cuFloatComplex*) A, lda, (cuFloatComplex*) AT, ldat, stream);
#endif
}

template <>
void transpose(
    int64_t m, int64_t n,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* AT, int64_t ldat,
    cudaStream_t stream)
{
#if !defined(SLATE_NO_CUDA) || defined(__NVCC__)
    transpose(m, n, (cuDoubleComplex*) A, lda, (cuDoubleComplex*) AT, ldat, stream);
#endif
}

//--------------------
template <>
void transpose_batch(
    int64_t m, int64_t n,
    std::complex<float>** Aarray, int64_t lda,
    std::complex<float>** ATarray, int64_t ldat,
    int64_t batch_count,
    cudaStream_t stream)
{
#if !defined(SLATE_NO_CUDA) || defined(__NVCC__)
    transpose_batch(m, n, (cuFloatComplex**) Aarray, lda, (cuFloatComplex**) ATarray, ldat, batch_count, stream);
#endif
}

template <>
void transpose_batch(
    int64_t m, int64_t n,
    std::complex<double>** Aarray, int64_t lda,
    std::complex<double>** ATarray, int64_t ldat,
    int64_t batch_count,
    cudaStream_t stream)
{
#if !defined(SLATE_NO_CUDA) || defined(__NVCC__)
    transpose_batch(m, n, (cuDoubleComplex**) Aarray, lda, (cuDoubleComplex**) ATarray, ldat, batch_count, stream);
#endif
}


//------------------------------------------------------------------------------
#if defined(SLATE_NO_CUDA)
// Specializations to allow compilation without CUDA.
template <>
void transpose(
    int64_t n,
    float* A, int64_t lda,
    cudaStream_t stream)
{
}

template <>
void transpose(
    int64_t n,
    double* A, int64_t lda,
    cudaStream_t stream)
{
}

//--------------------
template <>
void transpose_batch(
    int64_t n,
    float** Aarray, int64_t lda,
    int64_t batch_count,
    cudaStream_t stream)
{
}

template <>
void transpose_batch(
    int64_t n,
    double** Aarray, int64_t lda,
    int64_t batch_count,
    cudaStream_t stream)
{
}

//--------------------
template <>
void transpose(
    int64_t m, int64_t n,
    float* A, int64_t lda,
    float* AT, int64_t ldat,
    cudaStream_t stream)
{
}

template <>
void transpose(
    int64_t m, int64_t n,
    double* A, int64_t lda,
    double* AT, int64_t ldat,
    cudaStream_t stream)
{
}

//--------------------
template <>
void transpose_batch(
    int64_t m, int64_t n,
    float** Aarray, int64_t lda,
    float** ATarray, int64_t ldat,
    int64_t batch_count,
    cudaStream_t stream)
{
}

template <>
void transpose_batch(
    int64_t m, int64_t n,
    double** Aarray, int64_t lda,
    double** ATarray, int64_t ldat,
    int64_t batch_count,
    cudaStream_t stream)
{
}
#endif // not SLATE_NO_CUDA

} // namespace device
} // namespace slate
