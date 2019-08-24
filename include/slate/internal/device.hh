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

#ifndef SLATE_DEVICE_HH
#define SLATE_DEVICE_HH

#include "slate/internal/cuda.hh"
#include "slate/enums.hh"

#include <blas.hh>
#include <lapack.hh>

//------------------------------------------------------------------------------
// Extend BLAS real_type to cover cuComplex
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

namespace slate {

/// @namespace slate::device
/// GPU device implementations of kernels.
namespace device {

//------------------------------------------------------------------------------
template <typename src_scalar_t, typename dst_scalar_t>
void gecopy(
    int64_t m, int64_t n,
    src_scalar_t** Aarray, int64_t lda,
    dst_scalar_t** Barray, int64_t ldb,
    int64_t batch_count, cudaStream_t stream);

//------------------------------------------------------------------------------
template <typename src_scalar_t, typename dst_scalar_t>
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    src_scalar_t** Aarray, int64_t lda,
    dst_scalar_t** Barray, int64_t ldb,
    int64_t batch_count, cudaStream_t stream);

//------------------------------------------------------------------------------
template <typename scalar_t>
void geadd(
    int64_t m, int64_t n,
    scalar_t alpha, scalar_t** Aarray, int64_t lda,
    scalar_t beta, scalar_t** Barray, int64_t ldb,
    int64_t batch_count, cudaStream_t stream);

//------------------------------------------------------------------------------
template <typename scalar_t>
void geset(
    int64_t m, int64_t n,
    scalar_t alpha, scalar_t beta, scalar_t** Aarray, int64_t lda,
    int64_t batch_count, cudaStream_t stream);

//------------------------------------------------------------------------------
template <typename scalar_t>
void genorm(
    lapack::Norm norm, NormScope scope,
    int64_t m, int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* values, int64_t ldv,
    int64_t batch_count,
    cudaStream_t stream);

//------------------------------------------------------------------------------
template <typename scalar_t>
void henorm(
    lapack::Norm norm, lapack::Uplo uplo,
    int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* values, int64_t ldv,
    int64_t batch_count,
    cudaStream_t stream);

//------------------------------------------------------------------------------
template <typename scalar_t>
void synorm(
    lapack::Norm norm, lapack::Uplo uplo,
    int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* values, int64_t ldv,
    int64_t batch_count,
    cudaStream_t stream);

//------------------------------------------------------------------------------
template <typename scalar_t>
void synormOffdiag(
    lapack::Norm norm,
    int64_t m, int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* values, int64_t ldv,
    int64_t batch_count,
    cudaStream_t stream);

//------------------------------------------------------------------------------
template <typename scalar_t>
void trnorm(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag,
    int64_t m, int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* values, int64_t ldv,
    int64_t batch_count,
    cudaStream_t stream);

//------------------------------------------------------------------------------
template <typename scalar_t>
void transpose(
    int64_t n,
    scalar_t* A, int64_t lda,
    cudaStream_t stream);

//------------------------------------------------------------------------------
template <typename scalar_t>
void transpose_batch(
    int64_t n,
    scalar_t** Aarray, int64_t lda,
    int64_t batch_count,
    cudaStream_t stream);

//------------------------------------------------------------------------------
template <typename scalar_t>
void transpose(
    int64_t m, int64_t n,
    scalar_t* dA,  int64_t lda,
    scalar_t* dAT, int64_t ldat,
    cudaStream_t stream);

//------------------------------------------------------------------------------
template <typename scalar_t>
void transpose_batch(
    int64_t m, int64_t n,
    scalar_t** dA_array,  int64_t lda,
    scalar_t** dAT_array, int64_t ldat,
    int64_t batch_count,
    cudaStream_t stream);

} // namespace device
} // namespace slate

#endif // SLATE_DEVICE_HH
