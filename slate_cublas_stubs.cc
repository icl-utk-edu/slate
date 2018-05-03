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

///-----------------------------------------------------------------------------
/// \file
///
#include "slate_cublas.hh"

#include <cassert>

#ifdef __cplusplus
extern "C" {
#endif

cublasStatus_t cublasCreate(cublasHandle_t* handle)
{
    assert(0);
}

cublasStatus_t cublasDestroy(cublasHandle_t handle) 
{
    assert(0);
}

cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId)
{
    assert(0);
}

cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t* streamId)
{
    assert(0);
}

cublasStatus_t cublasDasum(cublasHandle_t handle, int n, const double* x, int incx, double* result)
{
    assert(0);
}

cublasStatus_t cublasSgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha, const float* Aarray[], int lda,
                        const float* Barray[], int ldb,
    const float* beta,        float* Carray[], int ldc,
    int batchCount)
{
    assert(0);
}

cublasStatus_t cublasDgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double* alpha, const double* Aarray[], int lda,
                         const double* Barray[], int ldb,
    const double* beta,        double* Carray[], int ldc,
    int batchCount)
{
    assert(0);
}

cublasStatus_t cublasCgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const cuComplex* alpha, const cuComplex* Aarray[], int lda,
                            const cuComplex* Barray[], int ldb,
    const cuComplex* beta,        cuComplex* Carray[], int ldc,
    int batchCount)
{
    assert(0);
}

cublasStatus_t cublasZgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const cuDoubleComplex* alpha, const cuDoubleComplex* Aarray[], int lda,
                                  const cuDoubleComplex* Barray[], int ldb,
    const cuDoubleComplex* beta,        cuDoubleComplex* Carray[], int ldc,
    int batchCount)
{
    assert(0);
}

#ifdef __cplusplus
}
#endif
