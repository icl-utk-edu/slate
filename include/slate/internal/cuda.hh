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

//------------------------------------------------------------------------------
/// @file
///
#ifndef SLATE_CUDA_HH
#define SLATE_CUDA_HH

#include "slate/Exception.hh"

#if (!defined(SLATE_NO_CUDA)) || defined(__NVCC__)
    #include <cuda_runtime.h>
    #include <cuComplex.h>
#else

#include <cstdlib>
#include <complex>

typedef int cudaError_t;
typedef void* cudaStream_t;
typedef int cudaMemcpyKind;
typedef std::complex<float> cuComplex;
typedef cuComplex cuFloatComplex;
typedef std::complex<double> cuDoubleComplex;

enum {
    cudaSuccess,
    cudaStreamNonBlocking,
    cudaMemcpyHostToHost,
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
    cudaMemcpyDeviceToDevice,
    cudaMemcpyDefault
};

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t cudaFree(void* devPtr);
cudaError_t cudaFreeHost(void* ptr);

cudaError_t cudaGetDevice(int* device);
cudaError_t cudaGetDeviceCount(int* count);

cudaError_t cudaMalloc(void** devPtr, size_t size);

cudaError_t cudaMallocHost(void** ptr, size_t size);

cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch,
                              const void* src, size_t spitch,
                              size_t width, size_t height,
                              cudaMemcpyKind kind, cudaStream_t stream = 0);

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream = 0);

cudaError_t cudaMemcpy(void* dst, const void*  src,
                       size_t count, cudaMemcpyKind kind);

cudaError_t cudaSetDevice(int device);

cudaError_t cudaStreamCreate(cudaStream_t* pStream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamDestroy(cudaStream_t pStream);

const char* cudaGetErrorName(cudaError_t error);
const char* cudaGetErrorString(cudaError_t error);

#ifdef __cplusplus
}
#endif

#endif // SLATE_NO_CUDA

namespace slate {

//------------------------------------------------------------------------------
/// Exception class for slate_cuda_call().
class CudaException : public Exception {
public:
    CudaException(const char* call,
                  cudaError_t code,
                  const char* func,
                  const char* file,
                  int line)
        : Exception()
    {
        const char* name = cudaGetErrorName(code);
        const char* string = cudaGetErrorString(code);

        what(std::string("SLATE CUDA ERROR: ")
             + call + " failed: " + string
             + " (" + name + "=" + std::to_string(code) + ")",
             func, file, line);
    }
};

/// Throws a CudaException if the CUDA call fails.
/// Example:
///
///     try {
///         slate_cuda_call( cudaSetDevice( device ) );
///     }
///     catch (CudaException& e) {
///         ...
///     }
///
#define slate_cuda_call(call) \
    do { \
        cudaError_t slate_cuda_call_ = call; \
        if (slate_cuda_call_ != cudaSuccess) \
            throw slate::CudaException( \
                #call, slate_cuda_call_, __func__, __FILE__, __LINE__); \
    } while(0)

} // namespace slate

#endif // SLATE_CUDA_HH
