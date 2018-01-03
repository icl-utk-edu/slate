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

#ifndef SLATE_TILE_HH
#define SLATE_TILE_HH

#include "slate_Memory.hh"

#include "blas.hh"
#include "lapack.hh"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

#ifdef SLATE_WITH_CUDA
    #include <cuda_runtime.h>
#else
    #include "slate_NoCuda.hh"
#endif

#ifdef SLATE_WITH_MPI
    #include <mpi.h>
#else
    #include "slate_NoMpi.hh"
#endif

#ifdef SLATE_WITH_OPENMP
    #include <omp.h>
#else
    #include "slate_NoOpenmp.hh"
#endif

extern "C" void trace_cpu_start();
extern "C" void trace_cpu_stop(const char *color);

namespace slate {

///-----------------------------------------------------------------------------
/// \class
/// \brief
///
template <typename FloatType>
class Tile {
public:
    Tile() {}

    Tile(int64_t mb, int64_t nb, Memory *memory)
        : mb_(mb), nb_(nb), memory_(memory),
          device_num_(host_num_), valid_(true), origin_(false) {}

    virtual ~Tile() {}

    virtual void copyTo(FloatType *a, int64_t lda) = 0;
    virtual void copyFrom(FloatType *a, int64_t lda) = 0;

    virtual Tile<FloatType>* copyToHost(cudaStream_t stream) = 0;
    virtual Tile<FloatType>* copyToDevice(int device_num,
                                          cudaStream_t stream) = 0;

    Tile<FloatType>* copyDataToHost(const Tile<FloatType> *dst_tile,
                                    cudaStream_t stream);
    Tile<FloatType>* copyDataToDevice(const Tile<FloatType> *dst_tile,
                                      cudaStream_t stream);

    static void gemm(blas::Op transa, blas::Op transb,
                     FloatType alpha, Tile<FloatType> *a,
                                      Tile<FloatType> *b,
                     FloatType beta,  Tile<FloatType> *c);

    static void potrf(blas::Uplo uplo, Tile<FloatType> *a);

    static void syrk(blas::Uplo uplo, blas::Op trans,
                     FloatType alpha, Tile<FloatType> *a,
                     FloatType beta,  Tile<FloatType> *c);

    static void trsm(blas::Side side, blas::Uplo uplo,
                     blas::Op transa, blas::Diag diag,
                     FloatType alpha, Tile<FloatType> *a,
                                      Tile<FloatType> *b);
    
    int64_t mb_;
    int64_t nb_;
    int64_t stride_;

    FloatType *data_;

    bool valid_;
    bool origin_;

protected:
    size_t size() { return sizeof(FloatType)*mb_*nb_; }
    void allocate();
    void deallocate();

    static int host_num_;
    int device_num_;

    Memory *memory_;
};

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
Tile<FloatType>* Tile<FloatType>::copyDataToHost(const Tile<FloatType> *dst_tile,
                                cudaStream_t stream)
{
    trace_cpu_start();
    cudaError_t error;
    error = cudaSetDevice(device_num_);
    assert(error == cudaSuccess);

    error = cudaMemcpyAsync(dst_tile->data_, data_, size(),
                            cudaMemcpyDeviceToHost, stream);
    assert(error == cudaSuccess);

    error = cudaStreamSynchronize(stream);
    assert(error == cudaSuccess);
    trace_cpu_stop("Gray");
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
Tile<FloatType>* Tile<FloatType>::copyDataToDevice(const Tile<FloatType> *dst_tile,
                                  cudaStream_t stream)
{
    trace_cpu_start();
    cudaError_t error;
    error = cudaSetDevice(dst_tile->device_num_);
    assert(error == cudaSuccess);

    error = cudaMemcpyAsync(dst_tile->data_, data_, size(),
                            cudaMemcpyHostToDevice, stream);
    assert(error == cudaSuccess);

    error = cudaStreamSynchronize(stream);
    assert(error == cudaSuccess);
    trace_cpu_stop("LightGray");
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
void Tile<FloatType>::gemm(blas::Op transa, blas::Op transb,
                 FloatType alpha, Tile<FloatType> *a,
                                  Tile<FloatType> *b,
                 FloatType beta,  Tile<FloatType> *c)
{
    trace_cpu_start();
    blas::gemm(blas::Layout::ColMajor,
               transa, transb,
               c->mb_, c->nb_, a->nb_,
               alpha, a->data_, a->stride_,
                      b->data_, b->stride_,
               beta,  c->data_, c->stride_);
    trace_cpu_stop("MediumAquamarine");
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
void Tile<FloatType>::potrf(blas::Uplo uplo, Tile<FloatType> *a)
{
    trace_cpu_start();
    lapack::potrf(blas::Layout::ColMajor,
                  uplo,
                  a->nb_,
                  a->data_, a->stride_);
    trace_cpu_stop("RosyBrown");
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
void Tile<FloatType>::syrk(blas::Uplo uplo, blas::Op trans,
                 FloatType alpha, Tile<FloatType> *a,
                 FloatType beta,  Tile<FloatType> *c)
{
    trace_cpu_start();
    blas::syrk(blas::Layout::ColMajor,
               uplo, trans,
               c->nb_, a->nb_,
               alpha, a->data_, a->stride_,
               beta,  c->data_, c->stride_);
    trace_cpu_stop("CornflowerBlue");
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
void Tile<FloatType>::trsm(blas::Side side, blas::Uplo uplo,
                 blas::Op transa, blas::Diag diag,
                 FloatType alpha, Tile<FloatType> *a,
                                  Tile<FloatType> *b)
{
    trace_cpu_start();
    blas::trsm(blas::Layout::ColMajor,
               side, uplo, transa, diag,
               b->mb_, b->nb_,
               alpha, a->data_, a->stride_,
                      b->data_, b->stride_);
    trace_cpu_stop("MediumPurple");
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
void Tile<FloatType>::allocate()
{
    trace_cpu_start();
    data_ = (FloatType*)memory_->alloc(device_num_);
    trace_cpu_stop("Orchid");
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
void Tile<FloatType>::deallocate()
{
    trace_cpu_start();
    memory_->free(data_, device_num_);
    data_ = nullptr;
    trace_cpu_stop("Crimson");
}

} // namespace slate

#endif // SLATE_TILE_HH
