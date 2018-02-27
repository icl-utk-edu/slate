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

#include "slate_Matrix.hh"
#include "slate_types.hh"
#include "slate_Tile_blas.hh"
#include "slate_internal.hh"

#ifdef SLATE_WITH_MKL
    #include <mkl_cblas.h>
#else
    #include <cblas.h>
#endif

namespace slate {
namespace internal {

///-----------------------------------------------------------------------------
/// \brief
/// Symmetric rank-k update of single block column (i.e., k = nb).
/// Dispatches to target implementations.
template <Target target, typename scalar_t>
void syrk(scalar_t alpha, Matrix< scalar_t >&& A,
          scalar_t beta,  SymmetricMatrix< scalar_t >&& C,
          int priority)
{
    syrk(internal::TargetType<target>(),
         alpha, A,
         beta,  C,
         priority);
}

///-----------------------------------------------------------------------------
/// \brief
/// Symmetric rank-k update of single block column (i.e., k = nb).
/// Host OpenMP task implementation.
template <typename scalar_t>
void syrk(internal::TargetType<Target::HostTask>,
          scalar_t alpha, Matrix< scalar_t >& A,
          scalar_t beta,  SymmetricMatrix< scalar_t >& C,
          int priority)
{
    // Lower, NoTrans
    for (int64_t j = 0; j < C.nt(); ++j)
        for (int64_t i = j; i < C.mt(); ++i)  // lower
            if (C.tileIsLocal(i, j)) {
                if (i == j) {
                    #pragma omp task shared(A, C) priority(priority)
                    {
                        A.tileCopyToHost(j, 0, A.tileDevice(j, 0));
                        C.tileMoveToHost(j, j, C.tileDevice(j, j));
                        syrk(-1.0, A(j, 0),
                             beta, C(j, j));
                        A.tileTick(j, 0);
                        A.tileTick(j, 0);
                    }
                }
                else {
                    #pragma omp task shared(A, C) priority(priority)
                    {
                        A.tileCopyToHost(i, 0, A.tileDevice(i, 0));
                        A.tileCopyToHost(j, 0, A.tileDevice(j, 0));
                        C.tileMoveToHost(i, j, C.tileDevice(i, j));
                        auto Aj0 = A(j, 0);
                        gemm(alpha, A(i, 0),
                                    transpose(Aj0),
                             beta,  C(i, j));
                        A.tileTick(i, 0);
                        A.tileTick(j, 0);
                    }
                }
            }

    #pragma omp taskwait
}

///-----------------------------------------------------------------------------
/// \brief
/// Symmetric rank-k update of single block column (i.e., k = nb).
/// Host nested OpenMP task implementation.
template <typename scalar_t>
void syrk(internal::TargetType<Target::HostNest>,
          scalar_t alpha, Matrix< scalar_t >& A,
          scalar_t beta,  SymmetricMatrix< scalar_t >& C,
          int priority)
{
    // Lower, NoTrans
    for (int64_t j = 0; j < C.nt(); ++j)
        if (C.tileIsLocal(j, j))
            #pragma omp task shared(A, C)
            {
                A.tileCopyToHost(j, 0, A.tileDevice(j, 0));
                C.tileMoveToHost(j, j, C.tileDevice(j, j));
                syrk(-1.0, A(j, 0),
                     beta, C(j, j));
                A.tileTick(j, 0);
                A.tileTick(j, 0);
            }

//  #pragma omp parallel for collapse(2) schedule(dynamic, 1) num_threads(...)
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int64_t j = 0; j < C.nt(); ++j)
        for (int64_t i = 0; i < C.mt(); ++i)  // full
            if (i >= j+1)                     // lower
                if (C.tileIsLocal(i, j))
                {
                    A.tileCopyToHost(i, 0, A.tileDevice(i, 0));
                    A.tileCopyToHost(j, 0, A.tileDevice(j, 0));
                    C.tileMoveToHost(i, j, C.tileDevice(i, j));
                    auto Aj0 = A(j, 0);
                    gemm(alpha, A(i, 0),
                                transpose(Aj0),
                         beta,  C(i, j));
                    A.tileTick(i, 0);
                    A.tileTick(j, 0);
                }

    #pragma omp taskwait
}

///-----------------------------------------------------------------------------
/// \brief
/// Symmetric rank-k update of single block column (i.e., k = nb).
/// Host batched implementation.
template <typename scalar_t>
void syrk(internal::TargetType<Target::HostBatch>,
          scalar_t alpha, Matrix< scalar_t >& A,
          scalar_t beta,  SymmetricMatrix< scalar_t >& C,
          int priority)
{
    // diagonal tiles by syrk on host
    for (int64_t j = 0; j < C.nt(); ++j)
        if (C.tileIsLocal(j, j))
            #pragma omp task shared(A, C)
            {
                A.tileCopyToHost(j, 0, A.tileDevice(j, 0));
                C.tileMoveToHost(j, j, C.tileDevice(j, j));
                syrk(-1.0, A(j, 0),
                     beta, C(j, j));
                A.tileTick(j, 0);
                A.tileTick(j, 0);
            }

    // load off-diagonal tiles to host, if not there
    for (int64_t j = 0; j < C.nt(); ++j)
        for (int64_t i = j+1; i < C.mt(); ++i)  // lower
            if (C.tileIsLocal(i, j)) {
                A.tileCopyToHost(i, 0, A.tileDevice(i, 0));
                A.tileCopyToHost(j, 0, A.tileDevice(j, 0));
                C.tileMoveToHost(i, j, C.tileDevice(i, j));
            }

    // off-diagonal tiles by batch gemm on host
    CBLAS_TRANSPOSE opa_array[1];
    CBLAS_TRANSPOSE opb_array[1];
    int m_array[1];
    int n_array[1];
    int k_array[1];
    scalar_t alpha_array[1];
    const scalar_t **a_array;
    int lda_array[1];
    const scalar_t **b_array;
    int ldb_array[1];
    scalar_t beta_array[1];
    scalar_t **c_array;
    int ldc_array[1];

    int nb = C.tileNb(0);
    opa_array[0] = (A.op() == Op::NoTrans ? CblasNoTrans : CblasTrans);
    opb_array[0] = (A.op() == Op::NoTrans ? CblasTrans : CblasNoTrans);
    m_array[0] = nb;
    n_array[0] = nb;
    k_array[0] = nb;
    alpha_array[0] = alpha;
    lda_array[0] = nb;
    ldb_array[0] = nb;
    beta_array[0] = beta;
    ldc_array[0] = nb;

    int group_size = 0;
    for (int64_t j = 0; j < C.nt(); ++j)
        for (int64_t i = j+1; i < C.mt(); ++i)  // lower
            if (C.tileIsLocal(i, j))
                ++group_size;

    a_array = new const scalar_t*[group_size];
    b_array = new const scalar_t*[group_size];
    c_array = new scalar_t*[group_size];

    int batch_index = 0;
    for (int64_t j = 0; j < C.nt(); ++j)
        for (int64_t i = j+1; i < C.mt(); ++i)  // lower
            if (C.tileIsLocal(i, j)) {
                a_array[ batch_index ] = A(i, 0).data();
                b_array[ batch_index ] = A(j, 0).data();
                c_array[ batch_index ] = C(i, j).data();
                ++batch_index;
            }

    {
        trace::Block trace_block("cblas_dgemm_batch");
        #ifdef SLATE_WITH_MKL
            // mkl_set_num_threads_local(...);
            cblas_dgemm_batch(CblasColMajor, opa_array, opb_array,
                               m_array, n_array, k_array,
                               alpha_array,
                               a_array, lda_array,
                               b_array, ldb_array,
                               beta_array,
                               c_array, ldc_array, 1, &group_size);
            // mkl_set_num_threads_local(1);
        #else
            assert(false);
        #endif
    }

    for (int64_t j = 0; j < C.nt(); ++j)
        for (int64_t i = j+1; i < C.mt(); ++i)  // lower
            if (C.tileIsLocal(i, j)) {
                A.tileTick(i, 0);
                A.tileTick(j, 0);
            }

    delete[] a_array;
    delete[] b_array;
    delete[] c_array;

    #pragma omp taskwait
}

///-----------------------------------------------------------------------------
/// \brief
/// Symmetric rank-k update of single block column (i.e., k = nb).
/// GPU device batched cuBLAS implementation.
template <typename scalar_t>
void syrk(internal::TargetType<Target::Devices>,
          scalar_t alpha, Matrix< scalar_t >& A,
          scalar_t beta,  SymmetricMatrix< scalar_t >& C,
          int priority)
{
    // Lower, NoTrans
    assert(C.uplo() == Uplo::Lower);
    assert(A.op() == Op::NoTrans);

    // off-diagonal tiles by batch gemm on device
    for (int device = 0; device < C.num_devices(); ++device) {
        #pragma omp task shared(A, C) priority(1)
        {
            scalar_t** a_array_host = C.a_array_host(device);
            scalar_t** b_array_host = C.b_array_host(device);
            scalar_t** c_array_host = C.c_array_host(device);

            int64_t batch_count = 0;
            for (int64_t j = 0; j < C.nt(); ++j) {
                for (int64_t i = j+1; i < C.mt(); ++i) {  // lower
                    if (C.tileIsLocal(i, j)) {
                        if (device == C.tileDevice(i, j)) {
                            A.tileCopyToDevice(i, 0, device);
                            A.tileCopyToDevice(j, 0, device);
                            C.tileMoveToDevice(i, j, device);
                            a_array_host[ batch_count ] = A(i, 0, device).data();
                            b_array_host[ batch_count ] = A(j, 0, device).data();
                            c_array_host[ batch_count ] = C(i, j, device).data();
                            ++batch_count;
                        }
                    }
                }
            }

            scalar_t** a_array_dev = C.a_array_device(device);
            scalar_t** b_array_dev = C.b_array_device(device);
            scalar_t** c_array_dev = C.c_array_device(device);
            cudaError_t error;
            error = cudaSetDevice(device);
            assert(error == cudaSuccess);

            // cublas_handle uses this stream
            cudaStream_t stream = C.compute_stream(device);
            cublasHandle_t cublas_handle = C.cublas_handle(device);

            error = cudaMemcpyAsync(a_array_dev, a_array_host,
                                    sizeof(scalar_t*)*batch_count,
                                    cudaMemcpyHostToDevice,
                                    stream);
            assert(error == cudaSuccess);

            error = cudaMemcpyAsync(b_array_dev, b_array_host,
                                    sizeof(scalar_t*)*batch_count,
                                    cudaMemcpyHostToDevice,
                                    stream);
            assert(error == cudaSuccess);

            error = cudaMemcpyAsync(c_array_dev, c_array_host,
                                    sizeof(scalar_t*)*batch_count,
                                    cudaMemcpyHostToDevice,
                                    stream);
            assert(error == cudaSuccess);

            {
                trace::Block trace_block("cublasDgemmBatched");
                int nb = C.tileNb(0);
                cublasOperation_t opa = (A.op() == Op::NoTrans ? CUBLAS_OP_N : CUBLAS_OP_C);
                cublasOperation_t opb = (A.op() == Op::NoTrans ? CUBLAS_OP_C : CUBLAS_OP_N);
                cublasStatus_t status =
                    cublasDgemmBatched(
                        cublas_handle,  // uses stream
                        opa, opb,
                        nb, nb, nb,
                        &alpha, (const double**) a_array_dev, nb,
                                (const double**) b_array_dev, nb,
                        &beta,  c_array_dev, nb,
                        batch_count);
                assert(status == CUBLAS_STATUS_SUCCESS);
                error = cudaStreamSynchronize(stream);
                assert(error == cudaSuccess);
            }

            for (int64_t j = 0; j < C.nt(); ++j) {
                for (int64_t i = j+1; i < C.mt(); ++i) {  // lower
                    if (C.tileIsLocal(i, j)) {
                        if (device == C.tileDevice(i, j)) {
                            //A.tileErase(i, 0, device);  // todo: why? shouldn't tileTick deal with this?
                            //A.tileErase(j, 0, device);  // ditto
                            A.tileTick(i, 0);
                            A.tileTick(j, 0);
                        }
                    }
                }
            }
        }
    }

    // diagonal tiles by syrk on host
    for (int64_t j = 0; j < C.nt(); ++j) {
        if (C.tileIsLocal(j, j)) {
            #pragma omp task shared(A, C)
            {
                A.tileCopyToHost(j, 0, A.tileDevice(j, 0));
                C.tileMoveToHost(j, j, C.tileDevice(j, j));
                syrk(-1.0, A(j, 0),
                     beta, C(j, j));
                A.tileTick(j, 0);
                A.tileTick(j, 0);
            }
        }
    }

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
// explicit instantiations
template
void syrk< Target::HostTask, double >(
    double alpha, Matrix< double >&& A,
    double beta,  SymmetricMatrix< double >&& C,
    int priority);

template
void syrk< Target::HostNest, double >(
    double alpha, Matrix< double >&& A,
    double beta,  SymmetricMatrix< double >&& C,
    int priority);

template
void syrk< Target::HostBatch, double >(
    double alpha, Matrix< double >&& A,
    double beta,  SymmetricMatrix< double >&& C,
    int priority);

template
void syrk< Target::Devices, double >(
    double alpha, Matrix< double >&& A,
    double beta,  SymmetricMatrix< double >&& C,
    int priority);

} // namespace internal
} // namespace slate
