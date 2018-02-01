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

#ifdef SLATE_WITH_MKL
    #include <mkl_cblas.h>
#else
    #include <cblas.h>
#endif

namespace slate {

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
template <Target target>
void Matrix<scalar_t>::syrk(blas::Uplo uplo, blas::Op op,
                             scalar_t alpha, Matrix &&a,
                             scalar_t beta,  Matrix &&c,
                             int priority)
{
    syrk(internal::TargetType<target>(),
        uplo, op,
        alpha, a,
        beta,  c);
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
void Matrix<scalar_t>::syrk(internal::TargetType<Target::HostTask>,
                             blas::Uplo uplo, blas::Op op,
                             scalar_t alpha, Matrix &a,
                             scalar_t beta,  Matrix &c,
                             int priority)
{
    // Lower, NoTrans
    for (int64_t n = 0; n < c.nt_; ++n)
        for (int64_t m = n; m < c.mt_; ++m)
            if (c.tileIsLocal(m, n)) {
                if (m == n) {
                    #pragma omp task shared(a, c) priority(priority)
                    {
                        a.tileCopyToHost(n, 0, a.tileDevice(n, 0));
                        c.tileMoveToHost(n, n, c.tileDevice(n, n));
                        Tile<scalar_t>::syrk(uplo, op,
                                              -1.0, a(n, 0),
                                              beta, c(n, n));
                        a.tileTick(n, 0);
                        a.tileTick(n, 0);
                    }
                }
                else {
                    #pragma omp task shared(a, c) priority(priority)
                    {
                        a.tileCopyToHost(m, 0, a.tileDevice(m, 0));
                        a.tileCopyToHost(n, 0, a.tileDevice(n, 0));
                        c.tileMoveToHost(m, n, c.tileDevice(m, n));
                        Tile<scalar_t>::gemm(op, blas::Op::Trans,
                                              alpha, a(m, 0),
                                                     a(n, 0),
                                              beta,  c(m, n));
                        a.tileTick(m, 0);
                        a.tileTick(n, 0);
                    }
                }
            }

    #pragma omp taskwait
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
void Matrix<scalar_t>::syrk(internal::TargetType<Target::HostNest>,
                             blas::Uplo uplo, blas::Op op,
                             scalar_t alpha, Matrix &a,
                             scalar_t beta,  Matrix &c,
                             int priority)
{
    // Lower, NoTrans
    for (int64_t n = 0; n < c.nt_; ++n)
        if (c.tileIsLocal(n, n))
            #pragma omp task shared(a, c)
            {
                a.tileCopyToHost(n, 0, a.tileDevice(n, 0));
                c.tileMoveToHost(n, n, c.tileDevice(n, n));
                Tile<scalar_t>::syrk(uplo, op,
                                      -1.0, a(n, 0),
                                      beta, c(n, n));
                a.tileTick(n, 0);
                a.tileTick(n, 0);
            }

//  #pragma omp parallel for collapse(2) schedule(dynamic, 1) num_threads(...)
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int64_t n = 0; n < c.nt_; ++n)
        for (int64_t m = 0; m < c.mt_; ++m)
            if (m >= n+1)
                if (c.tileIsLocal(m, n))
                {
                    a.tileCopyToHost(m, 0, a.tileDevice(m, 0));
                    a.tileCopyToHost(n, 0, a.tileDevice(n, 0));
                    c.tileMoveToHost(m, n, c.tileDevice(m, n));
                    Tile<scalar_t>::gemm(op, blas::Op::Trans,
                                          alpha, a(m, 0),
                                                 a(n, 0),
                                          beta,  c(m, n));
                    a.tileTick(m, 0);
                    a.tileTick(n, 0);
                }

    #pragma omp taskwait
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
void Matrix<scalar_t>::syrk(internal::TargetType<Target::HostBatch>,
                             blas::Uplo uplo, blas::Op op,
                             scalar_t alpha, Matrix &a,
                             scalar_t beta,  Matrix &c,
                             int priority)
{
    for (int64_t n = 0; n < c.nt_; ++n)
        if (c.tileIsLocal(n, n))
            #pragma omp task shared(a, c)
            {
                a.tileCopyToHost(n, 0, a.tileDevice(n, 0));
                c.tileMoveToHost(n, n, c.tileDevice(n, n));
                Tile<scalar_t>::syrk(uplo, op,
                                      -1.0, a(n, 0),
                                      beta, c(n, n));
                a.tileTick(n, 0);
                a.tileTick(n, 0);
            }

    for (int64_t n = 0; n < c.nt_; ++n)
        for (int64_t m = n+1; m < c.mt_; ++m)
            if (c.tileIsLocal(m, n)) {
                a.tileCopyToHost(m, 0, a.tileDevice(m, 0));
                a.tileCopyToHost(n, 0, a.tileDevice(n, 0));
                c.tileMoveToHost(m, n, c.tileDevice(m, n));
            }

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

    int nb = c.tileNb(0);
    opa_array[0] = CblasNoTrans;
    opb_array[0] = CblasTrans;
    m_array[0] = nb;
    n_array[0] = nb;
    k_array[0] = nb;
    alpha_array[0] = alpha;
    lda_array[0] = nb;
    ldb_array[0] = nb;
    beta_array[0] = beta;
    ldc_array[0] = nb;

    int group_size = 0;
    for (int64_t n = 0; n < c.nt_; ++n)
        for (int64_t m = n+1; m < c.mt_; ++m)
            if (c.tileIsLocal(m, n))
                ++group_size;

    a_array = new const scalar_t*[group_size];
    b_array = new const scalar_t*[group_size];
    c_array = new scalar_t*[group_size];

    int i = 0;
    for (int64_t n = 0; n < c.nt_; ++n)
        for (int64_t m = n+1; m < c.mt_; ++m)
            if (c.tileIsLocal(m, n)) {
                a_array[i] = a(m, 0)->data_;
                b_array[i] = a(n, 0)->data_;
                c_array[i] = c(m, n)->data_;
                ++i;
            }

    trace_cpu_start();
//  mkl_set_num_threads_local(...);
    cblas_dgemm_batch(CblasColMajor, opa_array, opb_array,
                      m_array, n_array, k_array, alpha_array,
                      a_array, lda_array, b_array, ldb_array, beta_array,
                      c_array, ldc_array, 1, &group_size);
//  mkl_set_num_threads_local(1);
    trace_cpu_stop("DarkGreen");

    for (int64_t n = 0; n < c.nt_; ++n)
        for (int64_t m = n+1; m < c.mt_; ++m)
            if (c.tileIsLocal(m, n)) {
                a.tileTick(m, 0);
                a.tileTick(n, 0);
            }

    delete[] a_array;
    delete[] b_array;
    delete[] c_array;

    #pragma omp taskwait
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
void Matrix<scalar_t>::syrk(internal::TargetType<Target::Devices>,
                             blas::Uplo uplo, blas::Op op,
                             scalar_t alpha, Matrix &a,
                             scalar_t beta,  Matrix &c,
                             int priority)
{
    for (int device = 0; device < c.num_devices_; ++device)
        #pragma omp task shared(a, c) priority (1)
        {
            int64_t i = 0;
            for (int64_t n = 0; n < c.nt_; ++n)
                for (int64_t m = n+1; m < c.mt_; ++m)
                    if (c.tileIsLocal(m, n))
                        if (device == c.tileDevice(m, n)) {
                            a.tileCopyToDevice(m, 0, device);
                            a.tileCopyToDevice(n, 0, device);
                            c.tileMoveToDevice(m, n, device);
                            c.a_array_h_[device][i] = a(m, 0, device)->data_;
                            c.b_array_h_[device][i] = a(n, 0, device)->data_;
                            c.c_array_h_[device][i] = c(m, n, device)->data_;
                            ++i;
                        }
            int64_t batch_count = i;

            cudaError_t error;
            error = cudaSetDevice(device);
            assert(error == cudaSuccess);

            error = cudaMemcpyAsync(c.a_array_d_[device], c.a_array_h_[device],
                                    sizeof(scalar_t*)*batch_count,
                                    cudaMemcpyHostToDevice,
                                    c.gemm_stream_[device]);
            assert(error == cudaSuccess);

            error = cudaMemcpyAsync(c.b_array_d_[device], c.b_array_h_[device],
                                    sizeof(scalar_t*)*batch_count,
                                    cudaMemcpyHostToDevice,
                                    c.gemm_stream_[device]);
            assert(error == cudaSuccess);

            error = cudaMemcpyAsync(c.c_array_d_[device], c.c_array_h_[device],
                                    sizeof(scalar_t*)*batch_count,
                                    cudaMemcpyHostToDevice,
                                    c.gemm_stream_[device]);
            assert(error == cudaSuccess);

            trace_cpu_start();
            int nb = c.tileNb(0);
            cublasStatus_t status =
                cublasDgemmBatched(
                    c.cublas_handle_[device],
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    nb, nb, nb,
                    &alpha, c.a_array_d_[device], nb,
                            c.b_array_d_[device], nb,
                    &beta,  c.c_array_d_[device], nb,
                    batch_count);
            assert(status == CUBLAS_STATUS_SUCCESS);
            error = cudaStreamSynchronize(c.gemm_stream_[device]);
            assert(error == cudaSuccess);
            trace_cpu_stop("PaleGreen");

            for (int64_t n = 0; n < c.nt_; ++n)
                for (int64_t m = n+1; m < c.mt_; ++m)
                    if (c.tileIsLocal(m, n))
                        if (device == c.tileDevice(m, n)) {
                            a.tileErase(m, 0, device);
                            a.tileErase(n, 0, device);
                            a.tileTick(m, 0);
                            a.tileTick(n, 0);
                        }
        }

    for (int64_t n = 0; n < c.nt_; ++n)
        if (c.tileIsLocal(n, n))
            #pragma omp task shared(a, c)
            {
                a.tileCopyToHost(n, 0, a.tileDevice(n, 0));
                c.tileMoveToHost(n, n, c.tileDevice(n, n));
                Tile<scalar_t>::syrk(uplo, op,
                                          -1.0, a(n, 0),
                                          beta, c(n, n));
                a.tileTick(n, 0);
                a.tileTick(n, 0);
            }

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
template
void Matrix<double>::syrk<Target::HostTask>(
    blas::Uplo uplo, blas::Op op,
    double alpha, Matrix &&a,
    double beta,  Matrix &&c,
    int priority);

template
void Matrix<double>::syrk<Target::HostNest>(
    blas::Uplo uplo, blas::Op op,
    double alpha, Matrix &&a,
    double beta,  Matrix &&c,
    int priority);

template
void Matrix<double>::syrk<Target::HostBatch>(
    blas::Uplo uplo, blas::Op op,
    double alpha, Matrix &&a,
    double beta,  Matrix &&c,
    int priority);

template
void Matrix<double>::syrk<Target::Devices>(
    blas::Uplo uplo, blas::Op op,
    double alpha, Matrix &&a,
    double beta,  Matrix &&c,
    int priority);

} // namespace slate
