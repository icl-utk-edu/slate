
#include "slate_Matrix.hh"

namespace slate {

//------------------------------------------------------------------------------
template<typename FloatType>
void Matrix<FloatType>::syrkTask(blas::Uplo uplo, blas::Op trans,
                                 FloatType alpha, const Matrix &that,
                                 FloatType beta)
{
    using namespace blas;

    Matrix<FloatType> c = *this;
    Matrix<FloatType> a = that;

    // Lower, NoTrans
    for (int64_t n = 0; n < c.nt_; ++n) {

        for (int64_t k = 0; k < a.nt_; ++k)
            #pragma omp task
            if (c.tileIsLocal(n, n)) {
                a.tileWait(n, k);
                c(n, n)->syrk(uplo, trans, -1.0, a(n, k), k == 0 ? beta : 1.0);
            }

        for (int64_t m = n+1; m < c.mt_; ++m)
            for (int64_t k = 0; k < a.nt_; ++k)
                #pragma omp task
                if (c.tileIsLocal(m, n)) {
                    a.tileWait(m, k);
                    a.tileWait(n, k);
                    c(m, n)->gemm(trans, Op::Trans,
                                  alpha, a(m, k), a(n, k), k == 0 ? beta : 1.0);
                }
    }
    #pragma omp taskwait
}

//------------------------------------------------------------------------------
template<typename FloatType>
void Matrix<FloatType>::syrkNest(blas::Uplo uplo, blas::Op trans,
                                 FloatType alpha, const Matrix &that,
                                 FloatType beta)
{
    using namespace blas;

    Matrix<FloatType> c = *this;
    Matrix<FloatType> a = that;

    for (int64_t n = 0; n < c.nt_; ++n) {
        for (int64_t k = 0; k < a.nt_; ++k)
            #pragma omp task
            if (c.tileIsLocal(n, n)) {
                a.tileWait(n, k);
                c(n, n)->syrk(uplo, trans, -1.0, a(n, k), k == 0 ? beta : 1.0);
            }
    }

//  #pragma omp parallel for collapse(3) schedule(dynamic, 1) num_threads(60)
    #pragma omp parallel for collapse(3) schedule(dynamic, 1)
    for (int64_t n = 0; n < c.nt_; ++n) {
        for (int64_t m = 0; m < c.mt_; ++m)
            for (int64_t k = 0; k < a.nt_; ++k)
                if (m >= n+1)
                    if (c.tileIsLocal(m, n)) {
                        a.tileWait(m, k);
                        a.tileWait(n, k);
                        c(m, n)->gemm(trans, Op::Trans,
                                      alpha, a(m, k), a(n, k),
                                      k == 0 ? beta : 1.0);
                    }
    }
    #pragma omp taskwait
}

//------------------------------------------------------------------------------
template<typename FloatType>
void Matrix<FloatType>::syrkBatch(blas::Uplo uplo, blas::Op trans,
                                  FloatType alpha, const Matrix &that,
                                  FloatType beta)
{
    using namespace blas;

    Matrix<FloatType> c = *this;
    Matrix<FloatType> a = that;

    // syrk tasks
    for (int64_t n = 0; n < c.nt_; ++n) {
        for (int64_t k = 0; k < a.nt_; ++k)
            #pragma omp task
            if (c.tileIsLocal(n, n)) {
                a.tileWait(n, k);
                c(n, n)->syrk(uplo, trans, -1.0, a(n, k), k == 0 ? beta : 1.0);
            }
    }

    CBLAS_TRANSPOSE transa_array[1];
    CBLAS_TRANSPOSE transb_array[1];
    int m_array[1];
    int n_array[1];
    int k_array[1];
    FloatType alpha_array[1];
    const FloatType **a_array;
    int lda_array[1];
    const FloatType **b_array;
    int ldb_array[1];
    FloatType beta_array[1];
    FloatType **c_array;
    int ldc_array[1];

    int nb = tileNb(0);
    transa_array[0] = CblasNoTrans;
    transb_array[0] = CblasTrans;
    m_array[0] = nb;
    n_array[0] = nb;
    k_array[0] = nb;
    alpha_array[0] = alpha;
    lda_array[0] = nb;
    ldb_array[0] = nb;
    beta_array[0] = beta;
    ldc_array[0] = nb;

    // Wait for remote tiles.
    // Compute group size.
    int group_size = 0;
    for (int64_t n = 0; n < c.nt_; ++n)
        for (int64_t m = n+1; m < c.mt_; ++m)
            for (int64_t k = 0; k < a.nt_; ++k)
                if (c.tileIsLocal(m, n)) {
                    a.tileWait(m, k);
                    a.tileWait(n, k);
                    ++group_size;
                }

    a_array = new const FloatType*[group_size];
    b_array = new const FloatType*[group_size];
    c_array = new FloatType*[group_size];

    int i = 0;
    for (int64_t n = 0; n < c.nt_; ++n)
        for (int64_t m = n+1; m < c.mt_; ++m)
            for (int64_t k = 0; k < a.nt_; ++k)
                if (c.tileIsLocal(m, n)) {
                    a_array[i] = a(m, k)->data_;
                    b_array[i] = a(n, k)->data_;
                    c_array[i] = c(m, n)->data_;
                    ++i;
                }

    trace_cpu_start();
//  mkl_set_num_threads_local(60);
    cblas_dgemm_batch(CblasColMajor, transa_array, transb_array,
                      m_array, n_array, k_array, alpha_array,
                      a_array, lda_array, b_array, ldb_array, beta_array,
                      c_array, ldc_array, 1, &group_size);
//  mkl_set_num_threads_local(1);
    trace_cpu_stop("DarkGreen");

    delete[] a_array;
    delete[] b_array;
    delete[] c_array;

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
template<typename FloatType>
void Matrix<FloatType>::syrkAcc(blas::Uplo uplo, blas::Op trans,
                                FloatType alpha, const Matrix &that,
                                FloatType beta)
{
    using namespace blas;

    Matrix<FloatType> c = *this;
    Matrix<FloatType> a = that;

    // Wait for MPI.
    for (int64_t n = 0; n < c.nt_; ++n)
        for (int64_t m = n+1; m < c.mt_; ++m)
            for (int64_t k = 0; k < a.nt_; ++k)
                if (c.tileIsLocal(m, n)) {
                    a.tileWait(m, k);
                    a.tileWait(n, k);
                }

    for (int device = 0; device < num_devices_; ++device)
        #pragma omp task priority (1)
        {
            // trace_cpu_start();
            int64_t i = 0;
            for (int64_t n = 0; n < c.nt_; ++n)
                for (int64_t m = n+1; m < c.mt_; ++m)
                    for (int64_t k = 0; k < a.nt_; ++k)
                        if (c.tileIsLocal(m, n))
                            if (device == tileDevice(m, n)) {
                                c.tileMoveToDevice(m, n, device);
                                a.tileCopyToDevice(m, k, device);
                                a.tileCopyToDevice(n, k, device);
                                a_array_h_[device][i] = a(m, k, device)->data_;
                                b_array_h_[device][i] = a(n, k, device)->data_;
                                c_array_h_[device][i] = c(m, n, device)->data_;
                                ++i;
                            }
            int64_t batch_count = i;
            // trace_cpu_stop("LightGray");

            cudaError_t error;
            error = cudaSetDevice(device);
            assert(error == cudaSuccess);

            error = cudaMemcpyAsync(a_array_d_[device], a_array_h_[device],
                                    sizeof(FloatType*)*batch_count,
                                    cudaMemcpyHostToDevice,
                                    gemm_stream_[device]);
            assert(error == cudaSuccess);
            error = cudaMemcpyAsync(b_array_d_[device], b_array_h_[device],
                                    sizeof(FloatType*)*batch_count,
                                    cudaMemcpyHostToDevice,
                                    gemm_stream_[device]);
            assert(error == cudaSuccess);
            error = cudaMemcpyAsync(c_array_d_[device], c_array_h_[device],
                                    sizeof(FloatType*)*batch_count,
                                    cudaMemcpyHostToDevice,
                                    gemm_stream_[device]);
            assert(error == cudaSuccess);

            trace_cpu_start();
            int nb = tileNb(0);
            cublasStatus_t status =
                cublasDgemmBatched(
                    cublas_handle_[device],
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    nb, nb, nb,
                    &alpha, a_array_d_[device], nb,
                            b_array_d_[device], nb,
                    &beta,  c_array_d_[device], nb,
                    batch_count);
            assert(status == CUBLAS_STATUS_SUCCESS);
            error = cudaStreamSynchronize(gemm_stream_[device]);
            assert(error == cudaSuccess);
            trace_cpu_stop("PaleGreen");

            for (int64_t n = 0; n < c.nt_; ++n)
                for (int64_t m = n+1; m < c.mt_; ++m)
                    for (int64_t k = 0; k < a.nt_; ++k)
                        if (c.tileIsLocal(m, n))
                            if (device == tileDevice(m, n)) {
                                a(m, k)->tick();
                                a(n, k)->tick();
                                a.tileErase(m, k, device);
                                a.tileErase(n, k, device);
                            }
        }

    // host syrk on diagonal tiles
    for (int64_t n = 0; n < c.nt_; ++n)
        for (int64_t k = 0; k < a.nt_; ++k)
            if (c.tileIsLocal(n, n))
                #pragma omp task
                {
                    a.tileWait(n, k);
                    c(n, n)->syrk(uplo, trans, -1.0, a(n, k), k == 0 ? beta : 1.0);
                }

    #pragma omp taskwait
}

template
void Matrix<double>::syrkAcc(blas::Uplo uplo, blas::Op trans,
                             double alpha, const Matrix &that,
                             double beta);

template
void Matrix<double>::syrkTask(blas::Uplo uplo, blas::Op trans,
                              double alpha, const Matrix &that,
                              double beta);

} // namespace slate
