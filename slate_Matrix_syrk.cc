
#include "slate_Matrix.hh"
#include "slate_types.hh"

namespace slate {

//------------------------------------------------------------------------------
template <typename FloatType>
template <Target target>
void Matrix<FloatType>::syrk(blas::Uplo uplo, blas::Op trans,
                             FloatType alpha, Matrix &a,
                             FloatType beta,  Matrix &c)
{
    syrk(internal::TargetType<target>(),
         uplo, trans,
         alpha, a,
         beta,  c);
}

//------------------------------------------------------------------------------
template <typename FloatType>
void Matrix<FloatType>::syrk(internal::TargetType<Target::HostTask>,
                             blas::Uplo uplo, blas::Op trans,
                             FloatType alpha, Matrix &a,
                             FloatType beta,  Matrix &c)
{
    // Lower, NoTrans
    for (int64_t n = 0; n < c.nt_; ++n)
        if (c.tileIsLocal(n, n))
            #pragma omp task shared(a, c)
            Tile<FloatType>::syrk(uplo, trans,
                                  -1.0, a(n, 0),
                                  beta, c(n, n));

    for (int64_t n = 0; n < c.nt_; ++n)
        for (int64_t m = n+1; m < c.mt_; ++m)
            for (int64_t k = 0; k < a.nt_; ++k)
                if (c.tileIsLocal(m, n))
                    #pragma omp task shared(a, c)
                    Tile<FloatType>::gemm(trans, blas::Op::Trans,
                                          alpha, a(m, k),
                                                 a(n, k),
                                          k == 0 ? beta : 1.0, c(m, n));

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
template <typename FloatType>
void Matrix<FloatType>::syrk(internal::TargetType<Target::HostNest>,
                             blas::Uplo uplo, blas::Op trans,
                             FloatType alpha, Matrix &a,
                             FloatType beta,  Matrix &c)
{
    // Lower, NoTrans
    for (int64_t n = 0; n < c.nt_; ++n)
        if (c.tileIsLocal(n, n))
            #pragma omp task shared(a, c)
            Tile<FloatType>::syrk(uplo, trans,
                                  -1.0, a(n, 0),
                                  beta, c(n, n));

//  #pragma omp parallel for collapse(2) schedule(dynamic, 1) num_threads(...)
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int64_t n = 0; n < c.nt_; ++n)
        for (int64_t m = 0; m < c.mt_; ++m)
            if (m >= n+1)
                if (c.tileIsLocal(m, n))
                    Tile<FloatType>::gemm(trans, blas::Op::Trans,
                                          alpha, a(m, 0),
                                                 a(n, 0),
                                          beta,  c(m, n));

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
template <typename FloatType>
void Matrix<FloatType>::syrk(internal::TargetType<Target::HostBatch>,
                             blas::Uplo uplo, blas::Op trans,
                             FloatType alpha, Matrix &a,
                             FloatType beta,  Matrix &c)
{
    for (int64_t n = 0; n < c.nt_; ++n)
        if (c.tileIsLocal(n, n))
            #pragma omp task shared(a, c)
            Tile<FloatType>::syrk(uplo, trans,
                                  -1.0, a(n, 0),
                                  beta, c(n, n));

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

    int nb = c.tileNb(0);
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

    int group_size = 0;
    for (int64_t n = 0; n < c.nt_; ++n)
        for (int64_t m = n+1; m < c.mt_; ++m)
            if (c.tileIsLocal(m, n))
                ++group_size;

    a_array = new const FloatType*[group_size];
    b_array = new const FloatType*[group_size];
    c_array = new FloatType*[group_size];

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
template <typename FloatType>
void Matrix<FloatType>::syrk(internal::TargetType<Target::Devices>,
                             blas::Uplo uplo, blas::Op trans,
                             FloatType alpha, Matrix &a,
                             FloatType beta,  Matrix &c)
{
    for (int device = 0; device < c.num_devices_; ++device)
        #pragma omp task shared(a, c) priority (1)
        {
            int64_t i = 0;
            for (int64_t n = 0; n < c.nt_; ++n)
                for (int64_t m = n+1; m < c.mt_; ++m)
                    if (c.tileIsLocal(m, n))
                        if (device == c.tileDevice(m, n)) {
                            c.tileMoveToDevice(m, n, device);
                            a.tileCopyToDevice(m, 0, device);
                            a.tileCopyToDevice(n, 0, device);
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
                                    sizeof(FloatType*)*batch_count,
                                    cudaMemcpyHostToDevice,
                                    c.gemm_stream_[device]);
            assert(error == cudaSuccess);

            error = cudaMemcpyAsync(c.b_array_d_[device], c.b_array_h_[device],
                                    sizeof(FloatType*)*batch_count,
                                    cudaMemcpyHostToDevice,
                                    c.gemm_stream_[device]);
            assert(error == cudaSuccess);

            error = cudaMemcpyAsync(c.c_array_d_[device], c.c_array_h_[device],
                                    sizeof(FloatType*)*batch_count,
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
                            a(m, 0)->tick();
                            a(n, 0)->tick();
                            a.tileErase(m, 0, device);
                            a.tileErase(n, 0, device);
                        }
        }

    for (int64_t n = 0; n < c.nt_; ++n)
        if (c.tileIsLocal(n, n))
            #pragma omp task shared(a, c)
            Tile<FloatType>::syrk(uplo, trans,
                                  -1.0, a(n, 0),
                                  beta, c(n, n));

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
template
void Matrix<double>::syrk<Target::HostTask>(
    blas::Uplo uplo, blas::Op trans,
    double alpha, Matrix &a,
    double beta,  Matrix &c);

template
void Matrix<double>::syrk<Target::HostNest>(
    blas::Uplo uplo, blas::Op trans,
    double alpha, Matrix &a,
    double beta,  Matrix &c);

template
void Matrix<double>::syrk<Target::HostBatch>(
    blas::Uplo uplo, blas::Op trans,
    double alpha, Matrix &a,
    double beta,  Matrix &c);

template
void Matrix<double>::syrk<Target::Devices>(
    blas::Uplo uplo, blas::Op trans,
    double alpha, Matrix &a,
    double beta,  Matrix &c);

} // namespace slate
