
#ifndef SLATE_NO_CUBLAS_HH
#define SLATE_NO_CUBLAS_HH

#include <cassert>

typedef int cublasHandle_t;
typedef int cublasStatus_t;
typedef int cublasOperation_t;

enum {
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    CUBLAS_STATUS_SUCCESS
};

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
cublasStatus_t cublasCreate(cublasHandle_t *handle)
{
    assert(0);
}

//------------------------------------------------------------------------------
cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId)
{
    assert(0);
}

//------------------------------------------------------------------------------
cublasStatus_t cublasDgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double *alpha, const double *Aarray[], int lda,
                         const double *Barray[], int ldb,
    const double *beta,        double *Carray[], int ldc,
    int batchCount)
{
    assert(0);
}

#ifdef __cplusplus
}
#endif

#endif // SLATE_NO_CUBLAS_HH
