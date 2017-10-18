
#ifndef SLATE_NOCUBLAS_HH
#define SLATE_NOCUBLAS_HH

#include <cassert>

typedef int cublasHandle_t;
typedef int cublasStatus_t;

enum {
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    CUBLAS_STATUS_SUCCESS
};

#ifdef __cplusplus
extern "C" {
#endif

cublasStatus_t cublasCreate(cublasHandle_t *handle)
{
    assert(0);
}

cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId)
{
    assert(0);
}

#ifdef __cplusplus
}
#endif

#endif // SLATE_NOCUBLAS_HH
