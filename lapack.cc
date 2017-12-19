
#include "lapack.hh"

namespace lapack {

void potrf(blas::Layout layout, blas::Uplo uplo, int64_t n, float *a,
           int64_t lda)
{
    LAPACKE_spotrf(LAPACK_COL_MAJOR, 'L', n, a, lda);
}

void potrf(blas::Layout layout, blas::Uplo uplo, int64_t n, double *a,
           int64_t lda) 
{
    LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, a, lda);
}

void potrf(blas::Layout layout, blas::Uplo uplo, int64_t n,
           std::complex<float> *a, int64_t lda)
{
    LAPACKE_cpotrf(LAPACK_COL_MAJOR, 'L', n, a, lda);
}

void potrf(blas::Layout layout, blas::Uplo uplo, int64_t n,
           std::complex<double> *a, int64_t lda)
{
    LAPACKE_zpotrf(LAPACK_COL_MAJOR, 'L', n, a, lda);
}

void larnv(int idist, int *iseed, int n, float *x)
{
    slarnv_(&idist, iseed, &n, x);
}

void larnv(int idist, int *iseed, int n, double *x)
{
    dlarnv_(&idist, iseed, &n, x);
}

void larnv(int idist, int *iseed, int n, std::complex<float> *x)
{
    clarnv_(&idist, iseed, &n, x);
}

void larnv(int idist, int *iseed, int n, std::complex<double> *x)
{
    zlarnv_(&idist, iseed, &n, x);
}

} // namespace lapack
