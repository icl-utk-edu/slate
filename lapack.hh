
#ifndef LAPACK_HH
#define LAPACK_HH

#include "blas.hh"

#ifdef SLATE_WITH_MKL
    #define MKL_Complex8  std::complex<float>
    #define MKL_Complex16 std::complex<double>
    #include <mkl_cblas.h>
    #include <mkl_lapacke.h>
#elif SLATE_WITH_ESSL
    #include <essl.h>
    #include <essl_lapacke.h>
#endif

extern "C" {
    void slarnv_(int *idist, int *iseed, int *n, float *x);
    void dlarnv_(int *idist, int *iseed, int *n, double *x);
    void clarnv_(int *idist, int *iseed, int *n, std::complex<float> *x);
    void zlarnv_(int *idist, int *iseed, int *n, std::complex<double> *x);
}

namespace lapack {

void potrf(blas::Layout layout, blas::Uplo uplo, int64_t n, float *a,
           int64_t lda);

void potrf(blas::Layout layout, blas::Uplo uplo, int64_t n, double *a,
           int64_t lda);

void potrf(blas::Layout layout, blas::Uplo uplo, int64_t n,
           std::complex<float> *a, int64_t lda);

void potrf(blas::Layout layout, blas::Uplo uplo, int64_t n,
           std::complex<double> *a, int64_t lda);

void larnv(int idist, int *iseed, int n, float *x);
void larnv(int idist, int *iseed, int n, double *x);
void larnv(int idist, int *iseed, int n, std::complex<float> *x);
void larnv(int idist, int *iseed, int n, std::complex<double> *x);

} // namespace lapack

#endif // LAPACK_HH
