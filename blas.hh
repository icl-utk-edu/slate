
#ifndef BLAS_HH
#define BLAS_HH

#include <complex>

#ifdef SLATE_WITH_MKL
    #include <mkl_cblas.h>
#elif SLATE_WITH_ESSL
    #include <essl.h>
#endif

namespace blas {

enum class Layout {ColMajor, RowMajor};
enum class Side   {Left, Right};
enum class Uplo   {Upper, Lower};
enum class Op     {Trans, NoTrans, ConjTrans};
enum class Diag   {Unit, NonUnit};

void trsm(blas::Layout layout,
          blas::Side side, blas::Uplo uplo, blas::Op transa, blas::Diag diag,
          int64_t m, int64_t n,
          float alpha, float *a, int64_t lda,
                       float *b, int64_t ldb);

void trsm(blas::Layout layout,
          blas::Side side, blas::Uplo uplo, blas::Op transa, blas::Diag diag,
          int64_t m, int64_t n,
          double alpha, double *a, int64_t lda,
                        double *b, int64_t ldb);

void trsm(blas::Layout layout,
          blas::Side side, blas::Uplo uplo, blas::Op transa, blas::Diag diag,
          int64_t m, int64_t n,
          std::complex<float> alpha, std::complex<float> *a, int64_t lda,
                                     std::complex<float> *b, int64_t ldb);

void trsm(blas::Layout layout,
          blas::Side side, blas::Uplo uplo, blas::Op transa, blas::Diag diag,
          int64_t m, int64_t n,
          std::complex<double> alpha, std::complex<double> *a, int64_t lda,
                                      std::complex<double> *b, int64_t ldb);

void syrk(blas::Layout layout,
          blas::Uplo uplo, blas::Op trans,
          int64_t n, int64_t k,
          float alpha, float *a, int64_t lda,
          float beta,  float *c, int64_t ldc);

void syrk(blas::Layout layout,
          blas::Uplo uplo, blas::Op trans,
          int64_t n, int64_t k,
          double alpha, double *a, int64_t lda,
          double beta,  double *c, int64_t ldc);

void syrk(blas::Layout layout,
          blas::Uplo uplo, blas::Op trans,
          int64_t n, int64_t k,
          float alpha, std::complex<float> *a, int64_t lda,
          float beta,  std::complex<float> *c, int64_t ldc);

void syrk(blas::Layout layout,
          blas::Uplo uplo, blas::Op trans,
          int64_t n, int64_t k,
          double alpha, std::complex<double> *a, int64_t lda,
          double beta,  std::complex<double> *c, int64_t ldc);

void gemm(blas::Layout layout,
         blas::Op transa, blas::Op transb,
          int64_t m, int64_t n, int64_t k,
          float alpha, float *a, int64_t lda,
                       float *b, int64_t ldb,
          float beta,  float *c, int64_t ldc);

void gemm(blas::Layout layout,
          blas::Op transa, blas::Op transb,
          int64_t m, int64_t n, int64_t k,
          double alpha, double *a, int64_t lda,
                        double *b, int64_t ldb,
          double beta,  double *c, int64_t ldc);

void gemm(blas::Layout layout,
          blas::Op transa, blas::Op transb,
          int64_t m, int64_t n, int64_t k,
          std::complex<float> alpha, std::complex<float> *a, int64_t lda,
                                     std::complex<float> *b, int64_t ldb,
          std::complex<float> beta,  std::complex<float> *c, int64_t ldc);

void gemm(blas::Layout layout,
          blas::Op transa, blas::Op transb,
          int64_t m, int64_t n, int64_t k,
          std::complex<double> alpha, std::complex<double> *a, int64_t lda,
                                      std::complex<double> *b, int64_t ldb,
          std::complex<double> beta,  std::complex<double> *c, int64_t ldc);

} // namespace blas

#endif // BLAS_HH
