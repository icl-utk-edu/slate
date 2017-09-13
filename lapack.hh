
#ifndef LAPACK_HH
#define LAPACK_HH

#include "blas.hh"

#ifdef ESSL
// extern "C" {
#include "essl.h"
// }
#else
#include "mkl_lapacke.h"
#endif

namespace lapack {

//------------------------------------------------------------------------------
void potrf(blas::Layout layout, blas::Uplo uplo, int64_t n, float *a,
           uint64_t lda)
{
    LAPACKE_spotrf(LAPACK_COL_MAJOR, 'L', n, a, lda);
}

void potrf(blas::Layout layout, blas::Uplo uplo, int64_t n, double *a,
           uint64_t lda) 
{
    LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, a, lda);
}

void potrf(blas::Layout layout, blas::Uplo uplo, int64_t n,
           std::complex<float> *a, uint64_t lda)
{
#ifdef ESSL
    LAPACKE_cpotrf(LAPACK_COL_MAJOR, 'L', n, a, lda);
#else
    LAPACKE_cpotrf(LAPACK_COL_MAJOR, 'L', n, (MKL_Complex8*)a, lda);
#endif
}

void potrf(blas::Layout layout, blas::Uplo uplo, int64_t n,
           std::complex<double> *a, uint64_t lda)
{
#ifdef ESSL
    LAPACKE_zpotrf(LAPACK_COL_MAJOR, 'L', n, a, lda);
#else
    LAPACKE_zpotrf(LAPACK_COL_MAJOR, 'L', n, (MKL_Complex16*)a, lda);
#endif
}

} // namespace LAPACK_HH

#endif // LAPACK_HH
