
#ifndef CCLAPACK_HH
#define CCLAPACK_HH

#include "Ccblas.hh"

#include "mkl_lapacke.h"

namespace Cclapack {

//------------------------------------------------------------------------------
void potrf(Ccblas::Order order, Ccblas::Uplo uplo, int64_t n, float *a,
           uint64_t lda)
{
    LAPACKE_spotrf(LAPACK_COL_MAJOR, 'L', n, a, lda);
}

void potrf(Ccblas::Order order, Ccblas::Uplo uplo, int64_t n, double *a,
           uint64_t lda) 
{
    LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, a, lda);
}

void potrf(Ccblas::Order order, Ccblas::Uplo uplo, int64_t n,
           std::complex<float> *a, uint64_t lda)
{
    LAPACKE_cpotrf(LAPACK_COL_MAJOR, 'L', n, (MKL_Complex8*)a, lda);
}

void potrf(Ccblas::Order order, Ccblas::Uplo uplo, int64_t n,
           std::complex<double> *a, uint64_t lda)
{
    LAPACKE_zpotrf(LAPACK_COL_MAJOR, 'L', n, (MKL_Complex16*)a, lda);
}

} // namespace CCLAPACK_HH

#endif // CCLAPACK_HH
