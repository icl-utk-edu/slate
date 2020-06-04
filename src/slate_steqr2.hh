#include "blas_fortran.hh"
#include "slate/Exception.hh"

#include <complex>
#include <limits>

#include <blas.hh>

// -----------------------------------------------------------------------------
// helper funtion to check and do type conversion
inline int int64_to_int(int64_t n)
{
    if (sizeof(int64_t) > sizeof(blas_int))
        slate_assert(n < std::numeric_limits<int>::max());
    int n_ = (int)n;
    return n_;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define slate_ssteqr2 BLAS_FORTRAN_NAME( slate_ssteqr2, SLATE_SSTEQR2 )
#define slate_dsteqr2 BLAS_FORTRAN_NAME( slate_dsteqr2, SLATE_DSTEQR2 )
#define slate_csteqr2 BLAS_FORTRAN_NAME( slate_csteqr2, SLATE_CSTEQR2 )
#define slate_zsteqr2 BLAS_FORTRAN_NAME( slate_zsteqr2, SLATE_ZSTEQR2 )

extern "C" void slate_ssteqr2(
    const char* compz, const blas_int* n,
    float* d, float* e,
    float* z, const blas_int* ldz, const blas_int* nr,
    float* work,
    blas_int* info);

extern "C" void slate_dsteqr2(
    const char* compz, const blas_int* n,
    double* d, double* e,
    double* z, const blas_int* ldz, const blas_int* nr,
    double* work,
    blas_int* info);

extern "C" void slate_csteqr2(
    const char* compz, const blas_int* n,
    float* d, float* e,
    std::complex<float>* z, const blas_int* ldz, const blas_int* nr,
    float* work,
    blas_int* info);

extern "C" void slate_zsteqr2(
    const char* compz, const blas_int* n,
    double* d, double* e,
    std::complex<double>* z, const blas_int* ldz, const blas_int* nr,
    double* work,
    blas_int* info);

// -----------------------------------------------------------------------------

inline void slate_steqr2(
    lapack::Job compz, blas_int* n,
    float* d, float* e,
    float* z, blas_int* ldz, blas_int* nr,
    float* work,
    blas_int* info)
{
    char compz_ = job_comp2char( compz );
    slate_ssteqr2(&compz_, n,
            d, e,
            z, ldz, nr,
            work, info);
}

inline void slate_steqr2(
    lapack::Job compz, blas_int* n,
    double* d, double* e,
    double* z, blas_int* ldz, blas_int* nr,
    double* work,
    blas_int* info)
{
    char compz_ = job_comp2char( compz );
    slate_dsteqr2(&compz_, n,
            d, e,
            z, ldz, nr,
            work, info);
}

inline void slate_steqr2(
    lapack::Job compz, blas_int* n,
    float* d, float* e,
    std::complex<float>* z, blas_int* ldz, blas_int* nr,
    float* work,
    blas_int* info)
{
    char compz_ = job_comp2char( compz );
    slate_csteqr2(&compz_, n,
            d, e,
            z, ldz, nr,
            work, info);
}

inline void slate_steqr2(
    lapack::Job compz, blas_int* n,
    double* d, double* e,
    std::complex<double>* z, blas_int* ldz, blas_int* nr,
    double* work,
    blas_int* info)
{
    char compz_ = job_comp2char( compz );
    slate_zsteqr2(&compz_, n,
            d, e,
            z, ldz, nr,
            work, info);
}

template <typename scalar_t>
inline void slate_steqr2(
    lapack::Job compz, int64_t n,
    blas::real_type<scalar_t>* d,
    blas::real_type<scalar_t>* e,
    scalar_t* z, int64_t ldz, int64_t nr,
    blas::real_type<scalar_t>* work,
    int64_t* info)
{
    // todo: int64_to_blas_int
    blas_int n_       = int64_to_int(n);
    blas_int ldz_     = int64_to_int(ldz);
    blas_int nr_      = int64_to_int(nr);
    blas_int info_    = int64_to_int(*info);
    slate_steqr2(compz, &n_,
                     d, e,
                     z, &ldz_, &nr_,
                     work,
                     &info_);
    *info = (int64_t)info_;
}
