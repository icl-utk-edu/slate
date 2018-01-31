#ifndef ICL_LAPACK_FLOPS_H
#define ICL_LAPACK_FLOPS_H

#include "lapack.hh"
#include "blas_flops.hh"

#include <complex>

namespace lapack {

//==============================================================================
// Generic formulas come from LAWN 41
// BLAS formulas generally assume alpha == 1 or -1, and beta == 1, -1, or 0;
// otherwise add some smaller order term.
// Some formulas are wrong when m, n, or k == 0; flops should be 0
// (e.g., syr2k, unmqr).
// Formulas may give negative results for invalid combinations of m, n, k
// (e.g., ungqr, unmqr).

//------------------------------------------------------------ getrf
// LAWN 41 omits (m < n) case
static double fmuls_getrf(double m, double n)
{
    return (m >= n)
        ? (0.5*m*n*n - 1./6*n*n*n + 0.5*m*n - 0.5*n*n + 2/3.*n)
        : (0.5*n*m*m - 1./6*m*m*m + 0.5*n*m - 0.5*m*m + 2/3.*m);
}

static double fadds_getrf(double m, double n)
{
    return (m >= n)
        ? (0.5*m*n*n - 1./6*n*n*n - 0.5*m*n + 1./6*n)
        : (0.5*n*m*m - 1./6*m*m*m - 0.5*n*m + 1./6*m);
}

//------------------------------------------------------------ getri
static double fmuls_getri(double n)
    { return 2/3.*n*n*n + 0.5*n*n + 5./6*n; }

static double fadds_getri(double n)
    { return 2/3.*n*n*n - 1.5*n*n + 5./6*n; }

//------------------------------------------------------------ getrs
static double fmuls_getrs(double n, double nrhs)
    { return nrhs*n*n; }

static double fadds_getrs(double n, double nrhs)
    { return nrhs*n*(n - 1); }

//------------------------------------------------------------ potrf
static double fmuls_potrf(double n)
    { return 1./6*n*n*n + 0.5*n*n + 1./3.*n; }

static double fadds_potrf(double n)
    { return 1./6*n*n*n - 1./6*n; }

//------------------------------------------------------------ potri
static double fmuls_potri(double n)
    { return 1./3.*n*n*n + n*n + 2/3.*n; }

static double fadds_potri(double n)
    { return 1./3.*n*n*n - 0.5*n*n + 1./6*n; }

//------------------------------------------------------------ potrs
static double fmuls_potrs(double n, double nrhs)
    { return nrhs*n*(n + 1); }

static double fadds_potrs(double n, double nrhs)
    { return nrhs*n*(n - 1); }

//------------------------------------------------------------ geqrf
static double fmuls_geqrf(double m, double n)
{
    return (m > n)
        ? (m*n*n - 1./3.*n*n*n +   m*n + 0.5*n*n + 23./6*n)
        : (n*m*m - 1./3.*m*m*m + 2*n*m - 0.5*m*m + 23./6*m);
}

static double fadds_geqrf(double m, double n)
{
    return (m > n)
        ? (m*n*n - 1./3.*n*n*n + 0.5*n*n       + 5./6*n)
        : (n*m*m - 1./3.*m*m*m + n*m - 0.5*m*m + 5./6*m);
}

//------------------------------------------------------------ geqrt
static double fmuls_geqrt(double m, double n)
    { return 0.5*m*n; }

static double fadds_geqrt(double m, double n)
    { return 0.5*m*n; }

//------------------------------------------------------------ geqlf
static double fmuls_geqlf(double m, double n)
    { return fmuls_geqrf(m, n); }

static double fadds_geqlf(double m, double n)
    { return fadds_geqrf(m, n); }

//------------------------------------------------------------ gerqf
static double fmuls_gerqf(double m, double n)
{
    return (m > n)
        ? (m*n*n - 1./3.*n*n*n +   m*n + 0.5*n*n + 29./6*n)
        : (n*m*m - 1./3.*m*m*m + 2*n*m - 0.5*m*m + 29./6*m);
}

static double fadds_gerqf(double m, double n)
{
    return (m > n)
        ? (m*n*n - 1./3.*n*n*n + m*n - 0.5*n*n + 5./6*n)
        : (n*m*m - 1./3.*m*m*m + 0.5*m*m       + 5./6*m);
}

//------------------------------------------------------------ gelqf
static double fmuls_gelqf(double m, double n)
    { return  fmuls_gerqf(m, n); }

static double fadds_gelqf(double m, double n)
    { return  fadds_gerqf(m, n); }

//------------------------------------------------------------ ungqr
static double fmuls_ungqr(double m, double n, double k)
    { return 2*m*n*k - (m + n)*k*k + 2/3.*k*k*k + 2*n*k - k*k - 5./3.*k; }

static double fadds_ungqr(double m, double n, double k)
    { return 2*m*n*k - (m + n)*k*k + 2/3.*k*k*k + n*k - m*k + 1./3.*k; }

//------------------------------------------------------------ ungql
static double fmuls_ungql(double m, double n, double k)
    { return  fmuls_ungqr(m, n, k); }

static double fadds_ungql(double m, double n, double k)
    { return fadds_ungqr(m, n, k); }

//------------------------------------------------------------ ungrq
static double fmuls_ungrq(double m, double n, double k)
    { return 2*m*n*k - (m + n)*k*k + 2/3.*k*k*k + m*k + n*k - k*k - 2/3.*k; }

static double fadds_ungrq(double m, double n, double k)
    { return 2*m*n*k - (m + n)*k*k + 2/3.*k*k*k + m*k - n*k + 1./3.*k; }

//------------------------------------------------------------ unglq
static double fmuls_unglq(double m, double n, double k)
    { return fmuls_ungrq(m, n, k); }

static double fadds_unglq(double m, double n, double k)
    { return fadds_ungrq(m, n, k); }

//------------------------------------------------------------ geqrs
static double fmuls_geqrs(double m, double n, double nrhs)
    { return nrhs*(2*m*n - 0.5*n*n + 25*n); }

static double fadds_geqrs(double m, double n, double nrhs)
    { return nrhs*(2*m*n - 0.5*n*n + 0.5*n); }

//------------------------------------------------------------ unmqr
static double fmuls_unmqr(lapack::Side side, double m, double n, double k)
{
    return (side == lapack::Side::Left)
        ? (2*n*m*k - n*k*k + 2*n*k)
        : (2*n*m*k - m*k*k + m*k + n*k - 0.5*k*k + 0.5*k);
}

static double fadds_unmqr(lapack::Side side, double m, double n, double k)
{
    return (side == lapack::Side::Left)
        ? (2*n*m*k - n*k*k + n*k)
        : (2*n*m*k - m*k*k + m*k);
}

//------------------------------------------------------------ unmql
static double fmuls_unmql(lapack::Side side, double m, double n, double k)
    { return fmuls_unmqr(side, m, n, k); }

static double fadds_unmql(lapack::Side side, double m, double n, double k)
    { return fadds_unmqr(side, m, n, k); }

//------------------------------------------------------------ unmrq
static double fmuls_unmrq(lapack::Side side, double m, double n, double k)
    { return fmuls_unmqr(side, m, n, k); }

static double fadds_unmrq(lapack::Side side, double m, double n, double k)
    { return fadds_unmqr(side, m, n, k); }

//------------------------------------------------------------ unmlq
static double fmuls_unmlq(lapack::Side side, double m, double n, double k)
    { return fmuls_unmqr(side, m, n, k); }

static double fadds_unmlq(lapack::Side side, double m, double n, double k)
    { return fadds_unmqr(side, m, n, k); }

//------------------------------------------------------------ trtri
static double fmuls_trtri(double n)
    { return 1./6*n*n*n + 0.5*n*n + 1./3.*n; }

static double fadds_trtri(double n)
    { return 1./6*n*n*n - 0.5*n*n + 1./3.*n; }

//------------------------------------------------------------ gehrd
static double fmuls_gehrd(double n)
    { return 5./3.*n*n*n + 0.5*n*n - 7./6*n; }

static double fadds_gehrd(double n)
    { return 5./3.*n*n*n - n*n - 2/3.*n; }

//------------------------------------------------------------ sytrd
static double fmuls_sytrd(double n)
    { return 2/3.*n*n*n + 25*n*n - 1./6*n; }

static double fadds_sytrd(double n)
    { return 2/3.*n*n*n + n*n - 8./3.*n; }

static double fmuls_hetrd(double n)
    { return fmuls_sytrd(n); }

static double fadds_hetrd(double n)
    { return fadds_sytrd(n); }

//------------------------------------------------------------ gebrd
static double fmuls_gebrd(double m, double n)
{
    return (m >= n)
        ? (2*m*n*n - 2/3.*n*n*n + 2*n*n + 20./3.*n)
        : (2*n*m*m - 2/3.*m*m*m + 2*m*m + 20./3.*m);
}

static double fadds_gebrd(double m, double n)
{
    return (m >= n)
        ? (2*m*n*n - 2/3.*n*n*n + n*n - m*n +  5./3.*n)
        : (2*n*m*m - 2/3.*m*m*m + m*m - n*m +  5./3.*m);
}

//------------------------------------------------------------ larfg
static double fmuls_larfg(double n)
    { return 2*n; }

static double fadds_larfg(double n)
    { return   n; }

//------------------------------------------------------------ geadd
static double fmuls_geadd(double m, double n)
    { return 2*m*n; }

static double fadds_geadd(double m, double n)
    { return   m*n; }

//------------------------------------------------------------ lauum
static double fmuls_lauum(double n)
    { return fmuls_potri(n) - fmuls_trtri(n); }

static double fadds_lauum(double n)
    { return fadds_potri(n) - fadds_trtri(n); }

//------------------------------------------------------------ lange
static double fmuls_lange(double m, double n, lapack::Norm norm)
    { return norm == lapack::Norm::Fro ? m*n : 0; }

static double fadds_lange(double m, double n, lapack::Norm norm)
{
    switch (norm) {
    case lapack::Norm::One: return (m-1)*n;
    case lapack::Norm::Inf: return (n-1)*m;
    case lapack::Norm::Fro: return m*n-1;
    default:                return 0;
    }
}

//------------------------------------------------------------ lanhe
static double fmuls_lanhe(double n, lapack::Norm norm)
    { return norm == lapack::Norm::Fro ? n*(n+1)/2 : 0; }

static double fadds_lanhe(double n, lapack::Norm norm)
{
    switch (norm) {
    case lapack::Norm::One: return (n-1)*n;
    case lapack::Norm::Inf: return (n-1)*n;
    case lapack::Norm::Fro: return n*(n+1)/2-1;
    default:                return 0;
    }
}

//==============================================================================
// template class. Example:
// gbyte< float >::gemv( m, n ) yields bytes transferred for sgemv.
// gbyte< std::complex<float> >::gemv( m, n ) yields bytes transferred for cgemv.
//==============================================================================
template< typename T >
class Gbyte:
    public blas::Gbyte<T>
{
};

//==============================================================================
// template class. Example:
// gflop< float >::getrf( m, n ) yields flops for sgetrf.
// gflop< std::complex<float> >::getrf( m, n ) yields flops for cgetrf.
//==============================================================================
template< typename T >
class Gflop:
    public blas::Gflop<T>
{
public:
    static double gesv(double n, double nrhs)
        { return getrf(n, n) + getrs(n, nrhs); }

    static double getrf(double m, double n)
        { return 1e-9 * (fmuls_getrf(m, n) + fadds_getrf(m, n)); }

    static double getri(double n)
        { return 1e-9 * (fmuls_getri(n) + fadds_getri(n)); }

    static double getrs(double n, double nrhs)
        { return 1e-9 * (fmuls_getrs(n, nrhs) + fadds_getrs(n, nrhs)); }

    static double posv(double n, double nrhs)
        { return potrf(n) + potrs(n, nrhs); }

    static double potrf(double n)
        { return 1e-9 * (fmuls_potrf(n) + fadds_potrf(n)); }

    static double potri(double n)
        { return 1e-9 * (fmuls_potri(n) + fadds_potri(n)); }

    static double potrs(double n, double nrhs)
        { return 1e-9 * (fmuls_potrs(n, nrhs) + fadds_potrs(n, nrhs)); }

    static double geqrf(double m, double n)
        { return 1e-9 * (fmuls_geqrf(m, n) + fadds_geqrf(m, n)); }

    static double geqrt(double m, double n)
        { return 1e-9 * (fmuls_geqrt(m, n) + fadds_geqrt(m, n)); }

    static double geqlf(double m, double n)
        { return 1e-9 * (fmuls_geqlf(m, n) + fadds_geqlf(m, n)); }

    static double gerqf(double m, double n)
        { return 1e-9 * (fmuls_gerqf(m, n) + fadds_gerqf(m, n)); }

    static double gelqf(double m, double n)
        { return 1e-9 * (fmuls_gelqf(m, n) + fadds_gelqf(m, n)); }

    static double orgqr(double m, double n, double k)
        { return 1e-9 * (fmuls_ungqr(m, n, k) + fadds_ungqr(m, n, k)); }

    static double orgql(double m, double n, double k)
        { return 1e-9 * (fmuls_ungql(m, n, k) + fadds_ungql(m, n, k)); }

    static double orgrq(double m, double n, double k)
        { return 1e-9 * (fmuls_ungrq(m, n, k) + fadds_ungrq(m, n, k)); }

    static double orglq(double m, double n, double k)
        { return 1e-9 * (fmuls_unglq(m, n, k) + fadds_unglq(m, n, k)); }

    static double geqrs(double m, double n, double nrhs)
        { return 1e-9 * (fmuls_geqrs(m, n, nrhs) + fadds_geqrs(m, n, nrhs)); }

    static double ormqr(lapack::Side side, double m, double n, double k)
        { return 1e-9 * (fmuls_unmqr(side, m, n, k) + fadds_unmqr(side, m, n, k)); }

    static double ormql(lapack::Side side, double m, double n, double k)
        { return 1e-9 * (fmuls_unmql(side, m, n, k) + fadds_unmql(side, m, n, k)); }

    static double ormrq(lapack::Side side, double m, double n, double k)
        { return 1e-9 * (fmuls_unmrq(side, m, n, k) + fadds_unmrq(side, m, n, k)); }

    static double ormlq(lapack::Side side, double m, double n, double k)
        { return 1e-9 * (fmuls_unmlq(side, m, n, k) + fadds_unmlq(side, m, n, k)); }

    static double trtri(double n)
        { return 1e-9 * (fmuls_trtri(n) + fadds_trtri(n)); }

    static double gehrd(double n)
        { return 1e-9 * (fmuls_gehrd(n) + fadds_gehrd(n)); }

    static double sytrd(double n)
        { return 1e-9 * (fmuls_sytrd(n) + fadds_sytrd(n)); }

    static double gebrd(double m, double n)
        { return 1e-9 * (fmuls_gebrd(m, n) + fadds_gebrd(m, n)); }

    static double larfg(double n)
        { return 1e-9 * (fmuls_larfg(n) + fadds_larfg(n)); }

    static double geadd(double m, double n)
        { return 1e-9 * (fmuls_geadd(m, n) + fadds_geadd(m, n)); }

    static double lauum(double n)
        { return 1e-9 * (fmuls_lauum(n) + fadds_lauum(n)); }

    static double lange(double m, double n, lapack::Norm norm)
        { return 1e-9 * (fmuls_lange(m, n, norm) + fadds_lange(m, n, norm)); }

    static double lansy(double n, lapack::Norm norm)
        { return 1e-9 * (fmuls_lanhe(n, norm) + fadds_lanhe(n, norm)); }

    // ------------------------------------------------------------------
    // Make complex function names available for tests

    static double ungqr(double m, double n, double k)
        { return 1e-9 * (6*fmuls_ungqr(m, n, k) + 2*fadds_ungqr(m, n, k)); }

    static double unglq(double m, double n, double k)
        { return 1e-9 * (6*fmuls_unglq(m, n, k) + 2*fadds_unglq(m, n, k)); }

    static double ungql(double m, double n, double k)
        { return 1e-9 * (6*fmuls_ungql(m, n, k) + 2*fadds_ungql(m, n, k)); }

    static double ungrq(double m, double n, double k)
        { return 1e-9 * (6*fmuls_ungrq(m, n, k) + 2*fadds_ungrq(m, n, k)); }

    static double unmqr(lapack::Side side, double m, double n, double k)
        { return 1e-9 * (6*fmuls_unmqr(side, m, n, k) + 2*fadds_unmqr(side, m, n, k)); }

    static double hetrd(double n)
        { return 1e-9 * (6*fmuls_hetrd(n) + 2*fadds_hetrd(n)); }
};

//==============================================================================
// specialization for complex
// flops = 6*muls + 2*adds
//==============================================================================
template< typename T >
class Gflop< std::complex<T> >:
    public blas::Gflop< std::complex<T> >
{
public:
    static double gesv(double n, double nrhs)
        { return getrf(n, n) + getrs(n, nrhs); }

    static double getrf(double m, double n)
        { return 1e-9 * (6*fmuls_getrf(m, n) + 2*fadds_getrf(m, n)); }

    static double getri(double n)
        { return 1e-9 * (6*fmuls_getri(n) + 2*fadds_getri(n)); }

    static double getrs(double n, double nrhs)
        { return 1e-9 * (6*fmuls_getrs(n, nrhs) + 2*fadds_getrs(n, nrhs)); }

    static double posv(double n, double nrhs)
        { return potrf(n) + potrs(n, nrhs); }

    static double potrf(double n)
        { return 1e-9 * (6*fmuls_potrf(n) + 2*fadds_potrf(n)); }

    static double potri(double n)
        { return 1e-9 * (6*fmuls_potri(n) + 2*fadds_potri(n)); }

    static double potrs(double n, double nrhs)
        { return 1e-9 * (6*fmuls_potrs(n, nrhs) + 2*fadds_potrs(n, nrhs)); }

    static double geqrf(double m, double n)
        { return 1e-9 * (6*fmuls_geqrf(m, n) + 2*fadds_geqrf(m, n)); }

    static double geqrt(double m, double n)
        { return 1e-9 * (6*fmuls_geqrt(m, n) + 2*fadds_geqrt(m, n)); }

    static double geqlf(double m, double n)
        { return 1e-9 * (6*fmuls_geqlf(m, n) + 2*fadds_geqlf(m, n)); }

    static double gerqf(double m, double n)
        { return 1e-9 * (6*fmuls_gerqf(m, n) + 2*fadds_gerqf(m, n)); }

    static double gelqf(double m, double n)
        { return 1e-9 * (6*fmuls_gelqf(m, n) + 2*fadds_gelqf(m, n)); }

    static double ungqr(double m, double n, double k)
        { return 1e-9 * (6*fmuls_ungqr(m, n, k) + 2*fadds_ungqr(m, n, k)); }

    static double ungql(double m, double n, double k)
        { return 1e-9 * (6*fmuls_ungql(m, n, k) + 2*fadds_ungql(m, n, k)); }

    static double ungrq(double m, double n, double k)
        { return 1e-9 * (6*fmuls_ungrq(m, n, k) + 2*fadds_ungrq(m, n, k)); }

    static double unglq(double m, double n, double k)
        { return 1e-9 * (6*fmuls_unglq(m, n, k) + 2*fadds_unglq(m, n, k)); }

    static double geqrs(double m, double n, double nrhs)
        { return 1e-9 * (6*fmuls_geqrs(m, n, nrhs) + 2*fadds_geqrs(m, n, nrhs)); }

    static double unmqr(lapack::Side side, double m, double n, double k)
        { return 1e-9 * (6*fmuls_unmqr(side, m, n, k) + 2*fadds_unmqr(side, m, n, k)); }

    static double unmql(lapack::Side side, double m, double n, double k)
        { return 1e-9 * (6*fmuls_unmql(side, m, n, k) + 2*fadds_unmql(side, m, n, k)); }

    static double unmrq(lapack::Side side, double m, double n, double k)
        { return 1e-9 * (6*fmuls_unmrq(side, m, n, k) + 2*fadds_unmrq(side, m, n, k)); }

    static double unmlq(lapack::Side side, double m, double n, double k)
        { return 1e-9 * (6*fmuls_unmlq(side, m, n, k) + 2*fadds_unmlq(side, m, n, k)); }

    static double trtri(double n)
        { return 1e-9 * (6*fmuls_trtri(n) + 2*fadds_trtri(n)); }

    static double gehrd(double n)
        { return 1e-9 * (6*fmuls_gehrd(n) + 2*fadds_gehrd(n)); }

    static double hetrd(double n)
        { return 1e-9 * (6*fmuls_hetrd(n) + 2*fadds_hetrd(n)); }

    static double gebrd(double m, double n)
        { return 1e-9 * (6*fmuls_gebrd(m, n) + 2*fadds_gebrd(m, n)); }

    static double larfg(double n)
        { return 1e-9 * (6*fmuls_larfg(n) + 2*fadds_larfg(n)); }

    static double geadd(double m, double n)
        { return 1e-9 * (6*fmuls_geadd(m, n) + 2*fadds_geadd(m, n)); }

    static double lauum(double n)
        { return 1e-9 * (6*fmuls_lauum(n) + 2*fadds_lauum(n)); }

    static double lange(double m, double n, lapack::Norm norm)
        { return 1e-9 * (6*fmuls_lange(m, n, norm) + 2*fadds_lange(m, n, norm)); }

    static double lanhe(double n, lapack::Norm norm)
        { return 1e-9 * (6*fmuls_lanhe(n, norm) + 2*fadds_lanhe(n, norm)); }
};

}  // namespace lapack

#endif  // ICL_LAPACK_FLOPS_H
