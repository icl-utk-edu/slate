#ifndef ICL_MYSCALAPACK_WRAPPERS_HH
#define ICL_MYSCALAPACK_WRAPPERS_HH

#include "myscalapack_fortran.h"

// -----------------------------------------------------------------------------
// Define the overloaded cpp wrappers to ScaLAPACK
// -----------------------------------------------------------------------------

static void sla_ppotrf( const char *uplo, int *n, float *a, int *ia, int *ja, int *desca, int *info )
{
    pspotrf_( uplo, n, a, ia, ja, desca, info );
}
static void sla_ppotrf( const char *uplo, int *n, double *a, int *ia, int *ja, int *desca, int *info )
{
    pdpotrf_( uplo, n, a, ia, ja, desca, info );
}
// void sla_ppotrf( const char *uplo, int *n, std::complex<float>*a, int *ia, int *ja, int *desca, int *info )
// {
//     pcpotrf_( uplo, n, a, ia, ja, desca, info );
// }
// void sla_ppotrf( const char *uplo, int *n, std::complex<double>*a, int *ia, int *ja, int *desca, int *info )
// {
//     pzpotrf_( uplo, n, a, ia, ja, desca, info );
// }

// -----------------------------------------------------------------------------

float sla_plansy( char *norm, char *uplo, int *n, float  *a, int *ia, int *ja, int *desca, float  *work )
{
    return pslansy_( norm, uplo, n, a, ia, ja, desca, work );
}
double sla_plansy( const char *norm, const char *uplo, int *n, double *a, int *ia, int *ja, int *desca, double *work )
{
    return pdlansy_( norm, uplo, n, a, ia, ja, desca, work );
}
// float sla_plansy( const char *norm, const char *uplo, int *n, std::complex<float> *a, int *ia, int *ja, int *desca, double *work )
// {
//     return pclansy_( norm, uplo, n, a, ia, ja, desca, work );
// }
// double sla_plansy( const char *norm, const char *uplo, int *n, std::complex<double> *a, int *ia, int *ja, int *desca, double *work )
// {
//     return pzlansy_( const char *norm, const char *uplo, int *n, lapack_complex_double *a, int *ia, int *ja, int *desca, double *work );
// }

// -----------------------------------------------------------------------------

void sla_psymm(const char *side, const char *uplo, int *m, int *n, float  *alpha, float  *a, int *ia, int *ja, int *desca, float  *b, int *ib, int *jb, int *descb, float  *beta, float  *c, int *ic, int *jc, int *descc)
{
    pssymm_(side, uplo, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}
void sla_psymm(const char *side, const char *uplo, int *m, int *n, double *alpha, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, double *beta, double *c, int *ic, int *jc, int *descc)
{
    pdsymm_(side, uplo, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

// -----------------------------------------------------------------------------

float sla_plange( const char *norm, int *m, int *n, float *A, int *ia, int *ja, int *descA, float *work)
{
    return pslange_( norm, m, n, A, ia, ja, descA, work);
}
double sla_plange( const char *norm, int *m, int *n, double *A, int *ia, int *ja, int *descA, double *work)
{
    return pdlange_( norm, m, n, A, ia, ja, descA, work);
}

// -----------------------------------------------------------------------------



void sla_ppotrs( const char *uplo, int *n, int *nrhs, float  *a, int *ia, int *ja, int *desca, float  *b, int *ib, int *jb, int *descb, int *info )
{
    pspotrs_( uplo, n, nrhs, a, ia, ja, desca, b, ib, jb, descb, info );
}
void sla_ppotrs( const char *uplo, int *n, int *nrhs, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, int *info )
{
    pdpotrs_( uplo, n, nrhs, a, ia, ja, desca, b, ib, jb, descb, info );
}


#endif
