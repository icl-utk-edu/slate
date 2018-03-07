/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2010      University of Denver, Colorado.
 */

#ifndef SCALAPACK_WRAPPERS_HH
#define SCALAPACK_WRAPPERS_HH

// get BLAS_FORTRAN_NAME and blas_int
#include "blas_fortran.hh"

#include <complex>

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Defined in scalapack_support_routines.cc
// -----------------------------------------------------------------------------

extern "C" void scalapack_pdplrnt( double *A, int m, int n, int mb, int nb, int myrow, int mycol, int nprow, int npcol, int mloc, int seed );
extern "C" void scalapack_pdplghe( double *A, int m, int n, int mb, int nb, int myrow, int mycol, int nprow, int npcol, int mloc, int seed );

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Required CBLACS calls
// -----------------------------------------------------------------------------

extern "C" void Cblacs_pinfo( int* mypnum, int* nprocs );
extern "C" void Cblacs_get( int context, int request, int* value );
extern "C" int  Cblacs_gridinit( int* context, const char * order, int np_row, int np_col );
extern "C" void Cblacs_gridinfo( int context, int*  np_row, int* np_col, int*  my_row, int*  my_col );
extern "C" void Cblacs_gridexit( int context );
extern "C" void Cblacs_exit( int error_code );
extern "C" void Cblacs_abort( int context, int error_code );

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Simple ScaLAPACK routine wrappers
// -----------------------------------------------------------------------------

#define scalapack_descinit BLAS_FORTRAN_NAME(descinit,DESCINIT)
extern "C" void scalapack_descinit( int *desc, int *m, int *n, int *mb, int *nb, int *irsrc, int *icsrc, int *ictxt, int *lld, int *info);

#define scalapack_numroc BLAS_FORTRAN_NAME(numroc,NUMROC)
extern "C" int scalapack_numroc( int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);

#define scalapack_ilcm BLAS_FORTRAN_NAME(ilcm,ILCM)
extern "C" int scalapack_ilcm( int *a, int *b );

#define scalapack_indxg2p BLAS_FORTRAN_NAME(indxg2p,INDXG2P)
extern "C" int scalapack_indxg2p( int *indxglob, int *nb, int *iproc, int *isrcproc, int *nprocs);

#define scalapack_indxg2l BLAS_FORTRAN_NAME(indxg2l,INDXG2L)
extern "C" int scalapack_indxg2l( int *indxglob, int *nb, int *iproc, int *isrcproc, int *nprocs);

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Type generic ScaLAPACK wrappers
// -----------------------------------------------------------------------------

#define scalapack_pslange BLAS_FORTRAN_NAME( pslange, PSLANGE )
#define scalapack_pdlange BLAS_FORTRAN_NAME( pdlange, PDLANGE )
#define scalapack_pclange BLAS_FORTRAN_NAME( pclange, PCLANGE )
#define scalapack_pzlange BLAS_FORTRAN_NAME( pzlange, PZLANGE )

extern "C" blas_float_return scalapack_pslange( const char *norm, blas_int *m, blas_int *n, float *A, blas_int *ia, blas_int *ja, blas_int *descA, float *work);

extern "C" double scalapack_pdlange( const char *norm, blas_int *m, blas_int *n, double *A, blas_int *ia, blas_int *ja, blas_int *descA, double *work);

extern "C" blas_float_return scalapack_pclange( const char *norm, blas_int *m, blas_int *n, std::complex<float> *A, blas_int *ia, blas_int *ja, blas_int *descA, float *work);

extern "C" double scalapack_pzlange( const char *norm, blas_int *m, blas_int *n, std::complex<double> *A, blas_int *ia, blas_int *ja, blas_int *descA, double *work);

// -----------------------------------------------------------------------------

inline blas_float_return scalapack_plange( const char *norm, blas_int *m, blas_int *n, float *A, blas_int *ia, blas_int *ja, blas_int *descA, float *work)
{
    return scalapack_pslange( norm, m, n, A, ia, ja, descA, work);
}

inline double scalapack_plange( const char *norm, blas_int *m, blas_int *n, double *A, blas_int *ia, blas_int *ja, blas_int *descA, double *work)
{
    return scalapack_pdlange( norm, m, n, A, ia, ja, descA, work);
}

inline blas_float_return scalapack_plange( const char *norm, blas_int *m, blas_int *n, std::complex<float> *A, blas_int *ia, blas_int *ja, blas_int *descA, float *work)
{
    return scalapack_pclange( norm, m, n, A, ia, ja, descA, work);
}

inline double scalapack_plange( const char *norm, blas_int *m, blas_int *n, std::complex<double> *A, blas_int *ia, blas_int *ja, blas_int *descA, double *work)
{
    return scalapack_pzlange( norm, m, n, A, ia, ja, descA, work);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pspotrf BLAS_FORTRAN_NAME( pspotrf, PSPOTRF )
#define scalapack_pdpotrf BLAS_FORTRAN_NAME( pdpotrf, PDPOTRF )
#define scalapack_pcpotrf BLAS_FORTRAN_NAME( pcpotrf, PCPOTRF )
#define scalapack_pzpotrf BLAS_FORTRAN_NAME( pzpotrf, PZPOTRF )

extern "C" void scalapack_pspotrf( const char *uplo, blas_int *n, float *a, blas_int *ia, blas_int *ja, blas_int *desca, blas_int *info );

extern "C" void scalapack_pdpotrf( const char *uplo, blas_int *n, double *a, blas_int *ia, blas_int *ja, blas_int *desca, blas_int *info );

extern "C" void scalapack_pcpotrf( const char *uplo, blas_int *n, std::complex<float> *a, blas_int *ia, blas_int *ja, blas_int *desca, blas_int *info );

extern void scalapack_pzpotrf( const char *uplo, blas_int *n, std::complex<double> *a, blas_int *ia, blas_int *ja, blas_int *desca, blas_int *info );

// -----------------------------------------------------------------------------

inline void scalapack_ppotrf( const char *uplo, blas_int *n, float *a, blas_int *ia, blas_int *ja, blas_int *desca, blas_int *info )
{
    scalapack_pspotrf( uplo, n, a, ia, ja, desca, info );
}

inline void scalapack_ppotrf( const char *uplo, blas_int *n, double *a, blas_int *ia, blas_int *ja, blas_int *desca, blas_int *info )
{
    scalapack_pdpotrf( uplo, n, a, ia, ja, desca, info );
}

inline void scalapack_ppotrf( const char *uplo, blas_int *n, std::complex<float>*a, blas_int *ia, blas_int *ja, blas_int *desca, blas_int *info )
{
    scalapack_pcpotrf( uplo, n, a, ia, ja, desca, info );
}

inline void scalapack_ppotrf( const char *uplo, blas_int *n, std::complex<double>*a, blas_int *ia, blas_int *ja, blas_int *desca, blas_int *info )
{
    scalapack_pzpotrf( uplo, n, a, ia, ja, desca, info );
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pslansy BLAS_FORTRAN_NAME(pslansy,PSLANSY)
#define scalapack_pdlansy BLAS_FORTRAN_NAME(pdlansy,PDLANSY)
#define scalapack_pclansy BLAS_FORTRAN_NAME(pclansy,PCLANSY)
#define scalapack_pzlansy BLAS_FORTRAN_NAME(pzlansy,PZLANSY)

extern "C" float scalapack_pslansy( const char *norm, const char *uplo, blas_int *n, float  *a, blas_int *ia, blas_int *ja, blas_int *desca, float  *work );
extern "C" double scalapack_pdlansy ( const char *norm, const char *uplo, blas_int *n, double *a, blas_int *ia, blas_int *ja, blas_int *desca, double *work );
extern "C" float scalapack_pclansy( const char *norm, const char *uplo, blas_int *n, std::complex<float> *a, blas_int *ia, blas_int *ja, blas_int *desca, double *work );
extern "C" double scalapack_pzlansy( const char *norm, const char *uplo, blas_int *n, std::complex<double> *a, blas_int *ia, blas_int *ja, blas_int *desca, double *work );

// -----------------------------------------------------------------------------

inline float scalapack_plansy( char *norm, char *uplo, blas_int *n, float *a, blas_int *ia, blas_int *ja, blas_int *desca, float *work )
{
    return scalapack_pslansy( norm, uplo, n, a, ia, ja, desca, work );
}

inline double scalapack_plansy( const char *norm, const char *uplo, blas_int *n, double *a, blas_int *ia, blas_int *ja, blas_int *desca, double *work )
{
    return scalapack_pdlansy( norm, uplo, n, a, ia, ja, desca, work );
}

inline float scalapack_plansy( const char *norm, const char *uplo, blas_int *n, std::complex<float> *a, blas_int *ia, blas_int *ja, blas_int *desca, double *work )
{
    return scalapack_pclansy( norm, uplo, n, a, ia, ja, desca, work );
}

inline double scalapack_plansy( const char *norm, const char *uplo, blas_int *n, std::complex<double> *a, blas_int *ia, blas_int *ja, blas_int *desca, double *work )
{
    return scalapack_pzlansy( norm, uplo, n, a, ia, ja, desca, work );

}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_psgemm BLAS_FORTRAN_NAME( psgemm, PSGEMM )
#define scalapack_pdgemm BLAS_FORTRAN_NAME( pdgemm, PDGEMM )

extern "C" void psgemm_( const char *transa, const char *transb, int *M, int *N, int *K, float *alpha, float *A, int *ia, int *ja, int *descA, float *B, int *ib, int *jb, int *descB, float *beta, float *C, int *ic, int *jc, int *descC );
extern "C" void pdgemm_( const char *transa, const char *transb, int *M, int *N, int *K, double *alpha, double *A, int *ia, int *ja, int *descA, double *B, int *ib, int *jb, int *descB, double *beta, double *C, int *ic, int *jc, int *descC );

// -----------------------------------------------------------------------------

inline void scalapack_pgemm( const char *transa, const char *transb, int *M, int *N, int *K, float *alpha, float *A, int *ia, int *ja, int *descA, float *B, int *ib, int *jb, int *descB, float *beta, float *C, int *ic, int *jc, int *descC )
{
    scalapack_psgemm( transa, transb, M, N, K, alpha, A, ia, ja, descA, B, ib, jb, descB, beta, C, ic, jc, descC );
}
inline void scalapack_pgemm( const char *transa, const char *transb, int *M, int *N, int *K, double *alpha, double *A, int *ia, int *ja, int *descA, double *B, int *ib, int *jb, int *descB, double *beta, double *C, int *ic, int *jc, int *descC )
{
    scalapack_pdgemm( transa, transb, M, N, K, alpha, A, ia, ja, descA, B, ib, jb, descB, beta, C, ic, jc, descC );
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pspotrs BLAS_FORTRAN_NAME(pspotrs,PSPOTRS)
#define scalapack_pdpotrs BLAS_FORTRAN_NAME(pdpotrs,PDPOTRS)

extern "C" void scalapack_pspotrs( const char *uplo, int *n, int *nrhs, float  *a, int *ia, int *ja, int *desca, float  *b, int *ib, int *jb, int *descb, int *info );

extern "C" void scalapack_pdpotrs( const char *uplo, int *n, int *nrhs, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, int *info );

// -----------------------------------------------------------------------------

inline void scalapack_ppotrs( const char *uplo, int *n, int *nrhs, float *a, int *ia, int *ja, int *desca, float *b, int *ib, int *jb, int *descb, int *info )
{
    scalapack_pspotrs( uplo, n, nrhs, a, ia, ja, desca, b, ib, jb, descb, info );
}
inline void scalapack_ppotrs( const char *uplo, int *n, int *nrhs, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, int *info )
{
    scalapack_pdpotrs( uplo, n, nrhs, a, ia, ja, desca, b, ib, jb, descb, info );
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pssymm BLAS_FORTRAN_NAME(pssymm,PSSYMM)
#define scalapack_pdsymm BLAS_FORTRAN_NAME(pdsymm,PDSYMM)

extern "C" void scalapack_pssymm(const char *side, const char *uplo, int *m, int *n, float  *alpha, float  *a, int *ia, int *ja, int *desca, float  *b, int *ib, int *jb, int *descb, float  *beta, float  *c, int *ic, int *jc, int *descc);

extern "C" void scalapack_pdsymm(const char *side, const char *uplo, int *m, int *n, double *alpha, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, double *beta, double *c, int *ic, int *jc, int *descc);

// -----------------------------------------------------------------------------

inline void scalapack_psymm(const char *side, const char *uplo, int *m, int *n, float *alpha, float *a, int *ia, int *ja, int *desca, float *b, int *ib, int *jb, int *descb, float *beta, float *c, int *ic, int *jc, int *descc)
{
    scalapack_pssymm(side, uplo, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

inline void scalapack_psymm(const char *side, const char *uplo, int *m, int *n, double *alpha, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, double *beta, double *c, int *ic, int *jc, int *descc)
{
    scalapack_pdsymm(side, uplo, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pstrmm BLAS_FORTRAN_NAME(pstrmm,PSTRMM)
#define scalapack_pdtrmm BLAS_FORTRAN_NAME(pdtrmm,PDTRMM)
#define scalapack_pctrmm BLAS_FORTRAN_NAME(pctrmm,PCTRMM)
#define scalapack_pztrmm BLAS_FORTRAN_NAME(pztrmm,PZTRMM)

extern "C" void scalapack_pstrmm (const char *side , const char *uplo , const char *transa , const char *diag , const blas_int *m , const blas_int *n , const float *alpha , const float *a , const blas_int *ia , const blas_int *ja , const blas_int *desca , float *b , const blas_int *ib , const blas_int *jb , const blas_int *descb );

extern "C" void scalapack_pdtrmm (const char *side , const char *uplo , const char *transa , const char *diag , const blas_int *m , const blas_int *n , const double *alpha , const double *a , const blas_int *ia , const blas_int *ja , const blas_int *desca , double *b , const blas_int *ib , const blas_int *jb , const blas_int *descb );

extern "C" void scalapack_pctrmm (const char *side , const char *uplo , const char *transa , const char *diag , const blas_int *m , const blas_int *n , const std::complex<float> *alpha , const std::complex<float> *a , const blas_int *ia , const blas_int *ja , const blas_int *desca , std::complex<float> *b , const blas_int *ib , const blas_int *jb , const blas_int *descb );

extern "C" void scalapack_pztrmm (const char *side , const char *uplo , const char *transa , const char *diag , const blas_int *m , const blas_int *n , const std::complex<double> *alpha , const std::complex<double> *a , const blas_int *ia , const blas_int *ja , const blas_int *desca , std::complex<double> *b , const blas_int *ib , const blas_int *jb , const blas_int *descb );

// -----------------------------------------------------------------------------

inline void scalapack_ptrmm (const char *side, const char *uplo, const char *transa, const char *diag, const blas_int *m, const blas_int *n, const float *alpha, const float *a, const blas_int *ia, const blas_int *ja, const blas_int *desca, float *b, const blas_int *ib, const blas_int *jb, const blas_int *descb )
{
    scalapack_pstrmm( side, uplo, transa, diag, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb );
}

inline void scalapack_ptrmm (const char *side, const char *uplo, const char *transa, const char *diag, const blas_int *m, const blas_int *n, const double *alpha, const double *a, const blas_int *ia, const blas_int *ja, const blas_int *desca, double *b, const blas_int *ib, const blas_int *jb, const blas_int *descb )
{
    scalapack_pdtrmm( side, uplo, transa, diag, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb );
}

inline void scalapack_ptrmm (const char *side, const char *uplo, const char *transa, const char *diag, const blas_int *m, const blas_int *n, const std::complex<float> *alpha, const std::complex<float> *a, const blas_int *ia, const blas_int *ja, const blas_int *desca, std::complex<float> *b, const blas_int *ib, const blas_int *jb, const blas_int *descb )
{
    scalapack_pctrmm( side, uplo, transa, diag, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb );
}

inline void scalapack_ptrmm (const char *side, const char *uplo, const char *transa, const char *diag, const blas_int *m, const blas_int *n, const std::complex<double> *alpha, const std::complex<double> *a, const blas_int *ia, const blas_int *ja, const blas_int *desca, std::complex<double> *b, const blas_int *ib, const blas_int *jb, const blas_int *descb )
{
    scalapack_pztrmm( side, uplo, transa, diag, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb );
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
#endif
