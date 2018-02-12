/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2010      University of Denver, Colorado.
 */

#ifndef MYSCALAPACK_FORTRAN_H
#define MYSCALAPACK_FORTRAN_H

#include "lapack_config.h"
#include "lapack_mangling.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define pspotrf_  LAPACK_GLOBAL(pspotrf,PSPOTRF)
extern void pspotrf_( const char *uplo, int *n, float *a, int *ia, int *ja, int *desca, int *info );
#define pdpotrf_  LAPACK_GLOBAL(pdpotrf,PDPOTRF)
extern void pdpotrf_( const char *uplo, int *n, double *a, int *ia, int *ja, int *desca, int *info );
#define pcpotrf_  LAPACK_GLOBAL(pcpotrf,PCPOTRF)
extern void pcpotrf_( const char *uplo, int *n, lapack_complex_float *a, int *ia, int *ja, int *desca, int *info );
#define pzpotrf_  LAPACK_GLOBAL(pzpotrf,PZPOTRF)
extern void pzpotrf_( const char *uplo, int *n, lapack_complex_double *a, int *ia, int *ja, int *desca, int *info );


#define pslansy_  LAPACK_GLOBAL(pslansy,PSLANSY)
extern float  pslansy_( const char *norm, const char *uplo, int *n, float  *a, int *ia, int *ja, int *desca, float  *work );
#define pdlansy_  LAPACK_GLOBAL(pdlansy,PDLANSY)
extern double pdlansy_ ( const char *norm, const char *uplo, int *n, double *a, int *ia, int *ja, int *desca, double *work );
#define pclansy_  LAPACK_GLOBAL(pclansy,PCLANSY)
extern float pclansy_( const char *norm, const char *uplo, int *n, lapack_complex_float *a, int *ia, int *ja, int *desca, double *work );
#define pzlansy_  LAPACK_GLOBAL(pzlansy,PZLANSY)
extern double pzlansy_( const char *norm, const char *uplo, int *n, lapack_complex_double *a, int *ia, int *ja, int *desca, double *work );


#define pssymm_   LAPACK_GLOBAL(pssymm,PSSYMM)
extern void pssymm_(const char *side, const char *uplo, int *m, int *n, float  *alpha, float  *a, int *ia, int *ja, int *desca, float  *b, int *ib, int *jb, int *descb, float  *beta, float  *c, int *ic, int *jc, int *descc);
#define pdsymm_   LAPACK_GLOBAL(pdsymm,PDSYMM)
extern void pdsymm_(const char *side, const char *uplo, int *m, int *n, double *alpha, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, double *beta, double *c, int *ic, int *jc, int *descc);


#define pslange_  LAPACK_GLOBAL(pslange,PSLANGE)
extern float  pslange_( const char *norm, int *m, int *n, float     *A, int *ia, int *ja, int *descA, float *work);
#define pdlange_  LAPACK_GLOBAL(pdlange,PDLANGE)
extern double pdlange_( const char *norm, int *m, int *n, double    *A, int *ia, int *ja, int *descA, double *work);


#define pspotrs_  LAPACK_GLOBAL(pspotrs,PSPOTRS)
extern void pspotrs_( const char *uplo, int *n, int *nrhs, float  *a, int *ia, int *ja, int *desca, float  *b, int *ib, int *jb, int *descb, int *info );
#define pdpotrs_  LAPACK_GLOBAL(pdpotrs,PDPOTRS)
extern void pdpotrs_( const char *uplo, int *n, int *nrhs, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, int *info );




#define pdpotri_  LAPACK_GLOBAL(pdpotri,PDPOTRI)
#define pdgetrf_  LAPACK_GLOBAL(pdgetrf,PDGETRF)
#define pdmatgen_ LAPACK_GLOBAL(pdmatgen,PDMATGEN)
#define pdtrsm_   LAPACK_GLOBAL(pdtrsm,PDTRSM)
#define psgesv_   LAPACK_GLOBAL(psgesv,PSGESV)
#define pdgesv_   LAPACK_GLOBAL(pdgesv,PDGESV)
#define psgemm_   LAPACK_GLOBAL(psgemm,PSGEMM)
#define pdgemm_   LAPACK_GLOBAL(pdgemm,PDGEMM)
#define numroc_   LAPACK_GLOBAL(numroc,NUMROC)
#define pslacpy_  LAPACK_GLOBAL(pslacpy,PSLACPY)
#define pdlacpy_  LAPACK_GLOBAL(pdlacpy,PDLACPY)
#define pdgeqrf_  LAPACK_GLOBAL(pdgeqrf,PDGEQRF)
#define pdormqr_  LAPACK_GLOBAL(pdormqr,PDORMQR)
#define psgesvd_  LAPACK_GLOBAL(psgesvd,PSGESVD)
#define pdgesvd_  LAPACK_GLOBAL(pdgesvd,PDGESVD)
#define pslaset_  LAPACK_GLOBAL(pslaset,PSLASET)
#define pdlaset_  LAPACK_GLOBAL(pdlaset,PDLASET)
#define pselset_  LAPACK_GLOBAL(pselset,PSELSET)
#define pdelset_  LAPACK_GLOBAL(pdelset,PDELSET)
#define pslamch_  LAPACK_GLOBAL(pslamch,PSLAMCH)
#define pdlamch_  LAPACK_GLOBAL(pdlamch,PDLAMCH)
#define pdaxpy_  LAPACK_GLOBAL(pdaxpy,PDAXPY)
#define indxg2p_  LAPACK_GLOBAL(indxg2p,INDXG2P)
#define indxg2l_  LAPACK_GLOBAL(indxg2l,INDXG2L)
#define descinit_ LAPACK_GLOBAL(descinit,DESCINIT)
#define pslawrite_ LAPACK_GLOBAL(pslawrite,PSLAWRITE)
#define pdlawrite_ LAPACK_GLOBAL(pdlawrite,PDLAWRITE)
#define blacs_get_      LAPACK_GLOBAL(blacs_get,BLACS_GET)
#define blacs_pinfo_    LAPACK_GLOBAL(blacs_pinfo,BLACS_PINFO)
#define blacs_gridinit_ LAPACK_GLOBAL(blacs_gridinit,BLACS_GRIDINIT)
#define blacs_gridinfo_ LAPACK_GLOBAL(blacs_gridinfo,BLACS_GRIDINFO)
#define blacs_gridexit_ LAPACK_GLOBAL(blacs_gridexit,BLACS_GRIDEXIT)
#define blacs_exit_     LAPACK_GLOBAL(blacs_exit,BLACS_EXIT)

extern void Cblacs_pinfo( int* mypnum, int* nprocs );
extern void Cblacs_get( int context, int request, int* value );
extern int  Cblacs_gridinit( int* context, const char * order, int np_row, int np_col );
extern void Cblacs_gridinfo( int context, int*  np_row, int* np_col, int*  my_row, int*  my_col );
extern void Cblacs_gridexit( int context );
extern void Cblacs_exit( int error_code );
extern void Cblacs_abort( int context, int error_code );

extern void blacs_pinfo_( int *mypnum, int *nprocs);
extern void blacs_get_( int *context, int *request, int* value);
extern void blacs_gridinit_( int* context, const char *order, int *np_row, int *np_col);
extern void blacs_gridinfo_( int *context, int *np_row, int *np_col, int *my_row, int *my_col);
extern void blacs_gridexit_( int *context);
extern void blacs_exit_( int *error_code);

extern void pdgeqrf_( int *m, int *n, double *a, int *ia, int *ja, int *desca, double *tau, double *work, int *lwork, int *info );
extern void pdormqr_( const char *side, const char *trans, int *m, int *n, int *k, double *a, int *ia,
                      int *ja, int *desca, double *tau, double *c, int *ic, int *jc, int *descc, double *work, int *lwork, int *info );
extern void pdtrsm_ ( const char *side, const char *uplo, const char *transa, const char *diag, int *m, int *n, double *alpha, double *a, int *ia,
                      int *ja, int *desca, double *b, int *ib, int *jb, int *descb );

extern void pslacpy_( const char *uplo, int *m, int *n, float     *A, int *ia, int *ja, int *descA,
                                                  float     *B, int *ib, int *jb, int *descB);
extern void pdlacpy_( const char *uplo, int *m, int *n, double     *A, int *ia, int *ja, int *descA,
                                                  double     *B, int *ib, int *jb, int *descB);

extern void psgesv_( int *n, int *nrhs, float     *A, int *ia, int *ja, int *descA, int *ipiv,
                                        float     *B, int *ib, int *jb, int *descB, int *info);
extern void pdgesv_( int *n, int *nrhs, double    *A, int *ia, int *ja, int *descA, int *ipiv,
                                        double    *B, int *ib, int *jb, int *descB, int *info);

extern void psgemm_( const char *transa, const char *transb, int *M, int *N, int *K,
                                          float     *alpha,
                                          float     *A, int *ia, int *ja, int *descA,
                                          float     *B, int *ib, int *jb, int *descB,
                                          float     *beta,
                                          float     *C, int *ic, int *jc, int *descC );
extern void pdgemm_( const char *transa, const char *transb, int *M, int *N, int *K,
                                          double    *alpha,
                                          double    *A, int *ia, int *ja, int *descA,
                                          double    *B, int *ib, int *jb, int *descB,
                                          double    *beta,
                                          double    *C, int *ic, int *jc, int *descC );

extern void pssyev_( const char *jobv, const char *uplo, int *m,
                                  float     *A, int *ia, int *ja, int *descA,
                                  float     *W,
                                  float     *Z, int *iz, int *jz, int *descZ,
                                  float     *work, int *lwork, int *info);
extern void pdsyev_( const char *jobv, const char *uplo, int *m,
                                  double    *A, int *ia, int *ja, int *descA,
                                  double    *W,
                                  double    *Z, int *iz, int *jz, int *descZ,
                                  double    *work, int *lwork, int *info);

extern void psgesvd_( const char *jobu, const char *jobvt, int *m, int *n,
                                  float     *A, int *ia, int *ja, int *descA,
                                  float     *s,
                                  float     *U, int *iu, int *ju, int *descU,
                                  float     *VT, int *ivt, int *jvt, int *descVT,
                                  float     *work, int *lwork, int *info);
extern void pdgesvd_( const char *jobu, const char *jobvt, int *m, int *n,
                                  double    *A, int *ia, int *ja, int *descA,
                                  double    *s,
                                  double    *U, int *iu, int *ju, int *descU,
                                  double    *VT, int *ivt, int *jvt, int *descVT,
                                  double    *work, int *lwork, int *info);

extern void pslaset_( const char *uplo, int *m, int *n, float     *alpha, float     *beta, float     *A, int *ia, int *ja, int *descA );
extern void pdlaset_( const char *uplo, int *m, int *n, double    *alpha, double    *beta, double    *A, int *ia, int *ja, int *descA );

extern void pselset_( float     *A, int *ia, int *ja, int *descA, float     *alpha);
extern void pdelset_( double    *A, int *ia, int *ja, int *descA, double    *alpha);

extern void pslawrite_( const char **filenam, int *m, int *n, float  *A, int *ia, int *ja, int *descA, int *irwrit, int *icwrit, float  *work);
extern void pdlawrite_( const char **filenam, int *m, int *n, double *A, int *ia, int *ja, int *descA, int *irwrit, int *icwrit, double *work);

extern float pslamch_( int *ictxt, const char *cmach);
extern double pdlamch_( int *ictxt, const char *cmach);

extern int ilcm_( int *a, int *b );
extern int indxg2p_( int *indxglob, int *nb, int *iproc, int *isrcproc, int *nprocs);
extern int indxg2l_( int *indxglob, int *nb, int *iproc, int *isrcproc, int *nprocs);
extern int numroc_( int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);
extern void descinit_( int *desc, int *m, int *n, int *mb, int *nb, int *irsrc, int *icsrc,
                       int *ictxt, int *lld, int *info);

extern void   pdgetrf_ ( int* m, int *n, double *a, int *i1, int *i2, int *desca, int* ipiv, int *info );
extern void   pdgetrs_ ( const char* trans, int* n, int* nrhs, double* A, int* ia, int* ja, int* descA, int* ippiv, double* B, int* ib, int* jb, int* descB, int* info);
extern void   pdmatgen_( int *ictxt, const char *aform, const char *diag, int *m, int *n, int *mb, int *nb, double *a, int *lda, int *iarow, int *iacol, int *iseed, int *iroff, int *irnum, int *icoff, int *icnum, int *myrow, int *mycol, int *nprow, int *npcol );


extern void pdpotri_( const char *uplo, int *n, double *a, int *ia, int *ja, int *desca, int *info );


extern void pdaxpy_( int *N, double *ALPHA, double * X, int * IX, int * JX, int * DESCX, int * INCX, double * Y, int * IY, int * JY, int * DESCY, int * INCY );


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif

