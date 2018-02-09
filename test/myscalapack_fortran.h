/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2010      University of Denver, Colorado.
 */

#ifndef MYSCALAPACK_H
#define MYSCALAPACK_H

#include "lapack_config.h"
#include "lapack_mangling.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifdef SUP_
#define pdpotrf_  pdpotrf
#define pdpotri_  pdpotri
#define pdpotrs_  pdpotrs
#define pdsymm_   pdsymm
#define pdgetrf_  pdgetrf
#define pdlansy_  pdlansy
#define pdmatgen_ pdmatgen
#define pdtrsm_   pdtrsm
#define psgesv_   psgesv
#define pdgesv_   pdgesv
#define psgemm_   psgemm
#define pdgemm_   pdgemm
#define numroc_   numroc
#define pslange_  pslange
#define pdlange_  pdlange
#define pslacpy_  pslacpy
#define pdlacpy_  pdlacpy
#define pdgeqrf_  pdgeqrf
#define pdormqr_  pdormqr
#define psgesvd_  psgesvd
#define pdgesvd_  pdgesvd
#define pslaset_  pslaset
#define pdlaset_  pdlaset
#define pselset_  pselset
#define pdelset_  pdelset
#define pslamch_  pslamch
#define pdlamch_  pdlamch
#define pdaxpy_  pdaxpy
#define indxg2p_  indxg2p
#define indxg2l_  indxg2l
#define descinit_ descinit
#define pslawrite_ pslawrite
#define pdlawrite_ pdlawrite
#define blacs_get_      blacs_get
#define blacs_pinfo_    blacs_pinfo
#define blacs_gridinit_ blacs_gridinit
#define blacs_gridinfo_ blacs_gridinfo
#define blacs_gridexit_ blacs_gridexit
#define blacs_exit_     blacs_exit
#endif

extern void Cblacs_pinfo( int* mypnum, int* nprocs );
extern void Cblacs_get( int context, int request, int* value );
extern int  Cblacs_gridinit( int* context, const char * order, int np_row, int np_col );
extern void Cblacs_gridinfo( int context, int*  np_row, int* np_col, int*  my_row, int*  my_col );
extern void Cblacs_gridexit( int context );
extern void Cblacs_exit( int error_code );
extern void Cblacs_abort( int context, int error_code );

extern void blacs_pinfo_( int *mypnum, int *nprocs);
extern void blacs_get_( int *context, int *request, int* value);
extern void blacs_gridinit_( int* context, char *order, int *np_row, int *np_col);
extern void blacs_gridinfo_( int *context, int *np_row, int *np_col, int *my_row, int *my_col);
extern void blacs_gridexit_( int *context);
extern void blacs_exit_( int *error_code);

extern void pdgeqrf_( int *m, int *n, double *a, int *ia, int *ja, int *desca, double *tau, double *work, int *lwork, int *info );
extern void pdormqr_( char *side, char *trans, int *m, int *n, int *k, double *a, int *ia,
                      int *ja, int *desca, double *tau, double *c, int *ic, int *jc, int *descc, double *work, int *lwork, int *info );
extern void pdtrsm_ ( char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha, double *a, int *ia,
                      int *ja, int *desca, double *b, int *ib, int *jb, int *descb );

extern float  pslange_( char *norm, int *m, int *n, float     *A, int *ia, int *ja, int *descA, float *work);
extern double pdlange_( const char *norm, int *m, int *n, double    *A, int *ia, int *ja, int *descA, double *work);


extern void pslacpy_( char *uplo, int *m, int *n, float     *A, int *ia, int *ja, int *descA,
                                                  float     *B, int *ib, int *jb, int *descB);
extern void pdlacpy_( const char *uplo, int *m, int *n, double     *A, int *ia, int *ja, int *descA,
                                                  double     *B, int *ib, int *jb, int *descB);

extern void psgesv_( int *n, int *nrhs, float     *A, int *ia, int *ja, int *descA, int *ipiv,
                                        float     *B, int *ib, int *jb, int *descB, int *info);
extern void pdgesv_( int *n, int *nrhs, double    *A, int *ia, int *ja, int *descA, int *ipiv,
                                        double    *B, int *ib, int *jb, int *descB, int *info);

extern void psgemm_( char *transa, char *transb, int *M, int *N, int *K,
                                          float     *alpha,
                                          float     *A, int *ia, int *ja, int *descA,
                                          float     *B, int *ib, int *jb, int *descB,
                                          float     *beta,
                                          float     *C, int *ic, int *jc, int *descC );
extern void pdgemm_( char *transa, char *transb, int *M, int *N, int *K,
                                          double    *alpha,
                                          double    *A, int *ia, int *ja, int *descA,
                                          double    *B, int *ib, int *jb, int *descB,
                                          double    *beta,
                                          double    *C, int *ic, int *jc, int *descC );

extern void pssyev_( char *jobv, char *uplo, int *m,
                                  float     *A, int *ia, int *ja, int *descA,
                                  float     *W,
                                  float     *Z, int *iz, int *jz, int *descZ,
                                  float     *work, int *lwork, int *info);
extern void pdsyev_( char *jobv, char *uplo, int *m,
                                  double    *A, int *ia, int *ja, int *descA,
                                  double    *W,
                                  double    *Z, int *iz, int *jz, int *descZ,
                                  double    *work, int *lwork, int *info);

extern void psgesvd_( char *jobu, char *jobvt, int *m, int *n,
                                  float     *A, int *ia, int *ja, int *descA,
                                  float     *s,
                                  float     *U, int *iu, int *ju, int *descU,
                                  float     *VT, int *ivt, int *jvt, int *descVT,
                                  float     *work, int *lwork, int *info);
extern void pdgesvd_( char *jobu, char *jobvt, int *m, int *n,
                                  double    *A, int *ia, int *ja, int *descA,
                                  double    *s,
                                  double    *U, int *iu, int *ju, int *descU,
                                  double    *VT, int *ivt, int *jvt, int *descVT,
                                  double    *work, int *lwork, int *info);

extern void pslaset_( char *uplo, int *m, int *n, float     *alpha, float     *beta, float     *A, int *ia, int *ja, int *descA );
extern void pdlaset_( char *uplo, int *m, int *n, double    *alpha, double    *beta, double    *A, int *ia, int *ja, int *descA );

extern void pselset_( float     *A, int *ia, int *ja, int *descA, float     *alpha);
extern void pdelset_( double    *A, int *ia, int *ja, int *descA, double    *alpha);

extern void pslawrite_( char **filenam, int *m, int *n, float  *A, int *ia, int *ja, int *descA, int *irwrit, int *icwrit, float  *work);
extern void pdlawrite_( char **filenam, int *m, int *n, double *A, int *ia, int *ja, int *descA, int *irwrit, int *icwrit, double *work);

extern float pslamch_( int *ictxt, char *cmach);
extern double pdlamch_( int *ictxt, const char *cmach);

extern int ilcm_( int *a, int *b );
extern int indxg2p_( int *indxglob, int *nb, int *iproc, int *isrcproc, int *nprocs);
extern int indxg2l_( int *indxglob, int *nb, int *iproc, int *isrcproc, int *nprocs);
extern int numroc_( int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);
extern void descinit_( int *desc, int *m, int *n, int *mb, int *nb, int *irsrc, int *icsrc,
                       int *ictxt, int *lld, int *info);

extern void   pdgetrf_ ( int* m, int *n, double *a, int *i1, int *i2, int *desca, int* ipiv, int *info );
extern void   pdgetrs_ ( char* trans, int* n, int* nrhs, double* A, int* ia, int* ja, int* descA, int* ippiv, double* B, int* ib, int* jb, int* descB, int* info);
extern double pdlansy_ ( const char *norm, const char *uplo, int *n, double *a, int *ia, int *ja, int *desca, double *work );
extern float  pslansy_ ( char *norm, char *uplo, int *n, float  *a, int *ia, int *ja, int *desca, float  *work );
extern void   pdmatgen_( int *ictxt, char *aform, char *diag, int *m, int *n, int *mb, int *nb, double *a, int *lda, int *iarow, int *iacol, int *iseed, int *iroff, int *irnum, int *icoff, int *icnum, int *myrow, int *mycol, int *nprow, int *npcol );


extern void pdpotrf_( const char *uplo, int *n, double *a, int *ia, int *ja, int *desca, int *info );
extern void pspotrf_( char *uplo, int *n, float  *a, int *ia, int *ja, int *desca, int *info );
extern void pdpotri_( char *uplo, int *n, double *a, int *ia, int *ja, int *desca, int *info );
extern void pdpotrs_( const char *uplo, int *n, int *nrhs, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, int *info );
extern void pspotrs_( char *uplo, int *n, int *nrhs, float  *a, int *ia, int *ja, int *desca, float  *b, int *ib, int *jb, int *descb, int *info );

extern void pdsymm_(const char *side, const char *uplo, int *m, int *n, double *alpha, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, double *beta, double *c, int *ic, int *jc, int *descc);
extern void pssymm_(char *side, char *uplo, int *m, int *n, float  *alpha, float  *a, int *ia, int *ja, int *desca, float  *b, int *ib, int *jb, int *descb, float  *beta, float  *c, int *ic, int *jc, int *descc);

extern void pdaxpy_( int *N, double *ALPHA, double * X, int * IX, int * JX, int * DESCX, int * INCX, double * Y, int * IY, int * JY, int * DESCY, int * INCY );


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
