/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2010      University of Denver, Colorado.
 */

#ifndef ICL_SLATE_SCALAPACK_WRAPPERS_HH
#define ICL_SLATE_SCALAPACK_WRAPPERS_HH

// get BLAS_FORTRAN_NAME and blas_int
#include "blas_fortran.hh"

#include <complex>

// -----------------------------------------------------------------------------
// helper funtion to check and do type conversion
inline int int64_to_int (int64_t n)
{
    if (sizeof (int64_t) > sizeof (blas_int))
        assert (n < std::numeric_limits<int>::max());
    int n_ = (int)n;
    return n_;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Required CBLACS calls
// -----------------------------------------------------------------------------

extern "C" void Cblacs_pinfo (int *mypnum, int *nprocs);
extern "C" void Cblacs_get (int context, int request, int *value);
extern "C" int  Cblacs_gridinit (int *context, const char *order, int np_row, int np_col);
extern "C" void Cblacs_gridinfo (int context, int  *np_row, int *np_col, int  *my_row, int  *my_col);
extern "C" void Cblacs_gridexit (int context);
extern "C" void Cblacs_exit (int error_code);
extern "C" void Cblacs_abort (int context, int error_code);

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Simple ScaLAPACK routine wrappers
// -----------------------------------------------------------------------------

#define scalapack_descinit BLAS_FORTRAN_NAME(descinit,DESCINIT)
extern "C" void scalapack_descinit (int *desc, int *m, int *n, int *mb, int *nb, int *irsrc, int *icsrc, int *ictxt, int *lld, int *info);
inline void scalapack_descinit (int *desc, int64_t m, int64_t n, int64_t mb, int64_t nb, int irsrc, int icsrc, int ictxt, int64_t lld, int *info)
{
    int m_ = int64_to_int (m);
    int n_ = int64_to_int (n);
    int mb_ = int64_to_int (mb);
    int nb_ = int64_to_int (nb);
    int lld_ = int64_to_int (lld);
    scalapack_descinit (desc, &m_, &n_, &mb_, &nb_, &irsrc, &icsrc, &ictxt, &lld_, info);
}

#define scalapack_numroc BLAS_FORTRAN_NAME(numroc,NUMROC)
extern "C" int scalapack_numroc (int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);
inline int64_t scalapack_numroc (int64_t n, int64_t nb, int iproc, int isrcproc, int nprocs)
{
    int n_ = int64_to_int (n);
    int nb_ = int64_to_int (nb);
    int nroc_ = scalapack_numroc (&n_, &nb_, &iproc, &isrcproc, &nprocs);
    int64_t nroc = (int64_t)nroc_;
    return nroc;
}

#define scalapack_ilcm BLAS_FORTRAN_NAME(ilcm,ILCM)
extern "C" int scalapack_ilcm (int *a, int *b);

#define scalapack_indxg2p BLAS_FORTRAN_NAME(indxg2p,INDXG2P)
extern "C" int scalapack_indxg2p (int *indxglob, int *nb, int *iproc, int *isrcproc, int *nprocs);

#define scalapack_indxg2l BLAS_FORTRAN_NAME(indxg2l,INDXG2L)
extern "C" int scalapack_indxg2l (int *indxglob, int *nb, int *iproc, int *isrcproc, int *nprocs);


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Type generic ScaLAPACK wrappers
// -----------------------------------------------------------------------------

#define scalapack_pslange BLAS_FORTRAN_NAME( pslange, PSLANGE )
#define scalapack_pdlange BLAS_FORTRAN_NAME( pdlange, PDLANGE )
#define scalapack_pclange BLAS_FORTRAN_NAME( pclange, PCLANGE )
#define scalapack_pzlange BLAS_FORTRAN_NAME( pzlange, PZLANGE )

extern "C" blas_float_return scalapack_pslange (const char *norm, blas_int *m, blas_int *n, float *A, blas_int *ia, blas_int *ja, blas_int *descA, float *work);

extern "C" double scalapack_pdlange (const char *norm, blas_int *m, blas_int *n, double *A, blas_int *ia, blas_int *ja, blas_int *descA, double *work);

extern "C" blas_float_return scalapack_pclange (const char *norm, blas_int *m, blas_int *n, std::complex<float> *A, blas_int *ia, blas_int *ja, blas_int *descA, float *work);

extern "C" double scalapack_pzlange (const char *norm, blas_int *m, blas_int *n, std::complex<double> *A, blas_int *ia, blas_int *ja, blas_int *descA, double *work);

// -----------------------------------------------------------------------------

inline blas_float_return scalapack_plange (const char *norm, blas_int *m, blas_int *n, float *A, blas_int *ia, blas_int *ja, blas_int *descA, float *work)
{
    return scalapack_pslange (norm, m, n, A, ia, ja, descA, work);
}
inline double scalapack_plange (const char *norm, blas_int *m, blas_int *n, double *A, blas_int *ia, blas_int *ja, blas_int *descA, double *work)
{
    return scalapack_pdlange (norm, m, n, A, ia, ja, descA, work);
}
inline blas_float_return scalapack_plange (const char *norm, blas_int *m, blas_int *n, std::complex<float> *A, blas_int *ia, blas_int *ja, blas_int *descA, float *work)
{
    return scalapack_pclange (norm, m, n, A, ia, ja, descA, work);
}
inline double scalapack_plange (const char *norm, blas_int *m, blas_int *n, std::complex<double> *A, blas_int *ia, blas_int *ja, blas_int *descA, double *work)
{
    return scalapack_pzlange (norm, m, n, A, ia, ja, descA, work);
}

template <typename scalar_t>
inline blas::real_type<scalar_t> scalapack_plange (const char *norm, int64_t m, int64_t n, scalar_t *A, int64_t ia, int64_t ja, blas_int *descA, blas::real_type<scalar_t> *work)
{
    int m_ = int64_to_int (m);
    int n_ = int64_to_int (n);
    int ia_ = int64_to_int (ia);
    int ja_ = int64_to_int (ja);
    return scalapack_plange (norm, &m_, &n_, A, &ia_, &ja_, descA, work);
}


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pspotrf BLAS_FORTRAN_NAME( pspotrf, PSPOTRF )
#define scalapack_pdpotrf BLAS_FORTRAN_NAME( pdpotrf, PDPOTRF )
#define scalapack_pcpotrf BLAS_FORTRAN_NAME( pcpotrf, PCPOTRF )
#define scalapack_pzpotrf BLAS_FORTRAN_NAME( pzpotrf, PZPOTRF )

extern "C" void scalapack_pspotrf (const char *uplo, blas_int *n, float *a, blas_int *ia, blas_int *ja, blas_int *desca, blas_int *info);

extern "C" void scalapack_pdpotrf (const char *uplo, blas_int *n, double *a, blas_int *ia, blas_int *ja, blas_int *desca, blas_int *info);

extern "C" void scalapack_pcpotrf (const char *uplo, blas_int *n, std::complex<float> *a, blas_int *ia, blas_int *ja, blas_int *desca, blas_int *info);

extern "C" void scalapack_pzpotrf (const char *uplo, blas_int *n, std::complex<double> *a, blas_int *ia, blas_int *ja, blas_int *desca, blas_int *info);

// -----------------------------------------------------------------------------

inline void scalapack_ppotrf (const char *uplo, blas_int *n, float *a, blas_int *ia, blas_int *ja, blas_int *desca, blas_int *info)
{
    scalapack_pspotrf (uplo, n, a, ia, ja, desca, info);
}

inline void scalapack_ppotrf (const char *uplo, blas_int *n, double *a, blas_int *ia, blas_int *ja, blas_int *desca, blas_int *info)
{
    scalapack_pdpotrf (uplo, n, a, ia, ja, desca, info);
}

inline void scalapack_ppotrf (const char *uplo, blas_int *n, std::complex<float> *a, blas_int *ia, blas_int *ja, blas_int *desca, blas_int *info)
{
    scalapack_pcpotrf (uplo, n, a, ia, ja, desca, info);
}

inline void scalapack_ppotrf (const char *uplo, blas_int *n, std::complex<double> *a, blas_int *ia, blas_int *ja, blas_int *desca, blas_int *info)
{
    scalapack_pzpotrf (uplo, n, a, ia, ja, desca, info);
}

template <typename scalar_t>
inline void scalapack_ppotrf (const char *uplo, int64_t n, scalar_t *a, int64_t ia, int64_t ja, int *desca, blas_int *info)
{
    int n_ = int64_to_int (n);
    int ia_ = int64_to_int (ia);
    int ja_ = int64_to_int (ja);
    scalapack_ppotrf (uplo, &n_, a, &ia_, &ja_, desca, info);
}


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pslansy BLAS_FORTRAN_NAME(pslansy,PSLANSY)
#define scalapack_pdlansy BLAS_FORTRAN_NAME(pdlansy,PDLANSY)
#define scalapack_pclansy BLAS_FORTRAN_NAME(pclansy,PCLANSY)
#define scalapack_pzlansy BLAS_FORTRAN_NAME(pzlansy,PZLANSY)

extern "C" float scalapack_pslansy (const char *norm, const char *uplo, blas_int *n, float  *a, blas_int *ia, blas_int *ja, blas_int *desca, float  *work);

extern "C" double scalapack_pdlansy (const char *norm, const char *uplo, blas_int *n, double *a, blas_int *ia, blas_int *ja, blas_int *desca, double *work);

extern "C" float scalapack_pclansy (const char *norm, const char *uplo, blas_int *n, std::complex<float> *a, blas_int *ia, blas_int *ja, blas_int *desca, float *work);

extern "C" double scalapack_pzlansy (const char *norm, const char *uplo, blas_int *n, std::complex<double> *a, blas_int *ia, blas_int *ja, blas_int *desca, double *work);

// -----------------------------------------------------------------------------

inline float scalapack_plansy (const char *norm, const char *uplo, blas_int *n, float *a, blas_int *ia, blas_int *ja, blas_int *desca, float *work)
{
    return scalapack_pslansy (norm, uplo, n, a, ia, ja, desca, work);
}

inline double scalapack_plansy (const char *norm, const char *uplo, blas_int *n, double *a, blas_int *ia, blas_int *ja, blas_int *desca, double *work)
{
    return scalapack_pdlansy (norm, uplo, n, a, ia, ja, desca, work);
}

inline float scalapack_plansy (const char *norm, const char *uplo, blas_int *n, std::complex<float> *a, blas_int *ia, blas_int *ja, blas_int *desca, float *work)
{
    return scalapack_pclansy (norm, uplo, n, a, ia, ja, desca, work);
}

inline double scalapack_plansy (const char *norm, const char *uplo, blas_int *n, std::complex<double> *a, blas_int *ia, blas_int *ja, blas_int *desca, double *work)
{
    return scalapack_pzlansy (norm, uplo, n, a, ia, ja, desca, work);
}

template <typename scalar_t>
inline double scalapack_plansy (const char *norm, const char *uplo, int64_t n, scalar_t *a, int64_t ia, int64_t ja, int *desca, blas::real_type<scalar_t> *work)
{
    int n_ = int64_to_int (n);
    int ia_ = int64_to_int (ia);
    int ja_ = int64_to_int (ja);
    return scalapack_plansy (norm, uplo, &n_, a, &ia_, &ja_, desca, work);
}


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_psgemm BLAS_FORTRAN_NAME( psgemm, PSGEMM )
#define scalapack_pdgemm BLAS_FORTRAN_NAME( pdgemm, PDGEMM )
#define scalapack_pcgemm BLAS_FORTRAN_NAME( pcgemm, PCGEMM )
#define scalapack_pzgemm BLAS_FORTRAN_NAME( pzgemm, PZGEMM )

extern "C" void scalapack_psgemm (const char *transa, const char *transb, int *M, int *N, int *K, float *alpha, float *A, int *ia, int *ja, int *descA, float *B, int *ib, int *jb, int *descB, float *beta, float *C, int *ic, int *jc, int *descC);

extern "C" void scalapack_pdgemm (const char *transa, const char *transb, int *M, int *N, int *K, double *alpha, double *A, int *ia, int *ja, int *descA, double *B, int *ib, int *jb, int *descB, double *beta, double *C, int *ic, int *jc, int *descC);

extern "C" void scalapack_pcgemm (const char *transa, const char *transb, int *M, int *N, int *K, std::complex<float> *alpha, std::complex<float> *A, int *ia, int *ja, int *descA, std::complex<float> *B, int *ib, int *jb, int *descB, std::complex<float> *beta, std::complex<float> *C, int *ic, int *jc, int *descC);

extern "C" void scalapack_pzgemm (const char *transa, const char *transb, int *M, int *N, int *K, std::complex<double> *alpha, std::complex<double> *A, int *ia, int *ja, int *descA, std::complex<double> *B, int *ib, int *jb, int *descB, std::complex<double> *beta, std::complex<double> *C, int *ic, int *jc, int *descC);

// -----------------------------------------------------------------------------

inline void scalapack_pgemm (const char *transa, const char *transb, int *M, int *N, int *K, float *alpha, float *A, int *ia, int *ja, int *descA, float *B, int *ib, int *jb, int *descB, float *beta, float *C, int *ic, int *jc, int *descC)
{
    scalapack_psgemm (transa, transb, M, N, K, alpha, A, ia, ja, descA, B, ib, jb, descB, beta, C, ic, jc, descC);
}

inline void scalapack_pgemm (const char *transa, const char *transb, int *M, int *N, int *K, double *alpha, double *A, int *ia, int *ja, int *descA, double *B, int *ib, int *jb, int *descB, double *beta, double *C, int *ic, int *jc, int *descC)
{
    scalapack_pdgemm (transa, transb, M, N, K, alpha, A, ia, ja, descA, B, ib, jb, descB, beta, C, ic, jc, descC);
}

inline void scalapack_pgemm (const char *transa, const char *transb, int *M, int *N, int *K, std::complex<float> *alpha, std::complex<float> *A, int *ia, int *ja, int *descA, std::complex<float> *B, int *ib, int *jb, int *descB, std::complex<float> *beta, std::complex<float> *C, int *ic, int *jc, int *descC)
{
    scalapack_pcgemm (transa, transb, M, N, K, alpha, A, ia, ja, descA, B, ib, jb, descB, beta, C, ic, jc, descC);
}

inline void scalapack_pgemm (const char *transa, const char *transb, int *M, int *N, int *K, std::complex<double> *alpha, std::complex<double> *A, int *ia, int *ja, int *descA, std::complex<double> *B, int *ib, int *jb, int *descB, std::complex<double> *beta, std::complex<double> *C, int *ic, int *jc, int *descC)
{
    scalapack_pzgemm (transa, transb, M, N, K, alpha, A, ia, ja, descA, B, ib, jb, descB, beta, C, ic, jc, descC);
}

template <typename scalar_t>
inline void scalapack_pgemm (const char *transa, const char *transb, int64_t M, int64_t N, int64_t K, scalar_t alpha, scalar_t *A, int64_t ia, int64_t ja, int *descA, scalar_t *B, int64_t ib, int64_t jb, int *descB, scalar_t beta, scalar_t *C, int64_t ic, int64_t jc, int *descC)
{
    int M_ = int64_to_int (M);
    int N_ = int64_to_int (N);
    int K_ = int64_to_int (K);
    int ia_ = int64_to_int (ia);
    int ja_ = int64_to_int (ja);
    int ib_ = int64_to_int (ib);
    int jb_ = int64_to_int (jb);
    int ic_ = int64_to_int (ic);
    int jc_ = int64_to_int (jc);
    scalapack_pgemm (transa, transb, &M_, &N_, &K_, &alpha, A, &ia_, &ja_, descA, B, &ib_, &jb_, descB, &beta, C, &ic_, &jc_, descC);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pspotrs BLAS_FORTRAN_NAME(pspotrs,PSPOTRS)
#define scalapack_pdpotrs BLAS_FORTRAN_NAME(pdpotrs,PDPOTRS)

extern "C" void scalapack_pspotrs (const char *uplo, int *n, int *nrhs, float  *a, int *ia, int *ja, int *desca, float  *b, int *ib, int *jb, int *descb, int *info);
extern "C" void scalapack_pdpotrs (const char *uplo, int *n, int *nrhs, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, int *info);

// -----------------------------------------------------------------------------

inline void scalapack_ppotrs (const char *uplo, int *n, int *nrhs, float *a, int *ia, int *ja, int *desca, float *b, int *ib, int *jb, int *descb, int *info)
{
    scalapack_pspotrs (uplo, n, nrhs, a, ia, ja, desca, b, ib, jb, descb, info);
}

inline void scalapack_ppotrs (const char *uplo, int *n, int *nrhs, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, int *info)
{
    scalapack_pdpotrs (uplo, n, nrhs, a, ia, ja, desca, b, ib, jb, descb, info);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pssymm BLAS_FORTRAN_NAME(pssymm,PSSYMM)
#define scalapack_pdsymm BLAS_FORTRAN_NAME(pdsymm,PDSYMM)
#define scalapack_pcsymm BLAS_FORTRAN_NAME(pcsymm,PCSYMM)
#define scalapack_pzsymm BLAS_FORTRAN_NAME(pzsymm,PZSYMM)

extern "C" void scalapack_pssymm (const char *side, const char *uplo, int *m, int *n, float  *alpha, float  *a, int *ia, int *ja, int *desca, float  *b, int *ib, int *jb, int *descb, float  *beta, float  *c, int *ic, int *jc, int *descc);

extern "C" void scalapack_pdsymm (const char *side, const char *uplo, int *m, int *n, double *alpha, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, double *beta, double *c, int *ic, int *jc, int *descc);

extern "C" void scalapack_pcsymm (const char *side, const char *uplo, int *m, int *n, const std::complex<float>  *alpha, const std::complex<float>  *a, int *ia, int *ja, int *desca, const std::complex<float>  *b, int *ib, int *jb, int *descb, const std::complex<float>  *beta, const std::complex<float>  *c, int *ic, int *jc, int *descc);

extern "C" void scalapack_pzsymm (const char *side, const char *uplo, int *m, int *n, const std::complex<double> *alpha, const std::complex<double> *a, int *ia, int *ja, int *desca, const std::complex<double> *b, int *ib, int *jb, int *descb, const std::complex<double> *beta, const std::complex<double> *c, int *ic, int *jc, int *descc);

// -----------------------------------------------------------------------------

inline void scalapack_psymm (const char *side, const char *uplo, int *m, int *n, float *alpha, float *a, int *ia, int *ja, int *desca, float *b, int *ib, int *jb, int *descb, float *beta, float *c, int *ic, int *jc, int *descc)
{
    scalapack_pssymm (side, uplo, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

inline void scalapack_psymm (const char *side, const char *uplo, int *m, int *n, double *alpha, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, double *beta, double *c, int *ic, int *jc, int *descc)
{
    scalapack_pdsymm (side, uplo, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

inline void scalapack_psymm (const char *side, const char *uplo, int *m, int *n, const std::complex<float>  *alpha, const std::complex<float>  *a, int *ia, int *ja, int *desca, const std::complex<float>  *b, int *ib, int *jb, int *descb, const std::complex<float>  *beta, const std::complex<float>  *c, int *ic, int *jc, int *descc)
{
    scalapack_pcsymm (side, uplo, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

inline void scalapack_psymm (const char *side, const char *uplo, int *m, int *n, const std::complex<double> *alpha, const std::complex<double> *a, int *ia, int *ja, int *desca, const std::complex<double> *b, int *ib, int *jb, int *descb, const std::complex<double> *beta, const std::complex<double> *c, int *ic, int *jc, int *descc)
{
    scalapack_pzsymm (side, uplo, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

template <typename scalar_t>
inline void scalapack_psymm (const char *side, const char *uplo, int64_t m, int64_t n, scalar_t alpha, scalar_t *a, int64_t ia, int64_t ja, int *desca, scalar_t *b, int64_t ib, int64_t jb, int *descb, scalar_t beta, scalar_t *c, int64_t ic, int64_t jc, int *descc)
{
    int m_ = int64_to_int (m);
    int n_ = int64_to_int (n);
    int ia_ = int64_to_int (ia);
    int ja_ = int64_to_int (ja);
    int ib_ = int64_to_int (ib);
    int jb_ = int64_to_int (jb);
    int ic_ = int64_to_int (ic);
    int jc_ = int64_to_int (jc);
    scalapack_psymm (side, uplo, &m_, &n_, &alpha, a, &ia_, &ja_, desca, b, &ib_, &jb_, descb, &beta, c, &ic_, &jc_, descc);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pstrmm BLAS_FORTRAN_NAME(pstrmm,PSTRMM)
#define scalapack_pdtrmm BLAS_FORTRAN_NAME(pdtrmm,PDTRMM)
#define scalapack_pctrmm BLAS_FORTRAN_NAME(pctrmm,PCTRMM)
#define scalapack_pztrmm BLAS_FORTRAN_NAME(pztrmm,PZTRMM)

extern "C" void scalapack_pstrmm (const char *side, const char *uplo, const char *transa, const char *diag, const blas_int *m, const blas_int *n, const float *alpha, const float *a, const blas_int *ia, const blas_int *ja, const blas_int *desca, float *b, const blas_int *ib, const blas_int *jb, const blas_int *descb);

extern "C" void scalapack_pdtrmm (const char *side, const char *uplo, const char *transa, const char *diag, const blas_int *m, const blas_int *n, const double *alpha, const double *a, const blas_int *ia, const blas_int *ja, const blas_int *desca, double *b, const blas_int *ib, const blas_int *jb, const blas_int *descb);

extern "C" void scalapack_pctrmm (const char *side, const char *uplo, const char *transa, const char *diag, const blas_int *m, const blas_int *n, const std::complex<float> *alpha, const std::complex<float> *a, const blas_int *ia, const blas_int *ja, const blas_int *desca, std::complex<float> *b, const blas_int *ib, const blas_int *jb, const blas_int *descb);

extern "C" void scalapack_pztrmm (const char *side, const char *uplo, const char *transa, const char *diag, const blas_int *m, const blas_int *n, const std::complex<double> *alpha, const std::complex<double> *a, const blas_int *ia, const blas_int *ja, const blas_int *desca, std::complex<double> *b, const blas_int *ib, const blas_int *jb, const blas_int *descb);

// -----------------------------------------------------------------------------

inline void scalapack_ptrmm (const char *side, const char *uplo, const char *transa, const char *diag, const blas_int *m, const blas_int *n, const float *alpha, const float *a, const blas_int *ia, const blas_int *ja, const blas_int *desca, float *b, const blas_int *ib, const blas_int *jb, const blas_int *descb)
{
    scalapack_pstrmm (side, uplo, transa, diag, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb);
}

inline void scalapack_ptrmm (const char *side, const char *uplo, const char *transa, const char *diag, const blas_int *m, const blas_int *n, const double *alpha, const double *a, const blas_int *ia, const blas_int *ja, const blas_int *desca, double *b, const blas_int *ib, const blas_int *jb, const blas_int *descb)
{
    scalapack_pdtrmm (side, uplo, transa, diag, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb);
}

inline void scalapack_ptrmm (const char *side, const char *uplo, const char *transa, const char *diag, const blas_int *m, const blas_int *n, const std::complex<float> *alpha, const std::complex<float> *a, const blas_int *ia, const blas_int *ja, const blas_int *desca, std::complex<float> *b, const blas_int *ib, const blas_int *jb, const blas_int *descb)
{
    scalapack_pctrmm (side, uplo, transa, diag, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb);
}

inline void scalapack_ptrmm (const char *side, const char *uplo, const char *transa, const char *diag, const blas_int *m, const blas_int *n, const std::complex<double> *alpha, const std::complex<double> *a, const blas_int *ia, const blas_int *ja, const blas_int *desca, std::complex<double> *b, const blas_int *ib, const blas_int *jb, const blas_int *descb)
{
    scalapack_pztrmm (side, uplo, transa, diag, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb);
}

template <typename scalar_t>
inline void scalapack_ptrmm (const char *side, const char *uplo, const char *transa, const char *diag, int64_t m, int64_t n, scalar_t alpha, scalar_t *a, int64_t ia, int64_t ja, const blas_int *desca, scalar_t *b, int64_t ib, int64_t jb, const blas_int *descb)
{
    int m_ = int64_to_int (m);
    int n_ = int64_to_int (n);
    int ia_ = int64_to_int (ia);
    int ja_ = int64_to_int (ja);
    int ib_ = int64_to_int (ib);
    int jb_ = int64_to_int (jb);
    scalapack_ptrmm (side, uplo, transa, diag, &m_, &n_, &alpha, a, &ia_, &ja_, desca, b, &ib_, &jb_, descb);
}


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pssyr2k BLAS_FORTRAN_NAME(pssyr2k,PSSYR2K)
#define scalapack_pdsyr2k BLAS_FORTRAN_NAME(pdsyr2k,PDSYR2K)
#define scalapack_pcsyr2k BLAS_FORTRAN_NAME(pcsyr2k,PCSYR2K)
#define scalapack_pzsyr2k BLAS_FORTRAN_NAME(pzsyr2k,PZSYR2K)

extern "C" void scalapack_pssyr2k (const char *uplo, const char *trans, blas_int *n, blas_int *k, float *alpha, float *a, blas_int *ia, blas_int *ja, blas_int *desca, float *b, blas_int *ib, blas_int *jb, blas_int *descb, float *beta, float *c, blas_int *ic, blas_int *jc, blas_int *descc);

extern "C" void scalapack_pdsyr2k (const char *uplo, const char *trans, blas_int *n, blas_int *k, double *alpha, double *a, blas_int *ia, blas_int *ja, blas_int *desca, double *b, blas_int *ib, blas_int *jb, blas_int *descb, double *beta, double *c, blas_int *ic, blas_int *jc, blas_int *descc);

extern "C" void scalapack_pcsyr2k (const char *uplo, const char *trans, blas_int *n, blas_int *k, std::complex<float> *alpha, std::complex<float> *a, blas_int *ia, blas_int *ja, blas_int *desca, std::complex<float> *b, blas_int *ib, blas_int *jb, blas_int *descb, std::complex<float> *beta, std::complex<float> *c, blas_int *ic, blas_int *jc, blas_int *descc);

extern "C" void scalapack_pzsyr2k (const char *uplo, const char *trans, blas_int *n, blas_int *k, std::complex<double> *alpha, std::complex<double> *a, blas_int *ia, blas_int *ja, blas_int *desca, std::complex<double> *b, blas_int *ib, blas_int *jb, blas_int *descb, std::complex<double> *beta, std::complex<double> *c, blas_int *ic, blas_int *jc, blas_int *descc);

// -----------------------------------------------------------------------------

inline void scalapack_psyr2k (const char *uplo, const char *trans, blas_int *n, blas_int *k, float *alpha, float *a, blas_int *ia, blas_int *ja, blas_int *desca, float *b, blas_int *ib, blas_int *jb, blas_int *descb, float *beta, float *c, blas_int *ic, blas_int *jc, blas_int *descc)
{
    scalapack_pssyr2k (uplo, trans, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

inline void scalapack_psyr2k (const char *uplo, const char *trans, blas_int *n, blas_int *k, double *alpha, double *a, blas_int *ia, blas_int *ja, blas_int *desca, double *b, blas_int *ib, blas_int *jb, blas_int *descb, double *beta, double *c, blas_int *ic, blas_int *jc, blas_int *descc)
{
    scalapack_pdsyr2k (uplo, trans, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

inline void scalapack_psyr2k (const char *uplo, const char *trans, blas_int *n, blas_int *k, std::complex<float> *alpha, std::complex<float> *a, blas_int *ia, blas_int *ja, blas_int *desca, std::complex<float> *b, blas_int *ib, blas_int *jb, blas_int *descb, std::complex<float> *beta, std::complex<float> *c, blas_int *ic, blas_int *jc, blas_int *descc)
{
    scalapack_pcsyr2k (uplo, trans, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

inline void scalapack_psyr2k (const char *uplo, const char *trans, blas_int *n, blas_int *k, std::complex<double> *alpha, std::complex<double> *a, blas_int *ia, blas_int *ja, blas_int *desca, std::complex<double> *b, blas_int *ib, blas_int *jb, blas_int *descb, std::complex<double> *beta, std::complex<double> *c, blas_int *ic, blas_int *jc, blas_int *descc)
{
    scalapack_pzsyr2k (uplo, trans, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

template <typename scalar_t>
inline void scalapack_psyr2k (const char *uplo, const char *trans, int64_t n, int64_t k, scalar_t alpha, scalar_t *a, int64_t ia, int64_t ja, int *desca, scalar_t *b, int64_t ib, int64_t jb, int *descb, scalar_t beta, scalar_t *c, int64_t ic, int64_t jc, int *descc)
{
    int n_ = int64_to_int (n);
    int k_ = int64_to_int (k);
    int ia_ = int64_to_int (ia);
    int ja_ = int64_to_int (ja);
    int ib_ = int64_to_int (ib);
    int jb_ = int64_to_int (jb);
    int ic_ = int64_to_int (ic);
    int jc_ = int64_to_int (jc);
    scalapack_psyr2k (uplo, trans, &n_, &k_, &alpha, a, &ia_, &ja_, desca, b, &ib_, &jb_, descb, &beta, c, &ic_, &jc_, descc);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pssyrk BLAS_FORTRAN_NAME(pssyrk,PSSYRK)
#define scalapack_pdsyrk BLAS_FORTRAN_NAME(pdsyrk,PDSYRK)
#define scalapack_pcsyrk BLAS_FORTRAN_NAME(pcsyrk,PCSYRK)
#define scalapack_pzsyrk BLAS_FORTRAN_NAME(pzsyrk,PZSYRK)

extern "C" void scalapack_pssyrk (const char *uplo, const char *trans, blas_int *n, blas_int *k, float *alpha, float *a, blas_int *ia, blas_int *ja, blas_int *desca, float *beta, float *c, blas_int *ic, blas_int *jc, blas_int *descc);

extern "C" void scalapack_pdsyrk (const char *uplo, const char *trans, blas_int *n, blas_int *k, double *alpha, double *a, blas_int *ia, blas_int *ja, blas_int *desca, double *beta, double *c, blas_int *ic, blas_int *jc, blas_int *descc);

extern "C" void scalapack_pcsyrk (const char *uplo, const char *trans, blas_int *n, blas_int *k, std::complex<float> *alpha, std::complex<float> *a, blas_int *ia, blas_int *ja, blas_int *desca, std::complex<float> *beta, std::complex<float> *c, blas_int *ic, blas_int *jc, blas_int *descc);

extern "C" void scalapack_pzsyrk (const char *uplo, const char *trans, blas_int *n, blas_int *k, std::complex<double> *alpha, std::complex<double> *a, blas_int *ia, blas_int *ja, blas_int *desca, std::complex<double> *beta, std::complex<double> *c, blas_int *ic, blas_int *jc, blas_int *descc);

// -----------------------------------------------------------------------------

inline void scalapack_psyrk (const char *uplo, const char *trans, blas_int *n, blas_int *k, float *alpha, float *a, blas_int *ia, blas_int *ja, blas_int *desca, float *beta, float *c, blas_int *ic, blas_int *jc, blas_int *descc)
{
    scalapack_pssyrk (uplo, trans, n, k, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

inline void scalapack_psyrk (const char *uplo, const char *trans, blas_int *n, blas_int *k, double *alpha, double *a, blas_int *ia, blas_int *ja, blas_int *desca, double *beta, double *c, blas_int *ic, blas_int *jc, blas_int *descc)
{
    scalapack_pdsyrk (uplo, trans, n, k, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

inline void scalapack_psyrk (const char *uplo, const char *trans, blas_int *n, blas_int *k, std::complex<float> *alpha, std::complex<float> *a, blas_int *ia, blas_int *ja, blas_int *desca, std::complex<float> *beta, std::complex<float> *c, blas_int *ic, blas_int *jc, blas_int *descc)
{
    scalapack_pcsyrk (uplo, trans, n, k, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

inline void scalapack_psyrk (const char *uplo, const char *trans, blas_int *n, blas_int *k, std::complex<double> *alpha, std::complex<double> *a, blas_int *ia, blas_int *ja, blas_int *desca, std::complex<double> *beta, std::complex<double> *c, blas_int *ic, blas_int *jc, blas_int *descc)
{
    scalapack_pzsyrk (uplo, trans, n, k, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

template <typename scalar_t>
inline void scalapack_psyrk (const char *uplo, const char *trans, int64_t n, int64_t k, scalar_t alpha, scalar_t *a, int64_t ia, int64_t ja, int *desca, scalar_t beta, scalar_t *c, int64_t ic, int64_t jc, int *descc)
{
    int n_ = int64_to_int (n);
    int k_ = int64_to_int (k);
    int ia_ = int64_to_int (ia);
    int ja_ = int64_to_int (ja);
    int ic_ = int64_to_int (ic);
    int jc_ = int64_to_int (jc);
    scalapack_psyrk (uplo, trans, &n_, &k_, &alpha, a, &ia_, &ja_, desca, &beta, c, &ic_, &jc_, descc);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pstrsm BLAS_FORTRAN_NAME(pstrsm,PSTRSM)
#define scalapack_pdtrsm BLAS_FORTRAN_NAME(pdtrsm,PDTRSM)
#define scalapack_pctrsm BLAS_FORTRAN_NAME(pctrsm,PCTRSM)
#define scalapack_pztrsm BLAS_FORTRAN_NAME(pztrsm,PZTRSM)

extern "C" void scalapack_pstrsm (const char *side, const char *uplo, const char *transa, const char *diag, const blas_int *m, const blas_int *n, const float *alpha, const float *a, const blas_int *ia, const blas_int *ja, const blas_int *desca, float *b, const blas_int *ib, const blas_int *jb, const blas_int *descb);

extern "C" void scalapack_pdtrsm (const char *side, const char *uplo, const char *transa, const char *diag, const blas_int *m, const blas_int *n, const double *alpha, const double *a, const blas_int *ia, const blas_int *ja, const blas_int *desca, double *b, const blas_int *ib, const blas_int *jb, const blas_int *descb);

extern "C" void scalapack_pctrsm (const char *side, const char *uplo, const char *transa, const char *diag, const blas_int *m, const blas_int *n, const std::complex<float> *alpha, const std::complex<float> *a, const blas_int *ia, const blas_int *ja, const blas_int *desca, std::complex<float> *b, const blas_int *ib, const blas_int *jb, const blas_int *descb);

extern "C" void scalapack_pztrsm (const char *side, const char *uplo, const char *transa, const char *diag, const blas_int *m, const blas_int *n, const std::complex<double> *alpha, const std::complex<double> *a, const blas_int *ia, const blas_int *ja, const blas_int *desca, std::complex<double> *b, const blas_int *ib, const blas_int *jb, const blas_int *descb);

// -----------------------------------------------------------------------------

inline void scalapack_ptrsm (const char *side, const char *uplo, const char *transa, const char *diag, const blas_int *m, const blas_int *n, const float *alpha, const float *a, const blas_int *ia, const blas_int *ja, const blas_int *desca, float *b, const blas_int *ib, const blas_int *jb, const blas_int *descb)
{
    scalapack_pstrsm (side, uplo, transa, diag, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb);
}

inline void scalapack_ptrsm (const char *side, const char *uplo, const char *transa, const char *diag, const blas_int *m, const blas_int *n, const double *alpha, const double *a, const blas_int *ia, const blas_int *ja, const blas_int *desca, double *b, const blas_int *ib, const blas_int *jb, const blas_int *descb)
{
    scalapack_pdtrsm (side, uplo, transa, diag, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb);
}

inline void scalapack_ptrsm (const char *side, const char *uplo, const char *transa, const char *diag, const blas_int *m, const blas_int *n, const std::complex<float> *alpha, const std::complex<float> *a, const blas_int *ia, const blas_int *ja, const blas_int *desca, std::complex<float> *b, const blas_int *ib, const blas_int *jb, const blas_int *descb)
{
    scalapack_pctrsm (side, uplo, transa, diag, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb);
}

inline void scalapack_ptrsm (const char *side, const char *uplo, const char *transa, const char *diag, const blas_int *m, const blas_int *n, const std::complex<double> *alpha, const std::complex<double> *a, const blas_int *ia, const blas_int *ja, const blas_int *desca, std::complex<double> *b, const blas_int *ib, const blas_int *jb, const blas_int *descb)
{
    scalapack_pztrsm (side, uplo, transa, diag, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb);
}

template <typename scalar_t>
inline void scalapack_ptrsm (const char *side, const char *uplo, const char *transa, const char *diag, int64_t m, int64_t n, scalar_t alpha, scalar_t *a, int64_t ia, int64_t ja, const blas_int *desca, scalar_t *b, int64_t ib, int64_t jb, const blas_int *descb)
{
    int m_ = int64_to_int (m);
    int n_ = int64_to_int (n);
    int ia_ = int64_to_int (ia);
    int ja_ = int64_to_int (ja);
    int ib_ = int64_to_int (ib);
    int jb_ = int64_to_int (jb);
    scalapack_ptrsm (side, uplo, transa, diag, &m_, &n_, &alpha, a, &ia_, &ja_, desca, b, &ib_, &jb_, descb);
}


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pslantr BLAS_FORTRAN_NAME( pslantr, PSLANTR )
#define scalapack_pdlantr BLAS_FORTRAN_NAME( pdlantr, PDLANTR )
#define scalapack_pclantr BLAS_FORTRAN_NAME( pclantr, PCLANTR )
#define scalapack_pzlantr BLAS_FORTRAN_NAME( pzlantr, PZLANTR )

extern "C" blas_float_return scalapack_pslantr (const char *norm, const char *uplo, const char *diag, blas_int *m, blas_int *n, float *A, blas_int *ia, blas_int *ja, blas_int *descA, float *work);

extern "C" double scalapack_pdlantr (const char *norm, const char *uplo, const char *diag, blas_int *m, blas_int *n, double *A, blas_int *ia, blas_int *ja, blas_int *descA, double *work);

extern "C" blas_float_return scalapack_pclantr (const char *norm, const char *uplo, const char *diag, blas_int *m, blas_int *n, std::complex<float> *A, blas_int *ia, blas_int *ja, blas_int *descA, float *work);

extern "C" double scalapack_pzlantr (const char *norm, const char *uplo, const char *diag, blas_int *m, blas_int *n, std::complex<double> *A, blas_int *ia, blas_int *ja, blas_int *descA, double *work);

// -----------------------------------------------------------------------------

inline blas_float_return scalapack_plantr (const char *norm, const char *uplo, const char *diag, blas_int *m, blas_int *n, float *A, blas_int *ia, blas_int *ja, blas_int *descA, float *work)
{
    return scalapack_pslantr (norm, uplo, diag, m, n, A, ia, ja, descA, work);
}
inline double scalapack_plantr (const char *norm, const char *uplo, const char *diag, blas_int *m, blas_int *n, double *A, blas_int *ia, blas_int *ja, blas_int *descA, double *work)
{
    return scalapack_pdlantr (norm, uplo, diag, m, n, A, ia, ja, descA, work);
}
inline blas_float_return scalapack_plantr (const char *norm, const char *uplo, const char *diag, blas_int *m, blas_int *n, std::complex<float> *A, blas_int *ia, blas_int *ja, blas_int *descA, float *work)
{
    return scalapack_pclantr (norm, uplo, diag, m, n, A, ia, ja, descA, work);
}
inline double scalapack_plantr (const char *norm, const char *uplo, const char *diag, blas_int *m, blas_int *n, std::complex<double> *A, blas_int *ia, blas_int *ja, blas_int *descA, double *work)
{
    return scalapack_pzlantr (norm, uplo, diag, m, n, A, ia, ja, descA, work);
}

template <typename scalar_t>
inline blas::real_type<scalar_t> scalapack_plantr (const char *norm, const char *uplo, const char *diag, int64_t m, int64_t n, scalar_t *A, int64_t ia, int64_t ja, blas_int *descA, blas::real_type<scalar_t> *work)
{
    int m_ = int64_to_int (m);
    int n_ = int64_to_int (n);
    int ia_ = int64_to_int (ia);
    int ja_ = int64_to_int (ja);
    return scalapack_plantr (norm, uplo, diag, &m_, &n_, A, &ia_, &ja_, descA, work);
}


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pchemm BLAS_FORTRAN_NAME(pchemm,PCHEMM)
#define scalapack_pzhemm BLAS_FORTRAN_NAME(pzhemm,PZHEMM)

extern "C" void scalapack_pchemm (const char *side, const char *uplo, int *m, int *n, const std::complex<float>  *alpha, const std::complex<float>  *a, int *ia, int *ja, int *desca, const std::complex<float>  *b, int *ib, int *jb, int *descb, const std::complex<float>  *beta, const std::complex<float>  *c, int *ic, int *jc, int *descc);

extern "C" void scalapack_pzhemm (const char *side, const char *uplo, int *m, int *n, const std::complex<double> *alpha, const std::complex<double> *a, int *ia, int *ja, int *desca, const std::complex<double> *b, int *ib, int *jb, int *descb, const std::complex<double> *beta, const std::complex<double> *c, int *ic, int *jc, int *descc);

// -----------------------------------------------------------------------------

inline void scalapack_phemm (const char *side, const char *uplo, int *m, int *n, float *alpha, float *a, int *ia, int *ja, int *desca, float *b, int *ib, int *jb, int *descb, float *beta, float *c, int *ic, int *jc, int *descc)
{
    scalapack_pssymm (side, uplo, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

inline void scalapack_phemm (const char *side, const char *uplo, int *m, int *n, double *alpha, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, double *beta, double *c, int *ic, int *jc, int *descc)
{
    scalapack_pdsymm (side, uplo, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

inline void scalapack_phemm (const char *side, const char *uplo, int *m, int *n, const std::complex<float>  *alpha, const std::complex<float>  *a, int *ia, int *ja, int *desca, const std::complex<float>  *b, int *ib, int *jb, int *descb, const std::complex<float>  *beta, const std::complex<float>  *c, int *ic, int *jc, int *descc)
{
    scalapack_pchemm (side, uplo, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

inline void scalapack_phemm (const char *side, const char *uplo, int *m, int *n, const std::complex<double> *alpha, const std::complex<double> *a, int *ia, int *ja, int *desca, const std::complex<double> *b, int *ib, int *jb, int *descb, const std::complex<double> *beta, const std::complex<double> *c, int *ic, int *jc, int *descc)
{
    scalapack_pzhemm (side, uplo, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

template <typename scalar_t>
inline void scalapack_phemm (const char *side, const char *uplo, int64_t m, int64_t n, scalar_t alpha, scalar_t *a, int64_t ia, int64_t ja, int *desca, scalar_t *b, int64_t ib, int64_t jb, int *descb, scalar_t beta, scalar_t *c, int64_t ic, int64_t jc, int *descc)
{
    int m_ = int64_to_int (m);
    int n_ = int64_to_int (n);
    int ia_ = int64_to_int (ia);
    int ja_ = int64_to_int (ja);
    int ib_ = int64_to_int (ib);
    int jb_ = int64_to_int (jb);
    int ic_ = int64_to_int (ic);
    int jc_ = int64_to_int (jc);
    scalapack_phemm (side, uplo, &m_, &n_, &alpha, a, &ia_, &ja_, desca, b, &ib_, &jb_, descb, &beta, c, &ic_, &jc_, descc);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pcher2k BLAS_FORTRAN_NAME(pcher2k,PCHER2K)
#define scalapack_pzher2k BLAS_FORTRAN_NAME(pzher2k,PZHER2K)

extern "C" void scalapack_pcher2k (const char *uplo, const char *trans, blas_int *n, blas_int *k, std::complex<float> *alpha, std::complex<float> *a, blas_int *ia, blas_int *ja, blas_int *desca, std::complex<float> *b, blas_int *ib, blas_int *jb, blas_int *descb, float *beta, std::complex<float> *c, blas_int *ic, blas_int *jc, blas_int *descc);

extern "C" void scalapack_pzher2k (const char *uplo, const char *trans, blas_int *n, blas_int *k, std::complex<double> *alpha, std::complex<double> *a, blas_int *ia, blas_int *ja, blas_int *desca, std::complex<double> *b, blas_int *ib, blas_int *jb, blas_int *descb, double *beta, std::complex<double> *c, blas_int *ic, blas_int *jc, blas_int *descc);

// -----------------------------------------------------------------------------

inline void scalapack_pher2k (const char *uplo, const char *trans, blas_int *n, blas_int *k, float *alpha, float *a, blas_int *ia, blas_int *ja, blas_int *desca, float *b, blas_int *ib, blas_int *jb, blas_int *descb, float *beta, float *c, blas_int *ic, blas_int *jc, blas_int *descc)
{
    scalapack_pssyr2k (uplo, trans, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

inline void scalapack_pher2k (const char *uplo, const char *trans, blas_int *n, blas_int *k, double *alpha, double *a, blas_int *ia, blas_int *ja, blas_int *desca, double *b, blas_int *ib, blas_int *jb, blas_int *descb, double *beta, double *c, blas_int *ic, blas_int *jc, blas_int *descc)
{
    scalapack_pdsyr2k (uplo, trans, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

inline void scalapack_pher2k (const char *uplo, const char *trans, blas_int *n, blas_int *k, std::complex<float> *alpha, std::complex<float> *a, blas_int *ia, blas_int *ja, blas_int *desca, std::complex<float> *b, blas_int *ib, blas_int *jb, blas_int *descb, float *beta, std::complex<float> *c, blas_int *ic, blas_int *jc, blas_int *descc)
{
    scalapack_pcher2k (uplo, trans, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

inline void scalapack_pher2k (const char *uplo, const char *trans, blas_int *n, blas_int *k, std::complex<double> *alpha, std::complex<double> *a, blas_int *ia, blas_int *ja, blas_int *desca, std::complex<double> *b, blas_int *ib, blas_int *jb, blas_int *descb, double *beta, std::complex<double> *c, blas_int *ic, blas_int *jc, blas_int *descc)
{
    scalapack_pzher2k (uplo, trans, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

template <typename scalar_t>
inline void scalapack_pher2k (const char *uplo, const char *trans, int64_t n, int64_t k, scalar_t alpha, scalar_t *a, int64_t ia, int64_t ja, int *desca, scalar_t *b, int64_t ib, int64_t jb, int *descb, blas::real_type<scalar_t> beta, scalar_t *c, int64_t ic, int64_t jc, int *descc)
{
    int n_ = int64_to_int (n);
    int k_ = int64_to_int (k);
    int ia_ = int64_to_int (ia);
    int ja_ = int64_to_int (ja);
    int ib_ = int64_to_int (ib);
    int jb_ = int64_to_int (jb);
    int ic_ = int64_to_int (ic);
    int jc_ = int64_to_int (jc);
    scalapack_pher2k (uplo, trans, &n_, &k_, &alpha, a, &ia_, &ja_, desca, b, &ib_, &jb_, descb, &beta, c, &ic_, &jc_, descc);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pcherk BLAS_FORTRAN_NAME(pcherk,PCHERK)
#define scalapack_pzherk BLAS_FORTRAN_NAME(pzherk,PZHERK)

extern "C" void scalapack_pcherk (const char *uplo, const char *trans, blas_int *n, blas_int *k, float *alpha, std::complex<float> *a, blas_int *ia, blas_int *ja, blas_int *desca, float *beta, std::complex<float> *c, blas_int *ic, blas_int *jc, blas_int *descc);

extern "C" void scalapack_pzherk (const char *uplo, const char *trans, blas_int *n, blas_int *k, double *alpha, std::complex<double> *a, blas_int *ia, blas_int *ja, blas_int *desca, double *beta, std::complex<double> *c, blas_int *ic, blas_int *jc, blas_int *descc);

// -----------------------------------------------------------------------------

inline void scalapack_pherk (const char *uplo, const char *trans, blas_int *n, blas_int *k, float *alpha, float *a, blas_int *ia, blas_int *ja, blas_int *desca, float *beta, float *c, blas_int *ic, blas_int *jc, blas_int *descc)
{
    scalapack_pssyrk (uplo, trans, n, k, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

inline void scalapack_pherk (const char *uplo, const char *trans, blas_int *n, blas_int *k, double *alpha, double *a, blas_int *ia, blas_int *ja, blas_int *desca, double *beta, double *c, blas_int *ic, blas_int *jc, blas_int *descc)
{
    scalapack_pdsyrk (uplo, trans, n, k, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

inline void scalapack_pherk (const char *uplo, const char *trans, blas_int *n, blas_int *k, float *alpha, std::complex<float> *a, blas_int *ia, blas_int *ja, blas_int *desca, float *beta, std::complex<float> *c, blas_int *ic, blas_int *jc, blas_int *descc)
{
    scalapack_pcherk (uplo, trans, n, k, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

inline void scalapack_pherk (const char *uplo, const char *trans, blas_int *n, blas_int *k, double *alpha, std::complex<double> *a, blas_int *ia, blas_int *ja, blas_int *desca, double *beta, std::complex<double> *c, blas_int *ic, blas_int *jc, blas_int *descc)
{
    scalapack_pzherk (uplo, trans, n, k, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

template <typename scalar_t>
inline void scalapack_pherk (const char *uplo, const char *trans, int64_t n, int64_t k, blas::real_type<scalar_t> alpha, scalar_t *a, int64_t ia, int64_t ja, int *desca, blas::real_type<scalar_t> beta, scalar_t *c, int64_t ic, int64_t jc, int *descc)
{
    int n_ = int64_to_int (n);
    int k_ = int64_to_int (k);
    int ia_ = int64_to_int (ia);
    int ja_ = int64_to_int (ja);
    int ic_ = int64_to_int (ic);
    int jc_ = int64_to_int (jc);
    scalapack_pherk (uplo, trans, &n_, &k_, &alpha, a, &ia_, &ja_, desca, &beta, c, &ic_, &jc_, descc);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
#endif // ICL_SLATE_SCALAPACK_WRAPPERS_HH
