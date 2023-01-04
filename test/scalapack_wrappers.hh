// Copyright (c) 2009-2022, University of Tennessee. All rights reserved.
// Copyright (c) 2010,      University of Denver, Colorado.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_SCALAPACK_WRAPPERS_HH
#define SLATE_SCALAPACK_WRAPPERS_HH

// get BLAS_FORTRAN_NAME and blas_int
#include "blas/fortran.h"

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
// Required CBLACS calls
// -----------------------------------------------------------------------------

extern "C" void Cblacs_pinfo(int* mypnum, int* nprocs);
extern "C" void Cblacs_get(int context, int request, int* value);
extern "C" int  Cblacs_gridinit(int* context, const char* order, int np_row, int np_col);
extern "C" void Cblacs_gridinfo(int context, int*  np_row, int* np_col, int*  my_row, int*  my_col);
extern "C" void Cblacs_gridexit(int context);
extern "C" void Cblacs_exit(int error_code);
extern "C" void Cblacs_abort(int context, int error_code);

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Simple ScaLAPACK routine wrappers
// -----------------------------------------------------------------------------

#define scalapack_descinit BLAS_FORTRAN_NAME(descinit,DESCINIT)
extern "C" void scalapack_descinit(int* desc, int* m, int* n, int* mb, int* nb, int* irsrc, int* icsrc, int* ictxt, int* lld, int* info);
inline void scalapack_descinit(int* desc, int64_t m, int64_t n, int64_t mb, int64_t nb, int irsrc, int icsrc, int ictxt, int64_t lld, int* info)
{
    int m_ = int64_to_int(m);
    int n_ = int64_to_int(n);
    int mb_ = int64_to_int(mb);
    int nb_ = int64_to_int(nb);
    int lld_ = std::max(1, int64_to_int(lld));
    scalapack_descinit(desc, &m_, &n_, &mb_, &nb_, &irsrc, &icsrc, &ictxt, &lld_, info);
}

#define scalapack_numroc BLAS_FORTRAN_NAME(numroc,NUMROC)
extern "C" int scalapack_numroc(int* n, int* nb, int* iproc, int* isrcproc, int* nprocs);
inline int64_t scalapack_numroc(int64_t n, int64_t nb, int iproc, int isrcproc, int nprocs)
{
    int n_ = int64_to_int(n);
    int nb_ = int64_to_int(nb);
    int nroc_ = scalapack_numroc(&n_, &nb_, &iproc, &isrcproc, &nprocs);
    int64_t nroc = (int64_t)nroc_;
    return nroc;
}

#define scalapack_ilcm BLAS_FORTRAN_NAME(ilcm,ILCM)
extern "C" int scalapack_ilcm(int* a, int* b);

#define scalapack_indxg2p BLAS_FORTRAN_NAME(indxg2p,INDXG2P)
extern "C" int scalapack_indxg2p(int* indxglob, int* nb, int* iproc, int* isrcproc, int* nprocs);

#define scalapack_indxg2l BLAS_FORTRAN_NAME(indxg2l,INDXG2L)
extern "C" int scalapack_indxg2l(int* indxglob, int* nb, int* iproc, int* isrcproc, int* nprocs);

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Type generic ScaLAPACK wrappers
// -----------------------------------------------------------------------------

#define scalapack_pslange BLAS_FORTRAN_NAME( pslange, PSLANGE )
#define scalapack_pdlange BLAS_FORTRAN_NAME( pdlange, PDLANGE )
#define scalapack_pclange BLAS_FORTRAN_NAME( pclange, PCLANGE )
#define scalapack_pzlange BLAS_FORTRAN_NAME( pzlange, PZLANGE )

extern "C" float scalapack_pslange(const char* norm, blas_int* m, blas_int* n, float* A, blas_int* ia, blas_int* ja, blas_int* descA, float* work);

extern "C" double scalapack_pdlange(const char* norm, blas_int* m, blas_int* n, double* A, blas_int* ia, blas_int* ja, blas_int* descA, double* work);

extern "C" float scalapack_pclange(const char* norm, blas_int* m, blas_int* n, std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA, float* work);

extern "C" double scalapack_pzlange(const char* norm, blas_int* m, blas_int* n, std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA, double* work);

// -----------------------------------------------------------------------------

inline float scalapack_plange(const char* norm, blas_int* m, blas_int* n, float* A, blas_int* ia, blas_int* ja, blas_int* descA, float* work)
{
    return scalapack_pslange(norm, m, n, A, ia, ja, descA, work);
}
inline double scalapack_plange(const char* norm, blas_int* m, blas_int* n, double* A, blas_int* ia, blas_int* ja, blas_int* descA, double* work)
{
    return scalapack_pdlange(norm, m, n, A, ia, ja, descA, work);
}
inline float scalapack_plange(const char* norm, blas_int* m, blas_int* n, std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA, float* work)
{
    return scalapack_pclange(norm, m, n, A, ia, ja, descA, work);
}
inline double scalapack_plange(const char* norm, blas_int* m, blas_int* n, std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA, double* work)
{
    return scalapack_pzlange(norm, m, n, A, ia, ja, descA, work);
}

template <typename scalar_t>
inline blas::real_type<scalar_t> scalapack_plange(const char* norm, int64_t m, int64_t n, scalar_t* A, int64_t ia, int64_t ja, blas_int* descA, blas::real_type<scalar_t>* work)
{
    int m_ = int64_to_int(m);
    int n_ = int64_to_int(n);
    int ia_ = int64_to_int(ia);
    int ja_ = int64_to_int(ja);
    return scalapack_plange(norm, &m_, &n_, A, &ia_, &ja_, descA, work);
}

//------------------------------------------------------------------------------
// Fortran prototypes
#define scalapack_psgeadd BLAS_FORTRAN_NAME( psgeadd, PSGEADD )
#define scalapack_pdgeadd BLAS_FORTRAN_NAME( pdgeadd, PDGEADD )
#define scalapack_pcgeadd BLAS_FORTRAN_NAME( pcgeadd, PCGEADD )
#define scalapack_pzgeadd BLAS_FORTRAN_NAME( pzgeadd, PZGEADD )

extern "C" float scalapack_psgeadd(const char* transa, blas_int* m, blas_int* n, float* alpha, float* A, blas_int* ia, blas_int* ja, blas_int* descA, float* beta, float* B, blas_int* ib, blas_int* jb, blas_int* descB, blas_int* info);

extern "C" double scalapack_pdgeadd(const char* transa, blas_int* m, blas_int* n, double* alpha, double* A, blas_int* ia, blas_int* ja, blas_int* descA, double* beta, double* B, blas_int* ib, blas_int* jb, blas_int* descB, blas_int* info);

extern "C" float scalapack_pcgeadd(const char* transa, blas_int* m, blas_int* n, std::complex<float>* alpha, std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA, std::complex<float>* beta, std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB, blas_int* info);

extern "C" double scalapack_pzgeadd(const char* transa, blas_int* m, blas_int* n, std::complex<double>* alpha, std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA, std::complex<double>* beta, std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB, blas_int* info);

//----------------------------------------
// Intermediate overloaded wrappers
inline void scalapack_pgeadd(const char* transa, blas_int* m, blas_int* n, float* alpha, float* A, blas_int* ia, blas_int* ja, blas_int* descA, float* beta, float* B, blas_int* ib, blas_int* jb, blas_int* descB, blas_int* info)
{
   scalapack_psgeadd(transa, m, n, alpha, A, ia, ja, descA, beta, B, ib, jb, descB, info);
}
inline void scalapack_pgeadd(const char* transa, blas_int* m, blas_int* n, double* alpha, double* A, blas_int* ia, blas_int* ja, blas_int* descA, double* beta, double* B, blas_int* ib, blas_int* jb, blas_int* descB, blas_int* info)
{
   scalapack_pdgeadd(transa, m, n, alpha, A, ia, ja, descA, beta, B, ib, jb, descB, info);
}
inline void scalapack_pgeadd(const char* transa, blas_int* m, blas_int* n, std::complex<float>* alpha, std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA, std::complex<float>* beta, std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB, blas_int* info)
{
    scalapack_pcgeadd(transa, m, n, alpha, A, ia, ja, descA, beta, B, ib, jb, descB, info);
}
inline void scalapack_pgeadd(const char* transa, blas_int* m, blas_int* n, std::complex<double>* alpha, std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA, std::complex<double>* beta, std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB, blas_int* info)
{
    scalapack_pzgeadd(transa, m, n, alpha, A, ia, ja, descA, beta, B, ib, jb, descB, info);
}

//----------------------------------------
// Templated wrapper
template <typename scalar_t>
inline void scalapack_pgeadd(const char* transa, int64_t m, int64_t n, scalar_t alpha, scalar_t* A, int64_t ia, int64_t ja, blas_int* descA, scalar_t beta, scalar_t* B, int64_t ib, int64_t jb, blas_int* descB, blas_int* info)
{
    int m_ = int64_to_int(m);
    int n_ = int64_to_int(n);
    int ia_ = int64_to_int(ia);
    int ja_ = int64_to_int(ja);
    int ib_ = int64_to_int(ib);
    int jb_ = int64_to_int(jb);
    scalapack_pgeadd(transa, &m_, &n_, &alpha, A, &ia_, &ja_, descA, &beta, B, &ib_, &jb_, descB, info);
}

//------------------------------------------------------------------------------
// Fortran prototypes
#define scalapack_pstradd BLAS_FORTRAN_NAME( pstradd, PSGEADD )
#define scalapack_pdtradd BLAS_FORTRAN_NAME( pdtradd, PDGEADD )
#define scalapack_pctradd BLAS_FORTRAN_NAME( pctradd, PCGEADD )
#define scalapack_pztradd BLAS_FORTRAN_NAME( pztradd, PZGEADD )

extern "C" float scalapack_pstradd(
    const char* uplo, const char* transA, blas_int* m, blas_int* n,
    float* alpha, float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* beta,  float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

extern "C" double scalapack_pdtradd(
    const char* uplo, const char* transA, blas_int* m, blas_int* n,
    double* alpha, double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* beta,  double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

extern "C" float scalapack_pctradd(
    const char* uplo, const char* transA, blas_int* m, blas_int* n,
    std::complex<float>* alpha,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* beta,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

extern "C" double scalapack_pztradd(
    const char* uplo, const char* transA, blas_int* m, blas_int* n,
    std::complex<double>* alpha,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* beta,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

//----------------------------------------
// Intermediate overloaded wrappers
inline void scalapack_ptradd(
    const char* uplo, const char* transA, blas_int* m, blas_int* n,
    float* alpha, float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* beta,  float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info)
{
    scalapack_pstradd(
        uplo, transA, m, n,
        alpha, A, ia, ja, descA,
        beta,  B, ib, jb, descB, info);
}

inline void scalapack_ptradd(
    const char* uplo, const char* transA, blas_int* m, blas_int* n,
    double* alpha, double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* beta,  double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info)
{
    scalapack_pdtradd(
        uplo, transA, m, n,
        alpha, A, ia, ja, descA,
        beta,  B, ib, jb, descB, info );
}

inline void scalapack_ptradd(
    const char* uplo, const char* transA, blas_int* m, blas_int* n,
    std::complex<float>* alpha,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* beta,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info)
{
    scalapack_pctradd(
        uplo, transA, m, n,
        alpha, A, ia, ja, descA,
        beta,  B, ib, jb, descB, info);
}

inline void scalapack_ptradd(
    const char* uplo, const char* transA, blas_int* m, blas_int* n,
    std::complex<double>* alpha,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* beta,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info)
{
    scalapack_pztradd(
        uplo, transA, m, n,
        alpha, A, ia, ja, descA,
        beta,  B, ib, jb, descB, info);
}

//----------------------------------------
// Templated wrapper
template <typename scalar_t>
inline void scalapack_ptradd(
    const char* uplo, const char* transA, int64_t m, int64_t n,
    scalar_t alpha, scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    scalar_t beta,  scalar_t* B, int64_t ib, int64_t jb, blas_int* descB,
    blas_int* info)
{
    int m_  = int64_to_int( m  );
    int n_  = int64_to_int( n  );
    int ia_ = int64_to_int( ia );
    int ja_ = int64_to_int( ja );
    int ib_ = int64_to_int( ib );
    int jb_ = int64_to_int( jb );
    scalapack_ptradd(
        uplo, transA, &m_, &n_,
        &alpha, A, &ia_, &ja_, descA,
        &beta,  B, &ib_, &jb_, descB, info);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pslascl BLAS_FORTRAN_NAME( pslascl, PSLASCL )
#define scalapack_pdlascl BLAS_FORTRAN_NAME( pdlascl, PDLASCL )
#define scalapack_pclascl BLAS_FORTRAN_NAME( pclascl, PCLASCL )
#define scalapack_pzlascl BLAS_FORTRAN_NAME( pzlascl, PZLASCL )

extern "C" float scalapack_pslascl(const char* uplo, float* numer,  float* denom,  blas_int* m, blas_int* n, float* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* info);

extern "C" double scalapack_pdlascl(const char* uplo, double* numer,  double* denom, blas_int* m, blas_int* n, double* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* info);

extern "C" float scalapack_pclascl(const char* uplo, float* numer,  float* denom, blas_int* m, blas_int* n, std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* info);

extern "C" double scalapack_pzlascl(const char* uplo, double* numer, double* denom, blas_int* m, blas_int* n, std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* info);


inline void scalapack_plascl(const char* uplo, float* numer,  float* denom,  blas_int* m, blas_int* n, float* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* info)
{
   scalapack_pslascl(uplo, denom, numer, m, n, A, ia, ja, descA, info);
}
inline void scalapack_plascl(const char* uplo, double* numer,  double* denom, blas_int* m, blas_int* n, double* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* info)
{
   scalapack_pdlascl(uplo, denom, numer, m, n, A, ia, ja, descA, info);
}
inline void scalapack_plascl(const char* uplo, float* numer,  float* denom, blas_int* m, blas_int* n, std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* info)
{
    scalapack_pclascl(uplo, denom, numer, m, n, A, ia, ja, descA, info);
}
inline void scalapack_plascl(const char* uplo, double* numer, double* denom, blas_int* m, blas_int* n, std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* info)
{
    scalapack_pzlascl(uplo, denom, numer, m, n, A, ia, ja, descA, info);
}

template <typename scalar_t>
inline void scalapack_plascl(const char* uplo, blas::real_type<scalar_t> numer, blas::real_type<scalar_t> denom, int64_t m, int64_t n, scalar_t* A, int64_t ia, int64_t ja, blas_int* descA, blas_int* info)
{
    int m_ = int64_to_int(m);
    int n_ = int64_to_int(n);
    int ia_ = int64_to_int(ia);
    int ja_ = int64_to_int(ja);
    scalapack_plascl(uplo, &numer, &denom, &m_, &n_, A, &ia_, &ja_, descA, info);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pspotrf BLAS_FORTRAN_NAME( pspotrf, PSPOTRF )
#define scalapack_pdpotrf BLAS_FORTRAN_NAME( pdpotrf, PDPOTRF )
#define scalapack_pcpotrf BLAS_FORTRAN_NAME( pcpotrf, PCPOTRF )
#define scalapack_pzpotrf BLAS_FORTRAN_NAME( pzpotrf, PZPOTRF )

extern "C" void scalapack_pspotrf(const char* uplo, blas_int* n, float* a, blas_int* ia, blas_int* ja, blas_int* desca, blas_int* info);

extern "C" void scalapack_pdpotrf(const char* uplo, blas_int* n, double* a, blas_int* ia, blas_int* ja, blas_int* desca, blas_int* info);

extern "C" void scalapack_pcpotrf(const char* uplo, blas_int* n, std::complex<float>* a, blas_int* ia, blas_int* ja, blas_int* desca, blas_int* info);

extern "C" void scalapack_pzpotrf(const char* uplo, blas_int* n, std::complex<double>* a, blas_int* ia, blas_int* ja, blas_int* desca, blas_int* info);

// -----------------------------------------------------------------------------

inline void scalapack_ppotrf(const char* uplo, blas_int* n, float* a, blas_int* ia, blas_int* ja, blas_int* desca, blas_int* info)
{
    scalapack_pspotrf(uplo, n, a, ia, ja, desca, info);
}

inline void scalapack_ppotrf(const char* uplo, blas_int* n, double* a, blas_int* ia, blas_int* ja, blas_int* desca, blas_int* info)
{
    scalapack_pdpotrf(uplo, n, a, ia, ja, desca, info);
}

inline void scalapack_ppotrf(const char* uplo, blas_int* n, std::complex<float>* a, blas_int* ia, blas_int* ja, blas_int* desca, blas_int* info)
{
    scalapack_pcpotrf(uplo, n, a, ia, ja, desca, info);
}

inline void scalapack_ppotrf(const char* uplo, blas_int* n, std::complex<double>* a, blas_int* ia, blas_int* ja, blas_int* desca, blas_int* info)
{
    scalapack_pzpotrf(uplo, n, a, ia, ja, desca, info);
}

template <typename scalar_t>
inline void scalapack_ppotrf(const char* uplo, int64_t n, scalar_t* a, int64_t ia, int64_t ja, int* desca, blas_int* info)
{
    int n_ = int64_to_int(n);
    int ia_ = int64_to_int(ia);
    int ja_ = int64_to_int(ja);
    scalapack_ppotrf(uplo, &n_, a, &ia_, &ja_, desca, info);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pspotrs BLAS_FORTRAN_NAME(pspotrs,PSPOTRS)
#define scalapack_pdpotrs BLAS_FORTRAN_NAME(pdpotrs,PDPOTRS)
#define scalapack_pcpotrs BLAS_FORTRAN_NAME(pcpotrs,PCPOTRS)
#define scalapack_pzpotrs BLAS_FORTRAN_NAME(pzpotrs,PZPOTRS)

extern "C" void scalapack_pspotrs(const char* uplo, int* n, int* nrhs, float*  a, int* ia, int* ja, int* desca, float*  b, int* ib, int* jb, int* descb, int* info);
extern "C" void scalapack_pdpotrs(const char* uplo, int* n, int* nrhs, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb, int* info);
extern "C" void scalapack_pcpotrs(const char* uplo, int* n, int* nrhs, std::complex<float>*  a, int* ia, int* ja, int* desca, std::complex<float>*  b, int* ib, int* jb, int* descb, int* info);
extern "C" void scalapack_pzpotrs(const char* uplo, int* n, int* nrhs, std::complex<double>* a, int* ia, int* ja, int* desca, std::complex<double>* b, int* ib, int* jb, int* descb, int* info);

// -----------------------------------------------------------------------------

inline void scalapack_ppotrs(const char* uplo, int* n, int* nrhs, float* a, int* ia, int* ja, int* desca, float* b, int* ib, int* jb, int* descb, int* info)
{
    scalapack_pspotrs(uplo, n, nrhs, a, ia, ja, desca, b, ib, jb, descb, info);
}

inline void scalapack_ppotrs(const char* uplo, int* n, int* nrhs, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb, int* info)
{
    scalapack_pdpotrs(uplo, n, nrhs, a, ia, ja, desca, b, ib, jb, descb, info);
}

inline void scalapack_ppotrs(const char* uplo, int* n, int* nrhs, std::complex<float>* a, int* ia, int* ja, int* desca, std::complex<float>* b, int* ib, int* jb, int* descb, int* info)
{
    scalapack_pcpotrs(uplo, n, nrhs, a, ia, ja, desca, b, ib, jb, descb, info);
}

inline void scalapack_ppotrs(const char* uplo, int* n, int* nrhs, std::complex<double>* a, int* ia, int* ja, int* desca, std::complex<double>* b, int* ib, int* jb, int* descb, int* info)
{
    scalapack_pzpotrs(uplo, n, nrhs, a, ia, ja, desca, b, ib, jb, descb, info);
}

template <typename scalar_t>
inline void scalapack_ppotrs(const char* uplo, int64_t n, int64_t nrhs, scalar_t* a, int64_t ia, int64_t ja, int* desca, scalar_t* b, int64_t ib, int64_t jb, int* descb, blas_int* info)
{
    int n_ = int64_to_int(n);
    int nrhs_ = int64_to_int(nrhs);
    int ia_ = int64_to_int(ia);
    int ja_ = int64_to_int(ja);
    int ib_ = int64_to_int(ib);
    int jb_ = int64_to_int(jb);
    scalapack_ppotrs(uplo, &n_, &nrhs_, a, &ia_, &ja_, desca, b, &ib_, &jb_, descb, info);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_psposv BLAS_FORTRAN_NAME(psposv,PSPOSV)
#define scalapack_pdposv BLAS_FORTRAN_NAME(pdposv,PDPOSV)
#define scalapack_pcposv BLAS_FORTRAN_NAME(pcposv,PCPOSV)
#define scalapack_pzposv BLAS_FORTRAN_NAME(pzposv,PZPOSV)

extern "C" void scalapack_psposv(const char* uplo, int* n, int* nrhs, float*  a, int* ia, int* ja, int* desca, float*  b, int* ib, int* jb, int* descb, int* info);
extern "C" void scalapack_pdposv(const char* uplo, int* n, int* nrhs, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb, int* info);
extern "C" void scalapack_pcposv(const char* uplo, int* n, int* nrhs, std::complex<float>*  a, int* ia, int* ja, int* desca, std::complex<float>*  b, int* ib, int* jb, int* descb, int* info);
extern "C" void scalapack_pzposv(const char* uplo, int* n, int* nrhs, std::complex<double>* a, int* ia, int* ja, int* desca, std::complex<double>* b, int* ib, int* jb, int* descb, int* info);

// -----------------------------------------------------------------------------

inline void scalapack_pposv(const char* uplo, int* n, int* nrhs, float* a, int* ia, int* ja, int* desca, float* b, int* ib, int* jb, int* descb, int* info)
{
    scalapack_psposv(uplo, n, nrhs, a, ia, ja, desca, b, ib, jb, descb, info);
}

inline void scalapack_pposv(const char* uplo, int* n, int* nrhs, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb, int* info)
{
    scalapack_pdposv(uplo, n, nrhs, a, ia, ja, desca, b, ib, jb, descb, info);
}

inline void scalapack_pposv(const char* uplo, int* n, int* nrhs, std::complex<float>* a, int* ia, int* ja, int* desca, std::complex<float>* b, int* ib, int* jb, int* descb, int* info)
{
    scalapack_pcposv(uplo, n, nrhs, a, ia, ja, desca, b, ib, jb, descb, info);
}

inline void scalapack_pposv(const char* uplo, int* n, int* nrhs, std::complex<double>* a, int* ia, int* ja, int* desca, std::complex<double>* b, int* ib, int* jb, int* descb, int* info)
{
    scalapack_pzposv(uplo, n, nrhs, a, ia, ja, desca, b, ib, jb, descb, info);
}

template <typename scalar_t>
inline void scalapack_pposv(const char* uplo, int64_t n, int64_t nrhs, scalar_t* a, int64_t ia, int64_t ja, int* desca, scalar_t* b, int64_t ib, int64_t jb, int* descb, blas_int* info)
{
    int n_ = int64_to_int(n);
    int nrhs_ = int64_to_int(nrhs);
    int ia_ = int64_to_int(ia);
    int ja_ = int64_to_int(ja);
    int ib_ = int64_to_int(ib);
    int jb_ = int64_to_int(jb);
    scalapack_pposv(uplo, &n_, &nrhs_, a, &ia_, &ja_, desca, b, &ib_, &jb_, descb, info);
}


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pslansy BLAS_FORTRAN_NAME(pslansy,PSLANSY)
#define scalapack_pdlansy BLAS_FORTRAN_NAME(pdlansy,PDLANSY)
#define scalapack_pclansy BLAS_FORTRAN_NAME(pclansy,PCLANSY)
#define scalapack_pzlansy BLAS_FORTRAN_NAME(pzlansy,PZLANSY)

extern "C" float scalapack_pslansy(const char* norm, const char* uplo, blas_int* n, float*  a, blas_int* ia, blas_int* ja, blas_int* desca, float*  work);

extern "C" double scalapack_pdlansy(const char* norm, const char* uplo, blas_int* n, double* a, blas_int* ia, blas_int* ja, blas_int* desca, double* work);

extern "C" float scalapack_pclansy(const char* norm, const char* uplo, blas_int* n, std::complex<float>* a, blas_int* ia, blas_int* ja, blas_int* desca, float* work);

extern "C" double scalapack_pzlansy(const char* norm, const char* uplo, blas_int* n, std::complex<double>* a, blas_int* ia, blas_int* ja, blas_int* desca, double* work);

// -----------------------------------------------------------------------------

inline float scalapack_plansy(const char* norm, const char* uplo, blas_int* n, float* a, blas_int* ia, blas_int* ja, blas_int* desca, float* work)
{
    return scalapack_pslansy(norm, uplo, n, a, ia, ja, desca, work);
}

inline double scalapack_plansy(const char* norm, const char* uplo, blas_int* n, double* a, blas_int* ia, blas_int* ja, blas_int* desca, double* work)
{
    return scalapack_pdlansy(norm, uplo, n, a, ia, ja, desca, work);
}

inline float scalapack_plansy(const char* norm, const char* uplo, blas_int* n, std::complex<float>* a, blas_int* ia, blas_int* ja, blas_int* desca, float* work)
{
    return scalapack_pclansy(norm, uplo, n, a, ia, ja, desca, work);
}

inline double scalapack_plansy(const char* norm, const char* uplo, blas_int* n, std::complex<double>* a, blas_int* ia, blas_int* ja, blas_int* desca, double* work)
{
    return scalapack_pzlansy(norm, uplo, n, a, ia, ja, desca, work);
}

template <typename scalar_t>
inline double scalapack_plansy(const char* norm, const char* uplo, int64_t n, scalar_t* a, int64_t ia, int64_t ja, int* desca, blas::real_type<scalar_t>* work)
{
    int n_ = int64_to_int(n);
    int ia_ = int64_to_int(ia);
    int ja_ = int64_to_int(ja);
    return scalapack_plansy(norm, uplo, &n_, a, &ia_, &ja_, desca, work);
}


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pclanhe BLAS_FORTRAN_NAME(pclanhe,PCLANHE)
#define scalapack_pzlanhe BLAS_FORTRAN_NAME(pzlanhe,PZLANHE)

extern "C" float scalapack_pclanhe(const char* norm, const char* uplo, blas_int* n, std::complex<float>* a, blas_int* ia, blas_int* ja, blas_int* desca, float* work);

extern "C" double scalapack_pzlanhe(const char* norm, const char* uplo, blas_int* n, std::complex<double>* a, blas_int* ia, blas_int* ja, blas_int* desca, double* work);

// -----------------------------------------------------------------------------

inline float scalapack_planhe(const char* norm, const char* uplo, blas_int* n, float* a, blas_int* ia, blas_int* ja, blas_int* desca, float* work)
{
    return scalapack_pslansy(norm, uplo, n, a, ia, ja, desca, work);
}

inline double scalapack_planhe(const char* norm, const char* uplo, blas_int* n, double* a, blas_int* ia, blas_int* ja, blas_int* desca, double* work)
{
    return scalapack_pdlansy(norm, uplo, n, a, ia, ja, desca, work);
}

inline float scalapack_planhe(const char* norm, const char* uplo, blas_int* n, std::complex<float>* a, blas_int* ia, blas_int* ja, blas_int* desca, float* work)
{
    return scalapack_pclanhe(norm, uplo, n, a, ia, ja, desca, work);
}

inline double scalapack_planhe(const char* norm, const char* uplo, blas_int* n, std::complex<double>* a, blas_int* ia, blas_int* ja, blas_int* desca, double* work)
{
    return scalapack_pzlanhe(norm, uplo, n, a, ia, ja, desca, work);
}

template <typename scalar_t>
inline double scalapack_planhe(const char* norm, const char* uplo, int64_t n, scalar_t* a, int64_t ia, int64_t ja, int* desca, blas::real_type<scalar_t>* work)
{
    int n_ = int64_to_int(n);
    int ia_ = int64_to_int(ia);
    int ja_ = int64_to_int(ja);
    return scalapack_planhe(norm, uplo, &n_, a, &ia_, &ja_, desca, work);
}


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_psgemm BLAS_FORTRAN_NAME( psgemm, PSGEMM )
#define scalapack_pdgemm BLAS_FORTRAN_NAME( pdgemm, PDGEMM )
#define scalapack_pcgemm BLAS_FORTRAN_NAME( pcgemm, PCGEMM )
#define scalapack_pzgemm BLAS_FORTRAN_NAME( pzgemm, PZGEMM )

extern "C" void scalapack_psgemm(const char* transa, const char* transb, int* M, int* N, int* K, float* alpha, float* A, int* ia, int* ja, int* descA, float* B, int* ib, int* jb, int* descB, float* beta, float* C, int* ic, int* jc, int* descC);

extern "C" void scalapack_pdgemm(const char* transa, const char* transb, int* M, int* N, int* K, double* alpha, double* A, int* ia, int* ja, int* descA, double* B, int* ib, int* jb, int* descB, double* beta, double* C, int* ic, int* jc, int* descC);

extern "C" void scalapack_pcgemm(const char* transa, const char* transb, int* M, int* N, int* K, std::complex<float>* alpha, std::complex<float>* A, int* ia, int* ja, int* descA, std::complex<float>* B, int* ib, int* jb, int* descB, std::complex<float>* beta, std::complex<float>* C, int* ic, int* jc, int* descC);

extern "C" void scalapack_pzgemm(const char* transa, const char* transb, int* M, int* N, int* K, std::complex<double>* alpha, std::complex<double>* A, int* ia, int* ja, int* descA, std::complex<double>* B, int* ib, int* jb, int* descB, std::complex<double>* beta, std::complex<double>* C, int* ic, int* jc, int* descC);

// -----------------------------------------------------------------------------

inline void scalapack_pgemm(const char* transa, const char* transb, int* M, int* N, int* K, float* alpha, float* A, int* ia, int* ja, int* descA, float* B, int* ib, int* jb, int* descB, float* beta, float* C, int* ic, int* jc, int* descC)
{
    scalapack_psgemm(transa, transb, M, N, K, alpha, A, ia, ja, descA, B, ib, jb, descB, beta, C, ic, jc, descC);
}

inline void scalapack_pgemm(const char* transa, const char* transb, int* M, int* N, int* K, double* alpha, double* A, int* ia, int* ja, int* descA, double* B, int* ib, int* jb, int* descB, double* beta, double* C, int* ic, int* jc, int* descC)
{
    scalapack_pdgemm(transa, transb, M, N, K, alpha, A, ia, ja, descA, B, ib, jb, descB, beta, C, ic, jc, descC);
}

inline void scalapack_pgemm(const char* transa, const char* transb, int* M, int* N, int* K, std::complex<float>* alpha, std::complex<float>* A, int* ia, int* ja, int* descA, std::complex<float>* B, int* ib, int* jb, int* descB, std::complex<float>* beta, std::complex<float>* C, int* ic, int* jc, int* descC)
{
    scalapack_pcgemm(transa, transb, M, N, K, alpha, A, ia, ja, descA, B, ib, jb, descB, beta, C, ic, jc, descC);
}

inline void scalapack_pgemm(const char* transa, const char* transb, int* M, int* N, int* K, std::complex<double>* alpha, std::complex<double>* A, int* ia, int* ja, int* descA, std::complex<double>* B, int* ib, int* jb, int* descB, std::complex<double>* beta, std::complex<double>* C, int* ic, int* jc, int* descC)
{
    scalapack_pzgemm(transa, transb, M, N, K, alpha, A, ia, ja, descA, B, ib, jb, descB, beta, C, ic, jc, descC);
}

template <typename scalar_t>
inline void scalapack_pgemm(const char* transa, const char* transb, int64_t M, int64_t N, int64_t K, scalar_t alpha, scalar_t* A, int64_t ia, int64_t ja, int* descA, scalar_t* B, int64_t ib, int64_t jb, int* descB, scalar_t beta, scalar_t* C, int64_t ic, int64_t jc, int* descC)
{
    int M_ = int64_to_int(M);
    int N_ = int64_to_int(N);
    int K_ = int64_to_int(K);
    int ia_ = int64_to_int(ia);
    int ja_ = int64_to_int(ja);
    int ib_ = int64_to_int(ib);
    int jb_ = int64_to_int(jb);
    int ic_ = int64_to_int(ic);
    int jc_ = int64_to_int(jc);
    scalapack_pgemm(transa, transb, &M_, &N_, &K_, &alpha, A, &ia_, &ja_, descA, B, &ib_, &jb_, descB, &beta, C, &ic_, &jc_, descC);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pssymm BLAS_FORTRAN_NAME(pssymm,PSSYMM)
#define scalapack_pdsymm BLAS_FORTRAN_NAME(pdsymm,PDSYMM)
#define scalapack_pcsymm BLAS_FORTRAN_NAME(pcsymm,PCSYMM)
#define scalapack_pzsymm BLAS_FORTRAN_NAME(pzsymm,PZSYMM)

extern "C" void scalapack_pssymm(const char* side, const char* uplo, int* m, int* n, float*  alpha, float*  a, int* ia, int* ja, int* desca, float*  b, int* ib, int* jb, int* descb, float*  beta, float*  c, int* ic, int* jc, int* descc);

extern "C" void scalapack_pdsymm(const char* side, const char* uplo, int* m, int* n, double* alpha, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb, double* beta, double* c, int* ic, int* jc, int* descc);

extern "C" void scalapack_pcsymm(const char* side, const char* uplo, int* m, int* n, const std::complex<float>*  alpha, const std::complex<float>*  a, int* ia, int* ja, int* desca, const std::complex<float>*  b, int* ib, int* jb, int* descb, const std::complex<float>*  beta, const std::complex<float>*  c, int* ic, int* jc, int* descc);

extern "C" void scalapack_pzsymm(const char* side, const char* uplo, int* m, int* n, const std::complex<double>* alpha, const std::complex<double>* a, int* ia, int* ja, int* desca, const std::complex<double>* b, int* ib, int* jb, int* descb, const std::complex<double>* beta, const std::complex<double>* c, int* ic, int* jc, int* descc);

// -----------------------------------------------------------------------------

inline void scalapack_psymm(const char* side, const char* uplo, int* m, int* n, float* alpha, float* a, int* ia, int* ja, int* desca, float* b, int* ib, int* jb, int* descb, float* beta, float* c, int* ic, int* jc, int* descc)
{
    scalapack_pssymm(side, uplo, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

inline void scalapack_psymm(const char* side, const char* uplo, int* m, int* n, double* alpha, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb, double* beta, double* c, int* ic, int* jc, int* descc)
{
    scalapack_pdsymm(side, uplo, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

inline void scalapack_psymm(const char* side, const char* uplo, int* m, int* n, const std::complex<float>*  alpha, const std::complex<float>*  a, int* ia, int* ja, int* desca, const std::complex<float>*  b, int* ib, int* jb, int* descb, const std::complex<float>*  beta, const std::complex<float>*  c, int* ic, int* jc, int* descc)
{
    scalapack_pcsymm(side, uplo, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

inline void scalapack_psymm(const char* side, const char* uplo, int* m, int* n, const std::complex<double>* alpha, const std::complex<double>* a, int* ia, int* ja, int* desca, const std::complex<double>* b, int* ib, int* jb, int* descb, const std::complex<double>* beta, const std::complex<double>* c, int* ic, int* jc, int* descc)
{
    scalapack_pzsymm(side, uplo, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

template <typename scalar_t>
inline void scalapack_psymm(const char* side, const char* uplo, int64_t m, int64_t n, scalar_t alpha, scalar_t* a, int64_t ia, int64_t ja, int* desca, scalar_t* b, int64_t ib, int64_t jb, int* descb, scalar_t beta, scalar_t* c, int64_t ic, int64_t jc, int* descc)
{
    int m_ = int64_to_int(m);
    int n_ = int64_to_int(n);
    int ia_ = int64_to_int(ia);
    int ja_ = int64_to_int(ja);
    int ib_ = int64_to_int(ib);
    int jb_ = int64_to_int(jb);
    int ic_ = int64_to_int(ic);
    int jc_ = int64_to_int(jc);
    scalapack_psymm(side, uplo, &m_, &n_, &alpha, a, &ia_, &ja_, desca, b, &ib_, &jb_, descb, &beta, c, &ic_, &jc_, descc);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pstrmm BLAS_FORTRAN_NAME(pstrmm,PSTRMM)
#define scalapack_pdtrmm BLAS_FORTRAN_NAME(pdtrmm,PDTRMM)
#define scalapack_pctrmm BLAS_FORTRAN_NAME(pctrmm,PCTRMM)
#define scalapack_pztrmm BLAS_FORTRAN_NAME(pztrmm,PZTRMM)

extern "C" void scalapack_pstrmm(const char* side, const char* uplo, const char* transa, const char* diag, const blas_int* m, const blas_int* n, const float* alpha, const float* a, const blas_int* ia, const blas_int* ja, const blas_int* desca, float* b, const blas_int* ib, const blas_int* jb, const blas_int* descb);

extern "C" void scalapack_pdtrmm(const char* side, const char* uplo, const char* transa, const char* diag, const blas_int* m, const blas_int* n, const double* alpha, const double* a, const blas_int* ia, const blas_int* ja, const blas_int* desca, double* b, const blas_int* ib, const blas_int* jb, const blas_int* descb);

extern "C" void scalapack_pctrmm(const char* side, const char* uplo, const char* transa, const char* diag, const blas_int* m, const blas_int* n, const std::complex<float>* alpha, const std::complex<float>* a, const blas_int* ia, const blas_int* ja, const blas_int* desca, std::complex<float>* b, const blas_int* ib, const blas_int* jb, const blas_int* descb);

extern "C" void scalapack_pztrmm(const char* side, const char* uplo, const char* transa, const char* diag, const blas_int* m, const blas_int* n, const std::complex<double>* alpha, const std::complex<double>* a, const blas_int* ia, const blas_int* ja, const blas_int* desca, std::complex<double>* b, const blas_int* ib, const blas_int* jb, const blas_int* descb);

// -----------------------------------------------------------------------------

inline void scalapack_ptrmm(const char* side, const char* uplo, const char* transa, const char* diag, const blas_int* m, const blas_int* n, const float* alpha, const float* a, const blas_int* ia, const blas_int* ja, const blas_int* desca, float* b, const blas_int* ib, const blas_int* jb, const blas_int* descb)
{
    scalapack_pstrmm(side, uplo, transa, diag, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb);
}

inline void scalapack_ptrmm(const char* side, const char* uplo, const char* transa, const char* diag, const blas_int* m, const blas_int* n, const double* alpha, const double* a, const blas_int* ia, const blas_int* ja, const blas_int* desca, double* b, const blas_int* ib, const blas_int* jb, const blas_int* descb)
{
    scalapack_pdtrmm(side, uplo, transa, diag, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb);
}

inline void scalapack_ptrmm(const char* side, const char* uplo, const char* transa, const char* diag, const blas_int* m, const blas_int* n, const std::complex<float>* alpha, const std::complex<float>* a, const blas_int* ia, const blas_int* ja, const blas_int* desca, std::complex<float>* b, const blas_int* ib, const blas_int* jb, const blas_int* descb)
{
    scalapack_pctrmm(side, uplo, transa, diag, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb);
}

inline void scalapack_ptrmm(const char* side, const char* uplo, const char* transa, const char* diag, const blas_int* m, const blas_int* n, const std::complex<double>* alpha, const std::complex<double>* a, const blas_int* ia, const blas_int* ja, const blas_int* desca, std::complex<double>* b, const blas_int* ib, const blas_int* jb, const blas_int* descb)
{
    scalapack_pztrmm(side, uplo, transa, diag, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb);
}

template <typename scalar_t>
inline void scalapack_ptrmm(const char* side, const char* uplo, const char* transa, const char* diag, int64_t m, int64_t n, scalar_t alpha, scalar_t* a, int64_t ia, int64_t ja, const blas_int* desca, scalar_t* b, int64_t ib, int64_t jb, const blas_int* descb)
{
    int m_ = int64_to_int(m);
    int n_ = int64_to_int(n);
    int ia_ = int64_to_int(ia);
    int ja_ = int64_to_int(ja);
    int ib_ = int64_to_int(ib);
    int jb_ = int64_to_int(jb);
    scalapack_ptrmm(side, uplo, transa, diag, &m_, &n_, &alpha, a, &ia_, &ja_, desca, b, &ib_, &jb_, descb);
}


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pssyr2k BLAS_FORTRAN_NAME(pssyr2k,PSSYR2K)
#define scalapack_pdsyr2k BLAS_FORTRAN_NAME(pdsyr2k,PDSYR2K)
#define scalapack_pcsyr2k BLAS_FORTRAN_NAME(pcsyr2k,PCSYR2K)
#define scalapack_pzsyr2k BLAS_FORTRAN_NAME(pzsyr2k,PZSYR2K)

extern "C" void scalapack_pssyr2k(const char* uplo, const char* trans, blas_int* n, blas_int* k, float* alpha, float* a, blas_int* ia, blas_int* ja, blas_int* desca, float* b, blas_int* ib, blas_int* jb, blas_int* descb, float* beta, float* c, blas_int* ic, blas_int* jc, blas_int* descc);

extern "C" void scalapack_pdsyr2k(const char* uplo, const char* trans, blas_int* n, blas_int* k, double* alpha, double* a, blas_int* ia, blas_int* ja, blas_int* desca, double* b, blas_int* ib, blas_int* jb, blas_int* descb, double* beta, double* c, blas_int* ic, blas_int* jc, blas_int* descc);

extern "C" void scalapack_pcsyr2k(const char* uplo, const char* trans, blas_int* n, blas_int* k, std::complex<float>* alpha, std::complex<float>* a, blas_int* ia, blas_int* ja, blas_int* desca, std::complex<float>* b, blas_int* ib, blas_int* jb, blas_int* descb, std::complex<float>* beta, std::complex<float>* c, blas_int* ic, blas_int* jc, blas_int* descc);

extern "C" void scalapack_pzsyr2k(const char* uplo, const char* trans, blas_int* n, blas_int* k, std::complex<double>* alpha, std::complex<double>* a, blas_int* ia, blas_int* ja, blas_int* desca, std::complex<double>* b, blas_int* ib, blas_int* jb, blas_int* descb, std::complex<double>* beta, std::complex<double>* c, blas_int* ic, blas_int* jc, blas_int* descc);

// -----------------------------------------------------------------------------

inline void scalapack_psyr2k(const char* uplo, const char* trans, blas_int* n, blas_int* k, float* alpha, float* a, blas_int* ia, blas_int* ja, blas_int* desca, float* b, blas_int* ib, blas_int* jb, blas_int* descb, float* beta, float* c, blas_int* ic, blas_int* jc, blas_int* descc)
{
    scalapack_pssyr2k(uplo, trans, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

inline void scalapack_psyr2k(const char* uplo, const char* trans, blas_int* n, blas_int* k, double* alpha, double* a, blas_int* ia, blas_int* ja, blas_int* desca, double* b, blas_int* ib, blas_int* jb, blas_int* descb, double* beta, double* c, blas_int* ic, blas_int* jc, blas_int* descc)
{
    scalapack_pdsyr2k(uplo, trans, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

inline void scalapack_psyr2k(const char* uplo, const char* trans, blas_int* n, blas_int* k, std::complex<float>* alpha, std::complex<float>* a, blas_int* ia, blas_int* ja, blas_int* desca, std::complex<float>* b, blas_int* ib, blas_int* jb, blas_int* descb, std::complex<float>* beta, std::complex<float>* c, blas_int* ic, blas_int* jc, blas_int* descc)
{
    scalapack_pcsyr2k(uplo, trans, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

inline void scalapack_psyr2k(const char* uplo, const char* trans, blas_int* n, blas_int* k, std::complex<double>* alpha, std::complex<double>* a, blas_int* ia, blas_int* ja, blas_int* desca, std::complex<double>* b, blas_int* ib, blas_int* jb, blas_int* descb, std::complex<double>* beta, std::complex<double>* c, blas_int* ic, blas_int* jc, blas_int* descc)
{
    scalapack_pzsyr2k(uplo, trans, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

template <typename scalar_t>
inline void scalapack_psyr2k(const char* uplo, const char* trans, int64_t n, int64_t k, scalar_t alpha, scalar_t* a, int64_t ia, int64_t ja, int* desca, scalar_t* b, int64_t ib, int64_t jb, int* descb, scalar_t beta, scalar_t* c, int64_t ic, int64_t jc, int* descc)
{
    int n_ = int64_to_int(n);
    int k_ = int64_to_int(k);
    int ia_ = int64_to_int(ia);
    int ja_ = int64_to_int(ja);
    int ib_ = int64_to_int(ib);
    int jb_ = int64_to_int(jb);
    int ic_ = int64_to_int(ic);
    int jc_ = int64_to_int(jc);
    scalapack_psyr2k(uplo, trans, &n_, &k_, &alpha, a, &ia_, &ja_, desca, b, &ib_, &jb_, descb, &beta, c, &ic_, &jc_, descc);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pssyrk BLAS_FORTRAN_NAME(pssyrk,PSSYRK)
#define scalapack_pdsyrk BLAS_FORTRAN_NAME(pdsyrk,PDSYRK)
#define scalapack_pcsyrk BLAS_FORTRAN_NAME(pcsyrk,PCSYRK)
#define scalapack_pzsyrk BLAS_FORTRAN_NAME(pzsyrk,PZSYRK)

extern "C" void scalapack_pssyrk(const char* uplo, const char* trans, blas_int* n, blas_int* k, float* alpha, float* a, blas_int* ia, blas_int* ja, blas_int* desca, float* beta, float* c, blas_int* ic, blas_int* jc, blas_int* descc);

extern "C" void scalapack_pdsyrk(const char* uplo, const char* trans, blas_int* n, blas_int* k, double* alpha, double* a, blas_int* ia, blas_int* ja, blas_int* desca, double* beta, double* c, blas_int* ic, blas_int* jc, blas_int* descc);

extern "C" void scalapack_pcsyrk(const char* uplo, const char* trans, blas_int* n, blas_int* k, std::complex<float>* alpha, std::complex<float>* a, blas_int* ia, blas_int* ja, blas_int* desca, std::complex<float>* beta, std::complex<float>* c, blas_int* ic, blas_int* jc, blas_int* descc);

extern "C" void scalapack_pzsyrk(const char* uplo, const char* trans, blas_int* n, blas_int* k, std::complex<double>* alpha, std::complex<double>* a, blas_int* ia, blas_int* ja, blas_int* desca, std::complex<double>* beta, std::complex<double>* c, blas_int* ic, blas_int* jc, blas_int* descc);

// -----------------------------------------------------------------------------

inline void scalapack_psyrk(const char* uplo, const char* trans, blas_int* n, blas_int* k, float* alpha, float* a, blas_int* ia, blas_int* ja, blas_int* desca, float* beta, float* c, blas_int* ic, blas_int* jc, blas_int* descc)
{
    scalapack_pssyrk(uplo, trans, n, k, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

inline void scalapack_psyrk(const char* uplo, const char* trans, blas_int* n, blas_int* k, double* alpha, double* a, blas_int* ia, blas_int* ja, blas_int* desca, double* beta, double* c, blas_int* ic, blas_int* jc, blas_int* descc)
{
    scalapack_pdsyrk(uplo, trans, n, k, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

inline void scalapack_psyrk(const char* uplo, const char* trans, blas_int* n, blas_int* k, std::complex<float>* alpha, std::complex<float>* a, blas_int* ia, blas_int* ja, blas_int* desca, std::complex<float>* beta, std::complex<float>* c, blas_int* ic, blas_int* jc, blas_int* descc)
{
    scalapack_pcsyrk(uplo, trans, n, k, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

inline void scalapack_psyrk(const char* uplo, const char* trans, blas_int* n, blas_int* k, std::complex<double>* alpha, std::complex<double>* a, blas_int* ia, blas_int* ja, blas_int* desca, std::complex<double>* beta, std::complex<double>* c, blas_int* ic, blas_int* jc, blas_int* descc)
{
    scalapack_pzsyrk(uplo, trans, n, k, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

template <typename scalar_t>
inline void scalapack_psyrk(const char* uplo, const char* trans, int64_t n, int64_t k, scalar_t alpha, scalar_t* a, int64_t ia, int64_t ja, int* desca, scalar_t beta, scalar_t* c, int64_t ic, int64_t jc, int* descc)
{
    int n_ = int64_to_int(n);
    int k_ = int64_to_int(k);
    int ia_ = int64_to_int(ia);
    int ja_ = int64_to_int(ja);
    int ic_ = int64_to_int(ic);
    int jc_ = int64_to_int(jc);
    scalapack_psyrk(uplo, trans, &n_, &k_, &alpha, a, &ia_, &ja_, desca, &beta, c, &ic_, &jc_, descc);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pstrsm BLAS_FORTRAN_NAME(pstrsm,PSTRSM)
#define scalapack_pdtrsm BLAS_FORTRAN_NAME(pdtrsm,PDTRSM)
#define scalapack_pctrsm BLAS_FORTRAN_NAME(pctrsm,PCTRSM)
#define scalapack_pztrsm BLAS_FORTRAN_NAME(pztrsm,PZTRSM)

extern "C" void scalapack_pstrsm(const char* side, const char* uplo, const char* transa, const char* diag, const blas_int* m, const blas_int* n, const float* alpha, const float* a, const blas_int* ia, const blas_int* ja, const blas_int* desca, float* b, const blas_int* ib, const blas_int* jb, const blas_int* descb);

extern "C" void scalapack_pdtrsm(const char* side, const char* uplo, const char* transa, const char* diag, const blas_int* m, const blas_int* n, const double* alpha, const double* a, const blas_int* ia, const blas_int* ja, const blas_int* desca, double* b, const blas_int* ib, const blas_int* jb, const blas_int* descb);

extern "C" void scalapack_pctrsm(const char* side, const char* uplo, const char* transa, const char* diag, const blas_int* m, const blas_int* n, const std::complex<float>* alpha, const std::complex<float>* a, const blas_int* ia, const blas_int* ja, const blas_int* desca, std::complex<float>* b, const blas_int* ib, const blas_int* jb, const blas_int* descb);

extern "C" void scalapack_pztrsm(const char* side, const char* uplo, const char* transa, const char* diag, const blas_int* m, const blas_int* n, const std::complex<double>* alpha, const std::complex<double>* a, const blas_int* ia, const blas_int* ja, const blas_int* desca, std::complex<double>* b, const blas_int* ib, const blas_int* jb, const blas_int* descb);

// -----------------------------------------------------------------------------

inline void scalapack_ptrsm(const char* side, const char* uplo, const char* transa, const char* diag, const blas_int* m, const blas_int* n, const float* alpha, const float* a, const blas_int* ia, const blas_int* ja, const blas_int* desca, float* b, const blas_int* ib, const blas_int* jb, const blas_int* descb)
{
    scalapack_pstrsm(side, uplo, transa, diag, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb);
}

inline void scalapack_ptrsm(const char* side, const char* uplo, const char* transa, const char* diag, const blas_int* m, const blas_int* n, const double* alpha, const double* a, const blas_int* ia, const blas_int* ja, const blas_int* desca, double* b, const blas_int* ib, const blas_int* jb, const blas_int* descb)
{
    scalapack_pdtrsm(side, uplo, transa, diag, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb);
}

inline void scalapack_ptrsm(const char* side, const char* uplo, const char* transa, const char* diag, const blas_int* m, const blas_int* n, const std::complex<float>* alpha, const std::complex<float>* a, const blas_int* ia, const blas_int* ja, const blas_int* desca, std::complex<float>* b, const blas_int* ib, const blas_int* jb, const blas_int* descb)
{
    scalapack_pctrsm(side, uplo, transa, diag, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb);
}

inline void scalapack_ptrsm(const char* side, const char* uplo, const char* transa, const char* diag, const blas_int* m, const blas_int* n, const std::complex<double>* alpha, const std::complex<double>* a, const blas_int* ia, const blas_int* ja, const blas_int* desca, std::complex<double>* b, const blas_int* ib, const blas_int* jb, const blas_int* descb)
{
    scalapack_pztrsm(side, uplo, transa, diag, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb);
}

template <typename scalar_t>
inline void scalapack_ptrsm(const char* side, const char* uplo, const char* transa, const char* diag, int64_t m, int64_t n, scalar_t alpha, scalar_t* a, int64_t ia, int64_t ja, const blas_int* desca, scalar_t* b, int64_t ib, int64_t jb, const blas_int* descb)
{
    int m_ = int64_to_int(m);
    int n_ = int64_to_int(n);
    int ia_ = int64_to_int(ia);
    int ja_ = int64_to_int(ja);
    int ib_ = int64_to_int(ib);
    int jb_ = int64_to_int(jb);
    scalapack_ptrsm(side, uplo, transa, diag, &m_, &n_, &alpha, a, &ia_, &ja_, desca, b, &ib_, &jb_, descb);
}


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pslantr BLAS_FORTRAN_NAME( pslantr, PSLANTR )
#define scalapack_pdlantr BLAS_FORTRAN_NAME( pdlantr, PDLANTR )
#define scalapack_pclantr BLAS_FORTRAN_NAME( pclantr, PCLANTR )
#define scalapack_pzlantr BLAS_FORTRAN_NAME( pzlantr, PZLANTR )

extern "C" float scalapack_pslantr(const char* norm, const char* uplo, const char* diag, blas_int* m, blas_int* n, float* A, blas_int* ia, blas_int* ja, blas_int* descA, float* work);

extern "C" double scalapack_pdlantr(const char* norm, const char* uplo, const char* diag, blas_int* m, blas_int* n, double* A, blas_int* ia, blas_int* ja, blas_int* descA, double* work);

extern "C" float scalapack_pclantr(const char* norm, const char* uplo, const char* diag, blas_int* m, blas_int* n, std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA, float* work);

extern "C" double scalapack_pzlantr(const char* norm, const char* uplo, const char* diag, blas_int* m, blas_int* n, std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA, double* work);

// -----------------------------------------------------------------------------

inline float scalapack_plantr(const char* norm, const char* uplo, const char* diag, blas_int* m, blas_int* n, float* A, blas_int* ia, blas_int* ja, blas_int* descA, float* work)
{
    return scalapack_pslantr(norm, uplo, diag, m, n, A, ia, ja, descA, work);
}
inline double scalapack_plantr(const char* norm, const char* uplo, const char* diag, blas_int* m, blas_int* n, double* A, blas_int* ia, blas_int* ja, blas_int* descA, double* work)
{
    return scalapack_pdlantr(norm, uplo, diag, m, n, A, ia, ja, descA, work);
}
inline float scalapack_plantr(const char* norm, const char* uplo, const char* diag, blas_int* m, blas_int* n, std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA, float* work)
{
    return scalapack_pclantr(norm, uplo, diag, m, n, A, ia, ja, descA, work);
}
inline double scalapack_plantr(const char* norm, const char* uplo, const char* diag, blas_int* m, blas_int* n, std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA, double* work)
{
    return scalapack_pzlantr(norm, uplo, diag, m, n, A, ia, ja, descA, work);
}

template <typename scalar_t>
inline blas::real_type<scalar_t> scalapack_plantr(const char* norm, const char* uplo, const char* diag, int64_t m, int64_t n, scalar_t* A, int64_t ia, int64_t ja, blas_int* descA, blas::real_type<scalar_t>* work)
{
    int m_ = int64_to_int(m);
    int n_ = int64_to_int(n);
    int ia_ = int64_to_int(ia);
    int ja_ = int64_to_int(ja);
    return scalapack_plantr(norm, uplo, diag, &m_, &n_, A, &ia_, &ja_, descA, work);
}


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pchemm BLAS_FORTRAN_NAME(pchemm,PCHEMM)
#define scalapack_pzhemm BLAS_FORTRAN_NAME(pzhemm,PZHEMM)

extern "C" void scalapack_pchemm(const char* side, const char* uplo, int* m, int* n, const std::complex<float>*  alpha, const std::complex<float>*  a, int* ia, int* ja, int* desca, const std::complex<float>*  b, int* ib, int* jb, int* descb, const std::complex<float>*  beta, const std::complex<float>*  c, int* ic, int* jc, int* descc);

extern "C" void scalapack_pzhemm(const char* side, const char* uplo, int* m, int* n, const std::complex<double>* alpha, const std::complex<double>* a, int* ia, int* ja, int* desca, const std::complex<double>* b, int* ib, int* jb, int* descb, const std::complex<double>* beta, const std::complex<double>* c, int* ic, int* jc, int* descc);

// -----------------------------------------------------------------------------

inline void scalapack_phemm(const char* side, const char* uplo, int* m, int* n, float* alpha, float* a, int* ia, int* ja, int* desca, float* b, int* ib, int* jb, int* descb, float* beta, float* c, int* ic, int* jc, int* descc)
{
    scalapack_pssymm(side, uplo, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

inline void scalapack_phemm(const char* side, const char* uplo, int* m, int* n, double* alpha, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb, double* beta, double* c, int* ic, int* jc, int* descc)
{
    scalapack_pdsymm(side, uplo, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

inline void scalapack_phemm(const char* side, const char* uplo, int* m, int* n, const std::complex<float>*  alpha, const std::complex<float>*  a, int* ia, int* ja, int* desca, const std::complex<float>*  b, int* ib, int* jb, int* descb, const std::complex<float>*  beta, const std::complex<float>*  c, int* ic, int* jc, int* descc)
{
    scalapack_pchemm(side, uplo, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

inline void scalapack_phemm(const char* side, const char* uplo, int* m, int* n, const std::complex<double>* alpha, const std::complex<double>* a, int* ia, int* ja, int* desca, const std::complex<double>* b, int* ib, int* jb, int* descb, const std::complex<double>* beta, const std::complex<double>* c, int* ic, int* jc, int* descc)
{
    scalapack_pzhemm(side, uplo, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

template <typename scalar_t>
inline void scalapack_phemm(const char* side, const char* uplo, int64_t m, int64_t n, scalar_t alpha, scalar_t* a, int64_t ia, int64_t ja, int* desca, scalar_t* b, int64_t ib, int64_t jb, int* descb, scalar_t beta, scalar_t* c, int64_t ic, int64_t jc, int* descc)
{
    int m_ = int64_to_int(m);
    int n_ = int64_to_int(n);
    int ia_ = int64_to_int(ia);
    int ja_ = int64_to_int(ja);
    int ib_ = int64_to_int(ib);
    int jb_ = int64_to_int(jb);
    int ic_ = int64_to_int(ic);
    int jc_ = int64_to_int(jc);
    scalapack_phemm(side, uplo, &m_, &n_, &alpha, a, &ia_, &ja_, desca, b, &ib_, &jb_, descb, &beta, c, &ic_, &jc_, descc);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pcher2k BLAS_FORTRAN_NAME(pcher2k,PCHER2K)
#define scalapack_pzher2k BLAS_FORTRAN_NAME(pzher2k,PZHER2K)

extern "C" void scalapack_pcher2k(const char* uplo, const char* trans, blas_int* n, blas_int* k, std::complex<float>* alpha, std::complex<float>* a, blas_int* ia, blas_int* ja, blas_int* desca, std::complex<float>* b, blas_int* ib, blas_int* jb, blas_int* descb, float* beta, std::complex<float>* c, blas_int* ic, blas_int* jc, blas_int* descc);

extern "C" void scalapack_pzher2k(const char* uplo, const char* trans, blas_int* n, blas_int* k, std::complex<double>* alpha, std::complex<double>* a, blas_int* ia, blas_int* ja, blas_int* desca, std::complex<double>* b, blas_int* ib, blas_int* jb, blas_int* descb, double* beta, std::complex<double>* c, blas_int* ic, blas_int* jc, blas_int* descc);

// -----------------------------------------------------------------------------

inline void scalapack_pher2k(const char* uplo, const char* trans, blas_int* n, blas_int* k, float* alpha, float* a, blas_int* ia, blas_int* ja, blas_int* desca, float* b, blas_int* ib, blas_int* jb, blas_int* descb, float* beta, float* c, blas_int* ic, blas_int* jc, blas_int* descc)
{
    scalapack_pssyr2k(uplo, trans, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

inline void scalapack_pher2k(const char* uplo, const char* trans, blas_int* n, blas_int* k, double* alpha, double* a, blas_int* ia, blas_int* ja, blas_int* desca, double* b, blas_int* ib, blas_int* jb, blas_int* descb, double* beta, double* c, blas_int* ic, blas_int* jc, blas_int* descc)
{
    scalapack_pdsyr2k(uplo, trans, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

inline void scalapack_pher2k(const char* uplo, const char* trans, blas_int* n, blas_int* k, std::complex<float>* alpha, std::complex<float>* a, blas_int* ia, blas_int* ja, blas_int* desca, std::complex<float>* b, blas_int* ib, blas_int* jb, blas_int* descb, float* beta, std::complex<float>* c, blas_int* ic, blas_int* jc, blas_int* descc)
{
    scalapack_pcher2k(uplo, trans, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

inline void scalapack_pher2k(const char* uplo, const char* trans, blas_int* n, blas_int* k, std::complex<double>* alpha, std::complex<double>* a, blas_int* ia, blas_int* ja, blas_int* desca, std::complex<double>* b, blas_int* ib, blas_int* jb, blas_int* descb, double* beta, std::complex<double>* c, blas_int* ic, blas_int* jc, blas_int* descc)
{
    scalapack_pzher2k(uplo, trans, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
}

template <typename scalar_t>
inline void scalapack_pher2k(const char* uplo, const char* trans, int64_t n, int64_t k, scalar_t alpha, scalar_t* a, int64_t ia, int64_t ja, int* desca, scalar_t* b, int64_t ib, int64_t jb, int* descb, blas::real_type<scalar_t> beta, scalar_t* c, int64_t ic, int64_t jc, int* descc)
{
    int n_ = int64_to_int(n);
    int k_ = int64_to_int(k);
    int ia_ = int64_to_int(ia);
    int ja_ = int64_to_int(ja);
    int ib_ = int64_to_int(ib);
    int jb_ = int64_to_int(jb);
    int ic_ = int64_to_int(ic);
    int jc_ = int64_to_int(jc);
    scalapack_pher2k(uplo, trans, &n_, &k_, &alpha, a, &ia_, &ja_, desca, b, &ib_, &jb_, descb, &beta, c, &ic_, &jc_, descc);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pcherk BLAS_FORTRAN_NAME(pcherk,PCHERK)
#define scalapack_pzherk BLAS_FORTRAN_NAME(pzherk,PZHERK)

extern "C" void scalapack_pcherk(const char* uplo, const char* trans, blas_int* n, blas_int* k, float* alpha, std::complex<float>* a, blas_int* ia, blas_int* ja, blas_int* desca, float* beta, std::complex<float>* c, blas_int* ic, blas_int* jc, blas_int* descc);

extern "C" void scalapack_pzherk(const char* uplo, const char* trans, blas_int* n, blas_int* k, double* alpha, std::complex<double>* a, blas_int* ia, blas_int* ja, blas_int* desca, double* beta, std::complex<double>* c, blas_int* ic, blas_int* jc, blas_int* descc);

// -----------------------------------------------------------------------------

inline void scalapack_pherk(const char* uplo, const char* trans, blas_int* n, blas_int* k, float* alpha, float* a, blas_int* ia, blas_int* ja, blas_int* desca, float* beta, float* c, blas_int* ic, blas_int* jc, blas_int* descc)
{
    scalapack_pssyrk(uplo, trans, n, k, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

inline void scalapack_pherk(const char* uplo, const char* trans, blas_int* n, blas_int* k, double* alpha, double* a, blas_int* ia, blas_int* ja, blas_int* desca, double* beta, double* c, blas_int* ic, blas_int* jc, blas_int* descc)
{
    scalapack_pdsyrk(uplo, trans, n, k, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

inline void scalapack_pherk(const char* uplo, const char* trans, blas_int* n, blas_int* k, float* alpha, std::complex<float>* a, blas_int* ia, blas_int* ja, blas_int* desca, float* beta, std::complex<float>* c, blas_int* ic, blas_int* jc, blas_int* descc)
{
    scalapack_pcherk(uplo, trans, n, k, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

inline void scalapack_pherk(const char* uplo, const char* trans, blas_int* n, blas_int* k, double* alpha, std::complex<double>* a, blas_int* ia, blas_int* ja, blas_int* desca, double* beta, std::complex<double>* c, blas_int* ic, blas_int* jc, blas_int* descc)
{
    scalapack_pzherk(uplo, trans, n, k, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

template <typename scalar_t>
inline void scalapack_pherk(const char* uplo, const char* trans, int64_t n, int64_t k, blas::real_type<scalar_t> alpha, scalar_t* a, int64_t ia, int64_t ja, int* desca, blas::real_type<scalar_t> beta, scalar_t* c, int64_t ic, int64_t jc, int* descc)
{
    int n_ = int64_to_int(n);
    int k_ = int64_to_int(k);
    int ia_ = int64_to_int(ia);
    int ja_ = int64_to_int(ja);
    int ic_ = int64_to_int(ic);
    int jc_ = int64_to_int(jc);
    scalapack_pherk(uplo, trans, &n_, &k_, &alpha, a, &ia_, &ja_, desca, &beta, c, &ic_, &jc_, descc);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_psgetrf BLAS_FORTRAN_NAME( psgetrf, PSGETRF )
#define scalapack_pdgetrf BLAS_FORTRAN_NAME( pdgetrf, PDGETRF )
#define scalapack_pcgetrf BLAS_FORTRAN_NAME( pcgetrf, PCGETRF )
#define scalapack_pzgetrf BLAS_FORTRAN_NAME( pzgetrf, PZGETRF )

extern "C" void scalapack_psgetrf(blas_int* M, blas_int* N, float* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* ipiv, blas_int* info);

extern "C" void scalapack_pdgetrf(blas_int* M, blas_int* N, double* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* ipiv, blas_int* info);

extern "C" void scalapack_pcgetrf(blas_int* M, blas_int* N, std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* ipiv, blas_int* info);

extern "C" void scalapack_pzgetrf(blas_int* M, blas_int* N, std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* ipiv, blas_int* info);

// -----------------------------------------------------------------------------

inline void scalapack_pgetrf(blas_int* M, blas_int* N, float* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* ipiv, blas_int* info)
{
    scalapack_psgetrf(M, N, A, ia, ja, descA, ipiv, info);
}

inline void scalapack_pgetrf(blas_int* M, blas_int* N, double* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* ipiv, blas_int* info)
{
    scalapack_pdgetrf(M, N, A, ia, ja, descA, ipiv, info);
}

inline void scalapack_pgetrf(blas_int* M, blas_int* N, std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* ipiv, blas_int* info)
{
    scalapack_pcgetrf(M, N, A, ia, ja, descA, ipiv, info);
}

inline void scalapack_pgetrf(blas_int* M, blas_int* N, std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* ipiv, blas_int* info)
{
    scalapack_pzgetrf(M, N, A, ia, ja, descA, ipiv, info);
}

template <typename scalar_t>
inline void scalapack_pgetrf(int64_t M, int64_t N, scalar_t* A, int64_t ia, int64_t ja, blas_int* descA, blas_int* ipiv, int64_t* info)
{
    blas_int M_ = int64_to_int(M);
    blas_int N_ = int64_to_int(N);
    blas_int ia_ = int64_to_int(ia);
    blas_int ja_ = int64_to_int(ja);
    blas_int info_ = int64_to_int(*info);
    scalapack_pgetrf(&M_, &N_, A, &ia_, &ja_, descA, ipiv, &info_);
    *info = (int64_t)info_;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_psgetrs BLAS_FORTRAN_NAME( psgetrs, PSGETRS )
#define scalapack_pdgetrs BLAS_FORTRAN_NAME( pdgetrs, PDGETRS )
#define scalapack_pcgetrs BLAS_FORTRAN_NAME( pcgetrs, PCGETRS )
#define scalapack_pzgetrs BLAS_FORTRAN_NAME( pzgetrs, PZGETRS )

extern "C" void scalapack_psgetrs(const char* trans, blas_int* N, blas_int* NRHS, float* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* ipiv, float* B, blas_int* ib, blas_int* jb, blas_int* descB, blas_int* info);

extern "C" void scalapack_pdgetrs(const char* trans, blas_int* N, blas_int* NRHS, double* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* ipiv, double* B, blas_int* ib, blas_int* jb, blas_int* descB, blas_int* info);

extern "C" void scalapack_pcgetrs(const char* trans, blas_int* N, blas_int* NRHS, std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* ipiv, std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB, blas_int* info);

extern "C" void scalapack_pzgetrs(const char* trans, blas_int* N, blas_int* NRHS, std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* ipiv, std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB, blas_int* info);

// -----------------------------------------------------------------------------

inline void scalapack_pgetrs(const char* trans, blas_int* N, blas_int* NRHS, float* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* ipiv, float* B, blas_int* ib, blas_int* jb, blas_int* descB, blas_int* info)
{
    scalapack_psgetrs(trans, N, NRHS, A, ia, ja, descA, ipiv, B, ib, jb, descB, info);
}

inline void scalapack_pgetrs(const char* trans, blas_int* N, blas_int* NRHS, double* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* ipiv, double* B, blas_int* ib, blas_int* jb, blas_int* descB, blas_int* info)
{
    scalapack_pdgetrs(trans, N, NRHS, A, ia, ja, descA, ipiv, B, ib, jb, descB, info);
}

inline void scalapack_pgetrs(const char* trans, blas_int* N, blas_int* NRHS, std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* ipiv, std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB, blas_int* info)
{
    scalapack_pcgetrs(trans, N, NRHS, A, ia, ja, descA, ipiv, B, ib, jb, descB, info);
}

inline void scalapack_pgetrs(const char* trans, blas_int* N, blas_int* NRHS, std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* ipiv, std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB, blas_int* info)
{
    scalapack_pzgetrs(trans, N, NRHS, A, ia, ja, descA, ipiv, B, ib, jb, descB, info);
}

template <typename scalar_t>
inline void scalapack_pgetrs(const char* trans, int64_t N, int64_t NRHS, scalar_t* A, int64_t ia, int64_t ja, blas_int* descA, blas_int* ipiv, scalar_t* B, int64_t ib, int64_t jb, blas_int* descB, int64_t* info)
{
    blas_int N_ = int64_to_int(N);
    blas_int NRHS_ = int64_to_int(NRHS);
    blas_int ia_ = int64_to_int(ia);
    blas_int ja_ = int64_to_int(ja);
    blas_int ib_ = int64_to_int(ib);
    blas_int jb_ = int64_to_int(jb);
    blas_int info_ = int64_to_int(*info);
    scalapack_pgetrs(trans, &N_, &NRHS_, A, &ia_, &ja_, descA, ipiv, B, &ib_, &jb_, descB, &info_);
    *info = (int64_t)info_;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_psgesv BLAS_FORTRAN_NAME( psgesv, PSGESV )
#define scalapack_pdgesv BLAS_FORTRAN_NAME( pdgesv, PDGESV )
#define scalapack_pcgesv BLAS_FORTRAN_NAME( pcgesv, PCGESV )
#define scalapack_pzgesv BLAS_FORTRAN_NAME( pzgesv, PZGESV )

extern "C" void scalapack_psgesv(blas_int* N, blas_int* NRHS, float* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* ipiv, float* B, blas_int* ib, blas_int* jb, blas_int* descB, blas_int* info);

extern "C" void scalapack_pdgesv(blas_int* N, blas_int* NRHS, double* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* ipiv, double* B, blas_int* ib, blas_int* jb, blas_int* descB, blas_int* info);

extern "C" void scalapack_pcgesv(blas_int* N, blas_int* NRHS, std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* ipiv, std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB, blas_int* info);

extern "C" void scalapack_pzgesv(blas_int* N, blas_int* NRHS, std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* ipiv, std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB, blas_int* info);

// -----------------------------------------------------------------------------

inline void scalapack_pgesv(blas_int* N, blas_int* NRHS, float* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* ipiv, float* B, blas_int* ib, blas_int* jb, blas_int* descB, blas_int* info)
{
    scalapack_psgesv(N, NRHS, A, ia, ja, descA, ipiv, B, ib, jb, descB, info);
}

inline void scalapack_pgesv(blas_int* N, blas_int* NRHS, double* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* ipiv, double* B, blas_int* ib, blas_int* jb, blas_int* descB, blas_int* info)
{
    scalapack_pdgesv(N, NRHS, A, ia, ja, descA, ipiv, B, ib, jb, descB, info);
}

inline void scalapack_pgesv(blas_int* N, blas_int* NRHS, std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* ipiv, std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB, blas_int* info)
{
    scalapack_pcgesv(N, NRHS, A, ia, ja, descA, ipiv, B, ib, jb, descB, info);
}

inline void scalapack_pgesv(blas_int* N, blas_int* NRHS, std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA, blas_int* ipiv, std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB, blas_int* info)
{
    scalapack_pzgesv(N, NRHS, A, ia, ja, descA, ipiv, B, ib, jb, descB, info);
}

template <typename scalar_t>
inline void scalapack_pgesv(int64_t N, int64_t NRHS, scalar_t* A, int64_t ia, int64_t ja, blas_int* descA, blas_int* ipiv, scalar_t* B, int64_t ib, int64_t jb, blas_int* descB, int64_t* info)
{
    blas_int N_ = int64_to_int(N);
    blas_int NRHS_ = int64_to_int(NRHS);
    blas_int ia_ = int64_to_int(ia);
    blas_int ja_ = int64_to_int(ja);
    blas_int ib_ = int64_to_int(ib);
    blas_int jb_ = int64_to_int(jb);
    blas_int info_ = int64_to_int(*info);
    scalapack_pgesv(&N_, &NRHS_, A, &ia_, &ja_, descA, ipiv, B, &ib_, &jb_, descB, &info_);
    *info = (int64_t)info_;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_psgeqrf BLAS_FORTRAN_NAME( psgeqrf, PSGEQRF )
#define scalapack_pdgeqrf BLAS_FORTRAN_NAME( pdgeqrf, PDGEQRF )
#define scalapack_pcgeqrf BLAS_FORTRAN_NAME( pcgeqrf, PCGEQRF )
#define scalapack_pzgeqrf BLAS_FORTRAN_NAME( pzgeqrf, PZGEQRF )

extern "C" void scalapack_psgeqrf(
    blas_int* M, blas_int* N,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* tau,
    float* work, blas_int* lwork,
    blas_int* info);

extern "C" void scalapack_pdgeqrf(
    blas_int* M, blas_int* N,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* tau,
    double* work, blas_int* lwork,
    blas_int* info);

extern "C" void scalapack_pcgeqrf(
    blas_int* M, blas_int* N,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* tau,
    std::complex<float>* work, blas_int* lwork,
    blas_int* info);

extern "C" void scalapack_pzgeqrf(
    blas_int* M, blas_int* N,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* tau,
    std::complex<double>* work, blas_int* lwork,
    blas_int* info);

// -----------------------------------------------------------------------------

inline void scalapack_pgeqrf(
    blas_int* M, blas_int* N,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* tau,
    float* work, blas_int* lwork,
    blas_int* info)
{
    scalapack_psgeqrf(M, N, A, ia, ja, descA, tau, work, lwork, info);
}

inline void scalapack_pgeqrf(
    blas_int* M, blas_int* N,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* tau,
    double* work, blas_int* lwork,
    blas_int* info)
{
    scalapack_pdgeqrf(M, N, A, ia, ja, descA, tau, work, lwork, info);
}

inline void scalapack_pgeqrf(
    blas_int* M, blas_int* N,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* tau,
    std::complex<float>* work, blas_int* lwork,
    blas_int* info)
{
    scalapack_pcgeqrf(M, N, A, ia, ja, descA, tau, work, lwork, info);
}

inline void scalapack_pgeqrf(
    blas_int* M, blas_int* N,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* tau,
    std::complex<double>* work, blas_int* lwork,
    blas_int* info)
{
    scalapack_pzgeqrf(M, N, A, ia, ja, descA, tau, work, lwork, info);
}

template <typename scalar_t>
inline void scalapack_pgeqrf(
    int64_t M, int64_t N,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    scalar_t* tau,
    scalar_t* work, int64_t lwork,
    int64_t* info)
{
    blas_int M_ = int64_to_int(M);
    blas_int N_ = int64_to_int(N);
    blas_int ia_ = int64_to_int(ia);
    blas_int ja_ = int64_to_int(ja);
    blas_int lwork_ = int64_to_int(lwork);
    blas_int info_ = int64_to_int(*info);
    scalapack_pgeqrf(&M_, &N_, A, &ia_, &ja_, descA, tau, work, &lwork_, &info_);
    *info = (int64_t)info_;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_psgelqf BLAS_FORTRAN_NAME( psgelqf, PSGELQF )
#define scalapack_pdgelqf BLAS_FORTRAN_NAME( pdgelqf, PDGELQF )
#define scalapack_pcgelqf BLAS_FORTRAN_NAME( pcgelqf, PCGELQF )
#define scalapack_pzgelqf BLAS_FORTRAN_NAME( pzgelqf, PZGELQF )

extern "C" void scalapack_psgelqf(
    blas_int* M, blas_int* N,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* tau,
    float* work, blas_int* lwork,
    blas_int* info);

extern "C" void scalapack_pdgelqf(
    blas_int* M, blas_int* N,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* tau,
    double* work, blas_int* lwork,
    blas_int* info);

extern "C" void scalapack_pcgelqf(
    blas_int* M, blas_int* N,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* tau,
    std::complex<float>* work, blas_int* lwork,
    blas_int* info);

extern "C" void scalapack_pzgelqf(
    blas_int* M, blas_int* N,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* tau,
    std::complex<double>* work, blas_int* lwork,
    blas_int* info);

// -----------------------------------------------------------------------------

inline void scalapack_pgelqf(
    blas_int* M, blas_int* N,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* tau,
    float* work, blas_int* lwork,
    blas_int* info)
{
    scalapack_psgelqf(M, N, A, ia, ja, descA, tau, work, lwork, info);
}

inline void scalapack_pgelqf(
    blas_int* M, blas_int* N,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* tau,
    double* work, blas_int* lwork,
    blas_int* info)
{
    scalapack_pdgelqf(M, N, A, ia, ja, descA, tau, work, lwork, info);
}

inline void scalapack_pgelqf(
    blas_int* M, blas_int* N,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* tau,
    std::complex<float>* work, blas_int* lwork,
    blas_int* info)
{
    scalapack_pcgelqf(M, N, A, ia, ja, descA, tau, work, lwork, info);
}

inline void scalapack_pgelqf(
    blas_int* M, blas_int* N,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* tau,
    std::complex<double>* work, blas_int* lwork,
    blas_int* info)
{
    scalapack_pzgelqf(M, N, A, ia, ja, descA, tau, work, lwork, info);
}

template <typename scalar_t>
inline void scalapack_pgelqf(
    int64_t M, int64_t N,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    scalar_t* tau,
    scalar_t* work, int64_t lwork,
    int64_t* info)
{
    blas_int M_ = int64_to_int(M);
    blas_int N_ = int64_to_int(N);
    blas_int ia_ = int64_to_int(ia);
    blas_int ja_ = int64_to_int(ja);
    blas_int lwork_ = int64_to_int(lwork);
    blas_int info_ = int64_to_int(*info);
    scalapack_pgelqf(&M_, &N_, A, &ia_, &ja_, descA, tau, work, &lwork_, &info_);
    *info = (int64_t)info_;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_psormqr BLAS_FORTRAN_NAME( psormqr, PSORMQR )
#define scalapack_pdormqr BLAS_FORTRAN_NAME( pdormqr, PDORMQR )
#define scalapack_pcunmqr BLAS_FORTRAN_NAME( pcunmqr, PCUNMQR )
#define scalapack_pzunmqr BLAS_FORTRAN_NAME( pzunmqr, PZUNMQR )

extern "C" void scalapack_psormqr(
    const char* side, const char* trans,
    blas_int* M, blas_int* N, blas_int* K,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* tau,
    float* C, blas_int* ic, blas_int* jc, blas_int* descC,
    float* work, blas_int* lwork,
    blas_int* info);

extern "C" void scalapack_pdormqr(
    const char* side, const char* trans,
    blas_int* M, blas_int* N, blas_int* K,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* tau,
    double* C, blas_int* ic, blas_int* jc, blas_int* descC,
    double* work, blas_int* lwork,
    blas_int* info);

extern "C" void scalapack_pcunmqr(
    const char* side, const char* trans,
    blas_int* M, blas_int* N, blas_int* K,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* tau,
    std::complex<float>* C, blas_int* ic, blas_int* jc, blas_int* descC,
    std::complex<float>* work, blas_int* lwork,
    blas_int* info);

extern "C" void scalapack_pzunmqr(
    const char* side, const char* trans,
    blas_int* M, blas_int* N, blas_int* K,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* tau,
    std::complex<double>* C, blas_int* ic, blas_int* jc, blas_int* descC,
    std::complex<double>* work, blas_int* lwork,
    blas_int* info);

// -----------------------------------------------------------------------------

inline void scalapack_punmqr(
    const char* side, const char* trans,
    blas_int* M, blas_int* N, blas_int* K,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* tau,
    float* C, blas_int* ic, blas_int* jc, blas_int* descC,
    float* work, blas_int* lwork,
    blas_int* info)
{
    scalapack_psormqr(side, trans, M, N, K, A, ia, ja, descA, tau,
                      C, ic, jc, descC, work, lwork, info);
}

inline void scalapack_punmqr(
    const char* side, const char* trans,
    blas_int* M, blas_int* N, blas_int* K,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* tau,
    double* C, blas_int* ic, blas_int* jc, blas_int* descC,
    double* work, blas_int* lwork,
    blas_int* info)
{
    scalapack_pdormqr(side, trans, M, N, K, A, ia, ja, descA, tau,
                      C, ic, jc, descC, work, lwork, info);
}

inline void scalapack_punmqr(
    const char* side, const char* trans,
    blas_int* M, blas_int* N, blas_int* K,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* tau,
    std::complex<float>* C, blas_int* ic, blas_int* jc, blas_int* descC,
    std::complex<float>* work, blas_int* lwork,
    blas_int* info)
{
    scalapack_pcunmqr(side, trans, M, N, K, A, ia, ja, descA, tau,
                      C, ic, jc, descC, work, lwork, info);
}

inline void scalapack_punmqr(
    const char* side, const char* trans,
    blas_int* M, blas_int* N, blas_int* K,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* tau,
    std::complex<double>* C, blas_int* ic, blas_int* jc, blas_int* descC,
    std::complex<double>* work, blas_int* lwork,
    blas_int* info)
{
    scalapack_pzunmqr(side, trans, M, N, K, A, ia, ja, descA, tau,
                      C, ic, jc, descC, work, lwork, info);
}

template <typename scalar_t>
inline void scalapack_punmqr(
    const char* side, const char* trans,
    int64_t M, int64_t N, int64_t K,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    scalar_t* tau,
    scalar_t* C, int64_t ic, int64_t jc, blas_int* descC,
    scalar_t* work, int64_t lwork,
    int64_t* info)
{
    blas_int M_ = int64_to_int(M);
    blas_int N_ = int64_to_int(N);
    blas_int K_ = int64_to_int(K);
    blas_int ia_ = int64_to_int(ia);
    blas_int ja_ = int64_to_int(ja);
    blas_int ic_ = int64_to_int(ic);
    blas_int jc_ = int64_to_int(jc);
    blas_int lwork_ = int64_to_int(lwork);
    blas_int info_ = int64_to_int(*info);
    scalapack_punmqr(side, trans, &M_, &N_, &K_, A, &ia_, &ja_, descA, tau,
                     C, &ic_, &jc_, descC, work, &lwork_, &info_);
    *info = (int64_t)info_;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_psormlq BLAS_FORTRAN_NAME( psormlq, PSORMLQ )
#define scalapack_pdormlq BLAS_FORTRAN_NAME( pdormlq, PDORMLQ )
#define scalapack_pcunmlq BLAS_FORTRAN_NAME( pcunmlq, PCUNMLQ )
#define scalapack_pzunmlq BLAS_FORTRAN_NAME( pzunmlq, PZUNMLQ )

extern "C" void scalapack_psormlq(
    const char* side, const char* trans,
    blas_int* M, blas_int* N, blas_int* K,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* tau,
    float* C, blas_int* ic, blas_int* jc, blas_int* descC,
    float* work, blas_int* lwork,
    blas_int* info);

extern "C" void scalapack_pdormlq(
    const char* side, const char* trans,
    blas_int* M, blas_int* N, blas_int* K,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* tau,
    double* C, blas_int* ic, blas_int* jc, blas_int* descC,
    double* work, blas_int* lwork,
    blas_int* info);

extern "C" void scalapack_pcunmlq(
    const char* side, const char* trans,
    blas_int* M, blas_int* N, blas_int* K,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* tau,
    std::complex<float>* C, blas_int* ic, blas_int* jc, blas_int* descC,
    std::complex<float>* work, blas_int* lwork,
    blas_int* info);

extern "C" void scalapack_pzunmlq(
    const char* side, const char* trans,
    blas_int* M, blas_int* N, blas_int* K,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* tau,
    std::complex<double>* C, blas_int* ic, blas_int* jc, blas_int* descC,
    std::complex<double>* work, blas_int* lwork,
    blas_int* info);

// -----------------------------------------------------------------------------

inline void scalapack_punmlq(
    const char* side, const char* trans,
    blas_int* M, blas_int* N, blas_int* K,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* tau,
    float* C, blas_int* ic, blas_int* jc, blas_int* descC,
    float* work, blas_int* lwork,
    blas_int* info)
{
    scalapack_psormlq(side, trans, M, N, K, A, ia, ja, descA, tau,
                      C, ic, jc, descC, work, lwork, info);
}

inline void scalapack_punmlq(
    const char* side, const char* trans,
    blas_int* M, blas_int* N, blas_int* K,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* tau,
    double* C, blas_int* ic, blas_int* jc, blas_int* descC,
    double* work, blas_int* lwork,
    blas_int* info)
{
    scalapack_pdormlq(side, trans, M, N, K, A, ia, ja, descA, tau,
                      C, ic, jc, descC, work, lwork, info);
}

inline void scalapack_punmlq(
    const char* side, const char* trans,
    blas_int* M, blas_int* N, blas_int* K,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* tau,
    std::complex<float>* C, blas_int* ic, blas_int* jc, blas_int* descC,
    std::complex<float>* work, blas_int* lwork,
    blas_int* info)
{
    scalapack_pcunmlq(side, trans, M, N, K, A, ia, ja, descA, tau,
                      C, ic, jc, descC, work, lwork, info);
}

inline void scalapack_punmlq(
    const char* side, const char* trans,
    blas_int* M, blas_int* N, blas_int* K,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* tau,
    std::complex<double>* C, blas_int* ic, blas_int* jc, blas_int* descC,
    std::complex<double>* work, blas_int* lwork,
    blas_int* info)
{
    scalapack_pzunmlq(side, trans, M, N, K, A, ia, ja, descA, tau,
                      C, ic, jc, descC, work, lwork, info);
}

template <typename scalar_t>
inline void scalapack_punmlq(
    const char* side, const char* trans,
    int64_t M, int64_t N, int64_t K,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    scalar_t* tau,
    scalar_t* C, int64_t ic, int64_t jc, blas_int* descC,
    scalar_t* work, int64_t lwork,
    int64_t* info)
{
    blas_int M_ = int64_to_int(M);
    blas_int N_ = int64_to_int(N);
    blas_int K_ = int64_to_int(K);
    blas_int ia_ = int64_to_int(ia);
    blas_int ja_ = int64_to_int(ja);
    blas_int ic_ = int64_to_int(ic);
    blas_int jc_ = int64_to_int(jc);
    blas_int lwork_ = int64_to_int(lwork);
    blas_int info_ = int64_to_int(*info);
    scalapack_punmlq(side, trans, &M_, &N_, &K_, A, &ia_, &ja_, descA, tau,
                     C, &ic_, &jc_, descC, work, &lwork_, &info_);
    *info = (int64_t)info_;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_psgels BLAS_FORTRAN_NAME( psgels, PSGELS )
#define scalapack_pdgels BLAS_FORTRAN_NAME( pdgels, PDGELS )
#define scalapack_pcgels BLAS_FORTRAN_NAME( pcgels, PCGELS )
#define scalapack_pzgels BLAS_FORTRAN_NAME( pzgels, PZGELS )

extern "C" void scalapack_psgels(
    const char* trans,
    blas_int* M, blas_int* N, blas_int* NRHS,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    float* work, blas_int* lwork,
    blas_int* info);

extern "C" void scalapack_pdgels(
    const char* trans,
    blas_int* M, blas_int* N, blas_int* NRHS,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* work, blas_int* lwork,
    blas_int* info);

extern "C" void scalapack_pcgels(
    const char* trans,
    blas_int* M, blas_int* N, blas_int* NRHS,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    std::complex<float>* work, blas_int* lwork,
    blas_int* info);

extern "C" void scalapack_pzgels(
    const char* trans,
    blas_int* M, blas_int* N, blas_int* NRHS,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    std::complex<double>* work, blas_int* lwork,
    blas_int* info);

// -----------------------------------------------------------------------------

inline void scalapack_pgels(
    const char* trans,
    blas_int* M, blas_int* N, blas_int* NRHS,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    float* work, blas_int* lwork,
    blas_int* info)
{
    char trans2 = *trans;
    if (trans2 == 'c' || trans2 == 'C')
        trans2 = 't';
    scalapack_psgels(&trans2, M, N, NRHS, A, ia, ja, descA,
                      B, ib, jb, descB, work, lwork, info);
}

inline void scalapack_pgels(
    const char* trans,
    blas_int* M, blas_int* N, blas_int* NRHS,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* work, blas_int* lwork,
    blas_int* info)
{
    char trans2 = *trans;
    if (trans2 == 'c' || trans2 == 'C')
        trans2 = 't';
    scalapack_pdgels(&trans2, M, N, NRHS, A, ia, ja, descA,
                      B, ib, jb, descB, work, lwork, info);
}

inline void scalapack_pgels(
    const char* trans,
    blas_int* M, blas_int* N, blas_int* NRHS,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    std::complex<float>* work, blas_int* lwork,
    blas_int* info)
{
    scalapack_pcgels(trans, M, N, NRHS, A, ia, ja, descA,
                      B, ib, jb, descB, work, lwork, info);
}

inline void scalapack_pgels(
    const char* trans,
    blas_int* M, blas_int* N, blas_int* NRHS,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    std::complex<double>* work, blas_int* lwork,
    blas_int* info)
{
    scalapack_pzgels(trans, M, N, NRHS, A, ia, ja, descA,
                      B, ib, jb, descB, work, lwork, info);
}

template <typename scalar_t>
inline void scalapack_pgels(
    const char* trans,
    int64_t M, int64_t N, int64_t NRHS,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    scalar_t* B, int64_t ib, int64_t jb, blas_int* descB,
    scalar_t* work, int64_t lwork,
    int64_t* info)
{
    blas_int M_     = int64_to_int(M);
    blas_int N_     = int64_to_int(N);
    blas_int NRHS_  = int64_to_int(NRHS);
    blas_int ia_    = int64_to_int(ia);
    blas_int ja_    = int64_to_int(ja);
    blas_int ib_    = int64_to_int(ib);
    blas_int jb_    = int64_to_int(jb);
    blas_int lwork_ = int64_to_int(lwork);
    blas_int info_  = int64_to_int(*info);
    scalapack_pgels(trans, &M_, &N_, &NRHS_, A, &ia_, &ja_, descA,
                     B, &ib_, &jb_, descB, work, &lwork_, &info_);
    *info = (int64_t)info_;
}


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_psgesvd BLAS_FORTRAN_NAME( psgesvd, PSGESVD )
#define scalapack_pdgesvd BLAS_FORTRAN_NAME( pdgesvd, PDGESVD )
#define scalapack_pcgesvd BLAS_FORTRAN_NAME( pcgesvd, PCGESVD )
#define scalapack_pzgesvd BLAS_FORTRAN_NAME( pzgesvd, PZGESVD )

extern "C" void scalapack_psgesvd(
    const char* jobu, const char* jobvt,
    blas_int* m, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* S,
    float* U, blas_int* iu, blas_int* ju, blas_int* descU,
    float* VT, blas_int* ivt, blas_int* jvt, blas_int* descVT,
    float* work, blas_int* lwork,
    blas_int* info);

extern "C" void scalapack_pdgesvd(
    const char* jobu, const char* jobvt,
    blas_int* m, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* S,
    double* U, blas_int* iu, blas_int* ju, blas_int* descU,
    double* VT, blas_int* ivt, blas_int* jvt, blas_int* descVT,
    double* work, blas_int* lwork,
    blas_int* info);

extern "C" void scalapack_pcgesvd(
    const char* jobu, const char* jobvt,
    blas_int* m, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* S,
    std::complex<float>* U, blas_int* iu, blas_int* ju, blas_int* descU,
    std::complex<float>* VT, blas_int* ivt, blas_int* jvt, blas_int* descVT,
    std::complex<float>* work, blas_int* lwork,
    float* rwork,
    blas_int* info);

extern "C" void scalapack_pzgesvd(
    const char* jobu, const char* jobvt,
    blas_int* m, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* S,
    std::complex<double>* U, blas_int* iu, blas_int* ju, blas_int* descU,
    std::complex<double>* VT, blas_int* ivt, blas_int* jvt, blas_int* descVT,
    std::complex<double>* work, blas_int* lwork,
    double* rwork,
    blas_int* info);

// -----------------------------------------------------------------------------

inline void scalapack_pgesvd(
    const char* jobu, const char* jobvt,
    blas_int* m, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* S,
    float* U, blas_int* iu, blas_int* ju, blas_int* descU,
    float* VT, blas_int* ivt, blas_int* jvt, blas_int* descVT,
    float* work, blas_int* lwork,
    float* rwork,
    blas_int* info)
{
    rwork[0] = 1;  // unused; lrwork = 1
    scalapack_psgesvd(jobu, jobvt, m, n,
                      A, ia, ja, descA, S,
                      U, iu, ju, descU,
                      VT, ivt, jvt, descVT,
                      work, lwork, info);
}

inline void scalapack_pgesvd(
    const char* jobu, const char* jobvt,
    blas_int* m, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* S,
    double* U, blas_int* iu, blas_int* ju, blas_int* descU,
    double* VT, blas_int* ivt, blas_int* jvt, blas_int* descVT,
    double* work, blas_int* lwork,
    double* rwork,
    blas_int* info)
{
    rwork[0] = 1;  // unused; lrwork = 1
    scalapack_pdgesvd(jobu, jobvt, m, n,
                      A, ia, ja, descA, S,
                      U, iu, ju, descU,
                      VT, ivt, jvt, descVT,
                      work, lwork, info);
}

inline void scalapack_pgesvd(
    const char* jobu, const char* jobvt,
    blas_int* m, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* S,
    std::complex<float>* U, blas_int* iu, blas_int* ju, blas_int* descU,
    std::complex<float>* VT, blas_int* ivt, blas_int* jvt, blas_int* descVT,
    std::complex<float>* work, blas_int* lwork,
    float* rwork,
    blas_int* info)
{
    scalapack_pcgesvd(jobu, jobvt, m, n,
                      A, ia, ja, descA, S,
                      U, iu, ju, descU,
                      VT, ivt, jvt, descVT,
                      work, lwork, rwork, info);
}

inline void scalapack_pgesvd(
    const char* jobu, const char* jobvt,
    blas_int* m, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* S,
    std::complex<double>* U, blas_int* iu, blas_int* ju, blas_int* descU,
    std::complex<double>* VT, blas_int* ivt, blas_int* jvt, blas_int* descVT,
    std::complex<double>* work, blas_int* lwork,
    double* rwork,
    blas_int* info)
{
    scalapack_pzgesvd(jobu, jobvt, m, n,
                      A, ia, ja, descA, S,
                      U, iu, ju, descU,
                      VT, ivt, jvt, descVT,
                      work, lwork, rwork, info);
}

template <typename scalar_t>
inline void scalapack_pgesvd(
    const char* jobu, const char* jobvt,
    int64_t m, int64_t n,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    blas::real_type<scalar_t>* S,
    scalar_t* U, int64_t iu, int64_t ju, blas_int* descU,
    scalar_t* VT, int64_t ivt, int64_t jvt, blas_int* descVT,
    scalar_t* work, int64_t lwork,
    blas::real_type<scalar_t>* rwork,
    int64_t* info)
{
    blas_int m_     = int64_to_int(m);
    blas_int n_     = int64_to_int(n);
    blas_int ia_    = int64_to_int(ia);
    blas_int ja_    = int64_to_int(ja);
    blas_int iu_    = int64_to_int(iu);
    blas_int ju_    = int64_to_int(ju);
    blas_int ivt_   = int64_to_int(ivt);
    blas_int jvt_   = int64_to_int(jvt);
    blas_int lwork_ = int64_to_int(lwork);
    blas_int info_  = int64_to_int(*info);
    scalapack_pgesvd(jobu, jobvt, &m_, &n_,
                     A, &ia_, &ja_, descA, S,
                     U, &iu_, &ju_, descU,
                     VT, &ivt_, &jvt_, descVT,
                     work, &lwork_, rwork, &info_);
    *info = (int64_t)info_;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pssyev BLAS_FORTRAN_NAME( pssyev, PSSYEV )
#define scalapack_pdsyev BLAS_FORTRAN_NAME( pdsyev, PDSYEV )
#define scalapack_pcheev BLAS_FORTRAN_NAME( pcheev, PCHEEV )
#define scalapack_pzheev BLAS_FORTRAN_NAME( pzheev, PZHEEV )

extern "C" void scalapack_pssyev(
    const char* jobz, const char* uplo, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA, float* W,
    float* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    float* work, blas_int* lwork,
    blas_int* info);

extern "C" void scalapack_pdsyev(
    const char* jobz, const char* uplo, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA, double* W,
    double* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    double* work, blas_int* lwork,
    blas_int* info);

extern "C" void scalapack_pcheev(
    const char* jobz, const char* uplo, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA, float* W,
    std::complex<float>* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    std::complex<float>* work, blas_int* lwork,
    float* rwork, blas_int* lrwork,
    blas_int* info);

extern "C" void scalapack_pzheev(
    const char* jobz, const char* uplo, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA, double* W,
    std::complex<double>* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    std::complex<double>* work, blas_int* lwork,
    double* rwork, blas_int* lrwork,
    blas_int* info);

// -----------------------------------------------------------------------------

inline void scalapack_pheev(
    const char* jobz, const char* uplo, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA, float* W,
    float* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    float* work, blas_int* lwork,
    float* rwork, blas_int* lrwork,
    blas_int* info)
{
    scalapack_pssyev(jobz, uplo, n, A, ia, ja, descA, W, Z, iz, jz, descZ, work, lwork, info);
}

inline void scalapack_pheev(
    const char* jobz, const char* uplo, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA, double* W,
    double* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    double* work, blas_int* lwork,
    double* rwork, blas_int* lrwork,
    blas_int* info)
{
    scalapack_pdsyev(jobz, uplo, n, A, ia, ja, descA, W, Z, iz, jz, descZ, work, lwork, info);
}

inline void scalapack_pheev(
    const char* jobz, const char* uplo, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA, float* W,
    std::complex<float>* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    std::complex<float>* work, blas_int* lwork,
    float* rwork, blas_int* lrwork,
    blas_int* info)
{
    scalapack_pcheev(jobz, uplo, n, A, ia, ja, descA, W, Z, iz, jz, descZ, work, lwork, rwork, lrwork, info);
}

inline void scalapack_pheev(
    const char* jobz, const char* uplo, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA, double* W,
    std::complex<double>* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    std::complex<double>* work, blas_int* lwork,
    double* rwork, blas_int* lrwork,
    blas_int* info)
{
    scalapack_pzheev(jobz, uplo, n, A, ia, ja, descA, W, Z, iz, jz, descZ, work, lwork, rwork, lrwork, info);
}

template <typename scalar_t>
inline void scalapack_pheev(
    const char* jobz, const char* uplo,
    int64_t n,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    blas::real_type<scalar_t>* W,
    scalar_t* Z, int64_t iz, int64_t jz, blas_int* descZ,
    scalar_t* work, int64_t lwork,
    blas::real_type<scalar_t>* rwork, int64_t lrwork,
    int64_t* info)
{
    blas_int n_     = int64_to_int(n);
    blas_int ia_    = int64_to_int(ia);
    blas_int ja_    = int64_to_int(ja);
    blas_int iz_    = int64_to_int(iz);
    blas_int jz_    = int64_to_int(jz);
    blas_int lwork_ = int64_to_int(lwork);
    blas_int lrwork_ = int64_to_int(lrwork);
    blas_int info_  = int64_to_int(*info);
    scalapack_pheev(jobz, uplo, &n_, A, &ia_, &ja_, descA, W,
                    Z, &iz_, &jz_, descZ,
                    work, &lwork_, rwork, &lrwork_, &info_);
    *info = (int64_t)info_;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pslaset BLAS_FORTRAN_NAME( pslaset, PSLASET )
#define scalapack_pdlaset BLAS_FORTRAN_NAME( pdlaset, PDLASET )
#define scalapack_pclaset BLAS_FORTRAN_NAME( pclaset, PCLASET )
#define scalapack_pzlaset BLAS_FORTRAN_NAME( pzlaset, PZLASET )

extern "C" void scalapack_pslaset(
    const char* uplo, blas_int* M, blas_int* N,
    float* offdiag, float* diag,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA);

extern "C" void scalapack_pdlaset(
    const char* uplo, blas_int* M, blas_int* N,
    double* offdiag, double* diag,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA);

extern "C" void scalapack_pclaset(
    const char* uplo, blas_int* M, blas_int* N,
    std::complex<float>* offdiag, std::complex<float>* diag,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA);

extern "C" void scalapack_pzlaset(
    const char* uplo, blas_int* M, blas_int* N,
    std::complex<double>* offdiag, std::complex<double>* diag,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA);

// -----------------------------------------------------------------------------

inline void scalapack_plaset(
    const char* uplo, blas_int* M, blas_int* N,
    float* offdiag, float* diag,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA)
{
    scalapack_pslaset(uplo, M, N, offdiag, diag, A, ia, ja, descA);
}

inline void scalapack_plaset(
    const char* uplo, blas_int* M, blas_int* N,
    double* offdiag, double* diag,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA)
{
    scalapack_pdlaset(uplo, M, N, offdiag, diag, A, ia, ja, descA);
}

inline void scalapack_plaset(
    const char* uplo, blas_int* M, blas_int* N,
    std::complex<float>* offdiag, std::complex<float>* diag,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA)
{
    scalapack_pclaset(uplo, M, N, offdiag, diag, A, ia, ja, descA);
}

inline void scalapack_plaset(
    const char* uplo, blas_int* M, blas_int* N,
    std::complex<double>* offdiag, std::complex<double>* diag,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA)
{
    scalapack_pzlaset(uplo, M, N, offdiag, diag, A, ia, ja, descA);
}

template <typename scalar_t>
inline void scalapack_plaset(
    const char* uplo, int64_t M, int64_t N,
    scalar_t offdiag, scalar_t diag,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA)
{
    blas_int M_ = int64_to_int(M);
    blas_int N_ = int64_to_int(N);
    blas_int ia_ = int64_to_int(ia);
    blas_int ja_ = int64_to_int(ja);
    scalapack_plaset(uplo, &M_, &N_, &offdiag, &diag, A, &ia_, &ja_, descA);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pslacpy BLAS_FORTRAN_NAME( pslacpy, PSLACPY )
#define scalapack_pdlacpy BLAS_FORTRAN_NAME( pdlacpy, PDLACPY )
#define scalapack_pclacpy BLAS_FORTRAN_NAME( pclacpy, PCLACPY )
#define scalapack_pzlacpy BLAS_FORTRAN_NAME( pzlacpy, PZLACPY )

extern "C" void scalapack_pslacpy(
    const char* uplo, blas_int* M, blas_int* N,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB);

extern "C" void scalapack_pdlacpy(
    const char* uplo, blas_int* M, blas_int* N,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB);

extern "C" void scalapack_pclacpy(
    const char* uplo, blas_int* M, blas_int* N,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB);

extern "C" void scalapack_pzlacpy(
    const char* uplo, blas_int* M, blas_int* N,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB);

// -----------------------------------------------------------------------------

inline void scalapack_placpy(
    const char* uplo, blas_int* M, blas_int* N,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB)
{
    scalapack_pslacpy(uplo, M, N, A, ia, ja, descA,
                                  B, ib, jb, descB);
}

inline void scalapack_placpy(
    const char* uplo, blas_int* M, blas_int* N,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB)
{
    scalapack_pdlacpy(uplo, M, N, A, ia, ja, descA,
                                  B, ib, jb, descB);
}

inline void scalapack_placpy(
    const char* uplo, blas_int* M, blas_int* N,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB)
{
    scalapack_pclacpy(uplo, M, N, A, ia, ja, descA,
                                  B, ib, jb, descB);
}

inline void scalapack_placpy(
    const char* uplo, blas_int* M, blas_int* N,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB)
{
    scalapack_pzlacpy(uplo, M, N, A, ia, ja, descA,
                                  B, ib, jb, descB);
}

template <typename scalar_t>
inline void scalapack_placpy(
    const char* uplo, int64_t M, int64_t N,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    scalar_t* B, int64_t ib, int64_t jb, blas_int* descB)
{
    blas_int M_ = int64_to_int(M);
    blas_int N_ = int64_to_int(N);
    blas_int ia_ = int64_to_int(ia);
    blas_int ja_ = int64_to_int(ja);
    blas_int ib_ = int64_to_int(ib);
    blas_int jb_ = int64_to_int(jb);
    scalapack_placpy(uplo, &M_, &N_, A, &ia_, &ja_, descA,
                                     B, &ib_, &jb_, descB);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pssygvx BLAS_FORTRAN_NAME( pssygvx, PSSYGVX )
#define scalapack_pdsygvx BLAS_FORTRAN_NAME( pdsygvx, PDSYGVX )
#define scalapack_pchegvx BLAS_FORTRAN_NAME( pchegvx, PCHEGVX )
#define scalapack_pzhegvx BLAS_FORTRAN_NAME( pzhegvx, PZHEGVX )

extern "C" void scalapack_pssygvx(
    blas_int* itype, const char* jobz, const char* range, const char* uplo, blas_int* n,
    float *A, blas_int* ia, blas_int* ja, blas_int* descA,
    float *B, blas_int* ib, blas_int* jb, blas_int* descB,
    float* vl, float* vu,  blas_int* il, blas_int* iu, float* abstol,
    blas_int* nfound, blas_int* nzfound, float* W, float* orfac,
    float* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    float* work, blas_int* lwork, blas_int* iwork, blas_int* liwork,
    blas_int* ifail, blas_int* iclustr, float* gap, blas_int* info);

extern "C" void scalapack_pdsygvx(
    blas_int* itype, const char* jobz, const char* range, const char* uplo, blas_int* n,
    double *A, blas_int* ia, blas_int* ja, blas_int* descA,
    double *B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* vl, double* vu,  blas_int* il, blas_int* iu, double* abstol,
    blas_int* nfound, blas_int* nzfound, double* W, double* orfac,
    double* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    double* work, blas_int* lwork, blas_int* iwork, blas_int* liwork,
    blas_int* ifail, blas_int* iclustr, double* gap, blas_int* info);

extern "C" void scalapack_pchegvx(
    blas_int* itype, const char* jobz, const char* range, const char* uplo, blas_int* n,
    std::complex<float> *A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float> *B, blas_int* ib, blas_int* jb, blas_int* descB,
    float* vl, float* vu,  blas_int* il, blas_int* iu, float* abstol,
    blas_int* nfound, blas_int* nzfound, float* W, float* orfac,
    std::complex<float>* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    std::complex<float>* work, blas_int* lwork,
    float* rwork, blas_int* lrwork, blas_int* iwork, blas_int* liwork,
    blas_int* ifail, blas_int* iclustr, float* gap, blas_int* info);

extern "C" void scalapack_pzhegvx(
    blas_int* itype, const char* jobz, const char* range, const char* uplo, blas_int* n,
    std::complex<double> *A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double> *B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* vl, double* vu,  blas_int* il, blas_int* iu, double* abstol,
    blas_int* nfound, blas_int* nzfound, double* W, double* orfac,
    std::complex<double>* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    std::complex<double>* work, blas_int* lwork,
    double* rwork, blas_int* lrwork, blas_int* iwork, blas_int* liwork,
    blas_int* ifail, blas_int* iclustr, double* gap, blas_int* info);

// -----------------------------------------------------------------------------

inline void scalapack_phegvx(
    blas_int* itype, const char* jobz, const char* range, const char* uplo, blas_int* n,
    float *A, blas_int* ia, blas_int* ja, blas_int* descA,
    float *B, blas_int* ib, blas_int* jb, blas_int* descB,
    float* vl, float* vu,  blas_int* il, blas_int* iu, float* abstol,
    blas_int* nfound, blas_int* nzfound, float* W, float* orfac,
    float* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    float* work, blas_int* lwork,
    float* rwork, blas_int* lrwork, blas_int* iwork, blas_int* liwork,
    blas_int* ifail, blas_int* iclustr, float* gap, blas_int* info)
{
    scalapack_pssygvx( itype, jobz, range, uplo, n, A, ia, ja, descA, B, ib, jb, descB, vl, vu, il, iu, abstol, nfound, nzfound, W, orfac, Z, iz, jz, descZ, work, lwork, iwork, liwork, ifail, iclustr, gap, info );
}

inline void scalapack_phegvx(
    blas_int* itype, const char* jobz, const char* range, const char* uplo, blas_int* n,
    double *A, blas_int* ia, blas_int* ja, blas_int* descA,
    double *B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* vl, double* vu,  blas_int* il, blas_int* iu, double* abstol,
    blas_int* nfound, blas_int* nzfound, double* W, double* orfac,
    double* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    double* work, blas_int* lwork,
    double* rwork, blas_int* lrwork, blas_int* iwork, blas_int* liwork,
    blas_int* ifail, blas_int* iclustr, double* gap, blas_int* info)
{
    scalapack_pdsygvx( itype, jobz, range, uplo, n, A, ia, ja, descA, B, ib, jb, descB, vl, vu, il, iu, abstol, nfound, nzfound, W, orfac, Z, iz, jz, descZ, work, lwork, iwork, liwork, ifail, iclustr, gap, info );
}

inline void scalapack_phegvx(
    blas_int* itype, const char* jobz, const char* range, const char* uplo, blas_int* n,
    std::complex<float> *A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float> *B, blas_int* ib, blas_int* jb, blas_int* descB,
    float* vl, float* vu,  blas_int* il, blas_int* iu, float* abstol,
    blas_int* nfound, blas_int* nzfound, float* W, float* orfac,
    std::complex<float>* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    std::complex<float>* work, blas_int* lwork,
    float* rwork, blas_int* lrwork, blas_int* iwork, blas_int* liwork,
    blas_int* ifail, blas_int* iclustr, float* gap, blas_int* info)
{
    scalapack_pchegvx( itype, jobz, range, uplo, n, A, ia, ja, descA, B, ib, jb, descB, vl, vu, il, iu, abstol, nfound, nzfound, W, orfac, Z, iz, jz, descZ, work, lwork, rwork, lrwork, iwork, liwork, ifail, iclustr, gap, info );
}

inline void scalapack_phegvx(
    blas_int* itype, const char* jobz, const char* range, const char* uplo, blas_int* n,
    std::complex<double> *A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double> *B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* vl, double* vu,  blas_int* il, blas_int* iu, double* abstol,
    blas_int* nfound, blas_int* nzfound, double* W, double* orfac,
    std::complex<double>* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    std::complex<double>* work, blas_int* lwork,
    double* rwork, blas_int* lrwork, blas_int* iwork, blas_int* liwork,
    blas_int* ifail, blas_int* iclustr, double* gap, blas_int* info)
{
    scalapack_pzhegvx( itype, jobz, range, uplo, n, A, ia, ja, descA, B, ib, jb, descB, vl, vu, il, iu, abstol, nfound, nzfound, W, orfac, Z, iz, jz, descZ, work, lwork, rwork, lrwork, iwork, liwork, ifail, iclustr, gap, info );
}

template <typename scalar_t>
inline void scalapack_phegvx(
    int64_t itype, const char* jobz, const char* range, const char* uplo, int64_t n,
    scalar_t *A, int64_t ia, int64_t ja, blas_int* descA,
    scalar_t *B, int64_t ib, int64_t jb, blas_int* descB,
    blas::real_type<scalar_t> vl, blas::real_type<scalar_t> vu,
    int64_t il, int64_t iu,
    blas::real_type<scalar_t> abstol,
    int64_t *nfound, int64_t *nzfound,
    blas::real_type<scalar_t>* W,
    blas::real_type<scalar_t> orfac,
    scalar_t* Z, int64_t iz, int64_t jz, blas_int* descZ,
    scalar_t* work, int64_t lwork,
    blas::real_type<scalar_t>* rwork, int64_t lrwork,
    blas_int* iwork, int64_t liwork,
    blas_int* ifail, blas_int* iclustr, blas::real_type<scalar_t>* gap,
    int64_t* info)
{
    blas_int itype_     = int64_to_int(itype);
    blas_int n_     = int64_to_int(n);
    blas_int ia_    = int64_to_int(ia);
    blas_int ja_    = int64_to_int(ja);
    blas_int ib_    = int64_to_int(ib);
    blas_int jb_    = int64_to_int(jb);
    blas_int il_    = int64_to_int(il);
    blas_int iu_    = int64_to_int(iu);
    blas_int nfound_ = int64_to_int(*nfound);
    blas_int nzfound_ = int64_to_int(*nzfound);
    blas_int iz_     = int64_to_int(iz);
    blas_int jz_     = int64_to_int(jz);
    blas_int lwork_     = int64_to_int(lwork);
    blas_int lrwork_     = int64_to_int(lrwork);
    blas_int liwork_     = int64_to_int(liwork);
    blas_int info_     = int64_to_int(*info);
    scalapack_phegvx( &itype_, jobz, range, uplo, &n_, A, &ia_, &ja_, descA, B, &ib_, &jb_, descB, &vl, &vu, &il_, &iu_, &abstol, &nfound_, &nzfound_, W, &orfac, Z, &iz_, &jz_, descZ, work, &lwork_, rwork, &lrwork_, iwork, &liwork_, ifail, iclustr, gap, &info_ );
    *nfound = (int64_t)nfound_;
    *nzfound = (int64_t)nzfound_;
    *info = (int64_t)info_;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pssygst BLAS_FORTRAN_NAME( pssygst, PSSYGST )
#define scalapack_pdsygst BLAS_FORTRAN_NAME( pdsygst, PDSYGST )
#define scalapack_pchegst BLAS_FORTRAN_NAME( pchegst, PCHEGST )
#define scalapack_pzhegst BLAS_FORTRAN_NAME( pzhegst, PZHEGST )

extern "C" void scalapack_pssygst(blas_int* itype, const char* uplo, blas_int* n, float* a, blas_int* ia, blas_int* ja, blas_int* desca, float* b, blas_int* ib, blas_int* jb, blas_int* descb, double* scale, blas_int* info);

extern "C" void scalapack_pdsygst(blas_int* itype, const char* uplo, blas_int* n, double* a, blas_int* ia, blas_int* ja, blas_int* desca, double* b, blas_int* ib, blas_int* jb, blas_int* descb, double* scale, blas_int* info);

extern "C" void scalapack_pchegst(blas_int* itype, const char* uplo, blas_int* n, std::complex<float>* a, blas_int* ia, blas_int* ja, blas_int* desca, std::complex<float>* b, blas_int* ib, blas_int* jb, blas_int* descb, double* scale, blas_int* info);

extern "C" void scalapack_pzhegst(blas_int* itype, const char* uplo, blas_int* n, std::complex<double>* a, blas_int* ia, blas_int* ja, blas_int* desca, std::complex<double>* b, blas_int* ib, blas_int* jb, blas_int* descb, double* scale, blas_int* info);

// -----------------------------------------------------------------------------

inline void scalapack_phegst(blas_int* itype, const char* uplo, blas_int* n, float* a, blas_int* ia, blas_int* ja, blas_int* desca, float* b, blas_int* ib, blas_int* jb, blas_int* descb, double* scale, blas_int* info)
{
    scalapack_pssygst(itype, uplo, n, a, ia, ja, desca, b, ib, jb, descb, scale, info);
}

inline void scalapack_phegst(blas_int* itype, const char* uplo, blas_int* n, double* a, blas_int* ia, blas_int* ja, blas_int* desca, double* b, blas_int* ib, blas_int* jb, blas_int* descb, double* scale, blas_int* info)
{
    scalapack_pdsygst(itype, uplo, n, a, ia, ja, desca, b, ib, jb, descb, scale, info);
}

inline void scalapack_phegst(blas_int* itype, const char* uplo, blas_int* n, std::complex<float>* a, blas_int* ia, blas_int* ja, blas_int* desca, std::complex<float>* b, blas_int* ib, blas_int* jb, blas_int* descb, double* scale, blas_int* info)
{
    scalapack_pchegst(itype, uplo, n, a, ia, ja, desca, b, ib, jb, descb, scale, info);
}

inline void scalapack_phegst(blas_int* itype, const char* uplo, blas_int* n, std::complex<double>* a, blas_int* ia, blas_int* ja, blas_int* desca, std::complex<double>* b, blas_int* ib, blas_int* jb, blas_int* descb, double* scale, blas_int* info)
{
    scalapack_pzhegst(itype, uplo, n, a, ia, ja, desca, b, ib, jb, descb, scale, info);
}

template <typename scalar_t>
inline void scalapack_phegst(int64_t itype, const char* uplo, int64_t n, scalar_t* a, int64_t ia, int64_t ja, int* desca, scalar_t* b, int64_t ib, int64_t jb, int* descb, double* scale, int* info)
{
    int itype_ = int64_to_int(itype);
    int n_ = int64_to_int(n);
    int ia_ = int64_to_int(ia);
    int ja_ = int64_to_int(ja);
    int ib_ = int64_to_int(ib);
    int jb_ = int64_to_int(jb);
    scalapack_phegst(&itype_, uplo, &n_, a, &ia_, &ja_, desca, b, &ib_, &jb_, descb, scale, info);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pslaqge BLAS_FORTRAN_NAME( pslaqge, PSlaqge )
#define scalapack_pdlaqge BLAS_FORTRAN_NAME( pdlaqge, PDlaqge )
#define scalapack_pclaqge BLAS_FORTRAN_NAME( pclaqge, PClaqge )
#define scalapack_pzlaqge BLAS_FORTRAN_NAME( pzlaqge, PZlaqge )

extern "C" {

void scalapack_pslaqge(
    blas_int* m, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* R, float* C,
    float* rowcnd, float* colcnd, float* Amax, char* equed);

void scalapack_pdlaqge(
    blas_int* m, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* R, double* C,
    double* rowcnd, double* colcnd, double* Amax, char* equed);

void scalapack_pclaqge(
    blas_int* m, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* R, float* C,
    float* rowcnd, float* colcnd, float* Amax, char* equed);

void scalapack_pzlaqge(
    blas_int* m, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* R, double* C,
    double* rowcnd, double* colcnd, double* Amax, char* equed);

} // extern C

// -----------------------------------------------------------------------------
inline void scalapack_plaqge(
    blas_int* m, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* R, float* C,
    float* rowcnd, float* colcnd, float* Amax, char* equed)
{
    scalapack_pslaqge( m, n, A, ia, ja, descA,
                       R, C, rowcnd, colcnd, Amax, equed );
}

inline void scalapack_plaqge(
    blas_int* m, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* R, double* C,
    double* rowcnd, double* colcnd, double* Amax, char* equed)
{
    scalapack_pdlaqge( m, n, A, ia, ja, descA,
                       R, C, rowcnd, colcnd, Amax, equed );
}

inline void scalapack_plaqge(
    blas_int* m, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* R, float* C,
    float* rowcnd, float* colcnd, float* Amax, char* equed)
{
    scalapack_pclaqge( m, n, A, ia, ja, descA,
                       R, C, rowcnd, colcnd, Amax, equed );
}

inline void scalapack_plaqge(
    blas_int* m, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* R, double* C,
    double* rowcnd, double* colcnd, double* Amax, char* equed)
{
    scalapack_pzlaqge( m, n, A, ia, ja, descA,
                       R, C, rowcnd, colcnd, Amax, equed );
}

// -----------------------------------------------------------------------------
// equed is an output, hence not const.
template <typename scalar_t>
inline void scalapack_plaqge(
    int64_t m, int64_t n,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    blas::real_type<scalar_t>* R,
    blas::real_type<scalar_t>* C,
    blas::real_type<scalar_t> rowcnd,
    blas::real_type<scalar_t> colcnd,
    blas::real_type<scalar_t> Amax, char* equed)
{
    blas_int m_  = int64_to_int( m  );
    blas_int n_  = int64_to_int( n  );
    blas_int ia_ = int64_to_int( ia );
    blas_int ja_ = int64_to_int( ja );
    scalapack_plaqge( &m_, &n_, A, &ia_, &ja_, descA,
                      R, C, &rowcnd, &colcnd, &Amax, equed );
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_psgecon BLAS_FORTRAN_NAME( psgecon, PSGECON )
#define scalapack_pdgecon BLAS_FORTRAN_NAME( pdgecon, PDGECON )
#define scalapack_pcgecon BLAS_FORTRAN_NAME( pcgecon, PCGECON )
#define scalapack_pzgecon BLAS_FORTRAN_NAME( pzgecon, PZGECON )

extern "C" {

void scalapack_psgecon(
    const char* norm, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* Anorm, float* rcond,
    float* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    int* info);

void scalapack_pdgecon(
    const char* norm, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* Anorm, double* rcond,
    double* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    int* info);

void scalapack_pcgecon(
    const char* norm, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* Anorm, float* rcond,
    std::complex<float>* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    int* info);

void scalapack_pzgecon(
    const char* norm, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* Anorm, double* rcond,
    std::complex<double>* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    int* info);

} // extern C

// -----------------------------------------------------------------------------
inline void scalapack_pgecon(
    const char* norm, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* Anorm, float* rcond,
    float* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    int* info)
{
    scalapack_psgecon( norm, n, A, ia, ja, descA,
                       Anorm, rcond, work, lwork, iwrok, liwork, info);
}

inline void scalapack_pgecon(
    const char* norm, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* Anorm, double* rcond,
    double* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    int* info)
{
    scalapack_pdgecon( norm, n, A, ia, ja, descA,
                       Anorm, rcond, work, lwork, iwrok, liwork, info);
}

inline void scalapack_pgecon(
    const char* norm, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* Anorm, float* rcond,
    std::complex<float>* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    int* info)
{
    scalapack_pcgecon( norm, n, A, ia, ja, descA,
                       Anorm, rcond, work, lwork, iwrok, liwork, info);
}

inline void scalapack_pgecon(
    const char* norm, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* Anorm, double* rcond,
    std::complex<double>* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    int* info)
{
    scalapack_pzgecon( norm, n, A, ia, ja, descA,
                       Anorm, rcond, work, lwork, iwrok, liwork, info);
}

template <typename scalar_t>
inline void scalapack_pgecon(
    const char* norm, int64_t n,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    blas::real_type<scalar_t>* Anorm, blas::real_type<scalar_t>* rcond,
    scalar_t* work, int64_t lwork,
    blas_int* iwork, int64_t liwork,
    int64_t info)
{
    int n_     = int64_to_int(n);
    int ia_    = int64_to_int(ia);
    int ja_    = int64_to_int(ja);
    int lwork_ = int64_to_int(lwork);
    int liwork_ = int64_to_int(liwork);
    int info_  = int64_to_int(info);
    scalapack_pgecon( norm, &n_, A, &ia_, &ja_, descA,
                      Anorm, rcond, work, &lwork_, iwork, &liwork_, &info_);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#define scalapack_pstrcon BLAS_FORTRAN_NAME( pstrcon, PSTRCON )
#define scalapack_pdtrcon BLAS_FORTRAN_NAME( pdtrcon, PDTRCON )
#define scalapack_pctrcon BLAS_FORTRAN_NAME( pctrcon, PCTRCON )
#define scalapack_pztrcon BLAS_FORTRAN_NAME( pztrcon, PZTRCON )

extern "C" {

void scalapack_pstrcon(
    const char* norm, const char* uplo, const char* diag, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* rcond,
    float* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    int* info);

void scalapack_pdtrcon(
    const char* norm, const char* uplo, const char* diag, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* rcond,
    double* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    int* info);

void scalapack_pctrcon(
    const char* norm, const char* uplo, const char* diag, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* rcond,
    std::complex<float>* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    int* info);

void scalapack_pztrcon(
    const char* norm, const char* uplo, const char* diag, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* rcond,
    std::complex<double>* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    int* info);

} // extern C

// -----------------------------------------------------------------------------
inline void scalapack_ptrcon(
    const char* norm, const char* uplo, const char* diag, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* rcond,
    float* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    int* info)
{
    scalapack_pstrcon( norm, uplo, diag, n, A, ia, ja, descA,
                       rcond, work, lwork, iwrok, liwork, info);
}

inline void scalapack_ptrcon(
    const char* norm, const char* uplo, const char* diag, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* rcond,
    double* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    int* info)
{
    scalapack_pdtrcon( norm, uplo, diag, n, A, ia, ja, descA,
                       rcond, work, lwork, iwrok, liwork, info);
}

inline void scalapack_ptrcon(
    const char* norm, const char* uplo, const char* diag, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* rcond,
    std::complex<float>* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    int* info)
{
    scalapack_pctrcon( norm, uplo, diag, n, A, ia, ja, descA,
                       rcond, work, lwork, iwrok, liwork, info);
}

inline void scalapack_ptrcon(
    const char* norm, const char* uplo, const char* diag, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* rcond,
    std::complex<double>* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    int* info)
{
    scalapack_pztrcon( norm, uplo, diag, n, A, ia, ja, descA,
                       rcond, work, lwork, iwrok, liwork, info);
}

template <typename scalar_t>
inline void scalapack_ptrcon(
    const char* norm, const char* uplo, const char* diag, int64_t n,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    blas::real_type<scalar_t>* rcond,
    scalar_t* work, int64_t lwork,
    blas_int* iwork, int64_t liwork,
    int64_t info)
{
    int n_     = int64_to_int(n);
    int ia_    = int64_to_int(ia);
    int ja_    = int64_to_int(ja);
    int lwork_ = int64_to_int(lwork);
    int liwork_ = int64_to_int(liwork);
    int info_  = int64_to_int(info);
    scalapack_ptrcon( norm, uplo, diag, &n_, A, &ia_, &ja_, descA,
                      rcond, work, &lwork_, iwork, &liwork_, &info_);
}

#endif // SLATE_SCALAPACK_WRAPPERS_HH
