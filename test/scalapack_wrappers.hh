// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
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

//==============================================================================
// In general, the only arguments left as blas_int are integer arrays
// such as descriptors, ipiv, iwork.

//==============================================================================
// to_blas_int copied from blaspp/src/blas_internal.hh
// Being in test/scalapack_wrappers.hh, the macro won't pollute
// the namespace when apps #include <slate.hh>.

//------------------------------------------------------------------------------
/// @see to_blas_int
///
inline blas_int to_blas_int_( int64_t x, const char* x_str )
{
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if_msg( x > std::numeric_limits<blas_int>::max(), "%s", x_str );
    }
    return blas_int( x );
}

//----------------------------------------
/// Convert int64_t to blas_int.
/// If blas_int is 64-bit, this does nothing.
/// If blas_int is 32-bit, throws if x > INT_MAX, so conversion would overflow.
///
#define to_blas_int( x ) to_blas_int_( x, #x )

//==============================================================================
// Required CBLACS calls

extern "C" {

void Cblacs_pinfo( blas_int* mypnum, blas_int* nprocs );

void Cblacs_get( blas_int context, blas_int request, blas_int* value );

blas_int Cblacs_gridinit(
    blas_int* context, const char* order, blas_int np_row, blas_int np_col );

void Cblacs_gridinfo(
    blas_int context, blas_int* np_row, blas_int* np_col,
    blas_int* my_row, blas_int* my_col );

void Cblacs_gridexit( blas_int context );

void Cblacs_exit( blas_int error_code );

void Cblacs_abort( blas_int context, blas_int error_code );

}  // extern "C"

//==============================================================================
// Fortran prototype
#define scalapack_descinit BLAS_FORTRAN_NAME( descinit, DESCINIT )

extern "C"
void scalapack_descinit(
    blas_int* desc, blas_int* m, blas_int* n, blas_int* mb, blas_int* nb,
    blas_int* irsrc, blas_int* icsrc, blas_int* ictxt, blas_int* lld,
    blas_int* info );

// High-level C++ wrapper.
inline void scalapack_descinit(
    blas_int* desc, int64_t m, int64_t n, int64_t mb,
    int64_t nb, blas_int irsrc, blas_int icsrc, blas_int ictxt,
    int64_t lld, int64_t* info )
{
    blas_int m_    = to_blas_int( m );
    blas_int n_    = to_blas_int( n );
    blas_int mb_   = to_blas_int( mb );
    blas_int nb_   = to_blas_int( nb );
    blas_int lld_  = blas::max( 1, to_blas_int( lld ) );
    blas_int info_ = 0;
    scalapack_descinit(
        desc, &m_, &n_, &mb_, &nb_, &irsrc, &icsrc, &ictxt,
        &lld_, &info_ );
    *info = info_;
}

//==============================================================================
// Fortran prototype
#define scalapack_numroc BLAS_FORTRAN_NAME( numroc, NUMROC )

extern "C"
blas_int scalapack_numroc(
    blas_int* n, blas_int* nb, blas_int* iproc, blas_int* isrcproc,
    blas_int* nprocs );

// High-level C++ wrapper
inline int64_t scalapack_numroc(
    int64_t n, int64_t nb, blas_int iproc, blas_int isrcproc, blas_int nprocs )
{
    blas_int n_    = to_blas_int( n );
    blas_int nb_   = to_blas_int( nb );
    blas_int nroc_ = scalapack_numroc( &n_, &nb_, &iproc, &isrcproc, &nprocs );
    return int64_t( nroc_ );
}

//==============================================================================
// Fortran prototype
#define scalapack_ilcm BLAS_FORTRAN_NAME( ilcm, ILCM )

extern "C"
blas_int scalapack_ilcm( blas_int* a, blas_int* b );

// High-level C++ wrapper
inline int64_t scalapack_ilcm( int64_t a, int64_t b )
{
    blas_int a_ = to_blas_int( a );
    blas_int b_ = to_blas_int( b );
    return scalapack_ilcm( &a_, &b_ );
}

//==============================================================================
// Fortran prototype
#define scalapack_indxg2p BLAS_FORTRAN_NAME( indxg2p, INDXG2P )

extern "C"
blas_int scalapack_indxg2p(
    blas_int* indxglob, blas_int* nb, blas_int* iproc, blas_int* isrcproc,
    blas_int* nprocs );

//==============================================================================
// Fortran prototype
#define scalapack_indxg2l BLAS_FORTRAN_NAME( indxg2l, INDXG2L )

extern "C"
blas_int scalapack_indxg2l(
    blas_int* indxglob, blas_int* nb, blas_int* iproc, blas_int* isrcproc,
    blas_int* nprocs );

//==============================================================================
// Fortran prototypes
#define scalapack_pslange BLAS_FORTRAN_NAME( pslange, PSLANGE )
#define scalapack_pdlange BLAS_FORTRAN_NAME( pdlange, PDLANGE )
#define scalapack_pclange BLAS_FORTRAN_NAME( pclange, PCLANGE )
#define scalapack_pzlange BLAS_FORTRAN_NAME( pzlange, PZLANGE )

extern "C" {

float scalapack_pslange(
    const char* norm, blas_int* m, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* work );

double scalapack_pdlange(
    const char* norm, blas_int* m, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* work );

float scalapack_pclange(
    const char* norm, blas_int* m, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* work );

double scalapack_pzlange(
    const char* norm, blas_int* m, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* work );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline float scalapack_plange(
    const char* norm, blas_int* m, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* work )
{
    return scalapack_pslange(
        norm, m, n,
        A, ia, ja, descA,
        work );
}

inline double scalapack_plange(
    const char* norm, blas_int* m, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* work )
{
    return scalapack_pdlange(
        norm, m, n,
        A, ia, ja, descA,
        work );
}

inline float scalapack_plange(
    const char* norm, blas_int* m, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* work )
{
    return scalapack_pclange(
        norm, m, n,
        A, ia, ja, descA,
        work );
}

inline double scalapack_plange(
    const char* norm, blas_int* m, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* work )
{
    return scalapack_pzlange(
        norm, m, n,
        A, ia, ja, descA,
        work );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
blas::real_type<scalar_t> scalapack_plange(
    const char* norm, int64_t m, int64_t n,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    blas::real_type<scalar_t>* work )
{
    blas_int m_  = to_blas_int( m );
    blas_int n_  = to_blas_int( n );
    blas_int ia_ = to_blas_int( ia );
    blas_int ja_ = to_blas_int( ja );
    return scalapack_plange(
        norm, &m_, &n_,
        A, &ia_, &ja_, descA,
        work );
}

//==============================================================================
// Fortran prototypes
#define scalapack_psgeadd BLAS_FORTRAN_NAME( psgeadd, PSGEADD )
#define scalapack_pdgeadd BLAS_FORTRAN_NAME( pdgeadd, PDGEADD )
#define scalapack_pcgeadd BLAS_FORTRAN_NAME( pcgeadd, PCGEADD )
#define scalapack_pzgeadd BLAS_FORTRAN_NAME( pzgeadd, PZGEADD )

extern "C" {

float scalapack_psgeadd(
    const char* transA, blas_int* m, blas_int* n,
    float* alpha, float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* beta,  float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

double scalapack_pdgeadd(
    const char* transA, blas_int* m, blas_int* n,
    double* alpha, double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* beta,  double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

float scalapack_pcgeadd(
    const char* transA, blas_int* m, blas_int* n,
    std::complex<float>* alpha,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* beta,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

double scalapack_pzgeadd(
    const char* transA, blas_int* m, blas_int* n,
    std::complex<double>* alpha,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* beta,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_pgeadd(
    const char* transA, blas_int* m, blas_int* n,
    float* alpha, float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* beta,  float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info )
{
    scalapack_psgeadd(
        transA, m, n,
        alpha, A, ia, ja, descA,
        beta,  B, ib, jb, descB,
        info );
}

inline void scalapack_pgeadd(
    const char* transA, blas_int* m, blas_int* n,
    double* alpha, double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* beta,  double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info )
{
    scalapack_pdgeadd(
        transA, m, n,
        alpha, A, ia, ja, descA,
        beta,  B, ib, jb, descB,
        info );
}

inline void scalapack_pgeadd(
    const char* transA, blas_int* m, blas_int* n,
    std::complex<float>* alpha,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* beta,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info )
{
    scalapack_pcgeadd(
        transA, m, n,
        alpha, A, ia, ja, descA,
        beta,  B, ib, jb, descB,
        info );
}

inline void scalapack_pgeadd(
    const char* transA, blas_int* m, blas_int* n,
    std::complex<double>* alpha,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* beta,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info )
{
    scalapack_pzgeadd(
        transA, m, n,
        alpha, A, ia, ja, descA,
        beta,  B, ib, jb, descB,
        info );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_pgeadd(
    const char* transA, int64_t m, int64_t n,
    scalar_t alpha,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    scalar_t beta,
    scalar_t* B, int64_t ib, int64_t jb, blas_int* descB,
    int64_t* info )
{
    blas_int m_  = to_blas_int( m );
    blas_int n_  = to_blas_int( n );
    blas_int ia_ = to_blas_int( ia );
    blas_int ja_ = to_blas_int( ja );
    blas_int ib_ = to_blas_int( ib );
    blas_int jb_ = to_blas_int( jb );
    blas_int info_ = 0;
    scalapack_pgeadd(
        transA, &m_, &n_,
        &alpha, A, &ia_, &ja_, descA,
        &beta,  B, &ib_, &jb_, descB,
        &info_ );
    *info = info_;
}

//==============================================================================
// Fortran prototypes
#define scalapack_pstradd BLAS_FORTRAN_NAME( pstradd, PSGEADD )
#define scalapack_pdtradd BLAS_FORTRAN_NAME( pdtradd, PDGEADD )
#define scalapack_pctradd BLAS_FORTRAN_NAME( pctradd, PCGEADD )
#define scalapack_pztradd BLAS_FORTRAN_NAME( pztradd, PZGEADD )

extern "C" {

float scalapack_pstradd(
    const char* uplo, const char* transA, blas_int* m, blas_int* n,
    float* alpha, float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* beta,  float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

double scalapack_pdtradd(
    const char* uplo, const char* transA, blas_int* m, blas_int* n,
    double* alpha, double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* beta,  double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

float scalapack_pctradd(
    const char* uplo, const char* transA, blas_int* m, blas_int* n,
    std::complex<float>* alpha,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* beta,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

double scalapack_pztradd(
    const char* uplo, const char* transA, blas_int* m, blas_int* n,
    std::complex<double>* alpha,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* beta,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_ptradd(
    const char* uplo, const char* transA, blas_int* m, blas_int* n,
    float* alpha, float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* beta,  float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info )
{
    scalapack_pstradd(
        uplo, transA, m, n,
        alpha, A, ia, ja, descA,
        beta,  B, ib, jb, descB, info );
}

inline void scalapack_ptradd(
    const char* uplo, const char* transA, blas_int* m, blas_int* n,
    double* alpha, double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* beta,  double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info )
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
    blas_int* info )
{
    scalapack_pctradd(
        uplo, transA, m, n,
        alpha, A, ia, ja, descA,
        beta,  B, ib, jb, descB, info );
}

inline void scalapack_ptradd(
    const char* uplo, const char* transA, blas_int* m, blas_int* n,
    std::complex<double>* alpha,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* beta,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info )
{
    scalapack_pztradd(
        uplo, transA, m, n,
        alpha, A, ia, ja, descA,
        beta,  B, ib, jb, descB, info );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_ptradd(
    const char* uplo, const char* transA, int64_t m, int64_t n,
    scalar_t alpha, scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    scalar_t beta,  scalar_t* B, int64_t ib, int64_t jb, blas_int* descB,
    int64_t* info )
{
    blas_int m_  = to_blas_int( m  );
    blas_int n_  = to_blas_int( n  );
    blas_int ia_ = to_blas_int( ia );
    blas_int ja_ = to_blas_int( ja );
    blas_int ib_ = to_blas_int( ib );
    blas_int jb_ = to_blas_int( jb );
    blas_int info_ = 0;
    scalapack_ptradd(
        uplo, transA, &m_, &n_,
        &alpha, A, &ia_, &ja_, descA,
        &beta,  B, &ib_, &jb_, descB, &info_ );
    *info = info_;
}

//==============================================================================
// Fortran prototypes
#define scalapack_pslascl BLAS_FORTRAN_NAME( pslascl, PSLASCL )
#define scalapack_pdlascl BLAS_FORTRAN_NAME( pdlascl, PDLASCL )
#define scalapack_pclascl BLAS_FORTRAN_NAME( pclascl, PCLASCL )
#define scalapack_pzlascl BLAS_FORTRAN_NAME( pzlascl, PZLASCL )

extern "C" {

float scalapack_pslascl(
    const char* uplo, float* numer,
    float* denom,  blas_int* m, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* info );

double scalapack_pdlascl(
    const char* uplo, double* numer,
    double* denom, blas_int* m, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* info );

float scalapack_pclascl(
    const char* uplo, float* numer,
    float* denom, blas_int* m, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* info );

double scalapack_pzlascl(
    const char* uplo, double* numer,
    double* denom, blas_int* m, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* info );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_plascl(
    const char* uplo, float* numer,  float* denom,
    blas_int* m, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* info )
{
    scalapack_pslascl(
        uplo, denom, numer, m, n,
        A, ia, ja, descA,
        info );
}

inline void scalapack_plascl(
    const char* uplo, double* numer,  double* denom,
    blas_int* m, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* info )
{
    scalapack_pdlascl(
        uplo, denom, numer, m, n,
        A, ia, ja, descA,
        info );
}

inline void scalapack_plascl(
    const char* uplo, float* numer,  float* denom,
    blas_int* m, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* info )
{
    scalapack_pclascl(
        uplo, denom, numer, m, n,
        A, ia, ja, descA,
        info );
}

inline void scalapack_plascl(
    const char* uplo, double* numer, double* denom,
    blas_int* m, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* info )
{
    scalapack_pzlascl(
        uplo, denom, numer, m, n,
        A, ia, ja, descA,
        info );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_plascl(
    const char* uplo,
    blas::real_type<scalar_t> numer, blas::real_type<scalar_t> denom,
    int64_t m, int64_t n,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    int64_t* info )
{
    blas_int m_  = to_blas_int( m );
    blas_int n_  = to_blas_int( n );
    blas_int ia_ = to_blas_int( ia );
    blas_int ja_ = to_blas_int( ja );
    blas_int info_ = 0;
    scalapack_plascl(
        uplo, &numer, &denom, &m_, &n_,
        A, &ia_, &ja_, descA,
        &info_ );
    *info = info_;
}

//==============================================================================
// Fortran prototypes
#define scalapack_pspotrf BLAS_FORTRAN_NAME( pspotrf, PSPOTRF )
#define scalapack_pdpotrf BLAS_FORTRAN_NAME( pdpotrf, PDPOTRF )
#define scalapack_pcpotrf BLAS_FORTRAN_NAME( pcpotrf, PCPOTRF )
#define scalapack_pzpotrf BLAS_FORTRAN_NAME( pzpotrf, PZPOTRF )

extern "C" {

void scalapack_pspotrf(
    const char* uplo, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* info );

void scalapack_pdpotrf(
    const char* uplo, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* info );

void scalapack_pcpotrf(
    const char* uplo, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* info );

void scalapack_pzpotrf(
    const char* uplo, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* info );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_ppotrf(
    const char* uplo, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* info )
{
    scalapack_pspotrf(
        uplo, n, A, ia, ja, descA, info );
}

inline void scalapack_ppotrf(
    const char* uplo, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* info )
{
    scalapack_pdpotrf(
        uplo, n, A, ia, ja, descA, info );
}

inline void scalapack_ppotrf(
    const char* uplo, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* info )
{
    scalapack_pcpotrf(
        uplo, n, A, ia, ja, descA, info );
}

inline void scalapack_ppotrf(
    const char* uplo, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* info )
{
    scalapack_pzpotrf(
        uplo, n, A, ia, ja, descA, info );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_ppotrf(
    const char* uplo, int64_t n,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    int64_t* info )
{
    blas_int n_  = to_blas_int( n );
    blas_int ia_ = to_blas_int( ia );
    blas_int ja_ = to_blas_int( ja );
    blas_int info_ = 0;
    scalapack_ppotrf(
        uplo, &n_, A, &ia_, &ja_, descA, &info_ );
    *info = info_;
}

//==============================================================================
// Fortran prototypes
#define scalapack_pspotrs BLAS_FORTRAN_NAME( pspotrs, PSPOTRS )
#define scalapack_pdpotrs BLAS_FORTRAN_NAME( pdpotrs, PDPOTRS )
#define scalapack_pcpotrs BLAS_FORTRAN_NAME( pcpotrs, PCPOTRS )
#define scalapack_pzpotrs BLAS_FORTRAN_NAME( pzpotrs, PZPOTRS )

extern "C" {

void scalapack_pspotrs(
    const char* uplo, blas_int* n, blas_int* nrhs,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

void scalapack_pdpotrs(
    const char* uplo, blas_int* n, blas_int* nrhs,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

void scalapack_pcpotrs(
    const char* uplo, blas_int* n, blas_int* nrhs,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

void scalapack_pzpotrs(
    const char* uplo, blas_int* n, blas_int* nrhs,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_ppotrs(
    const char* uplo, blas_int* n, blas_int* nrhs,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info )
{
    scalapack_pspotrs(
        uplo, n, nrhs,
        A, ia, ja, descA,
        B, ib, jb, descB,
        info );
}

inline void scalapack_ppotrs(
    const char* uplo, blas_int* n, blas_int* nrhs,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info )
{
    scalapack_pdpotrs(
        uplo, n, nrhs,
        A, ia, ja, descA,
        B, ib, jb, descB,
        info );
}

inline void scalapack_ppotrs(
    const char* uplo, blas_int* n, blas_int* nrhs,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info )
{
    scalapack_pcpotrs(
        uplo, n, nrhs,
        A, ia, ja, descA,
        B, ib, jb, descB,
        info );
}

inline void scalapack_ppotrs(
    const char* uplo, blas_int* n, blas_int* nrhs,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info )
{
    scalapack_pzpotrs(
        uplo, n, nrhs,
        A, ia, ja, descA,
        B, ib, jb, descB,
        info );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_ppotrs(
    const char* uplo, int64_t n, int64_t nrhs,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    scalar_t* B, int64_t ib, int64_t jb, blas_int* descB,
    int64_t* info )
{
    blas_int n_    = to_blas_int( n );
    blas_int nrhs_ = to_blas_int( nrhs );
    blas_int ia_   = to_blas_int( ia );
    blas_int ja_   = to_blas_int( ja );
    blas_int ib_   = to_blas_int( ib );
    blas_int jb_   = to_blas_int( jb );
    blas_int info_ = 0;
    scalapack_ppotrs(
        uplo, &n_, &nrhs_,
        A, &ia_, &ja_, descA,
        B, &ib_, &jb_, descB,
        &info_ );
    *info = info_;
}

//==============================================================================
// Fortran prototypes
#define scalapack_psposv BLAS_FORTRAN_NAME( psposv, PSPOSV )
#define scalapack_pdposv BLAS_FORTRAN_NAME( pdposv, PDPOSV )
#define scalapack_pcposv BLAS_FORTRAN_NAME( pcposv, PCPOSV )
#define scalapack_pzposv BLAS_FORTRAN_NAME( pzposv, PZPOSV )

extern "C" {

void scalapack_psposv(
    const char* uplo, blas_int* n, blas_int* nrhs,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

void scalapack_pdposv(
    const char* uplo, blas_int* n, blas_int* nrhs,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

void scalapack_pcposv(
    const char* uplo, blas_int* n, blas_int* nrhs,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

void scalapack_pzposv(
    const char* uplo, blas_int* n, blas_int* nrhs,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_pposv(
    const char* uplo, blas_int* n, blas_int* nrhs,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info )
{
    scalapack_psposv(
        uplo, n, nrhs,
        A, ia, ja, descA,
        B, ib, jb, descB,
        info );
}

inline void scalapack_pposv(
    const char* uplo, blas_int* n, blas_int* nrhs,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info )
{
    scalapack_pdposv(
        uplo, n, nrhs,
        A, ia, ja, descA,
        B, ib, jb, descB,
        info );
}

inline void scalapack_pposv(
    const char* uplo, blas_int* n, blas_int* nrhs,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info )
{
    scalapack_pcposv(
        uplo, n, nrhs,
        A, ia, ja, descA,
        B, ib, jb, descB,
        info );
}

inline void scalapack_pposv(
    const char* uplo, blas_int* n, blas_int* nrhs,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info )
{
    scalapack_pzposv(
        uplo, n, nrhs,
        A, ia, ja, descA,
        B, ib, jb, descB,
        info );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_pposv(
    const char* uplo, int64_t n, int64_t nrhs,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    scalar_t* B, int64_t ib, int64_t jb, blas_int* descB,
    int64_t* info )
{
    blas_int n_    = to_blas_int( n );
    blas_int nrhs_ = to_blas_int( nrhs );
    blas_int ia_   = to_blas_int( ia );
    blas_int ja_   = to_blas_int( ja );
    blas_int ib_   = to_blas_int( ib );
    blas_int jb_   = to_blas_int( jb );
    blas_int info_ = 0;
    scalapack_pposv(
        uplo, &n_, &nrhs_,
        A, &ia_, &ja_, descA,
        B, &ib_, &jb_, descB,
        &info_ );
    *info = info_;
}

//==============================================================================
// Fortran prototypes
#define scalapack_pslansy BLAS_FORTRAN_NAME( pslansy, PSLANSY )
#define scalapack_pdlansy BLAS_FORTRAN_NAME( pdlansy, PDLANSY )
#define scalapack_pclansy BLAS_FORTRAN_NAME( pclansy, PCLANSY )
#define scalapack_pzlansy BLAS_FORTRAN_NAME( pzlansy, PZLANSY )

extern "C" {

float scalapack_pslansy(
    const char* norm, const char* uplo,
    blas_int* n, float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* work );

double scalapack_pdlansy(
    const char* norm, const char* uplo,
    blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* work );

float scalapack_pclansy(
    const char* norm, const char* uplo, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* work );

double scalapack_pzlansy(
    const char* norm, const char* uplo, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* work );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline float scalapack_plansy(
    const char* norm, const char* uplo, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* work )
{
    return scalapack_pslansy(
        norm, uplo, n,
        A, ia, ja, descA,
        work );
}

inline double scalapack_plansy(
    const char* norm, const char* uplo, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* work )
{
    return scalapack_pdlansy(
        norm, uplo, n,
        A, ia, ja, descA,
        work );
}

inline float scalapack_plansy(
    const char* norm, const char* uplo, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* work )
{
    return scalapack_pclansy(
        norm, uplo, n,
        A, ia, ja, descA,
        work );
}

inline double scalapack_plansy(
    const char* norm, const char* uplo, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* work )
{
    return scalapack_pzlansy(
        norm, uplo, n,
        A, ia, ja, descA,
        work );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
double scalapack_plansy(
    const char* norm, const char* uplo, int64_t n,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    blas::real_type<scalar_t>* work )
{
    blas_int n_  = to_blas_int( n );
    blas_int ia_ = to_blas_int( ia );
    blas_int ja_ = to_blas_int( ja );
    return scalapack_plansy(
        norm, uplo, &n_,
        A, &ia_, &ja_, descA,
        work );
}

//==============================================================================
// Fortran prototypes
#define scalapack_pclanhe BLAS_FORTRAN_NAME( pclanhe, PCLANHE )
#define scalapack_pzlanhe BLAS_FORTRAN_NAME( pzlanhe, PZLANHE )

extern "C" {

float scalapack_pclanhe(
    const char* norm, const char* uplo, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* work );

double scalapack_pzlanhe(
    const char* norm, const char* uplo, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* work );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline float scalapack_planhe(
    const char* norm, const char* uplo, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* work )
{
    return scalapack_pslansy(
        norm, uplo, n,
        A, ia, ja, descA,
        work );
}

inline double scalapack_planhe(
    const char* norm, const char* uplo, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* work )
{
    return scalapack_pdlansy(
        norm, uplo, n,
        A, ia, ja, descA,
        work );
}

inline float scalapack_planhe(
    const char* norm, const char* uplo, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* work )
{
    return scalapack_pclanhe(
        norm, uplo, n,
        A, ia, ja, descA,
        work );
}

inline double scalapack_planhe(
    const char* norm, const char* uplo, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* work )
{
    return scalapack_pzlanhe(
        norm, uplo, n,
        A, ia, ja, descA,
        work );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
double scalapack_planhe(
    const char* norm, const char* uplo, int64_t n,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    blas::real_type<scalar_t>* work )
{
    blas_int n_  = to_blas_int( n );
    blas_int ia_ = to_blas_int( ia );
    blas_int ja_ = to_blas_int( ja );
    return scalapack_planhe(
        norm, uplo, &n_,
        A, &ia_, &ja_, descA,
        work );
}

//==============================================================================
// Fortran prototypes
#define scalapack_psgemm BLAS_FORTRAN_NAME( psgemm, PSGEMM )
#define scalapack_pdgemm BLAS_FORTRAN_NAME( pdgemm, PDGEMM )
#define scalapack_pcgemm BLAS_FORTRAN_NAME( pcgemm, PCGEMM )
#define scalapack_pzgemm BLAS_FORTRAN_NAME( pzgemm, PZGEMM )

extern "C" {

void scalapack_psgemm(
    const char* transA, const char* transB,
    blas_int* m, blas_int* n, blas_int* k,
    float* alpha,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    float* beta,
    float* C, blas_int* ic, blas_int* jc, blas_int* descC );

void scalapack_pdgemm(
    const char* transA, const char* transB,
    blas_int* m, blas_int* n, blas_int* k,
    double* alpha,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* beta,
    double* C, blas_int* ic, blas_int* jc, blas_int* descC );

void scalapack_pcgemm(
    const char* transA, const char* transB,
    blas_int* m, blas_int* n, blas_int* k,
    std::complex<float>* alpha,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    std::complex<float>* beta,
    std::complex<float>* C, blas_int* ic, blas_int* jc, blas_int* descC );

void scalapack_pzgemm(
    const char* transA, const char* transB,
    blas_int* m, blas_int* n, blas_int* k,
    std::complex<double>* alpha,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    std::complex<double>* beta,
    std::complex<double>* C, blas_int* ic, blas_int* jc, blas_int* descC );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_pgemm(
    const char* transA, const char* transB,
    blas_int* m, blas_int* n, blas_int* k,
    float* alpha,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    float* beta,
    float* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_psgemm(
        transA, transB, m, n, k, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB,
        beta,
        C, ic, jc, descC );
}

inline void scalapack_pgemm(
    const char* transA, const char* transB,
    blas_int* m, blas_int* n, blas_int* k,
    double* alpha,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* beta,
    double* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pdgemm(
        transA, transB, m, n, k, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB,
        beta,
        C, ic, jc, descC );
}

inline void scalapack_pgemm(
    const char* transA, const char* transB,
    blas_int* m, blas_int* n, blas_int* k,
    std::complex<float>* alpha,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    std::complex<float>* beta,
    std::complex<float>* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pcgemm(
        transA, transB, m, n, k, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB,
        beta,
        C, ic, jc, descC );
}

inline void scalapack_pgemm(
    const char* transA, const char* transB,
    blas_int* m, blas_int* n, blas_int* k,
    std::complex<double>* alpha,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    std::complex<double>* beta,
    std::complex<double>* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pzgemm(
        transA, transB, m, n, k, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB,
        beta,
        C, ic, jc, descC );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_pgemm(
    const char* transA, const char* transB, int64_t m,
    int64_t n, int64_t k,
    scalar_t alpha,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    scalar_t* B, int64_t ib, int64_t jb, blas_int* descB,
    scalar_t beta,
    scalar_t* C, int64_t ic, int64_t jc, blas_int* descC )
{
    blas_int m_  = to_blas_int( m );
    blas_int n_  = to_blas_int( n );
    blas_int k_  = to_blas_int( k );
    blas_int ia_ = to_blas_int( ia );
    blas_int ja_ = to_blas_int( ja );
    blas_int ib_ = to_blas_int( ib );
    blas_int jb_ = to_blas_int( jb );
    blas_int ic_ = to_blas_int( ic );
    blas_int jc_ = to_blas_int( jc );
    scalapack_pgemm(
        transA, transB, &m_, &n_, &k_, &alpha,
        A, &ia_, &ja_, descA,
        B, &ib_, &jb_, descB,
        &beta,
        C, &ic_, &jc_, descC );
}

//==============================================================================
// Fortran prototypes
#define scalapack_pssymm BLAS_FORTRAN_NAME( pssymm, PSSYMM )
#define scalapack_pdsymm BLAS_FORTRAN_NAME( pdsymm, PDSYMM )
#define scalapack_pcsymm BLAS_FORTRAN_NAME( pcsymm, PCSYMM )
#define scalapack_pzsymm BLAS_FORTRAN_NAME( pzsymm, PZSYMM )

extern "C" {

void scalapack_pssymm(
    const char* side, const char* uplo, blas_int* m, blas_int* n,
    float* alpha,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    float* beta,
    float* C, blas_int* ic, blas_int* jc, blas_int* descC );

void scalapack_pdsymm(
    const char* side, const char* uplo, blas_int* m, blas_int* n,
    double* alpha,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* beta,
    double* C, blas_int* ic, blas_int* jc, blas_int* descC );

void scalapack_pcsymm(
    const char* side, const char* uplo, blas_int* m, blas_int* n,
    const std::complex<float>* alpha,
    const std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    const std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    const std::complex<float>* beta,
    const std::complex<float>* C, blas_int* ic, blas_int* jc, blas_int* descC );

void scalapack_pzsymm(
    const char* side, const char* uplo, blas_int* m, blas_int* n,
    const std::complex<double>* alpha,
    const std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    const std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    const std::complex<double>* beta,
    const std::complex<double>* C, blas_int* ic, blas_int* jc, blas_int* descC );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_psymm(
    const char* side, const char* uplo, blas_int* m, blas_int* n,
    float* alpha,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    float* beta,
    float* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pssymm(
        side, uplo, m, n, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB,
        beta,
        C, ic, jc, descC );
}

inline void scalapack_psymm(
    const char* side, const char* uplo, blas_int* m, blas_int* n,
    double* alpha,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* beta,
    double* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pdsymm(
        side, uplo, m, n, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB,
        beta,
        C, ic, jc, descC );
}

inline void scalapack_psymm(
    const char* side, const char* uplo, blas_int* m, blas_int* n,
    const std::complex<float>* alpha,
    const std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    const std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    const std::complex<float>* beta,
    const std::complex<float>* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pcsymm(
        side, uplo, m, n, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB,
        beta,
        C, ic, jc, descC );
}

inline void scalapack_psymm(
    const char* side, const char* uplo, blas_int* m, blas_int* n,
    const std::complex<double>* alpha,
    const std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    const std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    const std::complex<double>* beta,
    const std::complex<double>* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pzsymm(
        side, uplo, m, n, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB,
        beta,
        C, ic, jc, descC );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_psymm(
    const char* side, const char* uplo, int64_t m, int64_t n,
    scalar_t alpha,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    scalar_t* B, int64_t ib, int64_t jb, blas_int* descB,
    scalar_t beta,
    scalar_t* C, int64_t ic, int64_t jc, blas_int* descC )
{
    blas_int m_  = to_blas_int( m );
    blas_int n_  = to_blas_int( n );
    blas_int ia_ = to_blas_int( ia );
    blas_int ja_ = to_blas_int( ja );
    blas_int ib_ = to_blas_int( ib );
    blas_int jb_ = to_blas_int( jb );
    blas_int ic_ = to_blas_int( ic );
    blas_int jc_ = to_blas_int( jc );
    scalapack_psymm(
        side, uplo, &m_, &n_, &alpha,
        A, &ia_, &ja_, descA,
        B, &ib_, &jb_, descB,
        &beta,
        C, &ic_, &jc_, descC );
}

//==============================================================================
// Fortran prototypes
#define scalapack_pstrmm BLAS_FORTRAN_NAME( pstrmm, PSTRMM )
#define scalapack_pdtrmm BLAS_FORTRAN_NAME( pdtrmm, PDTRMM )
#define scalapack_pctrmm BLAS_FORTRAN_NAME( pctrmm, PCTRMM )
#define scalapack_pztrmm BLAS_FORTRAN_NAME( pztrmm, PZTRMM )

extern "C" {

void scalapack_pstrmm(
    const char* side, const char* uplo,
    const char* transA, const char* diag,
    const blas_int* m, const blas_int* n,
    const float* alpha, const float* A,
    const blas_int* ia, const blas_int* ja,
    const blas_int* descA,
    float* B,
    const blas_int* ib, const blas_int* jb,
    const blas_int* descB );

void scalapack_pdtrmm(
    const char* side, const char* uplo,
    const char* transA, const char* diag,
    const blas_int* m, const blas_int* n,
    const double* alpha, const double* A,
    const blas_int* ia, const blas_int* ja,
    const blas_int* descA,
    double* B,
    const blas_int* ib, const blas_int* jb,
    const blas_int* descB );

void scalapack_pctrmm(
    const char* side, const char* uplo,
    const char* transA, const char* diag,
    const blas_int* m, const blas_int* n,
    const std::complex<float>* alpha,
    const std::complex<float>* A,
    const blas_int* ia, const blas_int* ja,
    const blas_int* descA,
    std::complex<float>* B,
    const blas_int* ib, const blas_int* jb,
    const blas_int* descB );

void scalapack_pztrmm(
    const char* side, const char* uplo,
    const char* transA, const char* diag,
    const blas_int* m, const blas_int* n,
    const std::complex<double>* alpha,
    const std::complex<double>* A,
    const blas_int* ia, const blas_int* ja,
    const blas_int* descA,
    std::complex<double>* B,
    const blas_int* ib, const blas_int* jb,
    const blas_int* descB );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_ptrmm(
    const char* side, const char* uplo, const char* transA, const char* diag,
    const blas_int* m, const blas_int* n,
    const float* alpha,
    const float* A, const blas_int* ia, const blas_int* ja,
    const blas_int* descA,
    float* B, const blas_int* ib, const blas_int* jb, const blas_int* descB )
{
    scalapack_pstrmm(
        side, uplo, transA, diag, m, n, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB );
}

inline void scalapack_ptrmm(
    const char* side, const char* uplo, const char* transA, const char* diag,
    const blas_int* m, const blas_int* n,
    const double* alpha,
    const double* A, const blas_int* ia, const blas_int* ja,
    const blas_int* descA,
    double* B, const blas_int* ib, const blas_int* jb, const blas_int* descB )
{
    scalapack_pdtrmm(
        side, uplo, transA, diag, m, n, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB );
}

inline void scalapack_ptrmm(
    const char* side, const char* uplo, const char* transA, const char* diag,
    const blas_int* m, const blas_int* n,
    const std::complex<float>* alpha,
    const std::complex<float>* A, const blas_int* ia, const blas_int* ja,
    const blas_int* descA,
    std::complex<float>* B, const blas_int* ib, const blas_int* jb,
    const blas_int* descB )
{
    scalapack_pctrmm(
        side, uplo, transA, diag, m, n, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB );
}

inline void scalapack_ptrmm(
    const char* side, const char* uplo, const char* transA, const char* diag,
    const blas_int* m, const blas_int* n,
    const std::complex<double>* alpha,
    const std::complex<double>* A, const blas_int* ia, const blas_int* ja,
    const blas_int* descA,
    std::complex<double>* B, const blas_int* ib, const blas_int* jb,
    const blas_int* descB )
{
    scalapack_pztrmm(
        side, uplo, transA, diag, m, n, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_ptrmm(
    const char* side, const char* uplo, const char* transA, const char* diag,
    int64_t m, int64_t n,
    scalar_t alpha,
    scalar_t* A, int64_t ia, int64_t ja, const blas_int* descA,
    scalar_t* B, int64_t ib, int64_t jb, const blas_int* descB )
{
    blas_int m_  = to_blas_int( m );
    blas_int n_  = to_blas_int( n );
    blas_int ia_ = to_blas_int( ia );
    blas_int ja_ = to_blas_int( ja );
    blas_int ib_ = to_blas_int( ib );
    blas_int jb_ = to_blas_int( jb );
    scalapack_ptrmm(
        side, uplo, transA, diag, &m_, &n_, &alpha,
        A, &ia_, &ja_, descA,
        B, &ib_, &jb_, descB );
}

//==============================================================================
// Fortran prototypes
#define scalapack_pssyr2k BLAS_FORTRAN_NAME( pssyr2k, PSSYR2K )
#define scalapack_pdsyr2k BLAS_FORTRAN_NAME( pdsyr2k, PDSYR2K )
#define scalapack_pcsyr2k BLAS_FORTRAN_NAME( pcsyr2k, PCSYR2K )
#define scalapack_pzsyr2k BLAS_FORTRAN_NAME( pzsyr2k, PZSYR2K )

extern "C" {

void scalapack_pssyr2k(
    const char* uplo, const char* trans,
    blas_int* n, blas_int* k, float* alpha,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    float* beta,
    float* C, blas_int* ic, blas_int* jc, blas_int* descC );

void scalapack_pdsyr2k(
    const char* uplo, const char* trans,
    blas_int* n, blas_int* k, double* alpha,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* beta,
    double* C, blas_int* ic, blas_int* jc, blas_int* descC );

void scalapack_pcsyr2k(
    const char* uplo, const char* trans,
    blas_int* n, blas_int* k,
    std::complex<float>* alpha,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    std::complex<float>* beta,
    std::complex<float>* C, blas_int* ic, blas_int* jc, blas_int* descC );

void scalapack_pzsyr2k(
    const char* uplo, const char* trans,
    blas_int* n, blas_int* k,
    std::complex<double>* alpha,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    std::complex<double>* beta,
    std::complex<double>* C, blas_int* ic, blas_int* jc, blas_int* descC );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_psyr2k(
    const char* uplo, const char* trans, blas_int* n, blas_int* k,
    float* alpha,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    float* beta,
    float* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pssyr2k(
        uplo, trans, n, k, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB,
        beta,
        C, ic, jc, descC );
}

inline void scalapack_psyr2k(
    const char* uplo, const char* trans, blas_int* n, blas_int* k,
    double* alpha,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* beta,
    double* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pdsyr2k(
        uplo, trans, n, k, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB,
        beta,
        C, ic, jc, descC );
}

inline void scalapack_psyr2k(
    const char* uplo, const char* trans, blas_int* n, blas_int* k,
    std::complex<float>* alpha,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    std::complex<float>* beta,
    std::complex<float>* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pcsyr2k(
        uplo, trans, n, k, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB,
        beta,
        C, ic, jc, descC );
}

inline void scalapack_psyr2k(
    const char* uplo, const char* trans, blas_int* n, blas_int* k,
    std::complex<double>* alpha,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    std::complex<double>* beta,
    std::complex<double>* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pzsyr2k(
        uplo, trans, n, k, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB,
        beta,
        C, ic, jc, descC );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_psyr2k(
    const char* uplo, const char* trans, int64_t n, int64_t k,
    scalar_t alpha,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    scalar_t* B, int64_t ib, int64_t jb, blas_int* descB,
    scalar_t beta,
    scalar_t* C, int64_t ic, int64_t jc, blas_int* descC )
{
    blas_int n_  = to_blas_int( n );
    blas_int k_  = to_blas_int( k );
    blas_int ia_ = to_blas_int( ia );
    blas_int ja_ = to_blas_int( ja );
    blas_int ib_ = to_blas_int( ib );
    blas_int jb_ = to_blas_int( jb );
    blas_int ic_ = to_blas_int( ic );
    blas_int jc_ = to_blas_int( jc );
    scalapack_psyr2k(
        uplo, trans, &n_, &k_, &alpha,
        A, &ia_, &ja_, descA,
        B, &ib_, &jb_, descB,
        &beta,
        C, &ic_, &jc_, descC );
}

//==============================================================================
// Fortran prototypes
#define scalapack_pssyrk BLAS_FORTRAN_NAME( pssyrk, PSSYRK )
#define scalapack_pdsyrk BLAS_FORTRAN_NAME( pdsyrk, PDSYRK )
#define scalapack_pcsyrk BLAS_FORTRAN_NAME( pcsyrk, PCSYRK )
#define scalapack_pzsyrk BLAS_FORTRAN_NAME( pzsyrk, PZSYRK )

extern "C" {

void scalapack_pssyrk(
    const char* uplo, const char* trans,
    blas_int* n, blas_int* k, float* alpha,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* beta,
    float* C, blas_int* ic, blas_int* jc, blas_int* descC );

void scalapack_pdsyrk(
    const char* uplo, const char* trans,
    blas_int* n, blas_int* k, double* alpha,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* beta,
    double* C, blas_int* ic, blas_int* jc, blas_int* descC );

void scalapack_pcsyrk(
    const char* uplo, const char* trans,
    blas_int* n, blas_int* k,
    std::complex<float>* alpha,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* beta,
    std::complex<float>* C, blas_int* ic, blas_int* jc, blas_int* descC );

void scalapack_pzsyrk(
    const char* uplo, const char* trans,
    blas_int* n, blas_int* k,
    std::complex<double>* alpha,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* beta,
    std::complex<double>* C, blas_int* ic, blas_int* jc, blas_int* descC );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_psyrk(
    const char* uplo, const char* trans, blas_int* n, blas_int* k,
    float* alpha,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* beta,
    float* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pssyrk(
        uplo, trans, n, k, alpha,
        A, ia, ja, descA,
        beta,
        C, ic, jc, descC );
}

inline void scalapack_psyrk(
    const char* uplo, const char* trans, blas_int* n, blas_int* k,
    double* alpha,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* beta,
    double* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pdsyrk(
        uplo, trans, n, k, alpha,
        A, ia, ja, descA,
        beta,
        C, ic, jc, descC );
}

inline void scalapack_psyrk(
    const char* uplo, const char* trans, blas_int* n, blas_int* k,
    std::complex<float>* alpha,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* beta,
    std::complex<float>* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pcsyrk(
        uplo, trans, n, k, alpha,
        A, ia, ja, descA,
        beta,
        C, ic, jc, descC );
}

inline void scalapack_psyrk(
    const char* uplo, const char* trans, blas_int* n, blas_int* k,
    std::complex<double>* alpha,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* beta,
    std::complex<double>* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pzsyrk(
        uplo, trans, n, k, alpha,
        A, ia, ja, descA,
        beta,
        C, ic, jc, descC );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_psyrk(
    const char* uplo, const char* trans, int64_t n,
    int64_t k, scalar_t alpha,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    scalar_t beta,
    scalar_t* C, int64_t ic, int64_t jc, blas_int* descC )
{
    blas_int n_  = to_blas_int( n );
    blas_int k_  = to_blas_int( k );
    blas_int ia_ = to_blas_int( ia );
    blas_int ja_ = to_blas_int( ja );
    blas_int ic_ = to_blas_int( ic );
    blas_int jc_ = to_blas_int( jc );
    scalapack_psyrk(
        uplo, trans, &n_, &k_, &alpha,
        A, &ia_, &ja_, descA,
        &beta,
        C, &ic_, &jc_, descC );
}

//==============================================================================
// Fortran prototypes
#define scalapack_pstrsm BLAS_FORTRAN_NAME( pstrsm, PSTRSM )
#define scalapack_pdtrsm BLAS_FORTRAN_NAME( pdtrsm, PDTRSM )
#define scalapack_pctrsm BLAS_FORTRAN_NAME( pctrsm, PCTRSM )
#define scalapack_pztrsm BLAS_FORTRAN_NAME( pztrsm, PZTRSM )

extern "C" {

void scalapack_pstrsm(
    const char* side, const char* uplo, const char* transA, const char* diag,
    const blas_int* m, const blas_int* n,
    const float* alpha, const float* A,
    const blas_int* ia, const blas_int* ja,
    const blas_int* descA,
    float* B,
    const blas_int* ib, const blas_int* jb,
    const blas_int* descB );

void scalapack_pdtrsm(
    const char* side, const char* uplo, const char* transA, const char* diag,
    const blas_int* m, const blas_int* n,
    const double* alpha, const double* A,
    const blas_int* ia, const blas_int* ja,
    const blas_int* descA,
    double* B,
    const blas_int* ib, const blas_int* jb,
    const blas_int* descB );

void scalapack_pctrsm(
    const char* side, const char* uplo,
    const char* transA, const char* diag,
    const blas_int* m, const blas_int* n,
    const std::complex<float>* alpha,
    const std::complex<float>* A,
    const blas_int* ia, const blas_int* ja,
    const blas_int* descA,
    std::complex<float>* B,
    const blas_int* ib, const blas_int* jb,
    const blas_int* descB );

void scalapack_pztrsm(
    const char* side, const char* uplo, const char* transA, const char* diag,
    const blas_int* m, const blas_int* n,
    const std::complex<double>* alpha,
    const std::complex<double>* A,
    const blas_int* ia, const blas_int* ja,
    const blas_int* descA,
    std::complex<double>* B,
    const blas_int* ib, const blas_int* jb,
    const blas_int* descB );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_ptrsm(
    const char* side, const char* uplo, const char* transA, const char* diag,
    const blas_int* m, const blas_int* n,
    const float* alpha, const float* A,
    const blas_int* ia, const blas_int* ja,
    const blas_int* descA,
    float* B, const blas_int* ib, const blas_int* jb, const blas_int* descB )
{
    scalapack_pstrsm(
        side, uplo, transA, diag, m, n, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB );
}

inline void scalapack_ptrsm(
    const char* side, const char* uplo, const char* transA, const char* diag,
    const blas_int* m, const blas_int* n,
    const double* alpha, const double* A,
    const blas_int* ia, const blas_int* ja,
    const blas_int* descA,
    double* B,
    const blas_int* ib, const blas_int* jb,
    const blas_int* descB )
{
    scalapack_pdtrsm(
        side, uplo, transA, diag, m, n, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB );
}

inline void scalapack_ptrsm(
    const char* side, const char* uplo, const char* transA, const char* diag,
    const blas_int* m, const blas_int* n,
    const std::complex<float>* alpha,
    const std::complex<float>* A, const blas_int* ia, const blas_int* ja,
    const blas_int* descA,
    std::complex<float>* B, const blas_int* ib, const blas_int* jb,
    const blas_int* descB )
{
    scalapack_pctrsm(
        side, uplo, transA, diag, m, n, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB );
}

inline void scalapack_ptrsm(
    const char* side, const char* uplo, const char* transA, const char* diag,
    const blas_int* m, const blas_int* n,
    const std::complex<double>* alpha,
    const std::complex<double>* A, const blas_int* ia, const blas_int* ja,
    const blas_int* descA,
    std::complex<double>* B, const blas_int* ib, const blas_int* jb,
    const blas_int* descB )
{
    scalapack_pztrsm(
        side, uplo, transA, diag, m, n, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_ptrsm(
    const char* side, const char* uplo, const char* transA, const char* diag,
    int64_t m, int64_t n,
    scalar_t alpha,
    scalar_t* A, int64_t ia, int64_t ja, const blas_int* descA,
    scalar_t* B, int64_t ib, int64_t jb, const blas_int* descB )
{
    blas_int m_  = to_blas_int( m );
    blas_int n_  = to_blas_int( n );
    blas_int ia_ = to_blas_int( ia );
    blas_int ja_ = to_blas_int( ja );
    blas_int ib_ = to_blas_int( ib );
    blas_int jb_ = to_blas_int( jb );
    scalapack_ptrsm(
        side, uplo, transA, diag, &m_, &n_, &alpha,
        A, &ia_, &ja_, descA,
        B, &ib_, &jb_, descB );
}

//==============================================================================
// Fortran prototypes
#define scalapack_pslantr BLAS_FORTRAN_NAME( pslantr, PSLANTR )
#define scalapack_pdlantr BLAS_FORTRAN_NAME( pdlantr, PDLANTR )
#define scalapack_pclantr BLAS_FORTRAN_NAME( pclantr, PCLANTR )
#define scalapack_pzlantr BLAS_FORTRAN_NAME( pzlantr, PZLANTR )

extern "C" {

float scalapack_pslantr(
    const char* norm, const char* uplo,
    const char* diag, blas_int* m, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* work );

double scalapack_pdlantr(
    const char* norm, const char* uplo,
    const char* diag, blas_int* m, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* work );

float scalapack_pclantr(
    const char* norm, const char* uplo,
    const char* diag, blas_int* m, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* work );

double scalapack_pzlantr(
    const char* norm, const char* uplo,
    const char* diag, blas_int* m, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* work );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline float scalapack_plantr(
    const char* norm, const char* uplo,
    const char* diag, blas_int* m, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* work )
{
    return scalapack_pslantr(
        norm, uplo, diag, m, n,
        A, ia, ja, descA,
        work );
}

inline double scalapack_plantr(
    const char* norm, const char* uplo,
    const char* diag, blas_int* m, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* work )
{
    return scalapack_pdlantr(
        norm, uplo, diag, m, n,
        A, ia, ja, descA,
        work );
}

inline float scalapack_plantr(
    const char* norm, const char* uplo,
    const char* diag, blas_int* m, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* work )
{
    return scalapack_pclantr(
        norm, uplo, diag, m, n,
        A, ia, ja, descA,
        work );
}

inline double scalapack_plantr(
    const char* norm, const char* uplo,
    const char* diag, blas_int* m, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* work )
{
    return scalapack_pzlantr(
        norm, uplo, diag, m, n,
        A, ia, ja, descA,
        work );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
blas::real_type<scalar_t> scalapack_plantr(
    const char* norm,
    const char* uplo,
    const char* diag, int64_t m,
    int64_t n,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    blas::real_type<scalar_t>* work )
{
    blas_int m_  = to_blas_int( m );
    blas_int n_  = to_blas_int( n );
    blas_int ia_ = to_blas_int( ia );
    blas_int ja_ = to_blas_int( ja );
    return scalapack_plantr(
        norm, uplo, diag, &m_, &n_,
        A, &ia_, &ja_, descA,
        work );
}

//==============================================================================
// Fortran prototypes
#define scalapack_pchemm BLAS_FORTRAN_NAME( pchemm, PCHEMM )
#define scalapack_pzhemm BLAS_FORTRAN_NAME( pzhemm, PZHEMM )

extern "C" {

void scalapack_pchemm(
    const char* side, const char* uplo, blas_int* m, blas_int* n,
    const std::complex<float>* alpha,
    const std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    const std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    const std::complex<float>* beta,
    const std::complex<float>* C, blas_int* ic, blas_int* jc, blas_int* descC );

void scalapack_pzhemm(
    const char* side, const char* uplo, blas_int* m, blas_int* n,
    const std::complex<double>* alpha,
    const std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    const std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    const std::complex<double>* beta,
    const std::complex<double>* C, blas_int* ic, blas_int* jc, blas_int* descC );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_phemm(
    const char* side, const char* uplo, blas_int* m, blas_int* n,
    float* alpha,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    float* beta,
    float* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pssymm(
        side, uplo, m, n, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB,
        beta,
        C, ic, jc, descC );
}

inline void scalapack_phemm(
    const char* side, const char* uplo, blas_int* m, blas_int* n,
    double* alpha,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* beta,
    double* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pdsymm(
        side, uplo, m, n, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB,
        beta,
        C, ic, jc, descC );
}

inline void scalapack_phemm(
    const char* side, const char* uplo, blas_int* m, blas_int* n,
    const std::complex<float>* alpha,
    const std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    const std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    const std::complex<float>* beta,
    const std::complex<float>* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pchemm(
        side, uplo, m, n, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB,
        beta,
        C, ic, jc, descC );
}

inline void scalapack_phemm(
    const char* side, const char* uplo, blas_int* m, blas_int* n,
    const std::complex<double>* alpha,
    const std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    const std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    const std::complex<double>* beta,
    const std::complex<double>* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pzhemm(
        side, uplo, m, n, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB,
        beta,
        C, ic, jc, descC );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_phemm(
    const char* side, const char* uplo, int64_t m,
    int64_t n, scalar_t alpha,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    scalar_t* B, int64_t ib, int64_t jb, blas_int* descB,
    scalar_t beta,
    scalar_t* C, int64_t ic, int64_t jc, blas_int* descC )
{
    blas_int m_  = to_blas_int( m );
    blas_int n_  = to_blas_int( n );
    blas_int ia_ = to_blas_int( ia );
    blas_int ja_ = to_blas_int( ja );
    blas_int ib_ = to_blas_int( ib );
    blas_int jb_ = to_blas_int( jb );
    blas_int ic_ = to_blas_int( ic );
    blas_int jc_ = to_blas_int( jc );
    scalapack_phemm(
        side, uplo, &m_, &n_, &alpha,
        A, &ia_, &ja_, descA,
        B, &ib_, &jb_, descB,
        &beta,
        C, &ic_, &jc_, descC );
}

//==============================================================================
// Fortran prototypes
#define scalapack_pcher2k BLAS_FORTRAN_NAME( pcher2k, PCHER2K )
#define scalapack_pzher2k BLAS_FORTRAN_NAME( pzher2k, PZHER2K )

extern "C" {

void scalapack_pcher2k(
    const char* uplo, const char* trans,
    blas_int* n, blas_int* k,
    std::complex<float>* alpha,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    float* beta,
    std::complex<float>* C, blas_int* ic, blas_int* jc, blas_int* descC );

void scalapack_pzher2k(
    const char* uplo, const char* trans,
    blas_int* n, blas_int* k,
    std::complex<double>* alpha,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* beta,
    std::complex<double>* C, blas_int* ic, blas_int* jc, blas_int* descC );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_pher2k(
    const char* uplo, const char* trans, blas_int* n, blas_int* k,
    float* alpha,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    float* beta,
    float* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pssyr2k(
        uplo, trans, n, k, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB,
        beta,
        C, ic, jc, descC );
}

inline void scalapack_pher2k(
    const char* uplo, const char* trans, blas_int* n, blas_int* k,
    double* alpha,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* beta,
    double* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pdsyr2k(
        uplo, trans, n, k, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB,
        beta,
        C, ic, jc, descC );
}

inline void scalapack_pher2k(
    const char* uplo, const char* trans, blas_int* n, blas_int* k,
    std::complex<float>* alpha,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    float* beta,
    std::complex<float>* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pcher2k(
        uplo, trans, n, k, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB,
        beta,
        C, ic, jc, descC );
}

inline void scalapack_pher2k(
    const char* uplo, const char* trans, blas_int* n, blas_int* k,
    std::complex<double>* alpha,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* beta,
    std::complex<double>* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pzher2k(
        uplo, trans, n, k, alpha,
        A, ia, ja, descA,
        B, ib, jb, descB,
        beta,
        C, ic, jc, descC );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_pher2k(
    const char* uplo, const char* trans, int64_t n,
    int64_t k, scalar_t alpha,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    scalar_t* B, int64_t ib, int64_t jb, blas_int* descB,
    blas::real_type<scalar_t> beta,
    scalar_t* C, int64_t ic, int64_t jc, blas_int* descC )
{
    blas_int n_  = to_blas_int( n );
    blas_int k_  = to_blas_int( k );
    blas_int ia_ = to_blas_int( ia );
    blas_int ja_ = to_blas_int( ja );
    blas_int ib_ = to_blas_int( ib );
    blas_int jb_ = to_blas_int( jb );
    blas_int ic_ = to_blas_int( ic );
    blas_int jc_ = to_blas_int( jc );
    scalapack_pher2k(
        uplo, trans, &n_, &k_, &alpha,
        A, &ia_, &ja_, descA,
        B, &ib_, &jb_, descB,
        &beta,
        C, &ic_, &jc_, descC );
}

//==============================================================================
// Fortran prototypes
#define scalapack_pcherk BLAS_FORTRAN_NAME( pcherk, PCHERK )
#define scalapack_pzherk BLAS_FORTRAN_NAME( pzherk, PZHERK )

extern "C" {

void scalapack_pcherk(
    const char* uplo, const char* trans,
    blas_int* n, blas_int* k, float* alpha,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* beta,
    std::complex<float>* C, blas_int* ic, blas_int* jc, blas_int* descC );

void scalapack_pzherk(
    const char* uplo, const char* trans,
    blas_int* n, blas_int* k, double* alpha,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* beta,
    std::complex<double>* C, blas_int* ic, blas_int* jc, blas_int* descC );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_pherk(
    const char* uplo, const char* trans, blas_int* n, blas_int* k,
    float* alpha,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* beta,
    float* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pssyrk(
        uplo, trans, n, k,
        alpha, A, ia, ja, descA,
        beta,  C, ic, jc, descC );
}

inline void scalapack_pherk(
    const char* uplo, const char* trans, blas_int* n, blas_int* k,
    double* alpha,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* beta,
    double* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pdsyrk(
        uplo, trans, n, k,
        alpha, A, ia, ja, descA,
        beta,  C, ic, jc, descC );
}

inline void scalapack_pherk(
    const char* uplo, const char* trans, blas_int* n, blas_int* k,
    float* alpha,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* beta,
    std::complex<float>* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pcherk(
        uplo, trans, n, k,
        alpha, A, ia, ja, descA,
        beta,  C, ic, jc, descC );
}

inline void scalapack_pherk(
    const char* uplo, const char* trans, blas_int* n, blas_int* k,
    double* alpha,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* beta,
    std::complex<double>* C, blas_int* ic, blas_int* jc, blas_int* descC )
{
    scalapack_pzherk(
        uplo, trans, n, k,
        alpha, A, ia, ja, descA,
        beta,  C, ic, jc, descC );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_pherk(
    const char* uplo, const char* trans, int64_t n,
    int64_t k, blas::real_type<scalar_t> alpha,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    blas::real_type<scalar_t> beta,
    scalar_t* C, int64_t ic, int64_t jc, blas_int* descC )
{
    blas_int n_  = to_blas_int( n );
    blas_int k_  = to_blas_int( k );
    blas_int ia_ = to_blas_int( ia );
    blas_int ja_ = to_blas_int( ja );
    blas_int ic_ = to_blas_int( ic );
    blas_int jc_ = to_blas_int( jc );
    scalapack_pherk(
        uplo, trans, &n_, &k_,
        &alpha, A, &ia_, &ja_, descA,
        &beta,  C, &ic_, &jc_, descC );
}

//==============================================================================
// Fortran prototypes
#define scalapack_psgetrf BLAS_FORTRAN_NAME( psgetrf, PSGETRF )
#define scalapack_pdgetrf BLAS_FORTRAN_NAME( pdgetrf, PDGETRF )
#define scalapack_pcgetrf BLAS_FORTRAN_NAME( pcgetrf, PCGETRF )
#define scalapack_pzgetrf BLAS_FORTRAN_NAME( pzgetrf, PZGETRF )

extern "C" {

void scalapack_psgetrf(
    blas_int* m, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* ipiv, blas_int* info );

void scalapack_pdgetrf(
    blas_int* m, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* ipiv, blas_int* info );

void scalapack_pcgetrf(
    blas_int* m, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* ipiv, blas_int* info );

void scalapack_pzgetrf(
    blas_int* m, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* ipiv, blas_int* info );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_pgetrf(
    blas_int* m, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* ipiv, blas_int* info )
{
    scalapack_psgetrf(
        m, n,
        A, ia, ja, descA,
        ipiv, info );
}

inline void scalapack_pgetrf(
    blas_int* m, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* ipiv, blas_int* info )
{
    scalapack_pdgetrf(
        m, n,
        A, ia, ja, descA,
        ipiv, info );
}

inline void scalapack_pgetrf(
    blas_int* m, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* ipiv, blas_int* info )
{
    scalapack_pcgetrf(
        m, n,
        A, ia, ja, descA,
        ipiv, info );
}

inline void scalapack_pgetrf(
    blas_int* m, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* ipiv, blas_int* info )
{
    scalapack_pzgetrf(
        m, n,
        A, ia, ja, descA,
        ipiv, info );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_pgetrf(
    int64_t m, int64_t n,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    blas_int* ipiv,
    int64_t* info )
{
    blas_int m_    = to_blas_int( m );
    blas_int n_    = to_blas_int( n );
    blas_int ia_   = to_blas_int( ia );
    blas_int ja_   = to_blas_int( ja );
    blas_int info_ = 0;
    scalapack_pgetrf(
        &m_, &n_,
        A, &ia_, &ja_, descA,
        ipiv, &info_ );
    *info = int64_t( info_ );
}

//==============================================================================
// Fortran prototypes
#define scalapack_psgetrs BLAS_FORTRAN_NAME( psgetrs, PSGETRS )
#define scalapack_pdgetrs BLAS_FORTRAN_NAME( pdgetrs, PDGETRS )
#define scalapack_pcgetrs BLAS_FORTRAN_NAME( pcgetrs, PCGETRS )
#define scalapack_pzgetrs BLAS_FORTRAN_NAME( pzgetrs, PZGETRS )

extern "C" {

void scalapack_psgetrs(
    const char* trans, blas_int* n, blas_int* nrhs,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* ipiv,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

void scalapack_pdgetrs(
    const char* trans, blas_int* n, blas_int* nrhs,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* ipiv,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

void scalapack_pcgetrs(
    const char* trans, blas_int* n, blas_int* nrhs,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* ipiv,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

void scalapack_pzgetrs(
    const char* trans, blas_int* n, blas_int* nrhs,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* ipiv,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_pgetrs(
    const char* trans, blas_int* n, blas_int* nrhs,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* ipiv,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info )
{
    scalapack_psgetrs(
        trans, n, nrhs,
        A, ia, ja, descA,
        ipiv,
        B, ib, jb, descB,
        info );
}

inline void scalapack_pgetrs(
    const char* trans, blas_int* n, blas_int* nrhs,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* ipiv,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info )
{
    scalapack_pdgetrs(
        trans, n, nrhs,
        A, ia, ja, descA,
        ipiv,
        B, ib, jb, descB,
        info );
}

inline void scalapack_pgetrs(
    const char* trans, blas_int* n, blas_int* nrhs,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* ipiv,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info )
{
    scalapack_pcgetrs(
        trans, n, nrhs,
        A, ia, ja, descA,
        ipiv,
        B, ib, jb, descB,
        info );
}

inline void scalapack_pgetrs(
    const char* trans, blas_int* n, blas_int* nrhs,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* ipiv,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info )
{
    scalapack_pzgetrs(
        trans, n, nrhs,
        A, ia, ja, descA,
        ipiv,
        B, ib, jb, descB,
        info );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_pgetrs(
    const char* trans, int64_t n, int64_t nrhs,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    blas_int* ipiv,
    scalar_t* B, int64_t ib, int64_t jb, blas_int* descB,
    int64_t* info )
{
    blas_int n_    = to_blas_int( n );
    blas_int nrhs_ = to_blas_int( nrhs );
    blas_int ia_   = to_blas_int( ia );
    blas_int ja_   = to_blas_int( ja );
    blas_int ib_   = to_blas_int( ib );
    blas_int jb_   = to_blas_int( jb );
    blas_int info_ = 0;
    scalapack_pgetrs(
        trans, &n_, &nrhs_,
        A, &ia_, &ja_, descA,
        ipiv,
        B, &ib_, &jb_, descB,
        &info_ );
    *info = int64_t( info_ );
}

//==============================================================================
// Fortran prototypes
#define scalapack_psgesv BLAS_FORTRAN_NAME( psgesv, PSGESV )
#define scalapack_pdgesv BLAS_FORTRAN_NAME( pdgesv, PDGESV )
#define scalapack_pcgesv BLAS_FORTRAN_NAME( pcgesv, PCGESV )
#define scalapack_pzgesv BLAS_FORTRAN_NAME( pzgesv, PZGESV )

extern "C" {

void scalapack_psgesv(
    blas_int* n, blas_int* nrhs,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* ipiv,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

void scalapack_pdgesv(
    blas_int* n, blas_int* nrhs,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* ipiv,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

void scalapack_pcgesv(
    blas_int* n, blas_int* nrhs,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* ipiv,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

void scalapack_pzgesv(
    blas_int* n, blas_int* nrhs,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* ipiv,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_pgesv(
    blas_int* n, blas_int* nrhs,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* ipiv,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info )
{
    scalapack_psgesv(
        n, nrhs,
        A, ia, ja, descA,
        ipiv,
        B, ib, jb, descB,
        info );
}

inline void scalapack_pgesv(
    blas_int* n, blas_int* nrhs,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* ipiv,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info )
{
    scalapack_pdgesv(
        n, nrhs,
        A, ia, ja, descA,
        ipiv,
        B, ib, jb, descB,
        info );
}

inline void scalapack_pgesv(
    blas_int* n, blas_int* nrhs,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* ipiv,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info )
{
    scalapack_pcgesv(
        n, nrhs,
        A, ia, ja, descA,
        ipiv,
        B, ib, jb, descB,
        info );
}

inline void scalapack_pgesv(
    blas_int* n, blas_int* nrhs,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    blas_int* ipiv,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    blas_int* info )
{
    scalapack_pzgesv(
        n, nrhs,
        A, ia, ja, descA,
        ipiv,
        B, ib, jb, descB,
        info );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_pgesv(
    int64_t n, int64_t nrhs,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    blas_int* ipiv,
    scalar_t* B, int64_t ib, int64_t jb, blas_int* descB,
    int64_t* info )
{
    blas_int n_    = to_blas_int( n );
    blas_int nrhs_ = to_blas_int( nrhs );
    blas_int ia_   = to_blas_int( ia );
    blas_int ja_   = to_blas_int( ja );
    blas_int ib_   = to_blas_int( ib );
    blas_int jb_   = to_blas_int( jb );
    blas_int info_ = 0;
    scalapack_pgesv(
        &n_, &nrhs_,
        A, &ia_, &ja_, descA,
        ipiv,
        B, &ib_, &jb_, descB,
        &info_ );
    *info = int64_t( info_ );
}

//==============================================================================
// Fortran prototypes
#define scalapack_psgeqrf BLAS_FORTRAN_NAME( psgeqrf, PSGEQRF )
#define scalapack_pdgeqrf BLAS_FORTRAN_NAME( pdgeqrf, PDGEQRF )
#define scalapack_pcgeqrf BLAS_FORTRAN_NAME( pcgeqrf, PCGEQRF )
#define scalapack_pzgeqrf BLAS_FORTRAN_NAME( pzgeqrf, PZGEQRF )

extern "C" {

void scalapack_psgeqrf(
    blas_int* m, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* tau,
    float* work, blas_int* lwork,
    blas_int* info );

void scalapack_pdgeqrf(
    blas_int* m, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* tau,
    double* work, blas_int* lwork,
    blas_int* info );

void scalapack_pcgeqrf(
    blas_int* m, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* tau,
    std::complex<float>* work, blas_int* lwork,
    blas_int* info );

void scalapack_pzgeqrf(
    blas_int* m, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* tau,
    std::complex<double>* work, blas_int* lwork,
    blas_int* info );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_pgeqrf(
    blas_int* m, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* tau,
    float* work, blas_int* lwork,
    blas_int* info )
{
    scalapack_psgeqrf(
        m, n,
        A, ia, ja, descA,
        tau, work, lwork, info );
}

inline void scalapack_pgeqrf(
    blas_int* m, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* tau,
    double* work, blas_int* lwork,
    blas_int* info )
{
    scalapack_pdgeqrf(
        m, n,
        A, ia, ja, descA,
        tau, work, lwork, info );
}

inline void scalapack_pgeqrf(
    blas_int* m, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* tau,
    std::complex<float>* work, blas_int* lwork,
    blas_int* info )
{
    scalapack_pcgeqrf(
        m, n,
        A, ia, ja, descA,
        tau, work, lwork, info );
}

inline void scalapack_pgeqrf(
    blas_int* m, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* tau,
    std::complex<double>* work, blas_int* lwork,
    blas_int* info )
{
    scalapack_pzgeqrf(
        m, n,
        A, ia, ja, descA,
        tau, work, lwork, info );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_pgeqrf(
    int64_t m, int64_t n,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    scalar_t* tau,
    scalar_t* work, int64_t lwork,
    int64_t* info )
{
    blas_int m_     = to_blas_int( m );
    blas_int n_     = to_blas_int( n );
    blas_int ia_    = to_blas_int( ia );
    blas_int ja_    = to_blas_int( ja );
    blas_int lwork_ = to_blas_int( lwork );
    blas_int info_  = 0;
    scalapack_pgeqrf(
        &m_, &n_,
        A, &ia_, &ja_, descA,
        tau, work, &lwork_,
        &info_ );
    *info = int64_t( info_ );
}

//==============================================================================
// Fortran prototypes
#define scalapack_psgelqf BLAS_FORTRAN_NAME( psgelqf, PSGELQF )
#define scalapack_pdgelqf BLAS_FORTRAN_NAME( pdgelqf, PDGELQF )
#define scalapack_pcgelqf BLAS_FORTRAN_NAME( pcgelqf, PCGELQF )
#define scalapack_pzgelqf BLAS_FORTRAN_NAME( pzgelqf, PZGELQF )

extern "C" {

void scalapack_psgelqf(
    blas_int* m, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* tau,
    float* work, blas_int* lwork,
    blas_int* info );

void scalapack_pdgelqf(
    blas_int* m, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* tau,
    double* work, blas_int* lwork,
    blas_int* info );

void scalapack_pcgelqf(
    blas_int* m, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* tau,
    std::complex<float>* work, blas_int* lwork,
    blas_int* info );

void scalapack_pzgelqf(
    blas_int* m, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* tau,
    std::complex<double>* work, blas_int* lwork,
    blas_int* info );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_pgelqf(
    blas_int* m, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* tau,
    float* work, blas_int* lwork,
    blas_int* info )
{
    scalapack_psgelqf(
        m, n,
        A, ia, ja, descA,
        tau, work, lwork, info );
}

inline void scalapack_pgelqf(
    blas_int* m, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* tau,
    double* work, blas_int* lwork,
    blas_int* info )
{
    scalapack_pdgelqf(
        m, n,
        A, ia, ja, descA,
        tau, work, lwork, info );
}

inline void scalapack_pgelqf(
    blas_int* m, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* tau,
    std::complex<float>* work, blas_int* lwork,
    blas_int* info )
{
    scalapack_pcgelqf(
        m, n,
        A, ia, ja, descA,
        tau, work, lwork, info );
}

inline void scalapack_pgelqf(
    blas_int* m, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* tau,
    std::complex<double>* work, blas_int* lwork,
    blas_int* info )
{
    scalapack_pzgelqf(
        m, n,
        A, ia, ja, descA,
        tau, work, lwork, info );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_pgelqf(
    int64_t m, int64_t n,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    scalar_t* tau,
    scalar_t* work, int64_t lwork,
    int64_t* info )
{
    blas_int m_     = to_blas_int( m );
    blas_int n_     = to_blas_int( n );
    blas_int ia_    = to_blas_int( ia );
    blas_int ja_    = to_blas_int( ja );
    blas_int lwork_ = to_blas_int( lwork );
    blas_int info_  = 0;
    scalapack_pgelqf(
        &m_, &n_,
        A, &ia_, &ja_, descA,
        tau, work, &lwork_,
        &info_ );
    *info = int64_t( info_ );
}

//==============================================================================
// Fortran prototypes
#define scalapack_psormqr BLAS_FORTRAN_NAME( psormqr, PSORMQR )
#define scalapack_pdormqr BLAS_FORTRAN_NAME( pdormqr, PDORMQR )
#define scalapack_pcunmqr BLAS_FORTRAN_NAME( pcunmqr, PCUNMQR )
#define scalapack_pzunmqr BLAS_FORTRAN_NAME( pzunmqr, PZUNMQR )

extern "C" {

void scalapack_psormqr(
    const char* side, const char* trans,
    blas_int* m, blas_int* n, blas_int* k,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* tau,
    float* C, blas_int* ic, blas_int* jc, blas_int* descC,
    float* work, blas_int* lwork,
    blas_int* info );

void scalapack_pdormqr(
    const char* side, const char* trans,
    blas_int* m, blas_int* n, blas_int* k,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* tau,
    double* C, blas_int* ic, blas_int* jc, blas_int* descC,
    double* work, blas_int* lwork,
    blas_int* info );

void scalapack_pcunmqr(
    const char* side, const char* trans,
    blas_int* m, blas_int* n, blas_int* k,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* tau,
    std::complex<float>* C, blas_int* ic, blas_int* jc, blas_int* descC,
    std::complex<float>* work, blas_int* lwork,
    blas_int* info );

void scalapack_pzunmqr(
    const char* side, const char* trans,
    blas_int* m, blas_int* n, blas_int* k,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* tau,
    std::complex<double>* C, blas_int* ic, blas_int* jc, blas_int* descC,
    std::complex<double>* work, blas_int* lwork,
    blas_int* info );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_punmqr(
    const char* side, const char* trans,
    blas_int* m, blas_int* n, blas_int* k,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* tau,
    float* C, blas_int* ic, blas_int* jc, blas_int* descC,
    float* work, blas_int* lwork,
    blas_int* info )
{
    scalapack_psormqr(
        side, trans, m, n, k,
        A, ia, ja, descA,
        tau,
        C, ic, jc, descC,
        work, lwork, info );
}

inline void scalapack_punmqr(
    const char* side, const char* trans,
    blas_int* m, blas_int* n, blas_int* k,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* tau,
    double* C, blas_int* ic, blas_int* jc, blas_int* descC,
    double* work, blas_int* lwork,
    blas_int* info )
{
    scalapack_pdormqr(
        side, trans, m, n, k,
        A, ia, ja, descA,
        tau,
        C, ic, jc, descC,
        work, lwork, info );
}

inline void scalapack_punmqr(
    const char* side, const char* trans,
    blas_int* m, blas_int* n, blas_int* k,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* tau,
    std::complex<float>* C, blas_int* ic, blas_int* jc, blas_int* descC,
    std::complex<float>* work, blas_int* lwork,
    blas_int* info )
{
    scalapack_pcunmqr(
        side, trans, m, n, k,
        A, ia, ja, descA,
        tau,
        C, ic, jc, descC,
        work, lwork, info );
}

inline void scalapack_punmqr(
    const char* side, const char* trans,
    blas_int* m, blas_int* n, blas_int* k,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* tau,
    std::complex<double>* C, blas_int* ic, blas_int* jc, blas_int* descC,
    std::complex<double>* work, blas_int* lwork,
    blas_int* info )
{
    scalapack_pzunmqr(
        side, trans, m, n, k,
        A, ia, ja, descA,
        tau,
        C, ic, jc, descC,
        work, lwork, info );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_punmqr(
    const char* side, const char* trans,
    int64_t m, int64_t n, int64_t k,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    scalar_t* tau,
    scalar_t* C, int64_t ic, int64_t jc, blas_int* descC,
    scalar_t* work, int64_t lwork,
    int64_t* info )
{
    blas_int m_     = to_blas_int( m );
    blas_int n_     = to_blas_int( n );
    blas_int k_     = to_blas_int( k );
    blas_int ia_    = to_blas_int( ia );
    blas_int ja_    = to_blas_int( ja );
    blas_int ic_    = to_blas_int( ic );
    blas_int jc_    = to_blas_int( jc );
    blas_int lwork_ = to_blas_int( lwork );
    blas_int info_  = 0;
    scalapack_punmqr(
        side, trans, &m_, &n_, &k_,
        A, &ia_, &ja_, descA,
        tau,
        C, &ic_, &jc_, descC,
        work, &lwork_, &info_ );
    *info = int64_t( info_ );
}

//==============================================================================
// Fortran prototypes
#define scalapack_psormlq BLAS_FORTRAN_NAME( psormlq, PSORMLQ )
#define scalapack_pdormlq BLAS_FORTRAN_NAME( pdormlq, PDORMLQ )
#define scalapack_pcunmlq BLAS_FORTRAN_NAME( pcunmlq, PCUNMLQ )
#define scalapack_pzunmlq BLAS_FORTRAN_NAME( pzunmlq, PZUNMLQ )

extern "C" {

void scalapack_psormlq(
    const char* side, const char* trans,
    blas_int* m, blas_int* n, blas_int* k,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* tau,
    float* C, blas_int* ic, blas_int* jc, blas_int* descC,
    float* work, blas_int* lwork,
    blas_int* info );

void scalapack_pdormlq(
    const char* side, const char* trans,
    blas_int* m, blas_int* n, blas_int* k,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* tau,
    double* C, blas_int* ic, blas_int* jc, blas_int* descC,
    double* work, blas_int* lwork,
    blas_int* info );

void scalapack_pcunmlq(
    const char* side, const char* trans,
    blas_int* m, blas_int* n, blas_int* k,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* tau,
    std::complex<float>* C, blas_int* ic, blas_int* jc, blas_int* descC,
    std::complex<float>* work, blas_int* lwork,
    blas_int* info );

void scalapack_pzunmlq(
    const char* side, const char* trans,
    blas_int* m, blas_int* n, blas_int* k,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* tau,
    std::complex<double>* C, blas_int* ic, blas_int* jc, blas_int* descC,
    std::complex<double>* work, blas_int* lwork,
    blas_int* info );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_punmlq(
    const char* side, const char* trans,
    blas_int* m, blas_int* n, blas_int* k,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* tau,
    float* C, blas_int* ic, blas_int* jc, blas_int* descC,
    float* work, blas_int* lwork,
    blas_int* info )
{
    scalapack_psormlq(
        side, trans, m, n, k,
        A, ia, ja, descA,
        tau,
        C, ic, jc, descC,
        work, lwork, info );
}

inline void scalapack_punmlq(
    const char* side, const char* trans,
    blas_int* m, blas_int* n, blas_int* k,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* tau,
    double* C, blas_int* ic, blas_int* jc, blas_int* descC,
    double* work, blas_int* lwork,
    blas_int* info )
{
    scalapack_pdormlq(
        side, trans, m, n, k,
        A, ia, ja, descA,
        tau,
        C, ic, jc, descC,
        work, lwork, info );
}

inline void scalapack_punmlq(
    const char* side, const char* trans,
    blas_int* m, blas_int* n, blas_int* k,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* tau,
    std::complex<float>* C, blas_int* ic, blas_int* jc, blas_int* descC,
    std::complex<float>* work, blas_int* lwork,
    blas_int* info )
{
    scalapack_pcunmlq(
        side, trans, m, n, k,
        A, ia, ja, descA,
        tau,
        C, ic, jc, descC,
        work, lwork, info );
}

inline void scalapack_punmlq(
    const char* side, const char* trans,
    blas_int* m, blas_int* n, blas_int* k,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* tau,
    std::complex<double>* C, blas_int* ic, blas_int* jc, blas_int* descC,
    std::complex<double>* work, blas_int* lwork,
    blas_int* info )
{
    scalapack_pzunmlq(
        side, trans, m, n, k,
        A, ia, ja, descA,
        tau,
        C, ic, jc, descC,
        work, lwork, info );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_punmlq(
    const char* side, const char* trans,
    int64_t m, int64_t n, int64_t k,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    scalar_t* tau,
    scalar_t* C, int64_t ic, int64_t jc, blas_int* descC,
    scalar_t* work, int64_t lwork,
    int64_t* info )
{
    blas_int m_     = to_blas_int( m );
    blas_int n_     = to_blas_int( n );
    blas_int k_     = to_blas_int( k );
    blas_int ia_    = to_blas_int( ia );
    blas_int ja_    = to_blas_int( ja );
    blas_int ic_    = to_blas_int( ic );
    blas_int jc_    = to_blas_int( jc );
    blas_int lwork_ = to_blas_int( lwork );
    blas_int info_  = 0;
    scalapack_punmlq(
        side, trans, &m_, &n_, &k_,
        A, &ia_, &ja_, descA,
        tau,
        C, &ic_, &jc_, descC,
        work, &lwork_, &info_ );
    *info = int64_t( info_ );
}

//==============================================================================
// Fortran prototypes
#define scalapack_psgels BLAS_FORTRAN_NAME( psgels, PSGELS )
#define scalapack_pdgels BLAS_FORTRAN_NAME( pdgels, PDGELS )
#define scalapack_pcgels BLAS_FORTRAN_NAME( pcgels, PCGELS )
#define scalapack_pzgels BLAS_FORTRAN_NAME( pzgels, PZGELS )

extern "C" {

void scalapack_psgels(
    const char* trans,
    blas_int* m, blas_int* n, blas_int* nrhs,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    float* work, blas_int* lwork,
    blas_int* info );

void scalapack_pdgels(
    const char* trans,
    blas_int* m, blas_int* n, blas_int* nrhs,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* work, blas_int* lwork,
    blas_int* info );

void scalapack_pcgels(
    const char* trans,
    blas_int* m, blas_int* n, blas_int* nrhs,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    std::complex<float>* work, blas_int* lwork,
    blas_int* info );

void scalapack_pzgels(
    const char* trans,
    blas_int* m, blas_int* n, blas_int* nrhs,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    std::complex<double>* work, blas_int* lwork,
    blas_int* info );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_pgels(
    const char* trans,
    blas_int* m, blas_int* n, blas_int* nrhs,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    float* work, blas_int* lwork,
    blas_int* info )
{
    char trans2 = *trans;
    if (trans2 == 'c' || trans2 == 'C')
        trans2 = 't';
    scalapack_psgels(
        &trans2, m, n, nrhs,
        A, ia, ja, descA,
        B, ib, jb, descB,
        work, lwork, info );
}

inline void scalapack_pgels(
    const char* trans,
    blas_int* m, blas_int* n, blas_int* nrhs,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* work, blas_int* lwork,
    blas_int* info )
{
    char trans2 = *trans;
    if (trans2 == 'c' || trans2 == 'C')
        trans2 = 't';
    scalapack_pdgels(
        &trans2, m, n, nrhs,
        A, ia, ja, descA,
        B, ib, jb, descB,
        work, lwork, info );
}

inline void scalapack_pgels(
    const char* trans,
    blas_int* m, blas_int* n, blas_int* nrhs,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    std::complex<float>* work, blas_int* lwork,
    blas_int* info )
{
    scalapack_pcgels(
        trans, m, n, nrhs,
        A, ia, ja, descA,
        B, ib, jb, descB,
        work, lwork, info );
}

inline void scalapack_pgels(
    const char* trans,
    blas_int* m, blas_int* n, blas_int* nrhs,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    std::complex<double>* work, blas_int* lwork,
    blas_int* info )
{
    scalapack_pzgels(
        trans, m, n, nrhs,
        A, ia, ja, descA,
        B, ib, jb, descB,
        work, lwork, info );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_pgels(
    const char* trans,
    int64_t m, int64_t n, int64_t nrhs,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    scalar_t* B, int64_t ib, int64_t jb, blas_int* descB,
    scalar_t* work, int64_t lwork,
    int64_t* info )
{
    blas_int m_     = to_blas_int( m );
    blas_int n_     = to_blas_int( n );
    blas_int nrhs_  = to_blas_int( nrhs );
    blas_int ia_    = to_blas_int( ia );
    blas_int ja_    = to_blas_int( ja );
    blas_int ib_    = to_blas_int( ib );
    blas_int jb_    = to_blas_int( jb );
    blas_int lwork_ = to_blas_int( lwork );
    blas_int info_  = 0;
    scalapack_pgels(
        trans, &m_, &n_, &nrhs_,
        A, &ia_, &ja_, descA,
        B, &ib_, &jb_, descB,
        work, &lwork_, &info_ );
    *info = int64_t( info_ );
}

//==============================================================================
// Fortran prototypes
#define scalapack_psgesvd BLAS_FORTRAN_NAME( psgesvd, PSGESVD )
#define scalapack_pdgesvd BLAS_FORTRAN_NAME( pdgesvd, PDGESVD )
#define scalapack_pcgesvd BLAS_FORTRAN_NAME( pcgesvd, PCGESVD )
#define scalapack_pzgesvd BLAS_FORTRAN_NAME( pzgesvd, PZGESVD )

extern "C" {

void scalapack_psgesvd(
    const char* jobu, const char* jobvt,
    blas_int* m, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* S,
    float* U, blas_int* iu, blas_int* ju, blas_int* descU,
    float* VT, blas_int* ivt, blas_int* jvt, blas_int* descVT,
    float* work, blas_int* lwork,
    blas_int* info );

void scalapack_pdgesvd(
    const char* jobu, const char* jobvt,
    blas_int* m, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* S,
    double* U, blas_int* iu, blas_int* ju, blas_int* descU,
    double* VT, blas_int* ivt, blas_int* jvt, blas_int* descVT,
    double* work, blas_int* lwork,
    blas_int* info );

void scalapack_pcgesvd(
    const char* jobu, const char* jobvt,
    blas_int* m, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* S,
    std::complex<float>* U, blas_int* iu, blas_int* ju, blas_int* descU,
    std::complex<float>* VT, blas_int* ivt, blas_int* jvt, blas_int* descVT,
    std::complex<float>* work, blas_int* lwork,
    float* rwork,
    blas_int* info );

void scalapack_pzgesvd(
    const char* jobu, const char* jobvt,
    blas_int* m, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* S,
    std::complex<double>* U, blas_int* iu, blas_int* ju, blas_int* descU,
    std::complex<double>* VT, blas_int* ivt, blas_int* jvt, blas_int* descVT,
    std::complex<double>* work, blas_int* lwork,
    double* rwork,
    blas_int* info );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_pgesvd(
    const char* jobu, const char* jobvt,
    blas_int* m, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* S,
    float* U, blas_int* iu, blas_int* ju, blas_int* descU,
    float* VT, blas_int* ivt, blas_int* jvt, blas_int* descVT,
    float* work, blas_int* lwork,
    float* rwork,
    blas_int* info )
{
    rwork[0] = 1;  // unused; lrwork = 1
    scalapack_psgesvd(
        jobu, jobvt, m, n,
        A, ia, ja, descA,
        S,
        U, iu, ju, descU,
        VT, ivt, jvt, descVT,
        work, lwork, info );
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
    blas_int* info )
{
    rwork[0] = 1;  // unused; lrwork = 1
    scalapack_pdgesvd(
        jobu, jobvt, m, n,
        A, ia, ja, descA,
        S,
        U, iu, ju, descU,
        VT, ivt, jvt, descVT,
        work, lwork, info );
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
    blas_int* info )
{
    scalapack_pcgesvd(
        jobu, jobvt, m, n,
        A, ia, ja, descA,
        S,
        U, iu, ju, descU,
        VT, ivt, jvt, descVT,
        work, lwork, rwork, info );
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
    blas_int* info )
{
    scalapack_pzgesvd(
        jobu, jobvt, m, n,
        A, ia, ja, descA,
        S,
        U, iu, ju, descU,
        VT, ivt, jvt, descVT,
        work, lwork, rwork, info );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_pgesvd(
    const char* jobu, const char* jobvt,
    int64_t m, int64_t n,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    blas::real_type<scalar_t>* S,
    scalar_t* U, int64_t iu, int64_t ju, blas_int* descU,
    scalar_t* VT, int64_t ivt, int64_t jvt, blas_int* descVT,
    scalar_t* work, int64_t lwork,
    blas::real_type<scalar_t>* rwork,
    int64_t* info )
{
    blas_int m_     = to_blas_int( m );
    blas_int n_     = to_blas_int( n );
    blas_int ia_    = to_blas_int( ia );
    blas_int ja_    = to_blas_int( ja );
    blas_int iu_    = to_blas_int( iu );
    blas_int ju_    = to_blas_int( ju );
    blas_int ivt_   = to_blas_int( ivt );
    blas_int jvt_   = to_blas_int( jvt );
    blas_int lwork_ = to_blas_int( lwork );
    blas_int info_  = 0;
    scalapack_pgesvd(
        jobu, jobvt, &m_, &n_,
        A, &ia_, &ja_, descA,
        S,
        U, &iu_, &ju_, descU,
        VT, &ivt_, &jvt_, descVT,
        work, &lwork_, rwork, &info_ );
    *info = int64_t( info_ );
}

//==============================================================================
// Fortran prototypes
#define scalapack_pssyev BLAS_FORTRAN_NAME( pssyev, PSSYEV )
#define scalapack_pdsyev BLAS_FORTRAN_NAME( pdsyev, PDSYEV )
#define scalapack_pcheev BLAS_FORTRAN_NAME( pcheev, PCHEEV )
#define scalapack_pzheev BLAS_FORTRAN_NAME( pzheev, PZHEEV )

extern "C" {

void scalapack_pssyev(
    const char* jobz, const char* uplo, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* W,
    float* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    float* work, blas_int* lwork,
    blas_int* info );

void scalapack_pdsyev(
    const char* jobz, const char* uplo, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* W,
    double* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    double* work, blas_int* lwork,
    blas_int* info );

void scalapack_pcheev(
    const char* jobz, const char* uplo, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* W,
    std::complex<float>* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    std::complex<float>* work, blas_int* lwork,
    float* rwork, blas_int* lrwork,
    blas_int* info );

void scalapack_pzheev(
    const char* jobz, const char* uplo, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* W,
    std::complex<double>* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    std::complex<double>* work, blas_int* lwork,
    double* rwork, blas_int* lrwork,
    blas_int* info );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_pheev(
    const char* jobz, const char* uplo, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* W,
    float* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    float* work, blas_int* lwork,
    float* rwork, blas_int* lrwork,
    blas_int* info )
{
    scalapack_pssyev(
        jobz, uplo, n,
        A, ia, ja, descA,
        W, Z, iz, jz, descZ,
        work,
        lwork, info );
}

inline void scalapack_pheev(
    const char* jobz, const char* uplo, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* W,
    double* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    double* work, blas_int* lwork,
    double* rwork, blas_int* lrwork,
    blas_int* info )
{
    scalapack_pdsyev(
        jobz, uplo, n,
        A, ia, ja, descA,
        W, Z, iz, jz, descZ,
        work,
        lwork, info );
}

inline void scalapack_pheev(
    const char* jobz, const char* uplo, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* W,
    std::complex<float>* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    std::complex<float>* work, blas_int* lwork,
    float* rwork, blas_int* lrwork,
    blas_int* info )
{
    scalapack_pcheev(
        jobz, uplo, n,
        A, ia, ja, descA,
        W, Z, iz, jz, descZ,
        work,
        lwork, rwork, lrwork, info );
}

inline void scalapack_pheev(
    const char* jobz, const char* uplo, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* W,
    std::complex<double>* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    std::complex<double>* work, blas_int* lwork,
    double* rwork, blas_int* lrwork,
    blas_int* info )
{
    scalapack_pzheev(
        jobz, uplo, n,
        A, ia, ja, descA,
        W, Z, iz, jz, descZ,
        work,
        lwork, rwork, lrwork, info );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_pheev(
    const char* jobz, const char* uplo,
    int64_t n,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    blas::real_type<scalar_t>* W,
    scalar_t* Z, int64_t iz, int64_t jz, blas_int* descZ,
    scalar_t* work, int64_t lwork,
    blas::real_type<scalar_t>* rwork, int64_t lrwork,
    int64_t* info )
{
    blas_int n_      = to_blas_int( n );
    blas_int ia_     = to_blas_int( ia );
    blas_int ja_     = to_blas_int( ja );
    blas_int iz_     = to_blas_int( iz );
    blas_int jz_     = to_blas_int( jz );
    blas_int lwork_  = to_blas_int( lwork );
    blas_int lrwork_ = to_blas_int( lrwork );
    blas_int info_   = 0;
    scalapack_pheev(
        jobz, uplo, &n_,
        A, &ia_, &ja_, descA,
        W,
        Z, &iz_, &jz_, descZ,
        work, &lwork_, rwork, &lrwork_, &info_ );
    *info = int64_t( info_ );
}

//==============================================================================
// Fortran prototypes
#define scalapack_psstedc BLAS_FORTRAN_NAME( psstedc, PSSTEDC )
#define scalapack_pdstedc BLAS_FORTRAN_NAME( pdstedc, PDSTEDC )

extern "C" {

void scalapack_psstedc(
    const char* jobz, blas_int* n,
    float* D, float* E,
    float* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    float* work, blas_int* lwork,
    blas_int* iwork, blas_int* liwork,
    blas_int* info );

void scalapack_pdstedc(
    const char* jobz, blas_int* n,
    double* D, double* E,
    double* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    double* work, blas_int* lwork,
    blas_int* iwork, blas_int* liwork,
    blas_int* info );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_pstedc(
    const char* jobz, blas_int* n,
    float* D, float* E,
    float* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    float* work, blas_int* lwork,
    blas_int* iwork, blas_int* liwork,
    blas_int* info )
{
    scalapack_psstedc(
        jobz, n, D, E, Z, iz, jz, descZ,
        work, lwork, iwork, liwork, info );
}

inline void scalapack_pstedc(
    const char* jobz, blas_int* n,
    double* D, double* E,
    double* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    double* work, blas_int* lwork,
    blas_int* iwork, blas_int* liwork,
    blas_int* info )
{
    scalapack_pdstedc(
        jobz, n, D, E, Z, iz, jz, descZ,
        work, lwork, iwork, liwork, info );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_pstedc(
    const char* jobz, int64_t n,
    scalar_t* D, scalar_t* E,
    scalar_t* Z, int64_t iz, int64_t jz, blas_int* descZ,
    scalar_t* work, int64_t lwork,
    blas_int* iwork, int64_t liwork,
    int64_t* info )
{
    blas_int n_      = to_blas_int( n );
    blas_int iz_     = to_blas_int( iz );
    blas_int jz_     = to_blas_int( jz );
    blas_int lwork_  = to_blas_int( lwork );
    blas_int liwork_ = to_blas_int( liwork );
    blas_int info_   = 0;
    scalapack_pstedc(
        jobz, &n_, D, E,
        Z, &iz_, &jz_, descZ,
        work, &lwork_, iwork, &liwork_, &info_ );
    *info = info_;
}

//==============================================================================
// Fortran prototypes
#define scalapack_pssyevd BLAS_FORTRAN_NAME( pssyevd, PSSYEVD )
#define scalapack_pdsyevd BLAS_FORTRAN_NAME( pdsyevd, PDSYEVD )
#define scalapack_pcheevd BLAS_FORTRAN_NAME( pcheevd, PCHEEVD )
#define scalapack_pzheevd BLAS_FORTRAN_NAME( pzheevd, PZHEEVD )

extern "C" {

void scalapack_pssyevd(
    const char* jobz, const char* uplo, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA, float* W,
    float* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    float* work, blas_int* lwork, blas_int* iwork, blas_int* liwork,
    blas_int* info );

void scalapack_pdsyevd(
    const char* jobz, const char* uplo, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA, double* W,
    double* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    double* work, blas_int* lwork, blas_int* iwork, blas_int* liwork,
    blas_int* info );

void scalapack_pcheevd(
    const char* jobz, const char* uplo, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA, float* W,
    std::complex<float>* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    std::complex<float>* work, blas_int* lwork,
    float* rwork, blas_int* lrwork, blas_int* iwork, blas_int* liwork,
    blas_int* info );

void scalapack_pzheevd(
    const char* jobz, const char* uplo, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA, double* W,
    std::complex<double>* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    std::complex<double>* work, blas_int* lwork,
    double* rwork, blas_int* lrwork, blas_int* iwork, blas_int* liwork,
    blas_int* info );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_pheevd(
    const char* jobz, const char* uplo, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA, float* W,
    float* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    float* work, blas_int* lwork,
    float* rwork, blas_int* lrwork, blas_int* iwork, blas_int* liwork,
    blas_int* info )
{
    scalapack_pssyevd(
        jobz, uplo, n,
        A, ia, ja, descA, W,
        Z, iz, jz, descZ,
        work, lwork, iwork, liwork, info );
}

inline void scalapack_pheevd(
    const char* jobz, const char* uplo, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA, double* W,
    double* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    double* work, blas_int* lwork,
    double* rwork, blas_int* lrwork, blas_int* iwork, blas_int* liwork,
    blas_int* info )
{
    scalapack_pdsyevd(
        jobz, uplo, n,
        A, ia, ja, descA, W,
        Z, iz, jz, descZ,
        work, lwork, iwork, liwork, info );
}

inline void scalapack_pheevd(
    const char* jobz, const char* uplo, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA, float* W,
    std::complex<float>* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    std::complex<float>* work, blas_int* lwork,
    float* rwork, blas_int* lrwork, blas_int* iwork, blas_int* liwork,
    blas_int* info )
{
    scalapack_pcheevd(
        jobz, uplo, n,
        A, ia, ja, descA, W,
        Z, iz, jz, descZ,
        work, lwork, rwork, lrwork, iwork, liwork, info );
}

inline void scalapack_pheevd(
    const char* jobz, const char* uplo, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA, double* W,
    std::complex<double>* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    std::complex<double>* work, blas_int* lwork,
    double* rwork, blas_int* lrwork, blas_int* iwork, blas_int* liwork,
    blas_int* info )
{
    scalapack_pzheevd(
        jobz, uplo, n,
        A, ia, ja, descA, W,
        Z, iz, jz, descZ,
        work, lwork, rwork, lrwork, iwork, liwork, info );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_pheevd(
    const char* jobz, const char* uplo,
    int64_t n,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    blas::real_type<scalar_t>* W,
    scalar_t* Z, int64_t iz, int64_t jz, blas_int* descZ,
    scalar_t* work, int64_t lwork,
    blas::real_type<scalar_t>* rwork, int64_t lrwork,
    blas_int* iwork, int64_t liwork,
    int64_t* info )
{
    blas_int n_      = to_blas_int( n );
    blas_int ia_     = to_blas_int( ia );
    blas_int ja_     = to_blas_int( ja );
    blas_int iz_     = to_blas_int( iz );
    blas_int jz_     = to_blas_int( jz );
    blas_int lwork_  = to_blas_int( lwork );
    blas_int lrwork_ = to_blas_int( lrwork );
    blas_int liwork_ = to_blas_int( n );
    blas_int info_   = 0;
    scalapack_pheevd(
        jobz, uplo, &n_,
        A, &ia_, &ja_, descA, W,
        Z, &iz_, &jz_, descZ,
        work, &lwork_, rwork, &lrwork_, iwork, &liwork_, &info_ );
    *info = int64_t( info_ );
}

//==============================================================================
// Fortran prototypes
#define scalapack_pslaset BLAS_FORTRAN_NAME( pslaset, PSLASET )
#define scalapack_pdlaset BLAS_FORTRAN_NAME( pdlaset, PDLASET )
#define scalapack_pclaset BLAS_FORTRAN_NAME( pclaset, PCLASET )
#define scalapack_pzlaset BLAS_FORTRAN_NAME( pzlaset, PZLASET )

extern "C" {

void scalapack_pslaset(
    const char* uplo, blas_int* m, blas_int* n,
    float* offdiag, float* diag,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA );

void scalapack_pdlaset(
    const char* uplo, blas_int* m, blas_int* n,
    double* offdiag, double* diag,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA );

void scalapack_pclaset(
    const char* uplo, blas_int* m, blas_int* n,
    std::complex<float>* offdiag, std::complex<float>* diag,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA );

void scalapack_pzlaset(
    const char* uplo, blas_int* m, blas_int* n,
    std::complex<double>* offdiag, std::complex<double>* diag,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_plaset(
    const char* uplo, blas_int* m, blas_int* n,
    float* offdiag, float* diag,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA )
{
    scalapack_pslaset(
        uplo, m, n, offdiag, diag,
        A, ia, ja, descA );
}

inline void scalapack_plaset(
    const char* uplo, blas_int* m, blas_int* n,
    double* offdiag, double* diag,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA )
{
    scalapack_pdlaset(
        uplo, m, n, offdiag, diag,
        A, ia, ja, descA );
}

inline void scalapack_plaset(
    const char* uplo, blas_int* m, blas_int* n,
    std::complex<float>* offdiag, std::complex<float>* diag,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA )
{
    scalapack_pclaset(
        uplo, m, n, offdiag, diag,
        A, ia, ja, descA );
}

inline void scalapack_plaset(
    const char* uplo, blas_int* m, blas_int* n,
    std::complex<double>* offdiag, std::complex<double>* diag,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA )
{
    scalapack_pzlaset(
        uplo, m, n, offdiag, diag,
        A, ia, ja, descA );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_plaset(
    const char* uplo, int64_t m, int64_t n,
    scalar_t offdiag, scalar_t diag,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA )
{
    blas_int m_  = to_blas_int( m );
    blas_int n_  = to_blas_int( n );
    blas_int ia_ = to_blas_int( ia );
    blas_int ja_ = to_blas_int( ja );
    scalapack_plaset(
        uplo, &m_, &n_, &offdiag, &diag,
        A, &ia_, &ja_, descA );
}

//==============================================================================
// Fortran prototypes
#define scalapack_pslacpy BLAS_FORTRAN_NAME( pslacpy, PSLACPY )
#define scalapack_pdlacpy BLAS_FORTRAN_NAME( pdlacpy, PDLACPY )
#define scalapack_pclacpy BLAS_FORTRAN_NAME( pclacpy, PCLACPY )
#define scalapack_pzlacpy BLAS_FORTRAN_NAME( pzlacpy, PZLACPY )

extern "C" {

void scalapack_pslacpy(
    const char* uplo, blas_int* m, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB );

void scalapack_pdlacpy(
    const char* uplo, blas_int* m, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB );

void scalapack_pclacpy(
    const char* uplo, blas_int* m, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB );

void scalapack_pzlacpy(
    const char* uplo, blas_int* m, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_placpy(
    const char* uplo, blas_int* m, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB )
{
    scalapack_pslacpy(
        uplo, m, n,
        A, ia, ja, descA,
        B, ib, jb, descB );
}

inline void scalapack_placpy(
    const char* uplo, blas_int* m, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB )
{
    scalapack_pdlacpy(
        uplo, m, n,
        A, ia, ja, descA,
        B, ib, jb, descB );
}

inline void scalapack_placpy(
    const char* uplo, blas_int* m, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB )
{
    scalapack_pclacpy(
        uplo, m, n,
        A, ia, ja, descA,
        B, ib, jb, descB );
}

inline void scalapack_placpy(
    const char* uplo, blas_int* m, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB )
{
    scalapack_pzlacpy(
        uplo, m, n,
        A, ia, ja, descA,
        B, ib, jb, descB );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_placpy(
    const char* uplo, int64_t m, int64_t n,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    scalar_t* B, int64_t ib, int64_t jb, blas_int* descB )
{
    blas_int m_  = to_blas_int( m );
    blas_int n_  = to_blas_int( n );
    blas_int ia_ = to_blas_int( ia );
    blas_int ja_ = to_blas_int( ja );
    blas_int ib_ = to_blas_int( ib );
    blas_int jb_ = to_blas_int( jb );
    scalapack_placpy(
        uplo, &m_, &n_,
        A, &ia_, &ja_, descA,
        B, &ib_, &jb_, descB );
}

//==============================================================================
// Fortran prototypes
#define scalapack_pssygvx BLAS_FORTRAN_NAME( pssygvx, PSSYGVX )
#define scalapack_pdsygvx BLAS_FORTRAN_NAME( pdsygvx, PDSYGVX )
#define scalapack_pchegvx BLAS_FORTRAN_NAME( pchegvx, PCHEGVX )
#define scalapack_pzhegvx BLAS_FORTRAN_NAME( pzhegvx, PZHEGVX )

extern "C" {

void scalapack_pssygvx(
    blas_int* itype, const char* jobz, const char* range, const char* uplo,
    blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    float* vl, float* vu,  blas_int* il, blas_int* iu, float* abstol,
    blas_int* nfound, blas_int* nzfound,
    float* W, float* orfac,
    float* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    float* work, blas_int* lwork, blas_int* iwork, blas_int* liwork,
    blas_int* ifail, blas_int* iclustr, float* gap, blas_int* info );

void scalapack_pdsygvx(
    blas_int* itype, const char* jobz, const char* range, const char* uplo,
    blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* vl, double* vu,  blas_int* il, blas_int* iu, double* abstol,
    blas_int* nfound, blas_int* nzfound,
    double* W, double* orfac,
    double* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    double* work, blas_int* lwork, blas_int* iwork, blas_int* liwork,
    blas_int* ifail, blas_int* iclustr, double* gap, blas_int* info );

void scalapack_pchegvx(
    blas_int* itype, const char* jobz, const char* range, const char* uplo,
    blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    float* vl, float* vu,  blas_int* il, blas_int* iu, float* abstol,
    blas_int* nfound, blas_int* nzfound,
    float* W, float* orfac,
    std::complex<float>* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    std::complex<float>* work, blas_int* lwork,
    float* rwork, blas_int* lrwork, blas_int* iwork, blas_int* liwork,
    blas_int* ifail, blas_int* iclustr, float* gap, blas_int* info );

void scalapack_pzhegvx(
    blas_int* itype, const char* jobz, const char* range, const char* uplo,
    blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* vl, double* vu,  blas_int* il, blas_int* iu, double* abstol,
    blas_int* nfound, blas_int* nzfound,
    double* W, double* orfac,
    std::complex<double>* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    std::complex<double>* work, blas_int* lwork,
    double* rwork, blas_int* lrwork, blas_int* iwork, blas_int* liwork,
    blas_int* ifail, blas_int* iclustr, double* gap, blas_int* info );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_phegvx(
    blas_int* itype, const char* jobz, const char* range, const char* uplo,
    blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    float* vl, float* vu,  blas_int* il, blas_int* iu, float* abstol,
    blas_int* nfound, blas_int* nzfound,
    float* W, float* orfac,
    float* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    float* work, blas_int* lwork,
    float* rwork, blas_int* lrwork, blas_int* iwork, blas_int* liwork,
    blas_int* ifail, blas_int* iclustr, float* gap, blas_int* info )
{
    scalapack_pssygvx(
        itype, jobz, range, uplo, n,
        A, ia, ja, descA,
        B, ib, jb, descB,
        vl, vu, il, iu, abstol, nfound,
        nzfound, W, orfac,
        Z, iz, jz, descZ,
        work, lwork, iwork, liwork,
        ifail, iclustr, gap, info );
}

inline void scalapack_phegvx(
    blas_int* itype, const char* jobz, const char* range, const char* uplo,
    blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* vl, double* vu,  blas_int* il, blas_int* iu, double* abstol,
    blas_int* nfound, blas_int* nzfound,
    double* W, double* orfac,
    double* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    double* work, blas_int* lwork,
    double* rwork, blas_int* lrwork, blas_int* iwork, blas_int* liwork,
    blas_int* ifail, blas_int* iclustr, double* gap, blas_int* info )
{
    scalapack_pdsygvx(
        itype, jobz, range, uplo, n,
        A, ia, ja, descA,
        B, ib, jb, descB,
        vl, vu, il, iu, abstol, nfound,
        nzfound, W, orfac,
        Z, iz, jz, descZ,
        work, lwork, iwork, liwork,
        ifail, iclustr, gap, info );
}

inline void scalapack_phegvx(
    blas_int* itype, const char* jobz, const char* range, const char* uplo,
    blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    float* vl, float* vu,  blas_int* il, blas_int* iu, float* abstol,
    blas_int* nfound, blas_int* nzfound,
    float* W, float* orfac,
    std::complex<float>* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    std::complex<float>* work, blas_int* lwork,
    float* rwork, blas_int* lrwork, blas_int* iwork, blas_int* liwork,
    blas_int* ifail, blas_int* iclustr, float* gap, blas_int* info )
{
    scalapack_pchegvx(
        itype, jobz, range, uplo, n,
        A, ia, ja, descA,
        B, ib, jb, descB,
        vl, vu, il, iu, abstol, nfound,
        nzfound, W, orfac,
        Z, iz, jz, descZ,
        work, lwork, rwork, lrwork, iwork, liwork,
        ifail, iclustr, gap, info );
}

inline void scalapack_phegvx(
    blas_int* itype, const char* jobz, const char* range, const char* uplo,
    blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* vl, double* vu,  blas_int* il, blas_int* iu, double* abstol,
    blas_int* nfound, blas_int* nzfound,
    double* W, double* orfac,
    std::complex<double>* Z, blas_int* iz, blas_int* jz, blas_int* descZ,
    std::complex<double>* work, blas_int* lwork,
    double* rwork, blas_int* lrwork, blas_int* iwork, blas_int* liwork,
    blas_int* ifail, blas_int* iclustr, double* gap, blas_int* info )
{
    scalapack_pzhegvx(
        itype, jobz, range, uplo, n,
        A, ia, ja, descA,
        B, ib, jb, descB,
        vl, vu, il, iu, abstol, nfound,
        nzfound, W, orfac,
        Z, iz, jz, descZ,
        work, lwork, rwork, lrwork, iwork, liwork,
        ifail, iclustr, gap, info );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_phegvx(
    int64_t itype, const char* jobz, const char* range, const char* uplo,
    int64_t n,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    scalar_t* B, int64_t ib, int64_t jb, blas_int* descB,
    blas::real_type<scalar_t> vl, blas::real_type<scalar_t> vu,
    int64_t il, int64_t iu,
    blas::real_type<scalar_t> abstol,
    int64_t* nfound, int64_t* nzfound,
    blas::real_type<scalar_t>* W,
    blas::real_type<scalar_t> orfac,
    scalar_t* Z, int64_t iz, int64_t jz, blas_int* descZ,
    scalar_t* work, int64_t lwork,
    blas::real_type<scalar_t>* rwork, int64_t lrwork,
    blas_int* iwork, int64_t liwork,
    blas_int* ifail, blas_int* iclustr, blas::real_type<scalar_t>* gap,
    int64_t* info )
{
    blas_int itype_   = to_blas_int( itype );
    blas_int n_       = to_blas_int( n );
    blas_int ia_      = to_blas_int( ia );
    blas_int ja_      = to_blas_int( ja );
    blas_int ib_      = to_blas_int( ib );
    blas_int jb_      = to_blas_int( jb );
    blas_int il_      = to_blas_int( il );
    blas_int iu_      = to_blas_int( iu );
    blas_int nfound_  = to_blas_int( *nfound );
    blas_int nzfound_ = to_blas_int( *nzfound );
    blas_int iz_      = to_blas_int( iz );
    blas_int jz_      = to_blas_int( jz );
    blas_int lwork_   = to_blas_int( lwork );
    blas_int lrwork_  = to_blas_int( lrwork );
    blas_int liwork_  = to_blas_int( liwork );
    blas_int info_    = 0;
    scalapack_phegvx(
        &itype_, jobz, range, uplo, &n_,
        A, &ia_, &ja_, descA,
        B, &ib_, &jb_, descB,
        &vl, &vu, &il_, &iu_, &abstol,
        &nfound_, &nzfound_, W, &orfac,
        Z, &iz_, &jz_, descZ,
        work, &lwork_, rwork, &lrwork_, iwork, &liwork_,
        ifail, iclustr, gap, &info_ );
    *nfound  = int64_t( nfound_ );
    *nzfound = int64_t( nzfound_ );
    *info    = int64_t( info_ );
}

//==============================================================================
// Fortran prototypes
#define scalapack_pssygst BLAS_FORTRAN_NAME( pssygst, PSSYGST )
#define scalapack_pdsygst BLAS_FORTRAN_NAME( pdsygst, PDSYGST )
#define scalapack_pchegst BLAS_FORTRAN_NAME( pchegst, PCHEGST )
#define scalapack_pzhegst BLAS_FORTRAN_NAME( pzhegst, PZHEGST )

extern "C" {

void scalapack_pssygst(
    blas_int* itype, const char* uplo, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* scale, blas_int* info );

void scalapack_pdsygst(
    blas_int* itype, const char* uplo, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* scale, blas_int* info );

void scalapack_pchegst(
    blas_int* itype, const char* uplo, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* scale,
    blas_int* info );

void scalapack_pzhegst(
    blas_int* itype, const char* uplo, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* scale,
    blas_int* info );

} // extern "C"

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_phegst(
    blas_int* itype, const char* uplo, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* scale,
    blas_int* info )
{
    scalapack_pssygst(
        itype, uplo, n,
        A, ia, ja, descA,
        B, ib, jb, descB,
        scale,
        info );
}

inline void scalapack_phegst(
    blas_int* itype, const char* uplo, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* scale,
    blas_int* info )
{
    scalapack_pdsygst(
        itype, uplo, n,
        A, ia, ja, descA,
        B, ib, jb, descB,
        scale,
        info );
}

inline void scalapack_phegst(
    blas_int* itype, const char* uplo, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<float>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* scale, blas_int* info )
{
    scalapack_pchegst(
        itype, uplo, n,
        A, ia, ja, descA,
        B, ib, jb, descB,
        scale, info );
}

inline void scalapack_phegst(
    blas_int* itype, const char* uplo, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    std::complex<double>* B, blas_int* ib, blas_int* jb, blas_int* descB,
    double* scale, blas_int* info )
{
    scalapack_pzhegst(
        itype, uplo, n,
        A, ia, ja, descA,
        B, ib, jb, descB,
        scale, info );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_phegst(
    int64_t itype, const char* uplo, int64_t n,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    scalar_t* B, int64_t ib, int64_t jb, blas_int* descB,
    double* scale,
    int64_t* info )
{
    blas_int itype_ = to_blas_int( itype );
    blas_int n_     = to_blas_int( n );
    blas_int ia_    = to_blas_int( ia );
    blas_int ja_    = to_blas_int( ja );
    blas_int ib_    = to_blas_int( ib );
    blas_int jb_    = to_blas_int( jb );
    blas_int info_  = 0;
    scalapack_phegst(
        &itype_, uplo, &n_,
        A, &ia_, &ja_, descA,
        B, &ib_, &jb_, descB,
        scale, &info_ );
    *info = info_;
}

//==============================================================================
// Fortran prototypes
#define scalapack_pslaqge BLAS_FORTRAN_NAME( pslaqge, PSlaqge )
#define scalapack_pdlaqge BLAS_FORTRAN_NAME( pdlaqge, PDlaqge )
#define scalapack_pclaqge BLAS_FORTRAN_NAME( pclaqge, PClaqge )
#define scalapack_pzlaqge BLAS_FORTRAN_NAME( pzlaqge, PZlaqge )

extern "C" {

void scalapack_pslaqge(
    blas_int* m, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* R, float* C,
    float* rowcnd, float* colcnd, float* Amax, char* equed );

void scalapack_pdlaqge(
    blas_int* m, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* R, double* C,
    double* rowcnd, double* colcnd, double* Amax, char* equed );

void scalapack_pclaqge(
    blas_int* m, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* R, float* C,
    float* rowcnd, float* colcnd, float* Amax, char* equed );

void scalapack_pzlaqge(
    blas_int* m, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* R, double* C,
    double* rowcnd, double* colcnd, double* Amax, char* equed );

} // extern C

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_plaqge(
    blas_int* m, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* R, float* C,
    float* rowcnd, float* colcnd, float* Amax, char* equed )
{
    scalapack_pslaqge(
        m, n, A, ia, ja, descA,
        R, C, rowcnd, colcnd, Amax, equed );
}

inline void scalapack_plaqge(
    blas_int* m, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* R, double* C,
    double* rowcnd, double* colcnd, double* Amax, char* equed )
{
    scalapack_pdlaqge(
        m, n, A, ia, ja, descA,
        R, C, rowcnd, colcnd, Amax, equed );
}

inline void scalapack_plaqge(
    blas_int* m, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* R, float* C,
    float* rowcnd, float* colcnd, float* Amax, char* equed )
{
    scalapack_pclaqge(
        m, n, A, ia, ja, descA,
        R, C, rowcnd, colcnd, Amax, equed );
}

inline void scalapack_plaqge(
    blas_int* m, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* R, double* C,
    double* rowcnd, double* colcnd, double* Amax, char* equed )
{
    scalapack_pzlaqge(
        m, n, A, ia, ja, descA,
        R, C, rowcnd, colcnd, Amax, equed );
}

//------------------------------------------------------------------------------
// Templated wrapper
// equed is an output, hence not const.
template <typename scalar_t>
void scalapack_plaqge(
    int64_t m, int64_t n,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    blas::real_type<scalar_t>* R,
    blas::real_type<scalar_t>* C,
    blas::real_type<scalar_t> rowcnd,
    blas::real_type<scalar_t> colcnd,
    blas::real_type<scalar_t> Amax, char* equed )
{
    blas_int m_  = to_blas_int( m );
    blas_int n_  = to_blas_int( n );
    blas_int ia_ = to_blas_int( ia );
    blas_int ja_ = to_blas_int( ja );
    scalapack_plaqge(
        &m_, &n_, A, &ia_, &ja_, descA,
        R, C, &rowcnd, &colcnd, &Amax, equed );
}

//==============================================================================
// Fortran prototypes
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
    blas_int* info );

void scalapack_pdgecon(
    const char* norm, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* Anorm, double* rcond,
    double* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    blas_int* info );

void scalapack_pcgecon(
    const char* norm, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* Anorm, float* rcond,
    std::complex<float>* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    blas_int* info );

void scalapack_pzgecon(
    const char* norm, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* Anorm, double* rcond,
    std::complex<double>* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    blas_int* info );

} // extern C

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_pgecon(
    const char* norm, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* Anorm, float* rcond,
    float* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    blas_int* info )
{
    scalapack_psgecon(
        norm, n, A, ia, ja, descA,
        Anorm, rcond, work, lwork, iwrok, liwork, info );
}

inline void scalapack_pgecon(
    const char* norm, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* Anorm, double* rcond,
    double* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    blas_int* info )
{
    scalapack_pdgecon(
        norm, n, A, ia, ja, descA,
        Anorm, rcond, work, lwork, iwrok, liwork, info );
}

inline void scalapack_pgecon(
    const char* norm, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* Anorm, float* rcond,
    std::complex<float>* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    blas_int* info )
{
    scalapack_pcgecon(
        norm, n, A, ia, ja, descA,
        Anorm, rcond, work, lwork, iwrok, liwork, info );
}

inline void scalapack_pgecon(
    const char* norm, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* Anorm, double* rcond,
    std::complex<double>* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    blas_int* info )
{
    scalapack_pzgecon(
        norm, n, A, ia, ja, descA,
        Anorm, rcond, work, lwork, iwrok, liwork, info );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_pgecon(
    const char* norm, int64_t n,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    blas::real_type<scalar_t>* Anorm, blas::real_type<scalar_t>* rcond,
    scalar_t* work, int64_t lwork,
    blas_int* iwork, int64_t liwork,
    int64_t* info )
{
    blas_int n_      = to_blas_int( n );
    blas_int ia_     = to_blas_int( ia );
    blas_int ja_     = to_blas_int( ja );
    blas_int lwork_  = to_blas_int( lwork );
    blas_int liwork_ = to_blas_int( liwork );
    blas_int info_   = 0;
    scalapack_pgecon(
        norm, &n_, A, &ia_, &ja_, descA,
        Anorm, rcond, work, &lwork_, iwork, &liwork_, &info_ );
}

//==============================================================================
// Fortran prototypes
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
    blas_int* info );

void scalapack_pdtrcon(
    const char* norm, const char* uplo, const char* diag, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* rcond,
    double* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    blas_int* info );

void scalapack_pctrcon(
    const char* norm, const char* uplo, const char* diag, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* rcond,
    std::complex<float>* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    blas_int* info );

void scalapack_pztrcon(
    const char* norm, const char* uplo, const char* diag, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* rcond,
    std::complex<double>* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    blas_int* info );

} // extern C

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_ptrcon(
    const char* norm, const char* uplo, const char* diag, blas_int* n,
    float* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* rcond,
    float* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    blas_int* info )
{
    scalapack_pstrcon(
        norm, uplo, diag, n, A, ia, ja, descA,
        rcond, work, lwork, iwrok, liwork, info );
}

inline void scalapack_ptrcon(
    const char* norm, const char* uplo, const char* diag, blas_int* n,
    double* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* rcond,
    double* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    blas_int* info )
{
    scalapack_pdtrcon(
        norm, uplo, diag, n, A, ia, ja, descA,
        rcond, work, lwork, iwrok, liwork, info );
}

inline void scalapack_ptrcon(
    const char* norm, const char* uplo, const char* diag, blas_int* n,
    std::complex<float>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    float* rcond,
    std::complex<float>* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    blas_int* info )
{
    scalapack_pctrcon(
        norm, uplo, diag, n, A, ia, ja, descA,
        rcond, work, lwork, iwrok, liwork, info );
}

inline void scalapack_ptrcon(
    const char* norm, const char* uplo, const char* diag, blas_int* n,
    std::complex<double>* A, blas_int* ia, blas_int* ja, blas_int* descA,
    double* rcond,
    std::complex<double>* work, blas_int* lwork,
    blas_int* iwrok, blas_int* liwork,
    blas_int* info )
{
    scalapack_pztrcon(
        norm, uplo, diag, n, A, ia, ja, descA,
        rcond, work, lwork, iwrok, liwork, info );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_ptrcon(
    const char* norm, const char* uplo, const char* diag, int64_t n,
    scalar_t* A, int64_t ia, int64_t ja, blas_int* descA,
    blas::real_type<scalar_t>* rcond,
    scalar_t* work, int64_t lwork,
    blas_int* iwork, int64_t liwork,
    int64_t info )
{
    blas_int n_      = to_blas_int( n );
    blas_int ia_     = to_blas_int( ia );
    blas_int ja_     = to_blas_int( ja );
    blas_int lwork_  = to_blas_int( lwork );
    blas_int liwork_ = to_blas_int( liwork );
    blas_int info_   = 0;
    scalapack_ptrcon(
        norm, uplo, diag, &n_, A, &ia_, &ja_, descA,
        rcond, work, &lwork_, iwork, &liwork_, &info_ );
}

//==============================================================================
// Fortran prototypes
#define scalapack_pslaed2 BLAS_FORTRAN_NAME( pslaed2, PSLAED2 )
#define scalapack_pdlaed2 BLAS_FORTRAN_NAME( pdlaed2, PDLAED2 )

extern "C"
void scalapack_pslaed2(
    blas_int* ictxt, blas_int* nsecular,
    blas_int* n, blas_int* n1, blas_int* nb,
    float* D, blas_int* drow, blas_int* dcol,
    float* Q, blas_int* ldq,
    float* rho, float* z, float* w, float* dlambda,
    float* Q2, blas_int* ldq2, float* qbuf,
    blas_int* ctot, blas_int* psm, blas_int* npcol,
    blas_int* indx, blas_int* indxc, blas_int* indxp,
    blas_int* indcol, blas_int* coltyp,
    blas_int* nn, blas_int* nn1, blas_int* nn2,
    blas_int* ib1, blas_int* ib2 );

extern "C"
void scalapack_pdlaed2(
    blas_int* ictxt, blas_int* nsecular,
    blas_int* n, blas_int* n1, blas_int* nb,
    double* D, blas_int* drow, blas_int* dcol,
    double* Q, blas_int* ldq,
    double* rho, double* z, double* w, double* dlambda,
    double* Q2, blas_int* ldq2, double* qbuf,
    blas_int* ctot, blas_int* psm, blas_int* npcol,
    blas_int* indx, blas_int* indxc, blas_int* indxp,
    blas_int* indcol, blas_int* coltyp,
    blas_int* nn, blas_int* nn1, blas_int* nn2,
    blas_int* ib1, blas_int* ib2 );

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_plaed2(
    blas_int* ictxt, blas_int* nsecular,
    blas_int* n, blas_int* n1, blas_int* nb,
    float* D, blas_int* drow, blas_int* dcol,
    float* Q, blas_int* ldq,
    float* rho, float* z, float* w, float* dlambda,
    float* Q2, blas_int* ldq2, float* qbuf,
    blas_int* ctot, blas_int* psm, blas_int* npcol,
    blas_int* indx, blas_int* indxc, blas_int* indxp,
    blas_int* indcol, blas_int* coltyp,
    blas_int* nn, blas_int* nn1, blas_int* nn2,
    blas_int* ib1, blas_int* ib2 )
{
    scalapack_pslaed2(
        ictxt, nsecular, n, n1, nb, D, drow, dcol,
        Q, ldq, rho, z, w, dlambda, Q2, ldq2, qbuf,
        ctot, psm, npcol, indx, indxc, indxp, indcol, coltyp,
        nn, nn1, nn2, ib1, ib2 );
}

inline void scalapack_plaed2(
    blas_int* ictxt, blas_int* nsecular,
    blas_int* n, blas_int* n1, blas_int* nb,
    double* D, blas_int* drow, blas_int* dcol,
    double* Q, blas_int* ldq,
    double* rho, double* z, double* w, double* dlambda,
    double* Q2, blas_int* ldq2, double* qbuf,
    blas_int* ctot, blas_int* psm, blas_int* npcol,
    blas_int* indx, blas_int* indxc, blas_int* indxp,
    blas_int* indcol, blas_int* coltyp,
    blas_int* nn, blas_int* nn1, blas_int* nn2,
    blas_int* ib1, blas_int* ib2 )
{
    scalapack_pdlaed2(
        ictxt, nsecular, n, n1, nb, D, drow, dcol,
        Q, ldq, rho, z, w, dlambda, Q2, ldq2, qbuf,
        ctot, psm, npcol, indx, indxc, indxp, indcol, coltyp,
        nn, nn1, nn2, ib1, ib2 );
}

//------------------------------------------------------------------------------
// Templated wrapper
// This takes a lot of output integers that are left as blas_int.
template <typename scalar_t>
void scalapack_plaed2(
    blas_int ictxt, blas_int* nsecular,
    int64_t n, int64_t n1, int64_t nb,
    scalar_t* D, int64_t drow, int64_t dcol,
    scalar_t* Q, int64_t ldq,
    scalar_t* rho, scalar_t* z, scalar_t* w, scalar_t* dlambda,
    scalar_t* Q2, int64_t ldq2, scalar_t* Qbuf,
    blas_int* ctot, blas_int* psm, blas_int  npcol,
    blas_int* indx, blas_int* indxc, blas_int* indxp,
    blas_int* indcol, blas_int* coltyp,
    blas_int* nn, blas_int* nn1, blas_int* nn2,
    blas_int* ib1, blas_int* ib2 )
{
    blas_int n_    = to_blas_int( n );
    blas_int n1_   = to_blas_int( n1 );
    blas_int nb_   = to_blas_int( nb );
    blas_int drow_ = to_blas_int( drow );
    blas_int dcol_ = to_blas_int( dcol );
    blas_int ldq_  = to_blas_int( ldq );
    blas_int ldq2_ = to_blas_int( ldq2 );
    scalapack_plaed2(
        &ictxt, nsecular, &n_, &n1_, &nb_, D, &drow_, &dcol_,
        Q, &ldq_, rho, z, w, dlambda, Q2, &ldq2_, Qbuf,
        ctot, psm, &npcol, indx, indxc, indxp, indcol, coltyp,
        nn, nn1, nn2, ib1, ib2 );
}

//==============================================================================
// Fortran prototypes
#define scalapack_pslaed3 BLAS_FORTRAN_NAME( pslaed3, PSLAED3 )
#define scalapack_pdlaed3 BLAS_FORTRAN_NAME( pdlaed3, PDLAED3 )

extern "C"
void scalapack_pslaed3(
    blas_int* ictxt, blas_int* nsecular, blas_int* n, blas_int* nb,
    float* Lambda, blas_int* drow, blas_int* dcol,
    float* rho, float* D, float* z, float* ztilde,
    float* U, blas_int* ldu, float* buf,
    blas_int* idx_Q_global, blas_int* pcols, blas_int* prows,
    blas_int* idx_row, blas_int* idx_col, blas_int* ct_count, blas_int* npcol,
    blas_int* info );

extern "C"
void scalapack_pdlaed3(
    blas_int* ictxt, blas_int* nsecular, blas_int* n, blas_int* nb,
    double* Lambda, blas_int* drow, blas_int* dcol,
    double* rho, double* D, double* z, double* ztilde,
    double* U, blas_int* ldu, double* buf,
    blas_int* idx_Q_global, blas_int* pcols, blas_int* prows,
    blas_int* idx_row, blas_int* idx_col, blas_int* ct_count, blas_int* npcol,
    blas_int* info );

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_plaed3(
    blas_int* ictxt, blas_int* nsecular, blas_int* n, blas_int* nb,
    float* Lambda, blas_int* drow, blas_int* dcol,
    float* rho, float* D, float* z, float* ztilde,
    float* U, blas_int* ldu, float* buf,
    blas_int* idx_Q_global, blas_int* pcols, blas_int* prows,
    blas_int* idx_row, blas_int* idx_col, blas_int* ct_count, blas_int* npcol,
    blas_int* info )
{
    scalapack_pslaed3(
        ictxt, nsecular, n, nb, Lambda, drow, dcol,
        rho, D, z, ztilde, U, ldu, buf,
        idx_Q_global, pcols, prows, idx_row, idx_col, ct_count,
        npcol, info );
}

inline void scalapack_plaed3(
    blas_int* ictxt, blas_int* nsecular, blas_int* n, blas_int* nb,
    double* Lambda, blas_int* drow, blas_int* dcol,
    double* rho, double* D, double* z, double* ztilde,
    double* U, blas_int* ldu, double* buf,
    blas_int* idx_Q_global, blas_int* pcols, blas_int* prows,
    blas_int* idx_row, blas_int* idx_col, blas_int* ct_count, blas_int* npcol,
    blas_int* info )
{
    scalapack_pdlaed3(
        ictxt, nsecular, n, nb, Lambda, drow, dcol,
        rho, D, z, ztilde, U, ldu, buf,
        idx_Q_global, pcols, prows, idx_row, idx_col, ct_count,
        npcol, info );
}

//------------------------------------------------------------------------------
// Templated wrapper
// This takes a lot of output integers that are left as blas_int.
template <typename scalar_t>
void scalapack_plaed3(
    blas_int ictxt, int64_t nsecular, int64_t n, int64_t nb,
    scalar_t* Lambda, int64_t drow, int64_t dcol,
    scalar_t rho, scalar_t* D, scalar_t* z, scalar_t* ztilde,
    scalar_t* U, int64_t ldu, scalar_t* buf,
    blas_int* idx_Q_global, blas_int* pcols, blas_int* prows,
    blas_int* idx_row, blas_int* idx_col, blas_int* ct_count, int64_t npcol,
    int64_t* info )
{
    blas_int nsecular_ = to_blas_int( nsecular );
    blas_int n_        = to_blas_int( n );
    blas_int nb_       = to_blas_int( nb );
    blas_int drow_     = to_blas_int( drow );
    blas_int dcol_     = to_blas_int( dcol );
    blas_int ldu_      = to_blas_int( ldu );
    blas_int npcol_    = to_blas_int( npcol );
    blas_int info_     = 0;
    scalapack_plaed3(
        &ictxt, &nsecular_, &n_, &nb_, Lambda, &drow_, &dcol_,
        &rho, D, z, ztilde, U, &ldu_, buf,
        idx_Q_global, pcols, prows, idx_row, idx_col, ct_count,
        &npcol_, &info_ );
    *info = info_;
}

//==============================================================================
// Fortran prototypes
#define scalapack_pslaedz BLAS_FORTRAN_NAME( pslaedz, PSLAEDZ )
#define scalapack_pdlaedz BLAS_FORTRAN_NAME( pdlaedz, PDLAEDZ )

extern "C" void scalapack_pslaedz(
    blas_int* n, blas_int* n1, blas_int* id,
    float* Q, blas_int* iq, blas_int* jq, blas_int* ldq, blas_int* descQ,
    float* z, float* work );

extern "C" void scalapack_pdlaedz(
    blas_int* n, blas_int* n1, blas_int* id,
    double* Q, blas_int* iq, blas_int* jq, blas_int* ldq, blas_int* descQ,
    double* z, double* work );

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_plaedz(
    blas_int* n, blas_int* n1, blas_int* id,
    float* Q, blas_int* iq, blas_int* jq, blas_int* ldq, blas_int* descQ,
    float* z, float* work )
{
    scalapack_pslaedz(
        n, n1, id, Q, iq, jq, ldq, descQ, z, work );
}

inline void scalapack_plaedz(
    blas_int* n, blas_int* n1, blas_int* id,
    double* Q, blas_int* iq, blas_int* jq, blas_int* ldq, blas_int* descQ,
    double* z, double* work )
{
    scalapack_pdlaedz(
        n, n1, id, Q, iq, jq, ldq, descQ, z, work );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_plaedz(
    int64_t n, int64_t n1, int64_t id,
    scalar_t* Q, int64_t iq, int64_t jq, int64_t ldq, blas_int* descQ,
    scalar_t* z, scalar_t* work )
{
    blas_int n_   = to_blas_int( n );
    blas_int n1_  = to_blas_int( n1 );
    blas_int id_  = to_blas_int( id );
    blas_int iq_  = to_blas_int( iq );
    blas_int jq_  = to_blas_int( jq );
    blas_int ldq_ = to_blas_int( ldq );
    scalapack_plaedz(
        &n_, &n1_, &id_, Q, &iq_, &jq_, &ldq_, descQ, z, work );
}

//==============================================================================
// Fortran prototypes
#define scalapack_pslasrt BLAS_FORTRAN_NAME( pslasrt, PSLASRT )
#define scalapack_pdlasrt BLAS_FORTRAN_NAME( pdlasrt, PDLASRT )

extern "C"
void scalapack_pslasrt(
    const char* id, blas_int* n, float* D,
    float* Q, blas_int* iq, blas_int* jq, blas_int* descQ,
    float* work, blas_int* lwork,
    blas_int* iwork, blas_int* liwork,
    blas_int* info );

extern "C"
void scalapack_pdlasrt(
    const char* id, blas_int* n, double* D,
    double* Q, blas_int* iq, blas_int* jq, blas_int* descQ,
    double* work, blas_int* lwork,
    blas_int* iwork, blas_int* liwork,
    blas_int* info );

//------------------------------------------------------------------------------
// Low-level overloaded wrappers
inline void scalapack_plasrt(
    const char* id, blas_int* n, float* D,
    float* Q, blas_int* iq, blas_int* jq, blas_int* descQ,
    float* work, blas_int* lwork,
    blas_int* iwork, blas_int* liwork,
    blas_int* info )
{
    scalapack_pslasrt(
        id, n, D, Q, iq, jq, descQ,
        work, lwork, iwork, liwork, info );
}

inline void scalapack_plasrt(
    const char* id, blas_int* n, double* D,
    double* Q, blas_int* iq, blas_int* jq, blas_int* descQ,
    double* work, blas_int* lwork,
    blas_int* iwork, blas_int* liwork,
    blas_int* info )
{
    scalapack_pdlasrt(
        id, n, D, Q, iq, jq, descQ,
        work, lwork, iwork, liwork, info );
}

//------------------------------------------------------------------------------
// Templated wrapper
template <typename scalar_t>
void scalapack_plasrt(
    const char* id, int64_t n, scalar_t* D,
    scalar_t* Q, int64_t iq, int64_t jq, blas_int* descQ,
    scalar_t* work, int64_t lwork,
    blas_int* iwork, int64_t liwork,
    int64_t* info )
{
    blas_int n_      = to_blas_int( n );
    blas_int iq_     = to_blas_int( iq );
    blas_int jq_     = to_blas_int( jq );
    blas_int lwork_  = to_blas_int( lwork );
    blas_int liwork_ = to_blas_int( liwork );
    blas_int info_   = 0;
    scalapack_plasrt(
        id, &n_, D, Q, &iq_, &jq_, descQ,
        work, &lwork_, iwork, &liwork_, &info_ );
    *info = int64_t( info_ );
}

#endif // SLATE_SCALAPACK_WRAPPERS_HH
