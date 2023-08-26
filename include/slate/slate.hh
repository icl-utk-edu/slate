// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_HH
#define SLATE_HH

#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/SymmetricMatrix.hh"
#include "slate/TriangularMatrix.hh"

#include "slate/BandMatrix.hh"
#include "slate/TriangularBandMatrix.hh"
#include "slate/HermitianBandMatrix.hh"

#include "slate/method.hh"

#include "slate/types.hh"
#include "slate/print.hh"

//------------------------------------------------------------------------------
/// @namespace slate
/// SLATE's top-level namespace.
///
namespace slate {

// Version is updated by make_release.py; DO NOT EDIT.
// Version 2023.08.25
#define SLATE_VERSION 20230825

int version();
const char* id();

//------------------------------------------------------------------------------
// Level 2 Auxiliary

//-----------------------------------------
// add()
template <typename scalar_t>
void add(
    scalar_t alpha, Matrix<scalar_t>& A,
    scalar_t beta,  Matrix<scalar_t>& B,
    Options const& opts = Options());

template <typename scalar_t>
void add(
     scalar_t alpha, BaseTrapezoidMatrix<scalar_t>& A,
     scalar_t beta,  BaseTrapezoidMatrix<scalar_t>& B,
     Options const& opts = Options());

//-----------------------------------------
// copy()
template <typename src_matrix_type, typename dst_matrix_type>
void copy(
    src_matrix_type& A,
    dst_matrix_type& B,
    Options const& opts = Options());

//-----------------------------------------
// scale()
// General matrix.
template <typename scalar_t>
void scale(
    blas::real_type<scalar_t> numer,
    blas::real_type<scalar_t> denom,
    Matrix<scalar_t>& A,
    Options const& opts = Options());

/// General matrix, version with denom = 1.0.
template <typename scalar_t>
void scale(
    blas::real_type<scalar_t> value,
    Matrix<scalar_t>& A,
    Options const& opts = Options())
{
    blas::real_type<scalar_t> one = 1.0;
    scale(value, one, A, opts);
}

// BaseTrapezoid matrix.
template <typename scalar_t>
void scale(
    blas::real_type<scalar_t> numer,
    blas::real_type<scalar_t> denom,
    BaseTrapezoidMatrix<scalar_t>& A,
    Options const& opts = Options());

/// BaseTrapezoid matrix, version with denom = 1.0.
template <typename scalar_t>
void scale(
    blas::real_type<scalar_t> value,
    BaseTrapezoidMatrix<scalar_t>& A,
    Options const& opts = Options())
{
    blas::real_type<scalar_t> one = 1.0;
    scale(value, one, A, opts);
}

//-----------------------------------------
// scale_row_col
// General matrix.
template <typename scalar_t, typename scalar_t2>
void scale_row_col(
    Equed equed,
    std::vector< scalar_t2 > const& R,
    std::vector< scalar_t2 > const& C,
    Matrix<scalar_t>& A,
    Options const& opts = Options());

//-----------------------------------------
// set()
template <typename scalar_t>
void set(
    scalar_t offdiag_value,
    scalar_t diag_value,
    Matrix<scalar_t>& A,
    Options const& opts = Options());

template <typename scalar_t>
void set(
    scalar_t value,
    Matrix<scalar_t>& A,
    Options const& opts = Options())
{
    set(value, value, A, opts);
}

template <typename scalar_t>
void set(
    scalar_t offdiag_value,
    scalar_t diag_value,
    BaseTrapezoidMatrix<scalar_t>& A,
    Options const& opts = Options());

template <typename scalar_t>
void set(
    scalar_t value,
    BaseTrapezoidMatrix<scalar_t>& A,
    Options const& opts = Options())
{
    set(value, value, A, opts);
}

//------------------------------------------------------------------------------
// Level 3 BLAS and LAPACK auxiliary

//-----------------------------------------
// gbmm()
template <typename scalar_t>
void gbmm(
    scalar_t alpha, BandMatrix<scalar_t>& A,
                        Matrix<scalar_t>& B,
    scalar_t beta,      Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
// gemm()
template <typename scalar_t>
void gemm(
    scalar_t alpha, Matrix<scalar_t>& A,
                    Matrix<scalar_t>& B,
    scalar_t beta,  Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
// gemmA()
template <typename scalar_t>
void gemmA(
    scalar_t alpha, Matrix<scalar_t>& A,
                    Matrix<scalar_t>& B,
    scalar_t beta,  Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
// gemmC()
template <typename scalar_t>
void gemmC(
    scalar_t alpha, Matrix<scalar_t>& A,
                    Matrix<scalar_t>& B,
    scalar_t beta,  Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
// hbmm()
template <typename scalar_t>
void hbmm(
    Side side,
    scalar_t alpha, HermitianBandMatrix<scalar_t>& A,
                                 Matrix<scalar_t>& B,
    scalar_t beta,               Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
// hemm()
template <typename scalar_t>
void hemm(
    Side side,
    scalar_t alpha, HermitianMatrix<scalar_t>& A,
                             Matrix<scalar_t>& B,
    scalar_t beta,           Matrix<scalar_t>& C,
    Options const& opts = Options());

// forward real-symmetric matrices to hemm;
// disabled for complex
template <typename scalar_t>
void hemm(
    Side side,
    scalar_t alpha, SymmetricMatrix<scalar_t>& A,
                             Matrix<scalar_t>& B,
    scalar_t beta,           Matrix<scalar_t>& C,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH(A);
    hemm(side, alpha, AH, B, beta, C, opts);
}

//-----------------------------------------
// hemmA()
template <typename scalar_t>
void hemmA(
    Side side,
    scalar_t alpha, HermitianMatrix<scalar_t>& A,
                             Matrix<scalar_t>& B,
    scalar_t beta,           Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
// hemmC()
template <typename scalar_t>
void hemmC(
    Side side,
    scalar_t alpha, HermitianMatrix<scalar_t>& A,
                             Matrix<scalar_t>& B,
    scalar_t beta,           Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
// symm()
template <typename scalar_t>
void symm(
    Side side,
    scalar_t alpha, SymmetricMatrix<scalar_t>& A,
                             Matrix<scalar_t>& B,
    scalar_t beta,           Matrix<scalar_t>& C,
    Options const& opts = Options());

// forward real-Hermitian matrices to symm;
// disabled for complex
template <typename scalar_t>
void symm(
    Side side,
    scalar_t alpha, HermitianMatrix<scalar_t>& A,
                             Matrix<scalar_t>& B,
    scalar_t beta,           Matrix<scalar_t>& C,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    SymmetricMatrix<scalar_t> AS(A);
    symm(side, alpha, AS, B, beta, C, opts);
}

//-----------------------------------------
// trmm()
template <typename scalar_t>
void trmm(
    Side side,
    scalar_t alpha, TriangularMatrix<scalar_t>& A,
                              Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// tbsm()
template <typename scalar_t>
void tbsm(
    Side side,
    scalar_t alpha, TriangularBandMatrix<scalar_t>& A, Pivots& pivots,
                                  Matrix<scalar_t>& B,
    Options const& opts = Options());

template <typename scalar_t>
void tbsm(
    Side side,
    scalar_t alpha, TriangularBandMatrix<scalar_t>& A,
                                  Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// trsm()
template <typename scalar_t>
void trsm(
    Side side,
    scalar_t alpha, TriangularMatrix<scalar_t>& A,
                              Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// trsmA()
template <typename scalar_t>
void trsmA(
    Side side,
    scalar_t alpha, TriangularMatrix<scalar_t>& A,
                              Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// trsmB()
template <typename scalar_t>
void trsmB(
    Side side,
    scalar_t alpha, TriangularMatrix<scalar_t>& A,
                              Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// trtri()
template <typename scalar_t>
void trtri(
    TriangularMatrix<scalar_t>& A,
    Options const& opts = Options());

//-----------------------------------------
// trtrm()
template <typename scalar_t>
void trtrm(
    TriangularMatrix<scalar_t>& A,
    Options const& opts = Options());

//-----------------------------------------
// herk()
template <typename scalar_t>
void herk(
    blas::real_type<scalar_t> alpha,          Matrix<scalar_t>& A,
    blas::real_type<scalar_t> beta,  HermitianMatrix<scalar_t>& C,
    Options const& opts = Options());

// forward real-symmetric matrices to herk;
// disabled for complex
template <typename scalar_t>
void herk(
    blas::real_type<scalar_t> alpha,          Matrix<scalar_t>& A,
    blas::real_type<scalar_t> beta,  SymmetricMatrix<scalar_t>& C,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> CH(C);
    herk(alpha, A, beta, CH, opts);
}

//-----------------------------------------
// syrk()
template <typename scalar_t>
void syrk(
    scalar_t alpha,          Matrix<scalar_t>& A,
    scalar_t beta,  SymmetricMatrix<scalar_t>& C,
    Options const& opts = Options());

// forward real-Hermitian matrices to syrk;
// disabled for complex
template <typename scalar_t>
void syrk(
    scalar_t alpha,          Matrix<scalar_t>& A,
    scalar_t beta,  HermitianMatrix<scalar_t>& C,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    SymmetricMatrix<scalar_t> CS(C);
    syrk(alpha, A, beta, CS, opts);
}

//-----------------------------------------
// her2k()
template <typename scalar_t>
void her2k(
    scalar_t alpha,                          Matrix<scalar_t>& A,
                                             Matrix<scalar_t>& B,
    blas::real_type<scalar_t> beta, HermitianMatrix<scalar_t>& C,
    Options const& opts = Options());

// forward real-symmetric matrices to her2k;
// disabled for complex
template <typename scalar_t>
void her2k(
    scalar_t alpha,                           Matrix<scalar_t>& A,
                                              Matrix<scalar_t>& B,
    blas::real_type<scalar_t> beta,  SymmetricMatrix<scalar_t>& C,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> CH(C);
    her2k(alpha, A, B, beta, CH, opts);
}

//-----------------------------------------
// redistribute()
template <typename scalar_t>
void redistribute(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// syr2k()
template <typename scalar_t>
void syr2k(
    scalar_t alpha,          Matrix<scalar_t>& A,
                             Matrix<scalar_t>& B,
    scalar_t beta,  SymmetricMatrix<scalar_t>& C,
    Options const& opts = Options());

// forward real-Hermitian matrices to syr2k;
// disabled for complex
template <typename scalar_t>
void syr2k(
    scalar_t alpha,          Matrix<scalar_t>& A,
                             Matrix<scalar_t>& B,
    scalar_t beta,  HermitianMatrix<scalar_t>& C,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    SymmetricMatrix<scalar_t> CS(C);
    syr2k(alpha, A, B, beta, CS, opts);
}

//------------------------------------------------------------------------------
// Norms

//-----------------------------------------
// norm()
template <typename matrix_type>
blas::real_type<typename matrix_type::value_type>
norm(
    Norm norm,
    matrix_type& A,
    Options const& opts = Options());

//-----------------------------------------
// norm for triangular case
template <typename scalar_t>
blas::real_type<scalar_t>
norm(
    Norm trnorm,
    TriangularMatrix<scalar_t>& A,
    Options const& opts = Options())
{
    return norm< TrapezoidMatrix<scalar_t> >( trnorm, A, opts );
}

//-----------------------------------------
// colNorms()
// all cols max norm
template <typename matrix_type>
void colNorms(
    Norm norm,
    matrix_type& A,
    blas::real_type<typename matrix_type::value_type>* values,
    Options const& opts = Options());

//------------------------------------------------------------------------------
// Linear systems

//-----------------------------------------
// LU

//-----------------------------------------
// gbsv()
template <typename scalar_t>
void gbsv(
    BandMatrix<scalar_t>& A, Pivots& pivots,
        Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// gesv()
template <typename scalar_t>
void gesv(
    Matrix<scalar_t>& A, Pivots& pivots,
    Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// gesv_nopiv()
// todo: deprecate, use gesv( ..., { MethodLU: NoPiv } )
template <typename scalar_t>
void gesv_nopiv(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// gesv_mixed()
template <typename scalar_t>
void gesv_mixed(
    Matrix<scalar_t>& A, Pivots& pivots,
    Matrix<scalar_t>& B,
    Matrix<scalar_t>& X,
    int& iter,
    Options const& opts = Options());

template <typename scalar_hi, typename scalar_lo>
void gesv_mixed(
    Matrix<scalar_hi>& A, Pivots& pivots,
    Matrix<scalar_hi>& B,
    Matrix<scalar_hi>& X,
    int& iter,
    Options const& opts = Options());

template <typename scalar_t>
[[deprecated( "Use gesv_mixed instead. Will be removed 2024-02." )]]
void gesvMixed(
    Matrix<scalar_t>& A, Pivots& pivots,
    Matrix<scalar_t>& B,
    Matrix<scalar_t>& X,
    int& iter,
    Options const& opts = Options())
{
    gesv_mixed( A, pivots, B, X, iter, opts );
}

template <typename scalar_hi, typename scalar_lo>
[[deprecated( "Use gesv_mixed instead. Will be removed 2024-02." )]]
void gesvMixed(
    Matrix<scalar_hi>& A, Pivots& pivots,
    Matrix<scalar_hi>& B,
    Matrix<scalar_hi>& X,
    int& iter,
    Options const& opts = Options())
{
    gesv_mixed( A, pivots, B, X, iter, opts );
}

//-----------------------------------------
// gesv_mixed_gmres()
template <typename scalar_t>
void gesv_mixed_gmres(
    Matrix<scalar_t>& A, Pivots& pivots,
    Matrix<scalar_t>& B,
    Matrix<scalar_t>& X,
    int& iter,
    Options const& opts = Options());

template <typename scalar_hi, typename scalar_lo>
void gesv_mixed_gmres(
    Matrix<scalar_hi>& A, Pivots& pivots,
    Matrix<scalar_hi>& B,
    Matrix<scalar_hi>& X,
    int& iter,
    Options const& opts = Options());

//-----------------------------------------
// gbtrf()
template <typename scalar_t>
void gbtrf(
    BandMatrix<scalar_t>& A, Pivots& pivots,
    Options const& opts = Options());

//-----------------------------------------
// getrf()
template <typename scalar_t>
void getrf(
    Matrix<scalar_t>& A, Pivots& pivots,
    Options const& opts = Options());

//-----------------------------------------
// getrf_nopiv()
template <typename scalar_t>
void getrf_nopiv(
    Matrix<scalar_t>& A,
    Options const& opts = Options());

//-----------------------------------------
// getrf_tntpiv()
template <typename scalar_t>
void getrf_tntpiv(
    Matrix<scalar_t>& A, Pivots& pivots,
    Options const& opts = Options());

//-----------------------------------------
// gbtrs()
template <typename scalar_t>
void gbtrs(
    BandMatrix<scalar_t>& A, Pivots& pivots,
        Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// getrs()
template <typename scalar_t>
void getrs(
    Matrix<scalar_t>& A, Pivots& pivots,
    Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// getrs_nopiv()
// todo: deprecate, use getrs( ..., { MethodLU: NoPiv } )
template <typename scalar_t>
void getrs_nopiv(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// getri()
// In-place
template <typename scalar_t>
void getri(
    Matrix<scalar_t>& A, Pivots& pivots,
    Options const& opts = Options());

// Out-of-place
template <typename scalar_t>
void getri(
    Matrix<scalar_t>& A, Pivots& pivots,
    Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// Cholesky

//-----------------------------------------
// pbsv()
template <typename scalar_t>
void pbsv(
    HermitianBandMatrix<scalar_t>& A,
                 Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// posv()
template <typename scalar_t>
void posv(
    HermitianMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
    Options const& opts = Options());

// forward real-symmetric matrices to potrf;
// disabled for complex
template <typename scalar_t>
void posv(
    SymmetricMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH(A);
    posv(AH, B, opts);
}

//-----------------------------------------
// posv_mixed()
template <typename scalar_t>
void posv_mixed(
    HermitianMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
             Matrix<scalar_t>& X,
    int& iter,
    Options const& opts = Options());

template <typename scalar_hi, typename scalar_lo>
void posv_mixed(
    HermitianMatrix<scalar_hi>& A,
             Matrix<scalar_hi>& B,
             Matrix<scalar_hi>& X,
    int& iter,
    Options const& opts = Options());

// todo: forward real-symmetric matrices to posv_mixed?

template <typename scalar_t>
[[deprecated( "Use posv_mixed instead. Will be removed 2024-02." )]]
void posvMixed(
    HermitianMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
             Matrix<scalar_t>& X,
    int& iter,
    Options const& opts = Options())
{
    posv_mixed( A, B, X, iter, opts );
}

template <typename scalar_hi, typename scalar_lo>
[[deprecated( "Use posv_mixed instead. Will be removed 2024-02." )]]
void posvMixed(
    HermitianMatrix<scalar_hi>& A,
             Matrix<scalar_hi>& B,
             Matrix<scalar_hi>& X,
    int& iter,
    Options const& opts = Options())
{
    posv_mixed( A, B, X, iter, opts );
}

//-----------------------------------------
// posv_mixed_gmres()
template <typename scalar_t>
void posv_mixed_gmres(
    HermitianMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
             Matrix<scalar_t>& X,
    int& iter,
    Options const& opts = Options());

template <typename scalar_hi, typename scalar_lo>
void posv_mixed_gmres(
    HermitianMatrix<scalar_hi>& A,
             Matrix<scalar_hi>& B,
             Matrix<scalar_hi>& X,
    int& iter,
    Options const& opts = Options());

// todo: forward real-symmetric matrices to posv_mixed_gmres?

//-----------------------------------------
// pbtrf()
template <typename scalar_t>
void pbtrf(
    HermitianBandMatrix<scalar_t>& A,
    Options const& opts = Options());

//-----------------------------------------
// potrf()
template <typename scalar_t>
void potrf(
    HermitianMatrix<scalar_t>& A,
    Options const& opts = Options());

// forward real-symmetric matrices to potrf;
// disabled for complex
template <typename scalar_t>
void potrf(
    SymmetricMatrix<scalar_t>& A,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH(A);
    potrf(AH, opts);
}

//-----------------------------------------
// pbtrs()
template <typename scalar_t>
void pbtrs(
    HermitianBandMatrix<scalar_t>& A,
                 Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// potrs()
template <typename scalar_t>
void potrs(
    HermitianMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
    Options const& opts = Options());

// forward real-symmetric matrices to potrs;
// disabled for complex
template <typename scalar_t>
void potrs(
    SymmetricMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH(A);
    potrs(AH, B, opts);
}

//-----------------------------------------
// potri()
template <typename scalar_t>
void potri(
    HermitianMatrix<scalar_t>& A,
    Options const& opts = Options());

// todo:
// forward real-symmetric matrices to potrs;
// disabled for complex

//-----------------------------------------
// Symmetric indefinite -- block Aasen's

//-----------------------------------------
// hesv()
template <typename scalar_t>
void hesv(
    HermitianMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& H,
             Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// sysv()
// forward real-symmetric matrices to hesv;
// disabled for complex
template <typename scalar_t>
void sysv(
    SymmetricMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& H,
             Matrix<scalar_t>& B,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH(A);
    hesv(AH, pivots, T, pivots2, H, B, opts);
}

//-----------------------------------------
// hetrf()
template <typename scalar_t>
void hetrf(
    HermitianMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& H,
    Options const& opts = Options());

//-----------------------------------------
// sytrf()
// forward real-symmetric matrices to hetrf;
// disabled for complex
template <typename scalar_t>
void sytrf(
    SymmetricMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& H,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH(A);
    hetrf(AH, pivots, T, pivots2, H, opts);
}

//-----------------------------------------
// hetrs()
template <typename scalar_t>
void hetrs(
    HermitianMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// sytrs()
// forward real-symmetric matrices to hetrs;
// disabled for complex
template <typename scalar_t>
void sytrs(
    SymmetricMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& B,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH(A);
    hetrs(AH, pivots, T, pivots2, B, opts);
}

//------------------------------------------------------------------------------
// QR

//-----------------------------------------
// auxiliary type for T factors
template <typename scalar_t>
using TriangularFactors = std::vector< Matrix<scalar_t> >;

//-----------------------------------------
// Least squares

//-----------------------------------------
// gels()

// Using QR
template <typename scalar_t>
void gels_qr(
    Matrix<scalar_t>& A, TriangularFactors<scalar_t>& T,
    Matrix<scalar_t>& BX,
    Options const& opts = Options());

// Using CholeskyQR
template <typename scalar_t>
void gels_cholqr(
    Matrix<scalar_t>& A, Matrix<scalar_t>& R,
    Matrix<scalar_t>& BX,
    Options const& opts = Options());

// Backward compatibility
template <typename scalar_t>
[[deprecated( "Use gels( A, BX[, opts] ) instead. Will be removed 2024-02." )]]
void gels(
    Matrix<scalar_t>& A, TriangularFactors<scalar_t>& T,
    Matrix<scalar_t>& BX,
    Options const& opts = Options())
{
    gels_qr( A, T, BX, opts );
}

// Routine selection
template <typename scalar_t>
void gels(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& BX,
    Options const& opts = Options());

//-----------------------------------------
// QR

//-----------------------------------------
// geqrf()
template <typename scalar_t>
void geqrf(
    Matrix<scalar_t>& A, TriangularFactors<scalar_t>& T,
    Options const& opts = Options());

//-----------------------------------------
// unmqr()
template <typename scalar_t>
void unmqr(
    Side side, Op op,
    Matrix<scalar_t>& A, TriangularFactors<scalar_t>& T,
    Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
// cholQR
template <typename scalar_t>
void cholqr(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& R,
    Options const& opts = Options());

//-----------------------------------------
// LQ

//-----------------------------------------
// gelqf()
template <typename scalar_t>
void gelqf(
    Matrix<scalar_t>& A, TriangularFactors<scalar_t>& T,
    Options const& opts = Options());

//-----------------------------------------
// unmlq()
template <typename scalar_t>
void unmlq(
    Side side, Op op,
    Matrix<scalar_t>& A, TriangularFactors<scalar_t>& T,
    Matrix<scalar_t>& C,
    Options const& opts = Options());

//------------------------------------------------------------------------------
// SVD

template <typename scalar_t>
void svd(
    Matrix<scalar_t> A,
    std::vector< blas::real_type<scalar_t> >& Sigma,
    Matrix<scalar_t>& U,
    Matrix<scalar_t>& VT,
    Options const& opts = Options());

/// Without U and VT, compute only singular values. Same as svd_vals.
template <typename scalar_t>
void svd(
    Matrix<scalar_t> A,
    std::vector< blas::real_type<scalar_t> >& Sigma,
    Options const& opts = Options())
{
    Matrix<scalar_t> U;
    Matrix<scalar_t> VT;
    svd( A, Sigma, U, VT, opts );
}

/// Compute only singular values. Same as svd without U and VT.
template <typename scalar_t>
void svd_vals(
    Matrix<scalar_t> A,
    std::vector< blas::real_type<scalar_t> >& Sigma,
    Options const& opts = Options())
{
    svd( A, Sigma, opts );
}

template <typename scalar_t>
[[deprecated( "Use svd instead. To be removed 2024-07." )]]
void gesvd(
    Matrix<scalar_t> A,
    std::vector< blas::real_type<scalar_t> >& Sigma,
    Matrix<scalar_t>& U,
    Matrix<scalar_t>& VT,
    Options const& opts = Options())
{
    svd( A, Sigma, U, VT, opts );
}

/// Without U and VT, compute only singular values.
template <typename scalar_t>
[[deprecated( "Use svd instead. To be removed 2024-07." )]]
void gesvd(
    Matrix<scalar_t> A,
    std::vector< blas::real_type<scalar_t> >& Sigma,
    Options const& opts = Options())
{
    svd( A, Sigma, opts );
}

//-----------------------------------------
// unmbr_ge2tb()
template <typename scalar_t>
void unmbr_ge2tb(
    Side side, Op op,
    Matrix<scalar_t>& A,
    TriangularFactors<scalar_t> T,
    Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
// ge2tb()
template <typename scalar_t>
void ge2tb(
    Matrix<scalar_t>& A,
    TriangularFactors<scalar_t>& TU,
    TriangularFactors<scalar_t>& TV,
    Options const& opts = Options());

//-----------------------------------------
// Bulge Chasing: TriangularBand to Bi-diagonal
// tb2bd()
template <typename scalar_t>
void tb2bd(
    TriangularBandMatrix<scalar_t>& A,
    Matrix<scalar_t>& U,
    Matrix<scalar_t>& V,
    Options const& opts = Options());

//-----------------------------------------
// Bi-diagonal SVD
// bdsqr()
template <typename scalar_t>
void bdsqr(
    Job jobu, Job jobvt,
    std::vector< blas::real_type<scalar_t> >& D,
    std::vector< blas::real_type<scalar_t> >& E,
    Matrix<scalar_t>& U,
    Matrix<scalar_t>& VT,
    Options const& opts = Options());

//------------------------------------------------------------------------------
// Symmetric/Hermitian eigenvalues

template <typename scalar_t>
void heev(
    HermitianMatrix<scalar_t>& A,
    std::vector< blas::real_type<scalar_t> >& Lambda,
    Matrix<scalar_t>& Z,
    Options const& opts = Options());

/// Without Z, compute only eigenvalues.
template <typename scalar_t>
void heev(
    HermitianMatrix<scalar_t>& A,
    std::vector< blas::real_type<scalar_t> >& Lambda,
    Options const& opts = Options())
{
    Matrix<scalar_t> Z;
    heev( A, Lambda, Z, opts );
}

//-----------------------------------------
// forward real-symmetric matrices to heev;
// disabled for complex
template <typename scalar_t>
void syev(
    SymmetricMatrix<scalar_t>& A,
    std::vector< blas::real_type<scalar_t> >& Lambda,
    Matrix<scalar_t>& Z,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH( A );
    heev( AH, Lambda, Z, opts );
}

/// Without Z, compute only eigenvalues.
template <typename scalar_t>
void syev(
    SymmetricMatrix<scalar_t>& A,
    std::vector< blas::real_type<scalar_t> >& Lambda,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH( A );
    Matrix<scalar_t> Z;
    heev( AH, Lambda, Z, opts );
}

//------------------------------------------------------------------------------
// Generalized symmetric/Hermitian eigenvalues

template <typename scalar_t>
void hegv(
    int64_t itype,
    HermitianMatrix<scalar_t>& A,
    HermitianMatrix<scalar_t>& B,
    std::vector< blas::real_type<scalar_t> >& Lambda,
    Matrix<scalar_t>& Z,
    Options const& opts = Options());

// Without Z, compute only eigenvalues.
template <typename scalar_t>
void hegv(
    int64_t itype,
    HermitianMatrix<scalar_t>& A,
    HermitianMatrix<scalar_t>& B,
    std::vector< blas::real_type<scalar_t> >& Lambda,
    Options const& opts = Options())
{
    Matrix<scalar_t> Z;
    hegv( itype, A, B, Lambda, Z, opts );
}

//-----------------------------------------
// forward real-symmetric matrices to hegv;
// disabled for complex
template <typename scalar_t>
void sygv(
    int64_t itype,
    SymmetricMatrix<scalar_t>& A,
    SymmetricMatrix<scalar_t>& B,
    std::vector< blas::real_type<scalar_t> >& Lambda,
    Matrix<scalar_t>& Z,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH( A );
    HermitianMatrix<scalar_t> BH( B );
    hegv( itype, AH, BH, Lambda, Z, opts );
}

/// Without Z, compute only eigenvalues.
template <typename scalar_t>
void sygv(
    int64_t itype,
    SymmetricMatrix<scalar_t>& A,
    SymmetricMatrix<scalar_t>& B,
    std::vector< blas::real_type<scalar_t> >& Lambda,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    Matrix<scalar_t> Z;
    sygv( itype, A, B, Lambda, Z, opts );
}

//-----------------------------------------
// hegst()
template <typename scalar_t>
void hegst(
    int64_t itype,
    HermitianMatrix<scalar_t>& A,
    HermitianMatrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// sygst()
// forward real-symmetric matrices to hegst;
// disabled for complex
template <typename scalar_t>
void sygst(
    int64_t itype,
    SymmetricMatrix<scalar_t>& A,
    SymmetricMatrix<scalar_t>& B,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH( A );
    HermitianMatrix<scalar_t> BH( B );
    hegst( itype, AH, BH, opts );
}

//------------------------------------------------------------------------------
// Symmetric/Hermitian eigenvalue reductions
// Reduction to band (he2hb), then tridiagonal (hb2st).

//-----------------------------------------
// he2hb()
template <typename scalar_t>
void he2hb(
    HermitianMatrix<scalar_t>& A,
    TriangularFactors<scalar_t>& T,
    Options const& opts = Options());

//-----------------------------------------
// unmtr_he2hb()
template <typename scalar_t>
void unmtr_he2hb(
    Side side, Op op,
    HermitianMatrix<scalar_t>& A,
    TriangularFactors<scalar_t> T,
    Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
// hb2st()
template <typename scalar_t>
void hb2st(
    HermitianBandMatrix<scalar_t>& A,
    Matrix<scalar_t>& V,
    Options const& opts = Options());

//-----------------------------------------
// unmtr_hb2st()
template <typename scalar_t>
void unmtr_hb2st(
    Side side, Op op,
    Matrix<scalar_t>& V,
    Matrix<scalar_t>& C,
    Options const& opts = Options());

//------------------------------------------------------------------------------
// Tridiagonal Symmetric eigenvalue solvers

template <typename real_t>
void stedc(
    std::vector<real_t>& D, std::vector<real_t>& E,
    Matrix<real_t>& Q,
    Options const& opts = Options());

template <typename real_t>
void stedc_deflate(
    int64_t n,
    int64_t n1,
    real_t& rho,
    real_t* D, real_t* Dhat,
    real_t* z, real_t* zhat,
    Matrix<real_t>& Q,
    Matrix<real_t>& Qtype,
    int64_t* itype,
    int64_t& nsecular,
    int64_t& Qtype12_begin, int64_t& Qtype12_end,
    int64_t& Qtype23_begin, int64_t& Qtype23_end,
    Options const& opts = Options());

template <typename real_t>
void stedc_merge(
    int64_t n, int64_t n1,
    real_t rho,
    real_t* D,
    Matrix<real_t>& Q,
    Matrix<real_t>& Qtype,
    Matrix<real_t>& U,
    Options const& opts = Options());

template <typename real_t>
void stedc_secular(
    int64_t nsecular, int64_t n,
    real_t rho,
    real_t* D,
    real_t* z,
    real_t* Lambda,
    Matrix<real_t>& U,
    int64_t* itype,
    Options const& opts = Options() );

template <typename real_t>
void stedc_solve(
    std::vector<real_t>& D, std::vector<real_t>& E,
    Matrix<real_t>& Q,
    Matrix<real_t>& W,
    Matrix<real_t>& U,
    Options const& opts = Options());

template <typename real_t>
void stedc_sort(
    std::vector<real_t>& D,
    Matrix<real_t>& Q,
    Matrix<real_t>& Qout,
    Options const& opts = Options());

template <typename real_t>
void stedc_z_vector(
    Matrix<real_t>& Q,
    std::vector<real_t>& z,
    Options const& opts = Options());

//-----------------------------------------
// unmbr_tb2bd()
template <typename scalar_t>
void unmbr_tb2bd(
    Side side, Op op,
    Matrix<scalar_t>& V,
    Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
// sterf()
template <typename scalar_t>
void sterf(
    std::vector< scalar_t >& D,
    std::vector< scalar_t >& E,
    Options const& opts = Options());

//-----------------------------------------
// steqr2()
template <typename scalar_t>
void steqr2(
    Job jobz,
    std::vector< blas::real_type<scalar_t> >& D,
    std::vector< blas::real_type<scalar_t> >& E,
    Matrix<scalar_t>& Z,
    Options const& opts = Options());

//------------------------------------------------------------------------------
// Condition number estimate

//-----------------------------------------
// gecondest()
template <typename scalar_t>
void gecondest(
        Norm in_norm,
        Matrix<scalar_t>& A,
        blas::real_type<scalar_t> *Anorm,
        blas::real_type<scalar_t> *rcond,
        Options const& opts = Options());

//-----------------------------------------
// trcondest()
template <typename scalar_t>
void trcondest(
        Norm in_norm,
        TriangularMatrix<scalar_t>& A,
        blas::real_type<scalar_t> *rcond,
        Options const& opts = Options());

} // namespace slate

//-----------------------------------------
// Simplified C++ API
#include "simplified_api.hh"

#endif // SLATE_HH
