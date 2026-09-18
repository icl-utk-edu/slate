// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
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

#include "slate/func.hh"
#include "slate/types.hh"
#include "slate/print.hh"

//------------------------------------------------------------------------------
/// @namespace slate
/// SLATE's top-level namespace.
///
namespace slate {

// Version is updated by make_release.py; DO NOT EDIT.
// Version 2025.05.28
#define SLATE_VERSION 20250528

int version();
const char* id();

/// Map of timers, in seconds, for top-level routines. For example:
/// `timers[ "gels" ]` is time for gels,
/// `timers[ "gels::geqrf" ]` is time for geqrf inside gels.
extern std::map< std::string, double > timers;

//------------------------------------------------------------------------------
// Level 2 Auxiliary

/// @defgroup aux Auxiliary routines
/// @brief Matrix addition, copy, scaling, setting, norms, etc.
/// @{

//-----------------------------------------
/// Matrix addition: \( B = \alpha A + \beta B \)
///
/// @param[in] alpha
///     Scalar multiplier for A.
///
/// @param[in] A
///     Matrix A.
///
/// @param[in] beta
///     Scalar multiplier for B.
///
/// @param[in,out] B
///     On entry, matrix B.
///     On exit, overwritten by \( \alpha A + \beta B \).
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup aux
///
template <typename scalar_t>
void add(
    scalar_t alpha, Matrix<scalar_t>& A,
    scalar_t beta,  Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
/// Matrix addition: \( B = \alpha A + \beta B \)
///
/// @param[in] alpha
///     Scalar multiplier for A.
///
/// @param[in] A
///     Trapezoid matrix A.
///
/// @param[in] beta
///     Scalar multiplier for B.
///
/// @param[in,out] B
///     On entry, trapezoid matrix B.
///     On exit, overwritten by \( \alpha A + \beta B \).
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup aux
///
template <typename scalar_t>
void add(
     scalar_t alpha, BaseTrapezoidMatrix<scalar_t>& A,
     scalar_t beta,  BaseTrapezoidMatrix<scalar_t>& B,
     Options const& opts = Options());

//-----------------------------------------
/// Matrix copy: \( B = A \)
///
/// @param[in] A
///     Source matrix.
///
/// @param[out] B
///     Destination matrix.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup aux
///
template <typename src_matrix_type, typename dst_matrix_type>
void copy(
    src_matrix_type& A,
    dst_matrix_type& B,
    Options const& opts = Options());

//-----------------------------------------
/// Matrix scaling: \( A = \text{numer} / \text{denom} \times A \)
///
/// @param[in] numer
///     Numerator of scalar.
///
/// @param[in] denom
///     Denominator of scalar.
///
/// @param[in,out] A
///     Matrix to scale.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup aux
///
template <typename scalar_t>
void scale(
    blas::real_type<scalar_t> numer,
    blas::real_type<scalar_t> denom,
    Matrix<scalar_t>& A,
    Options const& opts = Options());

/// General matrix, version with denom = 1.0.
/// @ingroup aux
template <typename scalar_t>
void scale(
    blas::real_type<scalar_t> value,
    Matrix<scalar_t>& A,
    Options const& opts = Options())
{
    blas::real_type<scalar_t> one = 1.0;
    scale(value, one, A, opts);
}

//-----------------------------------------
/// Matrix scaling: \( A = \text{numer} / \text{denom} \times A \)
///
/// @param[in] numer
///     Numerator of scalar.
///
/// @param[in] denom
///     Denominator of scalar.
///
/// @param[in,out] A
///     Trapezoid matrix to scale.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup aux
///
template <typename scalar_t>
void scale(
    blas::real_type<scalar_t> numer,
    blas::real_type<scalar_t> denom,
    BaseTrapezoidMatrix<scalar_t>& A,
    Options const& opts = Options());

/// BaseTrapezoid matrix, version with denom = 1.0.
/// @ingroup aux
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
/// Scale rows and columns: \( A = R A C \)
///
/// @param[in] equed
///     Form of equilibration:
///     - Equed::None: no equilibration
///     - Equed::Row: row equilibration, \( A = R A \)
///     - Equed::Col: column equilibration, \( A = A C \)
///     - Equed::Both: row and column equilibration, \( A = R A C \)
///
/// @param[in] R
///     Row scaling factors.
///
/// @param[in] C
///     Column scaling factors.
///
/// @param[in,out] A
///     Matrix to scale.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup aux
///
template <typename scalar_t, typename scalar_t2>
void scale_row_col(
    Equed equed,
    std::vector< scalar_t2 > const& R,
    std::vector< scalar_t2 > const& C,
    Matrix<scalar_t>& A,
    Options const& opts = Options());

//-----------------------------------------
/// Set matrix elements: \( A_{ij} = \text{diag\_value} \) if \( i=j \), else \( \text{offdiag\_value} \)
///
/// @param[in] offdiag_value
///     Value to set off-diagonal elements.
///
/// @param[in] diag_value
///     Value to set diagonal elements.
///
/// @param[out] A
///     Matrix to set.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup aux
///
template <typename scalar_t>
void set(
    scalar_t offdiag_value,
    scalar_t diag_value,
    Matrix<scalar_t>& A,
    Options const& opts = Options());

/// Set all matrix elements to same value.
/// @ingroup aux
template <typename scalar_t>
void set(
    scalar_t value,
    Matrix<scalar_t>& A,
    Options const& opts = Options())
{
    set(value, value, A, opts);
}

//-----------------------------------------
/// Set trapezoid matrix elements: \( A_{ij} = \text{diag\_value} \) if \( i=j \), else \( \text{offdiag\_value} \)
///
/// @param[in] offdiag_value
///     Value to set off-diagonal elements.
///
/// @param[in] diag_value
///     Value to set diagonal elements.
///
/// @param[out] A
///     Trapezoid matrix to set.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup aux
///
template <typename scalar_t>
void set(
    scalar_t offdiag_value,
    scalar_t diag_value,
    BaseTrapezoidMatrix<scalar_t>& A,
    Options const& opts = Options());

/// Set all trapezoid matrix elements to same value.
/// @ingroup aux
template <typename scalar_t>
void set(
    scalar_t value,
    BaseTrapezoidMatrix<scalar_t>& A,
    Options const& opts = Options())
{
    set(value, value, A, opts);
}

//-----------------------------------------
/// Set matrix elements using a function.
///
/// @param[in] value
///     Function mapping (i, j) global indices to value.
///
/// @param[out] A
///     Matrix to set.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup aux
///
template <typename scalar_t>
void set(
    std::function< scalar_t (int64_t i, int64_t j) > const& value,
    Matrix<scalar_t>& A,
    Options const& opts = Options());

//-----------------------------------------
/// Set trapezoid matrix elements using a function.
///
/// @param[in] value
///     Function mapping (i, j) global indices to value.
///
/// @param[out] A
///     Trapezoid matrix to set.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup aux
///
template <typename scalar_t>
void set(
    std::function< scalar_t (int64_t i, int64_t j) > const& value,
    BaseTrapezoidMatrix<scalar_t>& A,
    Options const& opts = Options());

/// @}


//------------------------------------------------------------------------------
// Level 3 BLAS and LAPACK auxiliary

/// @defgroup blas3 Level 3 BLAS and LAPACK routines
/// @brief Matrix multiplication, triangular solve, etc.
/// @{

//-----------------------------------------
/// General randomization of matrix (two-sided): \( A_{new} = U A V \)
///
/// @param[in] U
///     Random unitary matrix (left).
///
/// @param[in,out] A
///     Matrix to be randomized.
///
/// @param[in] V
///     Random unitary matrix (right).
///
/// @ingroup blas3
///
template<typename scalar_t>
void gerbt(Matrix<scalar_t>& U,
           Matrix<scalar_t>& A,
           Matrix<scalar_t>& V);

/// General randomization of matrix (one-sided): \( A_{new} = U A \)
/// @ingroup blas3
template<typename scalar_t>
void gerbt(Matrix<scalar_t>& U,
           Matrix<scalar_t>& A);

//-----------------------------------------
/// General band matrix multiply: \( C = \alpha A B + \beta C \)
///
/// @param[in] alpha
///     Scalar multiplier.
///
/// @param[in] A
///     Band matrix A.
///
/// @param[in] B
///     Matrix B.
///
/// @param[in] beta
///     Scalar multiplier for C.
///
/// @param[in,out] C
///     Matrix C.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup blas3
///
template <typename scalar_t>
void gbmm(
    scalar_t alpha, BandMatrix<scalar_t>& A,
                        Matrix<scalar_t>& B,
    scalar_t beta,      Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
/// General matrix multiply: \( C = \alpha op(A) op(B) + \beta C \)
///
/// @param[in] alpha
///     Scalar multiplier.
///
/// @param[in] A
///     Matrix A.
///
/// @param[in] B
///     Matrix B.
///
/// @param[in] beta
///     Scalar multiplier for C.
///
/// @param[in,out] C
///     Matrix C.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup blas3
///
template <typename scalar_t>
void gemm(
    scalar_t alpha, Matrix<scalar_t>& A,
                    Matrix<scalar_t>& B,
    scalar_t beta,  Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
/// General matrix multiply (variant A).
/// @see gemm
/// @ingroup blas3
template <typename scalar_t>
void gemmA(
    scalar_t alpha, Matrix<scalar_t>& A,
                    Matrix<scalar_t>& B,
    scalar_t beta,  Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
/// General matrix multiply (variant C).
/// @see gemm
/// @ingroup blas3
template <typename scalar_t>
void gemmC(
    scalar_t alpha, Matrix<scalar_t>& A,
                    Matrix<scalar_t>& B,
    scalar_t beta,  Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
/// Hermitian band matrix multiply: \( C = \alpha A B + \beta C \) (if side=Left)
/// or \( C = \alpha B A + \beta C \) (if side=Right).
///
/// @param[in] side
///     Side::Left or Side::Right.
///
/// @param[in] alpha
///     Scalar multiplier.
///
/// @param[in] A
///     Hermitian band matrix A.
///
/// @param[in] B
///     Matrix B.
///
/// @param[in] beta
///     Scalar multiplier for C.
///
/// @param[in,out] C
///     Matrix C.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup blas3
///
template <typename scalar_t>
void hbmm(
    Side side,
    scalar_t alpha, HermitianBandMatrix<scalar_t>& A,
                                 Matrix<scalar_t>& B,
    scalar_t beta,               Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
/// Hermitian matrix multiply: \( C = \alpha A B + \beta C \) (if side=Left)
/// or \( C = \alpha B A + \beta C \) (if side=Right).
///
/// @param[in] side
///     Side::Left or Side::Right.
///
/// @param[in] alpha
///     Scalar multiplier.
///
/// @param[in] A
///     Hermitian matrix A.
///
/// @param[in] B
///     Matrix B.
///
/// @param[in] beta
///     Scalar multiplier for C.
///
/// @param[in,out] C
///     Matrix C.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup blas3
///
template <typename scalar_t>
void hemm(
    Side side,
    scalar_t alpha, HermitianMatrix<scalar_t>& A,
                             Matrix<scalar_t>& B,
    scalar_t beta,           Matrix<scalar_t>& C,
    Options const& opts = Options());


/// Overload for real symmetric matrix (dispatched to Hermitian).
/// @ingroup blas3
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
/// Hermitian matrix multiply (variant A).
/// @see hemm
/// @ingroup blas3
template <typename scalar_t>
void hemmA(
    Side side,
    scalar_t alpha, HermitianMatrix<scalar_t>& A,
                             Matrix<scalar_t>& B,
    scalar_t beta,           Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
/// Hermitian matrix multiply (variant C).
/// @see hemm
/// @ingroup blas3
template <typename scalar_t>
void hemmC(
    Side side,
    scalar_t alpha, HermitianMatrix<scalar_t>& A,
                             Matrix<scalar_t>& B,
    scalar_t beta,           Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
/// Symmetric matrix multiply: \( C = \alpha A B + \beta C \) (if side=Left)
/// or \( C = \alpha B A + \beta C \) (if side=Right).
///
/// @param[in] side
///     Side::Left or Side::Right.
///
/// @param[in] alpha
///     Scalar multiplier.
///
/// @param[in] A
///     Symmetric matrix A.
///
/// @param[in] B
///     Matrix B.
///
/// @param[in] beta
///     Scalar multiplier for C.
///
/// @param[in,out] C
///     Matrix C.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup blas3
///
template <typename scalar_t>
void symm(
    Side side,
    scalar_t alpha, SymmetricMatrix<scalar_t>& A,
                             Matrix<scalar_t>& B,
    scalar_t beta,           Matrix<scalar_t>& C,
    Options const& opts = Options());

/// Overload for real Hermitian matrix (dispatched to Symmetric).
/// @ingroup blas3
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
/// Triangular matrix multiply: \( B = \alpha op(A) B \) (if side=Left)
/// or \( B = \alpha B op(A) \) (if side=Right).
///
/// @param[in] side
///     Side::Left or Side::Right.
///
/// @param[in] alpha
///     Scalar multiplier.
///
/// @param[in] A
///     Triangular matrix A.
///
/// @param[in,out] B
///     Matrix B.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup blas3
///
template <typename scalar_t>
void trmm(
    Side side,
    scalar_t alpha, TriangularMatrix<scalar_t>& A,
                              Matrix<scalar_t>& B,
    Options const& opts = Options());


//-----------------------------------------
/// Triangular band solve: \( op(A) X = \alpha B \) (if side=Left)
/// or \( X op(A) = \alpha B \) (if side=Right).
///
/// @param[in] side
///     Side::Left or Side::Right.
///
/// @param[in] alpha
///     Scalar multiplier.
///
/// @param[in] A
///     Triangular band matrix A.
///
/// @param[in] pivots
///     Pivot indices.
///
/// @param[in,out] B
///     On entry, right-hand side B. On exit, solution X.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup blas3
///
template <typename scalar_t>
void tbsm(
    Side side,
    scalar_t alpha, TriangularBandMatrix<scalar_t>& A, Pivots& pivots,
                                  Matrix<scalar_t>& B,
    Options const& opts = Options());

/// Triangular band solve (no pivots).
/// @ingroup blas3
template <typename scalar_t>
void tbsm(
    Side side,
    scalar_t alpha, TriangularBandMatrix<scalar_t>& A,
                                  Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
/// Triangular solve: \( op(A) X = \alpha B \) (if side=Left)
/// or \( X op(A) = \alpha B \) (if side=Right).
///
/// @param[in] side
///     Side::Left or Side::Right.
///
/// @param[in] alpha
///     Scalar multiplier.
///
/// @param[in] A
///     Triangular matrix A.
///
/// @param[in,out] B
///     On entry, right-hand side B. On exit, solution X.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup blas3
///
template <typename scalar_t>
void trsm(
    Side side,
    scalar_t alpha, TriangularMatrix<scalar_t>& A,
                              Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
/// Triangular solve (variant A).
/// @see trsm
/// @ingroup blas3
template <typename scalar_t>
void trsmA(
    Side side,
    scalar_t alpha, TriangularMatrix<scalar_t>& A,
                              Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
/// Triangular solve (variant B).
/// @see trsm
/// @ingroup blas3
template <typename scalar_t>
void trsmB(
    Side side,
    scalar_t alpha, TriangularMatrix<scalar_t>& A,
                              Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
/// Triangular matrix inverse: \( A = A^{-1} \).
///
/// @param[in,out] A
///     Triangular matrix A. On exit, overwritten by inverse.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup blas3
///
template <typename scalar_t>
void trtri(
    TriangularMatrix<scalar_t>& A,
    Options const& opts = Options());

//-----------------------------------------
/// Triangular matrix multiply: \( A = A * A \)? Or something similar.
/// Note: Description assumed from context, usually implies triangular multiply.
///
/// @param[in,out] A
///     Triangular matrix A.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup blas3
///
template <typename scalar_t>
void trtrm(
    TriangularMatrix<scalar_t>& A,
    Options const& opts = Options());

//-----------------------------------------
/// Hermitian rank-k update: \( C = \alpha A A^H + \beta C \).
///
/// @param[in] alpha
///     Real scalar multiplier.
///
/// @param[in] A
///     Matrix A.
///
/// @param[in] beta
///     Real scalar multiplier for C.
///
/// @param[in,out] C
///     Hermitian matrix C.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup blas3
///
template <typename scalar_t>
void herk(
    blas::real_type<scalar_t> alpha,          Matrix<scalar_t>& A,
    blas::real_type<scalar_t> beta,  HermitianMatrix<scalar_t>& C,
    Options const& opts = Options());

/// Overload for real symmetric matrix (dispatched to Hermitian).
/// @ingroup blas3
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
/// Symmetric rank-k update: \( C = \alpha A A^T + \beta C \).
///
/// @param[in] alpha
///     Scalar multiplier.
///
/// @param[in] A
///     Matrix A.
///
/// @param[in] beta
///     Scalar multiplier for C.
///
/// @param[in,out] C
///     Symmetric matrix C.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup blas3
///
template <typename scalar_t>
void syrk(
    scalar_t alpha,          Matrix<scalar_t>& A,
    scalar_t beta,  SymmetricMatrix<scalar_t>& C,
    Options const& opts = Options());

/// Overload for real Hermitian matrix (dispatched to Symmetric).
/// @ingroup blas3
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
/// Hermitian rank-2k update: \( C = \alpha A B^H + \overline{\alpha} B A^H + \beta C \).
///
/// @param[in] alpha
///     Scalar multiplier.
///
/// @param[in] A
///     Matrix A.
///
/// @param[in] B
///     Matrix B.
///
/// @param[in] beta
///     Real scalar multiplier for C.
///
/// @param[in,out] C
///     Hermitian matrix C.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup blas3
///
template <typename scalar_t>
void her2k(
    scalar_t alpha,                          Matrix<scalar_t>& A,
                                             Matrix<scalar_t>& B,
    blas::real_type<scalar_t> beta, HermitianMatrix<scalar_t>& C,
    Options const& opts = Options());

/// Overload for real symmetric matrix (dispatched to Hermitian).
/// @ingroup blas3
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
/// Redistribute matrix from A to B.
///
/// @param[in] A
///     Source matrix.
///
/// @param[out] B
///     Destination matrix.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup blas3
///
template <typename scalar_t>
void redistribute(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
/// Symmetric rank-2k update: \( C = \alpha A B^T + \alpha B A^T + \beta C \).
///
/// @param[in] alpha
///     Scalar multiplier.
///
/// @param[in] A
///     Matrix A.
///
/// @param[in] B
///     Matrix B.
///
/// @param[in] beta
///     Scalar multiplier for C.
///
/// @param[in,out] C
///     Symmetric matrix C.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup blas3
///
template <typename scalar_t>
void syr2k(
    scalar_t alpha,          Matrix<scalar_t>& A,
                             Matrix<scalar_t>& B,
    scalar_t beta,  SymmetricMatrix<scalar_t>& C,
    Options const& opts = Options());

/// Overload for real Hermitian matrix (dispatched to Symmetric).
/// @ingroup blas3
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

/// @}


//------------------------------------------------------------------------------
// Norms

/// @defgroup norm Matrix norms
/// @brief Matrix norm computations (One, Inf, Fro, Max, etc.)
/// @{

//-----------------------------------------
/// Compute matrix norm.
///
/// @param[in] norm
///     Type of norm:
///     - Norm::One: max column sum
///     - Norm::Two: max singular value (not supported yet?)
///     - Norm::Inf: max row sum
///     - Norm::Fro: Frobenius norm
///     - Norm::Max: max absolute element
///
/// @param[in] A
///     Matrix A.
///
/// @param[in] opts
///     Additional options.
///
/// @return The computed norm.
///
/// @ingroup norm
///
template <typename matrix_type>
blas::real_type<typename matrix_type::value_type>
norm(
    Norm norm,
    matrix_type& A,
    Options const& opts = Options());

/// Compute triangular matrix norm.
/// @ingroup norm
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
/// Compute column norms (e.g. max element in each column).
///
/// @param[in] norm
///     Type of norm.
///
/// @param[in] A
///     Matrix A.
///
/// @param[out] values
///     Array of size n, where n is number of columns.
///     On exit, contains norm of each column.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup norm
///
template <typename matrix_type>
void colNorms(
    Norm norm,
    matrix_type& A,
    blas::real_type<typename matrix_type::value_type>* values,
    Options const& opts = Options());

/// @}


//------------------------------------------------------------------------------
// Linear systems

/// @defgroup lu LU factorization and solve
/// @brief LU factorization, solve, inverse, condition number
/// @{

//-----------------------------------------
/// Solve linear system \( A X = B \) using LU factorization (band matrix).
///
/// @param[in,out] A
///     Band matrix A. On exit, overwritten by LU factors.
///
/// @param[out] pivots
///     Pivot indices.
///
/// @param[in,out] B
///     Right-hand side B. On exit, solution X.
///
/// @param[in] opts
///     Additional options.
///
/// @return Info code (0 = success).
///
/// @ingroup lu
///
template <typename scalar_t>
int64_t gbsv(
    BandMatrix<scalar_t>& A, Pivots& pivots,
        Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
/// Solve linear system \( A X = B \) using LU factorization.
///
/// @param[in,out] A
///     General matrix A. On exit, overwritten by LU factors.
///
/// @param[out] pivots
///     Pivot indices.
///
/// @param[in,out] B
///     Right-hand side B. On exit, solution X.
///
/// @param[in] opts
///     Additional options.
///
/// @return Info code.
///
/// @ingroup lu
///
template <typename scalar_t>
int64_t gesv(
    Matrix<scalar_t>& A, Pivots& pivots,
    Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
/// Solve linear system \( A X = B \) using LU factorization without pivoting.
/// @deprecated Use gesv( ..., { MethodLU: NoPiv } )
/// @ingroup lu
template <typename scalar_t>
int64_t gesv_nopiv(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
/// Solve linear system \( A X = B \) using mixed-precision iterative refinement.
///
/// @param[in,out] A
///     General matrix A.
///
/// @param[out] pivots
///     Pivot indices.
///
/// @param[in,out] B
///     Right-hand side B.
///
/// @param[out] X
///     Solution X.
///
/// @param[out] iter
///     Number of iterations performed.
///
/// @param[in] opts
///     Additional options.
///
/// @return Info code.
///
/// @ingroup lu
///
template <typename scalar_t>
int64_t gesv_mixed(
    Matrix<scalar_t>& A, Pivots& pivots,
    Matrix<scalar_t>& B,
    Matrix<scalar_t>& X,
    int& iter,
    Options const& opts = Options());

/// Mixed-precision solve (explicit types).
/// @ingroup lu
template <typename scalar_hi, typename scalar_lo>
int64_t gesv_mixed(
    Matrix<scalar_hi>& A, Pivots& pivots,
    Matrix<scalar_hi>& B,
    Matrix<scalar_hi>& X,
    int& iter,
    Options const& opts = Options());

//-----------------------------------------
/// Solve linear system \( A X = B \) using mixed-precision GMRES.
///
/// @param[in,out] A
///     General matrix A.
///
/// @param[out] pivots
///     Pivot indices.
///
/// @param[in,out] B
///     Right-hand side B.
///
/// @param[out] X
///     Solution X.
///
/// @param[out] iter
///     Number of iterations.
///
/// @param[in] opts
///     Additional options.
///
/// @return Info code.
///
/// @ingroup lu
///
template <typename scalar_t>
int64_t gesv_mixed_gmres(
    Matrix<scalar_t>& A, Pivots& pivots,
    Matrix<scalar_t>& B,
    Matrix<scalar_t>& X,
    int& iter,
    Options const& opts = Options());

/// Mixed-precision GMRES solve (explicit types).
/// @ingroup lu
template <typename scalar_hi, typename scalar_lo>
int64_t gesv_mixed_gmres(
    Matrix<scalar_hi>& A, Pivots& pivots,
    Matrix<scalar_hi>& B,
    Matrix<scalar_hi>& X,
    int& iter,
    Options const& opts = Options());

//-----------------------------------------
/// Solve linear system using random butterfly transform (RBT).
///
/// @param[in,out] A
///     General matrix A.
///
/// @param[in,out] B
///     Right-hand side B.
///
/// @param[out] X
///     Solution X.
///
/// @param[out] iter
///     Number of iterations.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup lu
///
template<typename scalar_t>
void gesv_rbt(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& B,
    Matrix<scalar_t>& X,
    int& iter,
    Options const& opts = Options());


//-----------------------------------------
/// LU factorization (band matrix): \( A = L U \).
///
/// @param[in,out] A
///     Band matrix A. On exit, overwritten by LU factors.
///
/// @param[out] pivots
///     Pivot indices.
///
/// @param[in] opts
///     Additional options.
///
/// @return Info code.
///
/// @ingroup lu
///
template <typename scalar_t>
int64_t gbtrf(
    BandMatrix<scalar_t>& A, Pivots& pivots,
    Options const& opts = Options());

//-----------------------------------------
/// LU factorization (general matrix): \( A = P L U \).
///
/// @param[in,out] A
///     General matrix A. On exit, overwritten by LU factors.
///
/// @param[out] pivots
///     Pivot indices representing permutation P.
///
/// @param[in] opts
///     Additional options.
///
/// @return Info code.
///
/// @ingroup lu
///
template <typename scalar_t>
int64_t getrf(
    Matrix<scalar_t>& A, Pivots& pivots,
    Options const& opts = Options());

//-----------------------------------------
/// LU factorization without pivoting.
///
/// @param[in,out] A
///     General matrix A. On exit, overwritten by LU factors.
///
/// @param[in] opts
///     Additional options.
///
/// @return Info code.
///
/// @ingroup lu
///
template <typename scalar_t>
int64_t getrf_nopiv(
    Matrix<scalar_t>& A,
    Options const& opts = Options());

//-----------------------------------------
/// LU factorization with tournament pivoting.
///
/// @param[in,out] A
///     General matrix A. On exit, overwritten by LU factors.
///
/// @param[out] pivots
///     Pivot indices.
///
/// @param[in] opts
///     Additional options.
///
/// @return Info code.
///
/// @ingroup lu
///
template <typename scalar_t>
int64_t getrf_tntpiv(
    Matrix<scalar_t>& A, Pivots& pivots,
    Options const& opts = Options());

//-----------------------------------------
/// Solve using partial LU factorization (band matrix).
///
/// @param[in] A
///     Band matrix with LU factors.
///
/// @param[in] pivots
///     Pivot indices.
///
/// @param[in,out] B
///     Right-hand side. On exit, solution.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup lu
///
template <typename scalar_t>
void gbtrs(
    BandMatrix<scalar_t>& A, Pivots& pivots,
        Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
/// Solve using partial LU factorization (general matrix).
///
/// @param[in] A
///     General matrix with LU factors.
///
/// @param[in] pivots
///     Pivot indices.
///
/// @param[in,out] B
///     Right-hand side. On exit, solution.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup lu
///
template <typename scalar_t>
void getrs(
    Matrix<scalar_t>& A, Pivots& pivots,
    Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
/// Solve using LU factorization without pivoting.
/// @deprecated Use getrs( ..., { MethodLU: NoPiv } )
/// @ingroup lu
template <typename scalar_t>
void getrs_nopiv(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
/// Compute matrix inverse using LU factorization (in-place).
///
/// @param[in,out] A
///     Matrix with LU factors. On exit, overwritten by inverse.
///
/// @param[in] pivots
///     Pivot indices.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup lu
///
template <typename scalar_t>
void getri(
    Matrix<scalar_t>& A, Pivots& pivots,
    Options const& opts = Options());

/// Compute matrix inverse using LU factorization (out-of-place).
/// @ingroup lu
template <typename scalar_t>
void getri(
    Matrix<scalar_t>& A, Pivots& pivots,
    Matrix<scalar_t>& B,
    Options const& opts = Options());

/// @}


/// @defgroup chol Cholesky factorization and solve
/// @brief Cholesky factorization, solve, inverse, condition number
/// @{

//-----------------------------------------
/// Solve linear system \( A X = B \) using Cholesky (Hermitian band).
///
/// @param[in,out] A
///     Hermitian band matrix A. On exit, overwritten by factor.
///
/// @param[in,out] B
///     Right-hand side B. On exit, solution X.
///
/// @param[in] opts
///     Additional options.
///
/// @return Info code (0 = success).
///
/// @ingroup chol
///
template <typename scalar_t>
int64_t pbsv(
    HermitianBandMatrix<scalar_t>& A,
                 Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
/// Solve linear system \( A X = B \) using Cholesky (Hermitian).
///
/// @param[in,out] A
///     Hermitian matrix A. On exit, overwritten by factor.
///
/// @param[in,out] B
///     Right-hand side B. On exit, solution X.
///
/// @param[in] opts
///     Additional options.
///
/// @return Info code (0 = success).
///
/// @ingroup chol
///
template <typename scalar_t>
int64_t posv(
    HermitianMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
    Options const& opts = Options());

/// Overload for real symmetric matrix.
/// @ingroup chol
template <typename scalar_t>
int64_t posv(
    SymmetricMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH(A);
    return posv( AH, B, opts );
}

//-----------------------------------------
/// Solve linear system using mixed-precision Cholesky iterative refinement.
///
/// @param[in,out] A
///     Hermitian matrix A.
///
/// @param[in,out] B
///     Right-hand side B.
///
/// @param[out] X
///     Solution X.
///
/// @param[out] iter
///     Number of iterations.
///
/// @param[in] opts
///     Additional options.
///
/// @return Info code.
///
/// @ingroup chol
///
template <typename scalar_t>
int64_t posv_mixed(
    HermitianMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
             Matrix<scalar_t>& X,
    int& iter,
    Options const& opts = Options());

/// Mixed-precision Cholesky solve (explicit types).
/// @ingroup chol
template <typename scalar_hi, typename scalar_lo>
int64_t posv_mixed(
    HermitianMatrix<scalar_hi>& A,
             Matrix<scalar_hi>& B,
             Matrix<scalar_hi>& X,
    int& iter,
    Options const& opts = Options());

// todo: forward real-symmetric matrices to posv_mixed?

//-----------------------------------------
/// Solve linear system using mixed-precision Cholesky GMRES.
///
/// @param[in,out] A
///     Hermitian matrix A.
///
/// @param[in,out] B
///     Right-hand side B.
///
/// @param[out] X
///     Solution X.
///
/// @param[out] iter
///     Number of iterations.
///
/// @param[in] opts
///     Additional options.
///
/// @return Info code.
///
/// @ingroup chol
///
template <typename scalar_t>
int64_t posv_mixed_gmres(
    HermitianMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
             Matrix<scalar_t>& X,
    int& iter,
    Options const& opts = Options());

/// Mixed-precision Cholesky GMRES solve (explicit types).
/// @ingroup chol
template <typename scalar_hi, typename scalar_lo>
int64_t posv_mixed_gmres(
    HermitianMatrix<scalar_hi>& A,
             Matrix<scalar_hi>& B,
             Matrix<scalar_hi>& X,
    int& iter,
    Options const& opts = Options());

// todo: forward real-symmetric matrices to posv_mixed_gmres?

//-----------------------------------------
/// Cholesky factorization (Hermitian band): \( A = L L^H \).
///
/// @param[in,out] A
///     Hermitian band matrix A. On exit, overwritten by factor.
///
/// @param[in] opts
///     Additional options.
///
/// @return Info code.
///
/// @ingroup chol
///
template <typename scalar_t>
int64_t pbtrf(
    HermitianBandMatrix<scalar_t>& A,
    Options const& opts = Options());

//-----------------------------------------
/// Cholesky factorization (Hermitian): \( A = L L^H \).
///
/// @param[in,out] A
///     Hermitian matrix A. On exit, overwritten by factor.
///
/// @param[in] opts
///     Additional options.
///
/// @return Info code.
///
/// @ingroup chol
///
template <typename scalar_t>
int64_t potrf(
    HermitianMatrix<scalar_t>& A,
    Options const& opts = Options());

/// Overload for real symmetric matrix.
/// @ingroup chol
template <typename scalar_t>
int64_t potrf(
    SymmetricMatrix<scalar_t>& A,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH(A);
    return potrf( AH, opts );
}

//-----------------------------------------
/// Solve using Cholesky factorization (Hermitian band).
///
/// @param[in] A
///     Hermitian band matrix with factor.
///
/// @param[in,out] B
///     Right-hand side. On exit, solution.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup chol
///
template <typename scalar_t>
void pbtrs(
    HermitianBandMatrix<scalar_t>& A,
                 Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
/// Solve using Cholesky factorization (Hermitian).
///
/// @param[in] A
///     Hermitian matrix with factor.
///
/// @param[in,out] B
///     Right-hand side. On exit, solution.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup chol
///
template <typename scalar_t>
void potrs(
    HermitianMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
    Options const& opts = Options());

/// Overload for real symmetric matrix.
/// @ingroup chol
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
/// Compute matrix inverse using Cholesky factorization (in-place).
///
/// @param[in,out] A
///     Matrix with factor. On exit, overwritten by inverse.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup chol
///
template <typename scalar_t>
void potri(
    HermitianMatrix<scalar_t>& A,
    Options const& opts = Options());

/// @}


// todo:
// forward real-symmetric matrices to potrs;
// disabled for complex

//------------------------------------------------------------------------------
// Symmetric indefinite -- block Aasen's

/// @defgroup sysv Symmetric/Hermitian indefinite factorization and solve
/// @brief Aasen's algorithm for indefinite systems
/// @{

//-----------------------------------------
/// Solve linear system \( A X = B \) using Aasen's algorithm (Hermitian).
///
/// @param[in,out] A
///     Hermitian matrix A. On exit, overwritten by factorization.
///
/// @param[out] pivots
///     Pivot indices.
///
/// @param[out] T
///     Band matrix factor T.
///
/// @param[out] pivots2
///     Secondary pivot indices.
///
/// @param[out] H
///     Auxiliary matrix H.
///
/// @param[in,out] B
///     Right-hand side B. On exit, solution X.
///
/// @param[in] opts
///     Additional options.
///
/// @return Info code.
///
/// @ingroup sysv
///
template <typename scalar_t>
int64_t hesv(
    HermitianMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& H,
             Matrix<scalar_t>& B,
    Options const& opts = Options());

/// Overload for real symmetric matrix.
/// @ingroup sysv
template <typename scalar_t>
int64_t sysv(
    SymmetricMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& H,
             Matrix<scalar_t>& B,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH( A );
    return hesv( AH, pivots, T, pivots2, H, B, opts );
}

//-----------------------------------------
/// Indefinite factorization (Hermitian): Aasen's algorithm.
///
/// @param[in,out] A
///     Hermitian matrix A. On exit, overwritten by factorization.
///
/// @param[out] pivots
///     Pivot indices.
///
/// @param[out] T
///     Band matrix factor T.
///
/// @param[out] pivots2
///     Secondary pivot indices.
///
/// @param[out] H
///     Auxiliary matrix H.
///
/// @param[in] opts
///     Additional options.
///
/// @return Info code.
///
/// @ingroup sysv
///
template <typename scalar_t>
int64_t hetrf(
    HermitianMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& H,
    Options const& opts = Options());

/// Overload for real symmetric matrix.
/// @ingroup sysv
template <typename scalar_t>
int64_t sytrf(
    SymmetricMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& H,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH( A );
    return hetrf( AH, pivots, T, pivots2, H, opts );
}

//-----------------------------------------
/// Solve using indefinite factorization (Hermitian).
///
/// @param[in] A
///     Hermitian matrix with factorization.
///
/// @param[in] pivots
///     Pivot indices.
///
/// @param[in] T
///     Band matrix factor T.
///
/// @param[in] pivots2
///     Secondary pivot indices.
///
/// @param[in,out] B
///     Right-hand side. On exit, solution.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup sysv
///
template <typename scalar_t>
void hetrs(
    HermitianMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& B,
    Options const& opts = Options());

/// Overload for real symmetric matrix.
/// @ingroup sysv
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

/// @}


//------------------------------------------------------------------------------
// QR

/// Vector of matrices for representing triangular factors T in QR/LQ.
template <typename scalar_t>
using TriangularFactors = std::vector< Matrix<scalar_t> >;

//------------------------------------------------------------------------------
// Least squares

/// @defgroup gels Least squares
/// @brief Solve over- or under-determined systems
/// @{

//-----------------------------------------
/// Solve least squares using QR factorization: minimize \( \| B - A X \|_2 \).
///
/// @param[in,out] A
///     General matrix A. On exit, overwritten by QR factors.
///
/// @param[out] T
///     Triangular factors.
///
/// @param[in,out] BX
///     Right-hand side B. On exit, solution X.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup gels
///
template <typename scalar_t>
void gels_qr(
    Matrix<scalar_t>& A, TriangularFactors<scalar_t>& T,
    Matrix<scalar_t>& BX,
    Options const& opts = Options());

//-----------------------------------------
/// Solve least squares using CholeskyQR: minimize \( \| B - A X \|_2 \).
/// A must be tall and skinny.
///
/// @param[in,out] A
///     General matrix A.
///
/// @param[out] R
///     Upper triangular factor R.
///
/// @param[in,out] BX
///     Right-hand side B. On exit, solution X.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup gels
///
template <typename scalar_t>
void gels_cholqr(
    Matrix<scalar_t>& A, Matrix<scalar_t>& R,
    Matrix<scalar_t>& BX,
    Options const& opts = Options());

//-----------------------------------------
/// Solve least squares problem (driver).
/// Dispatches to QR or CholeskyQR based on options.
///
/// @param[in,out] A
///     General matrix A.
///
/// @param[in,out] BX
///     Right-hand side B. On exit, solution X.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup gels
///
template <typename scalar_t>
void gels(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& BX,
    Options const& opts = Options());

/// @}

//------------------------------------------------------------------------------
// QR

/// @defgroup qr QR factorization
/// @brief QR factorization and multiplication by Q
/// @{

//-----------------------------------------
/// QR factorization: \( A = Q R \).
///
/// @param[in,out] A
///     General matrix A. On exit, overwritten by Householder vectors.
///
/// @param[out] T
///     Triangular factors T.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup qr
///
template <typename scalar_t>
void geqrf(
    Matrix<scalar_t>& A, TriangularFactors<scalar_t>& T,
    Options const& opts = Options());

//-----------------------------------------
/// Multiply by Q from QR factorization.
/// \( C = Q C \) or \( C = Q^H C \) (if side=Left)
/// \( C = C Q \) or \( C = C Q^H \) (if side=Right)
///
/// @param[in] side
///     Side::Left or Side::Right.
///
/// @param[in] op
///     Op::NoTrans or Op::ConjTrans (or Op::Trans for real).
///
/// @param[in] A
///     Matrix with Householder vectors.
///
/// @param[in] T
///     Triangular factors.
///
/// @param[in,out] C
///     Matrix C.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup qr
///
template <typename scalar_t>
void unmqr(
    Side side, Op op,
    Matrix<scalar_t>& A, TriangularFactors<scalar_t>& T,
    Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
/// CholeskyQR factorization: \( A = Q R \).
/// A must be tall and skinny.
///
/// @param[in,out] A
///     General matrix A. On exit, overwritten by Q.
///
/// @param[out] R
///     Upper triangular factor R.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup qr
///
template <typename scalar_t>
void cholqr(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& R,
    Options const& opts = Options());

/// @}

//------------------------------------------------------------------------------
// LQ

/// @defgroup lq LQ factorization
/// @brief LQ factorization and multiplication by Q
/// @{

//-----------------------------------------
/// LQ factorization: \( A = L Q \).
///
/// @param[in,out] A
///     General matrix A. On exit, overwritten by Householder vectors.
///
/// @param[out] T
///     Triangular factors T.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup lq
///
template <typename scalar_t>
void gelqf(
    Matrix<scalar_t>& A, TriangularFactors<scalar_t>& T,
    Options const& opts = Options());

//-----------------------------------------
/// Multiply by Q from LQ factorization.
///
/// @param[in] side
///     Side::Left or Side::Right.
///
/// @param[in] op
///     Op::NoTrans or Op::ConjTrans.
///
/// @param[in] A
///     Matrix with Householder vectors.
///
/// @param[in] T
///     Triangular factors.
///
/// @param[in,out] C
///     Matrix C.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup lq
///
template <typename scalar_t>
void unmlq(
    Side side, Op op,
    Matrix<scalar_t>& A, TriangularFactors<scalar_t>& T,
    Matrix<scalar_t>& C,
    Options const& opts = Options());

/// @}


//------------------------------------------------------------------------------
// SVD

/// @defgroup svd Singular Value Decomposition (SVD)
/// @brief SVD factorization
/// @{

//-----------------------------------------
/// Singular Value Decomposition: \( A = U \Sigma V^H \).
///
/// @param[in] A
///     Matrix A.
///
/// @param[out] Sigma
///     Vector of singular values.
///
/// @param[out] U
///     Matrix U.
///
/// @param[out] VT
///     Matrix \( V^H \).
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup svd
///
template <typename scalar_t>
void svd(
    Matrix<scalar_t> A,
    std::vector< blas::real_type<scalar_t> >& Sigma,
    Matrix<scalar_t>& U,
    Matrix<scalar_t>& VT,
    Options const& opts = Options());

/// Without U and VT, compute only singular values. Same as svd_vals.
/// @ingroup svd
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
/// @ingroup svd
template <typename scalar_t>
void svd_vals(
    Matrix<scalar_t> A,
    std::vector< blas::real_type<scalar_t> >& Sigma,
    Options const& opts = Options())
{
    svd( A, Sigma, opts );
}

//-----------------------------------------
/// Multiply by Q from GE2TB reduction (Bidiagonal reduction part 1).
///
/// @param[in] side
///     Side::Left or Side::Right.
///
/// @param[in] op
///     Op::NoTrans or Op::ConjTrans.
///
/// @param[in] A
///     Matrix with reduction factors.
///
/// @param[in] T
///     Triangular factors.
///
/// @param[in,out] C
///     Matrix C.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup svd
///
template <typename scalar_t>
void unmbr_ge2tb(
    Side side, Op op,
    Matrix<scalar_t>& A,
    TriangularFactors<scalar_t> T,
    Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
/// Reduction from general to triangular-band form (SVD step 1).
///
/// @param[in,out] A
///     General matrix A. On exit, reduced to triangular-band.
///
/// @param[out] TU
///     Triangular factors for U.
///
/// @param[out] TV
///     Triangular factors for V.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup svd
///
template <typename scalar_t>
void ge2tb(
    Matrix<scalar_t>& A,
    TriangularFactors<scalar_t>& TU,
    TriangularFactors<scalar_t>& TV,
    Options const& opts = Options());

//-----------------------------------------
/// Bulge Chasing: TriangularBand to Bi-diagonal (SVD step 2).
///
/// @param[in,out] A
///     Triangular band matrix. On exit, bidiagonal.
///
/// @param[out] U
///     Matrix U.
///
/// @param[out] V
///     Matrix V.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup svd
///
template <typename scalar_t>
void tb2bd(
    TriangularBandMatrix<scalar_t>& A,
    Matrix<scalar_t>& U,
    Matrix<scalar_t>& V,
    Options const& opts = Options());

//-----------------------------------------
/// Bi-diagonal SVD (SVD step 3).
///
/// @param[in] jobu
///     Job::NoVec or Job::Vec (compute U).
///
/// @param[in] jobvt
///     Job::NoVec or Job::Vec (compute VT).
///
/// @param[in,out] D
///     Diagonal elements.
///
/// @param[in,out] E
///     Off-diagonal elements.
///
/// @param[out] U
///     Matrix U.
///
/// @param[out] VT
///     Matrix VT.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup svd
///
template <typename scalar_t>
void bdsqr(
    Job jobu, Job jobvt,
    std::vector< blas::real_type<scalar_t> >& D,
    std::vector< blas::real_type<scalar_t> >& E,
    Matrix<scalar_t>& U,
    Matrix<scalar_t>& VT,
    Options const& opts = Options());

/// @}


//------------------------------------------------------------------------------
// Symmetric/Hermitian eigenvalues

/// @defgroup heev Symmetric/Hermitian eigenvalue problems
/// @brief Compute eigenvalues and eigenvectors
/// @{

//-----------------------------------------
/// Compute eigenvalues and eigenvectors (Hermitian matrix).
/// \( A Z = Z \Lambda \).
///
/// @param[in,out] A
///     Hermitian matrix A. On exit, overwritten.
///
/// @param[out] Lambda
///     Eigenvalues.
///
/// @param[out] Z
///     Eigenvectors.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup heev
///
template <typename scalar_t>
void heev(
    HermitianMatrix<scalar_t>& A,
    std::vector< blas::real_type<scalar_t> >& Lambda,
    Matrix<scalar_t>& Z,
    Options const& opts = Options());

/// Without Z, compute only eigenvalues.
/// @ingroup heev
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
/// Overload for real symmetric matrix (dispatched to heev).
/// @ingroup heev
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
/// @ingroup heev
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

/// @}

//------------------------------------------------------------------------------
// Generalized symmetric/Hermitian eigenvalues

/// @defgroup hegv Generalized symmetric/Hermitian eigenvalue problems
/// @brief Solve \( A v = \lambda B v \) and related problems
/// @{

//-----------------------------------------
/// Compute eigenvalues and eigenvectors of generalized Hermitian eigenproblem.
///
/// @param[in] itype
///     1: \( A x = \lambda B x \)
///     2: \( A B x = \lambda x \)
///     3: \( B A x = \lambda x \)
///
/// @param[in,out] A
///     Hermitian matrix A.
///
/// @param[in,out] B
///     Hermitian positive definite matrix B.
///
/// @param[out] Lambda
///     Eigenvalues.
///
/// @param[out] Z
///     Eigenvectors.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup hegv
///
template <typename scalar_t>
void hegv(
    int64_t itype,
    HermitianMatrix<scalar_t>& A,
    HermitianMatrix<scalar_t>& B,
    std::vector< blas::real_type<scalar_t> >& Lambda,
    Matrix<scalar_t>& Z,
    Options const& opts = Options());

/// Without Z, compute only eigenvalues.
/// @ingroup hegv
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
/// Overload for real symmetric matrix (dispatched to hegv).
/// @ingroup hegv
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
/// @ingroup hegv
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
/// Reduce generalized Hermitian eigenproblem to standard form.
///
/// @param[in] itype
///     Problem type.
///
/// @param[in,out] A
///     Hermitian matrix A. On exit, reduced matrix.
///
/// @param[in,out] B
///     Hermitian positive definite matrix B.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup hegv
///
template <typename scalar_t>
void hegst(
    int64_t itype,
    HermitianMatrix<scalar_t>& A,
    HermitianMatrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
/// Overload for real symmetric matrix (dispatched to hegst).
/// @ingroup hegv
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

/// @}


//------------------------------------------------------------------------------
// Symmetric/Hermitian eigenvalue reductions

/// @defgroup he_reduction Symmetric/Hermitian reductions
/// @brief Reduction to band and tridiagonal forms
/// @{

//-----------------------------------------
/// Reduce Hermitian matrix to band form.
///
/// @param[in,out] A
///     Hermitian matrix A. On exit, reduced to band form.
///
/// @param[out] T
///     Triangular factors.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup he_reduction
///
template <typename scalar_t>
void he2hb(
    HermitianMatrix<scalar_t>& A,
    TriangularFactors<scalar_t>& T,
    Options const& opts = Options());

//-----------------------------------------
/// Multiply by Q from HE2HB reduction.
///
/// @param[in] side
///     Side::Left or Side::Right.
///
/// @param[in] op
///     Op::NoTrans or Op::ConjTrans.
///
/// @param[in] A
///     Matrix with reduction factors.
///
/// @param[in] T
///     Triangular factors.
///
/// @param[in,out] C
///     Matrix C.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup he_reduction
///
template <typename scalar_t>
void unmtr_he2hb(
    Side side, Op op,
    HermitianMatrix<scalar_t>& A,
    TriangularFactors<scalar_t> T,
    Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
/// Reduce Hermitian band matrix to real symmetric tridiagonal form.
///
/// @param[in,out] A
///     Hermitian band matrix A.
///
/// @param[out] V
///     Matrix V with Householder vectors.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup he_reduction
///
template <typename scalar_t>
void hb2st(
    HermitianBandMatrix<scalar_t>& A,
    Matrix<scalar_t>& V,
    Options const& opts = Options());

//-----------------------------------------
/// Multiply by Q from HB2ST reduction.
///
/// @param[in] side
///     Side::Left or Side::Right.
///
/// @param[in] op
///     Op::NoTrans or Op::ConjTrans.
///
/// @param[in] V
///     Matrix with Householder vectors.
///
/// @param[in,out] C
///     Matrix C.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup he_reduction
///
template <typename scalar_t>
void unmtr_hb2st(
    Side side, Op op,
    Matrix<scalar_t>& V,
    Matrix<scalar_t>& C,
    Options const& opts = Options());

/// @}

//------------------------------------------------------------------------------
// Tridiagonal Symmetric eigenvalue solvers

/// @defgroup heev_tridiag Tridiagonal eigenvalue solvers
/// @brief Divide and conquer, QR, etc.
/// @{

//-----------------------------------------
/// Solve tridiagonal eigenvalue problem using divide and conquer.
///
/// @param[in,out] D
///     Diagonal elements.
///
/// @param[in,out] E
///     Off-diagonal elements.
///
/// @param[out] Q
///     Eigenvectors.
///
/// @param[in] opts
///     Additional options.
///
/// @ingroup heev_tridiag
///
template <typename real_t>
void stedc(
    std::vector<real_t>& D, std::vector<real_t>& E,
    Matrix<real_t>& Q,
    Options const& opts = Options());

// Internal routines for stedc...
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
/// Multiply by Q from TB2BD reduction (Bidiagonal reduction part 2?).
/// Actually this seems to be related to SVD but placed here?
/// It says unmbr_tb2bd.
///
/// @ingroup svd
template <typename scalar_t>
void unmbr_tb2bd(
    Side side, Op op,
    Matrix<scalar_t>& V,
    Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
/// Solve tridiagonal eigenvalue problem using QR / Pal-Walker-Kahan variant (sterf).
/// Computes eigenvalues only.
///
/// @param[in,out] D Diagonal.
/// @param[in,out] E Off-diagonal.
/// @param[in] opts Options.
///
/// @ingroup heev_tridiag
///
template <typename scalar_t>
void sterf(
    std::vector< scalar_t >& D,
    std::vector< scalar_t >& E,
    Options const& opts = Options());

//-----------------------------------------
/// Solve tridiagonal eigenvalue problem using QR (steqr).
/// Computes eigenvalues and optionally eigenvectors.
///
/// @param[in] jobz Job::NoVec or Job::Vec.
/// @param[in,out] D Diagonal.
/// @param[in,out] E Off-diagonal.
/// @param[out] Z Eigenvectors.
/// @param[in] opts Options.
///
/// @ingroup heev_tridiag
///
template <typename scalar_t>
void steqr(
    Job jobz,
    std::vector< blas::real_type<scalar_t> >& D,
    std::vector< blas::real_type<scalar_t> >& E,
    Matrix<scalar_t>& Z,
    Options const& opts = Options());

// low-level implementation
template <typename scalar_t>
int64_t steqr(
    int64_t n,
    blas::real_type<scalar_t>* D,
    blas::real_type<scalar_t>* E,
    scalar_t* Z, int64_t ldz,
    int64_t nrows,
    blas::real_type<scalar_t>* work, int64_t lwork );

/// @}

//------------------------------------------------------------------------------
// Condition number estimate

/// @defgroup cond Condition number estimate
/// @brief Estimate condition numbers
/// @{

//-----------------------------------------
/// Estimate condition number of general matrix.
///
/// @param[in] in_norm
///     Norm to use.
///
/// @param[in] A
///     Matrix A.
///
/// @param[in] Anorm
///     Norm of A.
///
/// @param[in] opts
///     Additional options.
///
/// @return Reciprocal condition number estimate.
///
/// @ingroup cond
///
template <typename scalar_t>
blas::real_type<scalar_t> gecondest(
    Norm in_norm,
    Matrix<scalar_t>& A,
    blas::real_type<scalar_t> Anorm,
    Options const& opts = Options());

//-----------------------------------------
/// Estimate condition number of Hermitian positive definite matrix.
///
/// @param[in] in_norm
///     Norm to use.
///
/// @param[in] A
///     Hermitian matrix A.
///
/// @param[in] Anorm
///     Norm of A.
///
/// @param[in] opts
///     Additional options.
///
/// @return Reciprocal condition number estimate.
///
/// @ingroup cond
///
template <typename scalar_t>
blas::real_type<scalar_t> pocondest(
    Norm in_norm,
    HermitianMatrix<scalar_t>& A,
    blas::real_type<scalar_t> Anorm,
    Options const& opts = Options());

//-----------------------------------------
/// Estimate condition number of triangular matrix.
///
/// @param[in] in_norm
///     Norm to use.
///
/// @param[in] A
///     Triangular matrix A.
///
/// @param[in] Anorm
///     Norm of A.
///
/// @param[in] opts
///     Additional options.
///
/// @return Reciprocal condition number estimate.
///
/// @ingroup cond
///
template <typename scalar_t>
blas::real_type<scalar_t> trcondest(
    Norm in_norm,
    TriangularMatrix<scalar_t>& A,
    blas::real_type<scalar_t> Anorm,
    Options const& opts = Options());

/// @}


} // namespace slate

//-----------------------------------------
// Simplified C++ API
#include "simplified_api.hh"

#endif // SLATE_HH
