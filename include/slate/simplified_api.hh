// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_SIMPLIFIED_API_HH
#define SLATE_SIMPLIFIED_API_HH

namespace slate {

//------------------------------------------------------------------------------
/// @defgroup multiply Matrix-matrix multiplication routines
/// @brief Simplified interface for matrix-matrix products
///
/// These functions provide a simplified interface to BLAS-3 matrix-matrix 
/// multiplication routines, automatically dispatching to the appropriate
/// underlying function (gemm, hemm, symm, etc.) based on matrix types.
///
/// @{

//-----------------------------------------
/// Matrix-matrix multiply: \\( C = \\alpha A B + \\beta C \\)
///
/// Band matrix times general matrix.
///
/// @param[in] alpha
///     Scalar multiplier for A*B product
///
/// @param[in] A
///     Band matrix operand
///
/// @param[in] B
///     General matrix operand
///
/// @param[in] beta
///     Scalar multiplier for C
///
/// @param[in,out] C
///     On entry, the matrix C. On exit, overwritten by \\( \\alpha A B + \\beta C \\)
///
/// @param[in] opts
///     Additional options (target, lookahead, etc.)
///
/// @ingroup multiply
///
template <typename scalar_t>
void multiply(
    scalar_t alpha, BandMatrix<scalar_t>& A,
                        Matrix<scalar_t>& B,
    scalar_t beta,      Matrix<scalar_t>& C,
    Options const& opts = Options())
{
    gbmm(alpha, A, B, beta, C, opts);
}

//-----------------------------------------
/// Matrix-matrix multiply: \\( C = \\alpha A B + \\beta C \\)
///
/// General matrix times general matrix (GEMM).
///
/// @param[in] alpha
///     Scalar multiplier for A*B product
///
/// @param[in] A
///     General matrix operand, m-by-k
///
/// @param[in] B
///     General matrix operand, k-by-n
///
/// @param[in] beta
///     Scalar multiplier for C
///
/// @param[in,out] C
///     On entry, m-by-n matrix. On exit, overwritten by \\( \\alpha A B + \\beta C \\)
///
/// @param[in] opts
///     Additional options:
///     - target: Target device (host, devices)
///     - lookahead: Number of blocks to lookahead (default 1)
///
/// @ingroup multiply
///
template <typename scalar_t>
void multiply(
    scalar_t alpha, Matrix<scalar_t>& A,
                    Matrix<scalar_t>& B,
    scalar_t beta,  Matrix<scalar_t>& C,
    Options const& opts = Options())
{
    gemm(alpha, A, B, beta, C, opts);
}

//-----------------------------------------
/// Hermitian band matrix multiply (left): \\( C = \\alpha A B + \\beta C \\)
///
/// @param[in] alpha Scalar multiplier
/// @param[in] A Hermitian band matrix (left operand)
/// @param[in] B General matrix
/// @param[in] beta Scalar multiplier for C
/// @param[in,out] C On exit, \\( \\alpha A B + \\beta C \\)
/// @param[in] opts Additional options
/// @ingroup multiply
///
template <typename scalar_t>
void multiply(
    scalar_t alpha, HermitianBandMatrix<scalar_t>& A,
                                 Matrix<scalar_t>& B,
    scalar_t beta,               Matrix<scalar_t>& C,
    Options const& opts = Options())
{
    hbmm(Side::Left, alpha, A, B, beta, C, opts);
}

//-----------------------------------------
/// Hermitian band matrix multiply (right): \\( C = \\alpha A B + \\beta C \\)
///
/// @param[in] alpha Scalar multiplier
/// @param[in] A General matrix
/// @param[in] B Hermitian band matrix (right operand)
/// @param[in] beta Scalar multiplier for C
/// @param[in,out] C On exit, \\( \\alpha A B + \\beta C \\)
/// @param[in] opts Additional options
/// @ingroup multiply
///
template <typename scalar_t>
void multiply(
    scalar_t alpha,              Matrix<scalar_t>& A,
                    HermitianBandMatrix<scalar_t>& B,
    scalar_t beta,               Matrix<scalar_t>& C,
    Options const& opts = Options())
{
    hbmm(Side::Right, alpha, B, A, beta, C, opts);
}

//-----------------------------------------
/// Hermitian matrix multiply (left): \\( C = \\alpha A B + \\beta C \\)
///
/// where A is Hermitian.
///
/// @param[in] alpha Scalar multiplier
/// @param[in] A Hermitian matrix (left operand)
/// @param[in] B General matrix
/// @param[in] beta Scalar multiplier for C
/// @param[in,out] C On exit, \\( \\alpha A B + \\beta C \\)
/// @param[in] opts Additional options
/// @ingroup multiply
///
template <typename scalar_t>
void multiply(
    scalar_t alpha, HermitianMatrix<scalar_t>& A,
                             Matrix<scalar_t>& B,
    scalar_t beta,           Matrix<scalar_t>& C,
    Options const& opts = Options())
{
    hemm(Side::Left, alpha, A, B, beta, C, opts);
}

//-----------------------------------------
/// Hermitian matrix multiply (right): \\( C = \\alpha A B + \\beta C \\)
///
/// where B is Hermitian.
///
/// @param[in] alpha Scalar multiplier
/// @param[in] A General matrix
/// @param[in] B Hermitian matrix (right operand)
/// @param[in] beta Scalar multiplier for C
/// @param[in,out] C On exit, \\( \\alpha A B + \\beta C \\)
/// @param[in] opts Additional options
/// @ingroup multiply
///
template <typename scalar_t>
void multiply(
    scalar_t alpha,          Matrix<scalar_t>& A,
                    HermitianMatrix<scalar_t>& B,
    scalar_t beta,           Matrix<scalar_t>& C,
    Options const& opts = Options())
{
    hemm(Side::Right, alpha, B, A, beta, C, opts);
}

//-----------------------------------------
/// Symmetric matrix multiply (left): \\( C = \\alpha A B + \\beta C \\)
///
/// where A is symmetric.
///
/// @param[in] alpha Scalar multiplier
/// @param[in] A Symmetric matrix (left operand)
/// @param[in] B General matrix
/// @param[in] beta Scalar multiplier for C
/// @param[in,out] C On exit, \\( \\alpha A B + \\beta C \\)
/// @param[in] opts Additional options
/// @ingroup multiply
///
template <typename scalar_t>
void multiply(
    scalar_t alpha, SymmetricMatrix<scalar_t>& A,
                             Matrix<scalar_t>& B,
    scalar_t beta,           Matrix<scalar_t>& C,
    Options const& opts = Options())
{
    symm(Side::Left, alpha, A, B, beta, C, opts);
}

//-----------------------------------------
/// Symmetric matrix multiply (right): \\( C = \\alpha A B + \\beta C \\)
///
/// where B is symmetric.
///
/// @param[in] alpha Scalar multiplier
/// @param[in] A General matrix
/// @param[in] B Symmetric matrix (right operand)
/// @param[in] beta Scalar multiplier for C
/// @param[in,out] C On exit, \\( \\alpha A B + \\beta C \\)
/// @param[in] opts Additional options
/// @ingroup multiply
///
template <typename scalar_t>
void multiply(
    scalar_t alpha,          Matrix<scalar_t>& A,
                    SymmetricMatrix<scalar_t>& B,
    scalar_t beta,           Matrix<scalar_t>& C,
    Options const& opts = Options())
{
    symm(Side::Right, alpha, B, A, beta, C, opts);
}

/// @}

//------------------------------------------------------------------------------
/// @defgroup triangular_multiply Triangular matrix multiplication
/// @brief Multiply by triangular matrices
///
/// Computes \\( B = \\alpha op(A) B \\) or \\( B = \\alpha B op(A) \\)
/// where A is triangular.
///
/// @{

//-----------------------------------------
/// Triangular matrix multiply (left): \\( B = \\alpha A B \\)
///
/// @param[in] alpha
///     Scalar multiplier
///
/// @param[in] A
///     Triangular matrix
///
/// @param[in,out] B
///     On entry, the matrix B. On exit, \\( \\alpha A B \\)
///
/// @param[in] opts
///     Additional options:
///     - uplo: Whether A is upper or lower triangular
///     - diag: Whether A has unit diagonal
///     - trans: Operation on A (NoTrans, Trans, ConjTrans)
///
/// @ingroup triangular_multiply
///
template <typename scalar_t>
void triangular_multiply(
    scalar_t alpha, TriangularMatrix<scalar_t>& A,
                              Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    trmm(Side::Left, alpha, A, B, opts);
}

//-----------------------------------------
/// Triangular matrix multiply (right): \\( B = \\alpha B A \\)
///
/// @param[in] alpha
///     Scalar multiplier
///
/// @param[in,out] B
///     On entry, the matrix B. On exit, \\( \\alpha B A \\)
///
/// @param[in] A
///     Triangular matrix
///
/// @param[in] opts
///     Additional options
///
/// @ingroup triangular_multiply
///
template <typename scalar_t>
void triangular_multiply(
    scalar_t alpha,           Matrix<scalar_t>& B,
                    TriangularMatrix<scalar_t>& A,
    Options const& opts = Options())
{
    trmm(Side::Right, alpha, A, B, opts);
}

/// @}

//------------------------------------------------------------------------------
/// @defgroup triangular_solve Triangular system solve
/// @brief Solve triangular systems
///
/// Solves \\( op(A) X = \\alpha B \\) or \\( X op(A) = \\alpha B \\)
/// where A is triangular.
///
/// @{

//-----------------------------------------
/// Triangular band solve (left): \\( A X = \\alpha B \\)
///
/// @param[in] alpha
///     Scalar multiplier for B
///
/// @param[in] A
///     Triangular band matrix
///
/// @param[in,out] B
///     On entry, the right-hand side B. On exit, the solution X
///
/// @param[in] opts
///     Additional options
///
/// @ingroup triangular_solve
///
template <typename scalar_t>
void triangular_solve(
    scalar_t alpha, TriangularBandMatrix<scalar_t>& A,
                                  Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    tbsm(Side::Left, alpha, A, B, opts);
}

//-----------------------------------------
/// Triangular band solve (right): \\( X A = \\alpha B \\)
///
/// @param[in] alpha
///     Scalar multiplier for B
///
/// @param[in,out] B
///     On entry, the right-hand side B. On exit, the solution X
///
/// @param[in] A
///     Triangular band matrix
///
/// @param[in] opts
///     Additional options
///
/// @ingroup triangular_solve
///
template <typename scalar_t>
void triangular_solve(
    scalar_t alpha,               Matrix<scalar_t>& B,
                    TriangularBandMatrix<scalar_t>& A,
    Options const& opts = Options())
{
    tbsm(Side::Right, alpha, A, B, opts);
}

//-----------------------------------------
/// Triangular solve (left): \\( A X = \\alpha B \\)
///
/// @param[in] alpha
///     Scalar multiplier for B
///
/// @param[in] A
///     Triangular matrix
///
/// @param[in,out] B
///     On entry, the right-hand side B. On exit, the solution X
///
/// @param[in] opts
///     Additional options:
///     - uplo: Whether A is upper or lower triangular
///     - diag: Whether A has unit diagonal
///     - trans: Operation on A
///
/// @ingroup triangular_solve
///
template <typename scalar_t>
void triangular_solve(
    scalar_t alpha, TriangularMatrix<scalar_t>& A,
                              Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    trsm(Side::Left, alpha, A, B, opts);
}

//-----------------------------------------
/// Triangular solve (right): \\( X A = \\alpha B \\)
///
/// @param[in] alpha
///     Scalar multiplier for B
///
/// @param[in,out] B
///     On entry, the right-hand side B. On exit, the solution X
///
/// @param[in] A
///     Triangular matrix
///
/// @param[in] opts
///     Additional options
///
/// @ingroup triangular_solve
///
template <typename scalar_t>
void triangular_solve(
    scalar_t alpha,           Matrix<scalar_t>& B,
                    TriangularMatrix<scalar_t>& A,
    Options const& opts = Options())
{
    trsm(Side::Right, alpha, A, B, opts);
}

/// @}

//------------------------------------------------------------------------------
/// @defgroup rank_update Rank-k and rank-2k updates
/// @brief Hermitian and symmetric rank updates
///
/// @{

//-----------------------------------------
/// Hermitian rank-k update: \\( C = \\alpha A A^H + \\beta C \\)
///
/// @param[in] alpha
///     Real scalar multiplier for \\( A A^H \\)
///
/// @param[in] A
///     Matrix operand of size n-by-k
///
/// @param[in] beta
///     Real scalar multiplier for C
///
/// @param[in,out] C
///     Hermitian matrix. On exit, updated by \\( \\alpha A A^H + \\beta C \\)
///
/// @param[in] opts
///     Additional options:
///     - uplo: Which triangle of C to update
///     - trans: Whether to use A or A^H
///
/// @ingroup rank_update
///
template <typename scalar_t>
void rank_k_update(
    blas::real_type<scalar_t> alpha,          Matrix<scalar_t>& A,
    blas::real_type<scalar_t> beta,  HermitianMatrix<scalar_t>& C,
    Options const& opts = Options())
{
    herk(alpha, A, beta, C, opts);
}

//-----------------------------------------
/// Symmetric rank-k update: \\( C = \\alpha A A^T + \\beta C \\)
///
/// @param[in] alpha
///     Scalar multiplier for \\( A A^T \\)
///
/// @param[in] A
///     Matrix operand of size n-by-k
///
/// @param[in] beta
///     Scalar multiplier for C
///
/// @param[in,out] C
///     Symmetric matrix. On exit, updated by \\( \\alpha A A^T + \\beta C \\)
///
/// @param[in] opts
///     Additional options
///
/// @ingroup rank_update
///
template <typename scalar_t>
void rank_k_update(
    scalar_t alpha,           Matrix<scalar_t>& A,
    scalar_t beta,   SymmetricMatrix<scalar_t>& C,
    Options const& opts = Options())
{
    syrk(alpha, A, beta, C, opts);
}

//-----------------------------------------
/// Hermitian rank-2k update: \\( C = \\alpha A B^H + \\overline{\\alpha} B A^H + \\beta C \\)
///
/// @param[in] alpha
///     Complex scalar multiplier
///
/// @param[in] A
///     Matrix operand A, size n-by-k
///
/// @param[in] B
///     Matrix operand B, size n-by-k
///
/// @param[in] beta
///     Real scalar multiplier for C
///
/// @param[in,out] C
///     Hermitian matrix. On exit, updated
///
/// @param[in] opts
///     Additional options
///
/// @ingroup rank_update
///
template <typename scalar_t>
void rank_2k_update(
    scalar_t alpha,                           Matrix<scalar_t>& A,
                                              Matrix<scalar_t>& B,
    blas::real_type<scalar_t> beta,  HermitianMatrix<scalar_t>& C,
    Options const& opts = Options())
{
    her2k(alpha, A, B, beta, C, opts);
}

//-----------------------------------------
/// Symmetric rank-2k update: \\( C = \\alpha A B^T + \\alpha B A^T + \\beta C \\)
///
/// @param[in] alpha
///     Scalar multiplier
///
/// @param[in] A
///     Matrix operand A, size n-by-k
///
/// @param[in] B
///     Matrix operand B, size n-by-k
///
/// @param[in] beta
///     Scalar multiplier for C
///
/// @param[in,out] C
///     Symmetric matrix. On exit, updated
///
/// @param[in] opts
///     Additional options
///
/// @ingroup rank_update
///
template <typename scalar_t>
void rank_2k_update(
    scalar_t alpha,           Matrix<scalar_t>& A,
                              Matrix<scalar_t>& B,
    scalar_t beta,   SymmetricMatrix<scalar_t>& C,
    Options const& opts = Options())
{
    syr2k(alpha, A, B, beta, C, opts);
}

/// @}

//------------------------------------------------------------------------------
/// @defgroup lu_solve LU factorization and solve
/// @brief Solve \\( A X = B \\) using LU decomposition
///
/// @{

//-----------------------------------------
/// Solve \\( A X = B \\) using LU factorization (band matrix)
///
/// @param[in,out] A
///     Band matrix. On exit, overwritten by LU factors
///
/// @param[in,out] B
///     On entry, right-hand side matrix. On exit, solution X
///
/// @param[in] opts
///     Additional options
///
/// @ingroup lu_solve
///
template <typename scalar_t>
void lu_solve(
    BandMatrix<scalar_t>& A,
        Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    Pivots pivots;
    gbsv(A, pivots, B, opts);
}

//-----------------------------------------
/// Solve \\( A X = B \\) using LU factorization (general matrix)
///
/// @param[in,out] A
///     General matrix. On exit, overwritten by LU factors
///
/// @param[in,out] B
///     On entry, right-hand side matrix. On exit, solution X
///
/// @param[in] opts
///     Additional options
///
/// @return Info code (0 = success, >0 = singular)
///
/// @ingroup lu_solve
///
template <typename scalar_t>
int64_t lu_solve(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    Pivots pivots;
    return gesv( A, pivots, B, opts );
}

//-----------------------------------------
/// LU factorization (band matrix): \\( A = L U \\)
///
/// @param[in,out] A
///     Band matrix. On exit, overwritten by LU factors
///
/// @param[out] pivots
///     Pivot indices
///
/// @param[in] opts
///     Additional options
///
/// @ingroup lu_solve
///
template <typename scalar_t>
void lu_factor(
    BandMatrix<scalar_t>& A, Pivots& pivots,
    Options const& opts = Options())
{
    gbtrf(A, pivots, opts);
}

//-----------------------------------------
/// LU factorization (general matrix): \\( A = P L U \\)
///
/// @param[in,out] A
///     General matrix. On exit, overwritten by LU factors
///
/// @param[out] pivots
///     Pivot indices representing permutation P
///
/// @param[in] opts
///     Additional options
///
/// @return Info code (0 = success, >0 = singular)
///
/// @ingroup lu_solve
///
template <typename scalar_t>
int64_t lu_factor(
    Matrix<scalar_t>& A, Pivots& pivots,
    Options const& opts = Options())
{
    return getrf( A, pivots, opts );
}

//-----------------------------------------
/// Solve using existing LU factorization (band): \\( A X = B \\)
///
/// @param[in] A
///     Band matrix with LU factors from lu_factor
///
/// @param[in] pivots
///     Pivot indices from lu_factor
///
/// @param[in,out] B
///     On entry, right-hand side. On exit, solution X
///
/// @param[in] opts
///     Additional options
///
/// @ingroup lu_solve
///
template <typename scalar_t>
void lu_solve_using_factor(
    BandMatrix<scalar_t>& A, Pivots& pivots,
        Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    gbtrs(A, pivots, B, opts);
}

//-----------------------------------------
/// Solve using existing LU factorization: \\( A X = B \\)
///
/// @param[in] A
///     Matrix with LU factors from lu_factor
///
/// @param[in] pivots
///     Pivot indices from lu_factor
///
/// @param[in,out] B
///     On entry, right-hand side. On exit, solution X
///
/// @param[in] opts
///     Additional options
///
/// @ingroup lu_solve
///
template <typename scalar_t>
void lu_solve_using_factor(
    Matrix<scalar_t>& A, Pivots& pivots,
    Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    getrs(A, pivots, B, opts);
}

//-----------------------------------------
/// Compute matrix inverse using LU factorization (in-place)
///
/// @param[in,out] A
///     Matrix with LU factors from lu_factor. On exit, overwritten by \\( A^{-1} \\)
///
/// @param[in] pivots
///     Pivot indices from lu_factor
///
/// @param[in] opts
///     Additional options
///
/// @ingroup lu_solve
///
template <typename scalar_t>
void lu_inverse_using_factor(
    Matrix<scalar_t>& A, Pivots& pivots,
    Options const& opts = Options())
{
    getri(A, pivots, opts);
}

//-----------------------------------------
/// Compute matrix inverse using LU factorization (out-of-place)
///
/// @param[in] A
///     Matrix with LU factors from lu_factor
///
/// @param[in] pivots
///     Pivot indices from lu_factor
///
/// @param[out] A_inverse
///     On exit, contains \\( A^{-1} \\)
///
/// @param[in] opts
///     Additional options
///
/// @ingroup lu_solve
///
template <typename scalar_t>
void lu_inverse_using_factor_out_of_place(
    Matrix<scalar_t>& A, Pivots& pivots,
    Matrix<scalar_t>& A_inverse,
    Options const& opts = Options())

{
    getri(A, pivots, A_inverse, opts);
}

//-----------------------------------------
/// Estimate reciprocal condition number using LU factorization
///
/// @param[in] in_norm
///     Norm to use (One, Inf, Fro)
///
/// @param[in] A
///     Matrix with LU factors from lu_factor
///
/// @param[in] Anorm
///     Norm of original matrix A before factorization
///
/// @param[in] opts
///     Additional options
///
/// @return Reciprocal condition number estimate \\( 1 / \\kappa(A) \\)
///
/// @ingroup lu_solve
///
template <typename scalar_t>
blas::real_type<scalar_t> lu_rcondest_using_factor(
    Norm in_norm,
    Matrix<scalar_t>& A,
    blas::real_type<scalar_t> Anorm,
    Options const& opts = Options())
{
    return gecondest( in_norm, A, Anorm, opts );
}

/// @}

//------------------------------------------------------------------------------
/// @defgroup chol_solve Cholesky factorization and solve
/// @brief Solve \\( A X = B \\) for positive definite A using Cholesky
///
/// @{

//-----------------------------------------
/// Solve \\( A X = B \\) using Cholesky (Hermitian band matrix)
///
/// @param[in,out] A
///     Hermitian band matrix. On exit, overwritten by Cholesky factor
///
/// @param[in,out] B
///     On entry, right-hand side. On exit, solution X
///
/// @param[in] opts
///     Additional options
///
/// @ingroup chol_solve
///
template <typename scalar_t>
void chol_solve(
    HermitianBandMatrix<scalar_t>& A,
                 Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    pbsv(A, B, opts);
}

//-----------------------------------------
/// Solve \\( A X = B \\) using Cholesky (Hermitian matrix)
///
/// @param[in,out] A
///     Hermitian positive definite matrix. On exit, overwritten by Cholesky factor
///
/// @param[in,out] B
///     On entry, right-hand side. On exit, solution X
///
/// @param[in] opts
///     Additional options
///
/// @return Info code (0 = success, >0 = not positive definite)
///
/// @ingroup chol_solve
///
template <typename scalar_t>
int64_t chol_solve(
    HermitianMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    return posv( A, B, opts );
}

//-----------------------------------------
/// Solve \\( A X = B \\) using Cholesky (real symmetric matrix)
///
/// @param[in,out] A
///     Symmetric positive definite matrix. On exit, overwritten by Cholesky factor
///
/// @param[in,out] B
///     On entry, right-hand side. On exit, solution X
///
/// @param[in] opts
///     Additional options
///
/// @return Info code
///
/// @ingroup chol_solve
///
template <typename scalar_t>
int64_t chol_solve(
    SymmetricMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    return posv( A, B, opts );
}

//-----------------------------------------
/// Cholesky factorization (Hermitian band): \\( A = L L^H \\)
///
/// @param[in,out] A
///     Hermitian band matrix. On exit, overwritten by Cholesky factor
///
/// @param[in] opts
///     Additional options
///
/// @return Info code (0 = success, >0 = not positive definite)
///
/// @ingroup chol_solve
///
template <typename scalar_t>
int64_t chol_factor(
    HermitianBandMatrix<scalar_t>& A,
    Options const& opts = Options())
{
    return pbtrf( A, opts );
}

//-----------------------------------------
/// Cholesky factorization (Hermitian): \\( A = L L^H \\) or \\( A = U^H U \\)
///
/// @param[in,out] A
///     Hermitian positive definite matrix. On exit, overwritten by Cholesky factor
///
/// @param[in] opts
///     Additional options:
///     - uplo: Whether to compute lower (L) or upper (U) factor
///
/// @return Info code (0 = success, >0 = not positive definite)
///
/// @ingroup chol_solve
///
template <typename scalar_t>
int64_t chol_factor(
    HermitianMatrix<scalar_t>& A,
    Options const& opts = Options())
{
    return potrf( A, opts );
}

//-----------------------------------------
/// Cholesky factorization (real symmetric): \\( A = L L^T \\) or \\( A = U^T U \\)
///
/// @param[in,out] A
///     Symmetric positive definite matrix. On exit, overwritten by Cholesky factor
///
/// @param[in] opts
///     Additional options
///
/// @return Info code
///
/// @ingroup chol_solve
///
template <typename scalar_t>
int64_t chol_factor(
    SymmetricMatrix<scalar_t>& A,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    return potrf( A, opts );
}

//-----------------------------------------
/// Solve using existing Cholesky factorization (Hermitian band)
///
/// @param[in] A
///     Hermitian band matrix with Cholesky factor from chol_factor
///
/// @param[in,out] B
///     On entry, right-hand side. On exit, solution X
///
/// @param[in] opts
///     Additional options
///
/// @ingroup chol_solve
///
template <typename scalar_t>
void chol_solve_using_factor(
    HermitianBandMatrix<scalar_t>& A,
                 Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    pbtrs(A, B, opts);
}

//-----------------------------------------
/// Solve using existing Cholesky factorization (Hermitian)
///
/// @param[in] A
///     Hermitian matrix with Cholesky factor from chol_factor
///
/// @param[in,out] B
///     On entry, right-hand side. On exit, solution X
///
/// @param[in] opts
///     Additional options
///
/// @ingroup chol_solve
///
template <typename scalar_t>
void chol_solve_using_factor(
    HermitianMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    potrs(A, B, opts);
}

//-----------------------------------------
/// Solve using existing Cholesky factorization (real symmetric)
///
/// @param[in] A
///     Symmetric matrix with Cholesky factor
///
/// @param[in,out] B
///     On entry, right-hand side. On exit, solution X
///
/// @param[in] opts
///     Additional options
///
/// @ingroup chol_solve
///
template <typename scalar_t>
void chol_solve_using_factor(
    SymmetricMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    potrs(A, B, opts);
}

//-----------------------------------------
/// Compute inverse using Cholesky factorization
///
/// @param[in,out] A
///     Hermitian matrix with Cholesky factor. On exit, overwritten by \\( A^{-1} \\)
///
/// @param[in] opts
///     Additional options
///
/// @ingroup chol_solve
///
template <typename scalar_t>
void chol_inverse_using_factor(
    HermitianMatrix<scalar_t>& A,
    Options const& opts = Options())
{
    potri(A, opts);
}

//-----------------------------------------
/// Estimate reciprocal condition number using Cholesky factorization
///
/// @param[in] in_norm
///     Norm to use
///
/// @param[in] A
///     Hermitian matrix with Cholesky factor
///
/// @param[in] Anorm
///     Norm of original matrix before factorization
///
/// @param[in] opts
///     Additional options
///
/// @return Reciprocal condition number estimate
///
/// @ingroup chol_solve
///
template <typename scalar_t>
blas::real_type<scalar_t> chol_rcondest_using_factor(
    Norm in_norm,
    HermitianMatrix<scalar_t>& A,
    blas::real_type<scalar_t> Anorm,
    Options const& opts = Options())
{
    return pocondest( in_norm, A, Anorm, opts );
}

/// @}

//------------------------------------------------------------------------------
/// @defgroup indefinite_solve Symmetric indefinite factorization
/// @brief Solve \\( A X = B \\) for indefinite symmetric/Hermitian A using Aasen's algorithm
///
/// @{

//-----------------------------------------
/// Solve \\( A X = B \\) using Aasen's algorithm (Hermitian indefinite)
///
/// Aasen's algorithm uses a block tridiagonal factorization suitable for indefinite matrices.
///
/// @param[in,out] A
///     Hermitian indefinite matrix. On exit, overwritten by factorization
///
/// @param[in,out] B
///     On entry, right-hand side. On exit, solution X
///
/// @param[in] opts
///     Additional options
///
/// @return Info code (0 = success)
///
/// @ingroup indefinite_solve
///
template <typename scalar_t>
int64_t indefinite_solve(
    HermitianMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    // auxiliary matrices
    auto H = slate::Matrix<scalar_t>::emptyLike(A);

    int64_t kl = A.tileNb(0);
    int64_t ku = A.tileNb(0);
    auto T = slate::BandMatrix<scalar_t>::emptyLike(A, kl, ku);

    Pivots pivots, pivots2;
    return hesv( A, pivots, T, pivots2, H, B, opts );
}

//-----------------------------------------
/// Solve \\( A X = B \\) using Aasen's algorithm (real symmetric indefinite)
///
/// @param[in,out] A
///     Symmetric indefinite matrix
///
/// @param[in,out] B
///     On entry, right-hand side. On exit, solution X
///
/// @param[in] opts
///     Additional options
///
/// @return Info code
///
/// @ingroup indefinite_solve
///
template <typename scalar_t>
int64_t indefinite_solve(
    SymmetricMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    // auxiliary matrices
    auto H = slate::Matrix<scalar_t>::emptyLike(A);

    int64_t kl = A.tileNb(0);
    int64_t ku = A.tileNb(0);
    auto T = slate::BandMatrix<scalar_t>::emptyLike(A, kl, ku);

    Pivots pivots, pivots2;
    return sysv( A, pivots, T, pivots2, H, B, opts );
}

//-----------------------------------------
/// Aasen's factorization (Hermitian indefinite)
///
/// @param[in,out] A
///     Hermitian matrix. On exit, overwritten by factorization
///
/// @param[out] pivots
///     First set of pivot indices
///
/// @param[out] T
///     Band matrix containing tridiagonal factors
///
/// @param[out] pivots2
///     Second set of pivot indices
///
/// @param[out] H
///     Auxiliary matrix for Householder reflectors
///
/// @param[in] opts
///     Additional options
///
/// @return Info code
///
/// @ingroup indefinite_solve
///
template <typename scalar_t>
int64_t indefinite_factor(
    HermitianMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& H,
    Options const& opts = Options())
{
    return hetrf( A, pivots, T, pivots2, H, opts );
}

//-----------------------------------------
/// Aasen's factorization (real symmetric indefinite)
///
/// @param[in,out] A
///     Symmetric matrix. On exit, overwritten by factorization
///
/// @param[out] pivots
///     First set of pivot indices
///
/// @param[out] T
///     Band matrix containing tridiagonal factors
///
/// @param[out] pivots2
///     Second set of pivot indices
///
/// @param[out] H
///     Auxiliary matrix
///
/// @param[in] opts
///     Additional options
///
/// @return Info code
///
/// @ingroup indefinite_solve
///
template <typename scalar_t>
int64_t indefinite_factor(
    SymmetricMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& H,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    return sytrf( A, pivots, T, pivots2, H, opts );
}

//-----------------------------------------
/// Solve using Aasen's factorization (Hermitian)
///
/// @param[in] A
///     Hermitian matrix with Aasen factors
///
/// @param[in] pivots
///     First set of pivots from indefinite_factor
///
/// @param[in] T
///     Band matrix from indefinite_factor
///
/// @param[in] pivots2
///     Second set of pivots from indefinite_factor
///
/// @param[in,out] B
///     On entry, right-hand side. On exit, solution X
///
/// @param[in] opts
///     Additional options
///
/// @ingroup indefinite_solve
///
template <typename scalar_t>
void indefinite_solve_using_factor(
    HermitianMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    hetrs(A, pivots, T, pivots2, B, opts);
}

//-----------------------------------------
/// Solve using Aasen's factorization (real symmetric)
///
/// @param[in] A
///     Symmetric matrix with Aasen factors
///
/// @param[in] pivots
///     First set of pivots
///
/// @param[in] T
///     Band matrix from factorization
///
/// @param[in] pivots2
///     Second set of pivots
///
/// @param[in,out] B
///     On entry, right-hand side. On exit, solution X
///
/// @param[in] opts
///     Additional options
///
/// @ingroup indefinite_solve
///
template <typename scalar_t>
void indefinite_solve_using_factor(
    SymmetricMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& B,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    sytrs(A, pivots, T, pivots2, B, opts);
}

/// @}

//------------------------------------------------------------------------------
/// @defgroup least_squares Least squares problems
/// @brief Solve overdetermined or underdetermined systems
///
/// @{

//-----------------------------------------
/// Solve least squares problem: minimize \\( \\|B - A X\\|_2 \\)
///
/// Solves overdetermined (m > n) or underdetermined (m < n) systems using QR or LQ.
///
/// @param[in,out] A
///     General matrix, m-by-n. On exit, overwritten by QR or LQ factors
///
/// @param[in,out] BX
///     On entry, right-hand side B (m-by-nrhs for overdetermined, n-by-nrhs for underdetermined).
///     On exit, solution X
///
/// @param[in] opts
///     Additional options:
///     - method: QR method to use (cholqr, geqrf)
///
/// @ingroup least_squares
///
template <typename scalar_t>
void least_squares_solve(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& BX,
    Options const& opts = Options())
{
    gels( A, BX, opts );
}

/// @}

//------------------------------------------------------------------------------
/// @defgroup qr_factor QR factorization
/// @brief Compute and apply QR factorization
///
/// @{

//-----------------------------------------
/// QR factorization: \\( A = Q R \\)
///
/// @param[in,out] A
///     General matrix, m-by-n. On exit, overwritten by QR factors
///
/// @param[out] T
///     Triangular factors for representing Q
///
/// @param[in] opts
///     Additional options:
///     - method: QR algorithm (geqrf, cholqr)
///     - ib: Inner blocking size
///
/// @ingroup qr_factor
///
template <typename scalar_t>
void qr_factor(
    Matrix<scalar_t>& A, TriangularFactors<scalar_t>& T,
    Options const& opts = Options())
{
    geqrf(A, T, opts);
}

//-----------------------------------------
/// Multiply by Q from QR factorization: \\( C = Q C \\) or \\( C = Q^H C \\) or \\( C = C Q \\) or \\( C = C Q^H \\)
///
/// @param[in] side
///     Whether Q multiplies from left or right
///
/// @param[in] op
///     Operation on Q (NoTrans for Q, ConjTrans for Q^H)
///
/// @param[in] A
///     Matrix with QR factors from qr_factor
///
/// @param[in] T
///     Triangular factors from qr_factor
///
/// @param[in,out] C
///     On entry, the matrix C. On exit, the product
///
/// @param[in] opts
///     Additional options
///
/// @ingroup qr_factor
///
template <typename scalar_t>
void qr_multiply_by_q(
    Side side, Op op,
    Matrix<scalar_t>& A, TriangularFactors<scalar_t>& T,
    Matrix<scalar_t>& C,
    Options const& opts = Options())
{
    unmqr(side, op, A, T, C, opts);
}

/// @}

//------------------------------------------------------------------------------
/// @defgroup lq_factor LQ factorization
/// @brief Compute and apply LQ factorization
///
/// @{

//-----------------------------------------
/// LQ factorization: \\( A = L Q \\)
///
/// @param[in,out] A
///     General matrix, m-by-n. On exit, overwritten by LQ factors
///
/// @param[out] T
///     Triangular factors for representing Q
///
/// @param[in] opts
///     Additional options
///
/// @ingroup lq_factor
///
template <typename scalar_t>
void lq_factor(
    Matrix<scalar_t>& A, TriangularFactors<scalar_t>& T,
    Options const& opts = Options())
{
    gelqf(A, T, opts);
}

//-----------------------------------------
/// Multiply by Q from LQ factorization: \\( C = Q C \\) or \\( C = Q^H C \\) or \\( C = C Q \\) or \\( C = C Q^H \\)
///
/// @param[in] side
///     Whether Q multiplies from left or right
///
/// @param[in] op
///     Operation on Q (NoTrans for Q, ConjTrans for Q^H)
///
/// @param[in] A
///     Matrix with LQ factors from lq_factor
///
/// @param[in] T
///     Triangular factors from lq_factor
///
/// @param[in,out] C
///     On entry, the matrix C. On exit, the product
///
/// @param[in] opts
///     Additional options
///
/// @ingroup lq_factor
///
template <typename scalar_t>
void lq_multiply_by_q(
    Side side, Op op,
    Matrix<scalar_t>& A, TriangularFactors<scalar_t>& T,
    Matrix<scalar_t>& C,
    Options const& opts = Options())
{
    unmlq(side, op, A, T, C, opts);
}

/// @}

//------------------------------------------------------------------------------
/// @defgroup triangular_condest Triangular condition number estimation
/// @brief Estimate condition numbers of triangular matrices
///
/// @{

//-----------------------------------------
/// Estimate reciprocal condition number of triangular matrix
///
/// @param[in] in_norm
///     Norm to use (One or Inf)
///
/// @param[in] A
///     Triangular matrix
///
/// @param[in] Anorm
///     Norm of A computed before calling this function
///
/// @param[in] opts
///     Additional options
///
/// @return Reciprocal condition number estimate \\( 1 / \\kappa(A) \\)
///
/// @ingroup triangular_condest
///
template <typename scalar_t>
blas::real_type<scalar_t> triangular_rcondest(
    Norm in_norm,
    TriangularMatrix<scalar_t>& A,
    blas::real_type<scalar_t> Anorm,
    Options const& opts = Options())
{
    return trcondest( in_norm, A, Anorm, opts );
}

/// @}

//------------------------------------------------------------------------------
/// @defgroup heev Hermitian/symmetric eigenvalue problems
/// @brief Compute eigenvalues and eigenvectors of Hermitian/symmetric matrices
///
/// @{

//-----------------------------------------
/// Compute eigenvalues only (Hermitian matrix)
///
/// @param[in,out] A
///     Hermitian matrix. On exit, overwritten
///
/// @param[out] Lambda
///     Vector of eigenvalues in ascending order
///
/// @param[in] opts
///     Additional options
///
/// @ingroup heev
///
template <typename scalar_t>
void eig_vals(
    HermitianMatrix<scalar_t>& A,
    std::vector< blas::real_type<scalar_t> >& Lambda,
    Options const& opts = Options())
{
    Matrix<scalar_t> Z;
    heev( A, Lambda, Z, opts );
}

//-----------------------------------------
/// Compute eigenvalues only (Hermitian). Alias for eig_vals.
///
/// @param[in,out] A Hermitian matrix
/// @param[out] Lambda Eigenvalues
/// @param[in] opts Additional options
/// @ingroup heev
///
template <typename scalar_t>
void eig(
    HermitianMatrix<scalar_t>& A,
    std::vector< blas::real_type<scalar_t> >& Lambda,
    Options const& opts = Options())
{
    eig_vals( A, Lambda, opts );
}

//-----------------------------------------
/// Compute eigenvalues and eigenvectors (Hermitian matrix)
///
/// @param[in,out] A
///     Hermitian matrix. On exit, overwritten
///
/// @param[out] Lambda
///     Vector of eigenvalues in ascending order
///
/// @param[out] Z
///     Eigenvector matrix. Column i contains eigenvector for Lambda[i]
///
/// @param[in] opts
///     Additional options
///
/// @ingroup heev
///
template <typename scalar_t>
void eig(
    HermitianMatrix<scalar_t>& A,
    std::vector< blas::real_type<scalar_t> >& Lambda,
    Matrix<scalar_t>& Z,
    Options const& opts = Options())
{
    heev( A, Lambda, Z, opts );
}

//-----------------------------------------
/// Compute eigenvalues only (real symmetric matrix)
///
/// @param[in,out] A Symmetric matrix
/// @param[out] Lambda Eigenvalues
/// @param[in] opts Additional options
/// @ingroup heev
///
template <typename scalar_t>
void eig_vals(
    SymmetricMatrix<scalar_t>& A,
    std::vector< blas::real_type<scalar_t> >& Lambda,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    Matrix<scalar_t> Z;
    syev( A, Lambda, Z, opts );
}

//-----------------------------------------
/// Compute eigenvalues only (symmetric). Alias for eig_vals.
///
/// @param[in,out] A Symmetric matrix
/// @param[out] Lambda Eigenvalues
/// @param[in] opts Additional options
/// @ingroup heev
///
template <typename scalar_t>
void eig(
    SymmetricMatrix<scalar_t>& A,
    std::vector< blas::real_type<scalar_t> >& Lambda,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    eig_vals( A, Lambda, opts );
}

//-----------------------------------------
/// Compute eigenvalues and eigenvectors (symmetric matrix)
///
/// @param[in,out] A Symmetric matrix
/// @param[out] Lambda Eigenvalues
/// @param[out] Z Eigenvector matrix
/// @param[in] opts Additional options
/// @ingroup heev
///
template <typename scalar_t>
void eig(
    SymmetricMatrix<scalar_t>& A,
    std::vector< blas::real_type<scalar_t> >& Lambda,
    Matrix<scalar_t>& Z,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    syev( A, Lambda, Z, opts );
}

/// @}

//------------------------------------------------------------------------------
/// @defgroup hegv Generalized Hermitian eigenvalue problems
/// @brief Solve \\( A v = \\lambda B v \\) and related problems
///
/// @{

//-----------------------------------------
/// Compute eigenvalues of generalized Hermitian eigenproblem
///
/// Solves one of:
/// - itype = 1: \\( A v = \\lambda B v \\)
/// - itype = 2: \\( A B v = \\lambda v \\)
/// - itype = 3: \\( B A v = \\lambda v \\)
///
/// @param[in] itype
///     Problem type (1, 2, or 3)
///
/// @param[in,out] A
///     Hermitian matrix A
///
/// @param[in,out] B
///     Hermitian positive definite matrix B
///
/// @param[out] Lambda
///     Eigenvalues
///
/// @param[in] opts
///     Additional options
///
/// @ingroup hegv
///
template <typename scalar_t>
void eig_vals(
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
/// Compute eigenvalues of generalized problem. Alias for eig_vals.
///
/// @param[in] itype Problem type
/// @param[in,out] A Hermitian matrix A
/// @param[in,out] B Hermitian positive definite matrix B
/// @param[out] Lambda Eigenvalues
/// @param[in] opts Additional options
/// @ingroup hegv
///
template <typename scalar_t>
void eig(
    int64_t itype,
    HermitianMatrix<scalar_t>& A,
    HermitianMatrix<scalar_t>& B,
    std::vector< blas::real_type<scalar_t> >& Lambda,
    Options const& opts = Options())
{
    eig_vals( itype, A, B, Lambda, opts );
}

//-----------------------------------------
/// Compute eigenvalues and eigenvectors of generalized Hermitian problem
///
/// @param[in] itype Problem type (1, 2, or 3)
/// @param[in,out] A Hermitian matrix A
/// @param[in,out] B Hermitian positive definite matrix B
/// @param[out] Lambda Eigenvalues
/// @param[out] Z Eigenvector matrix
/// @param[in] opts Additional options
/// @ingroup hegv
///
template <typename scalar_t>
void eig(
    int64_t itype,
    HermitianMatrix<scalar_t>& A,
    HermitianMatrix<scalar_t>& B,
    std::vector< blas::real_type<scalar_t> >& Lambda,
    Matrix<scalar_t>& Z,
    Options const& opts = Options())
{
    hegv( itype, A, B, Lambda, Z, opts );
}

//-----------------------------------------
/// Compute eigenvalues of generalized symmetric problem
///
/// @param[in] itype Problem type (1, 2, or 3)
/// @param[in,out] A Symmetric matrix A
/// @param[in,out] B Symmetric positive definite matrix B
/// @param[out] Lambda Eigenvalues
/// @param[in] opts Additional options
/// @ingroup hegv
///
template <typename scalar_t>
void eig_vals(
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
/// Compute eigenvalues of generalized symmetric problem. Alias for eig_vals.
///
/// @param[in] itype Problem type
/// @param[in,out] A Symmetric matrix A
/// @param[in,out] B Symmetric positive definite matrix B
/// @param[out] Lambda Eigenvalues
/// @param[in] opts Additional options
/// @ingroup hegv
///
template <typename scalar_t>
void eig(
    int64_t itype,
    SymmetricMatrix<scalar_t>& A,
    SymmetricMatrix<scalar_t>& B,
    std::vector< blas::real_type<scalar_t> >& Lambda,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    eig_vals( itype, A, B, Lambda, opts );
}

//-----------------------------------------
/// Compute eigenvalues and eigenvectors of generalized symmetric problem
///
/// @param[in] itype Problem type (1, 2, or 3)
/// @param[in,out] A Symmetric matrix A
/// @param[in,out] B Symmetric positive definite matrix B
/// @param[out] Lambda Eigenvalues
/// @param[out] Z Eigenvector matrix
/// @param[in] opts Additional options
/// @ingroup hegv
///
template <typename scalar_t>
void eig(
    int64_t itype,
    SymmetricMatrix<scalar_t>& A,
    SymmetricMatrix<scalar_t>& B,
    std::vector< blas::real_type<scalar_t> >& Lambda,
    Matrix<scalar_t>& Z,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    sygv( itype, A, B, Lambda, Z, opts );
}

/// @}

} // namespace slate

#endif // SLATE_SIMPLIFIED_API_HH
