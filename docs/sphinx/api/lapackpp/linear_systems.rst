Linear Systems
==============

Solvers for systems of linear equations: Ax = b or AX = B

LAPACK++ provides direct methods for solving linear systems through factorization approaches. All routines are templated on scalar type and support both real and complex arithmetic.

General (Non-Symmetric) Systems
--------------------------------

**gesv** - General linear system solve (simple driver)

Computes LU factorization with partial pivoting then solves Ax = b

**gesvx** - General linear system solve (expert driver)

Extended version with equilibration, condition estimates, and error bounds

**getrf** - LU factorization with partial pivoting

Computes :math:`A = PLU` where P is permutation, L is unit lower triangular, U is upper triangular

**getrs** - Solve using LU factorization

Solves :math:`A x = b`, :math:`A^T x = b`, or :math:`A^H x = b` using factorization from getrf

**getri** - Matrix inverse using LU factorization

Computes :math:`A^{-1}` using factorization from getrf

**gecon** - Condition number estimate

Estimates reciprocal condition number in 1-norm or infinity-norm

**geequ** - Row/column equilibration

Computes row and column scalings to equilibrate the matrix

Symmetric/Hermitian Positive Definite
--------------------------------------

**posv** - Positive definite solve (simple driver)

Cholesky factorization then solve

**posvx** - Positive definite solve (expert driver)

With equilibration and condition estimates

**potrf** - Cholesky factorization

Computes :math:`A = U^H U` (upper) or :math:`A = L L^H` (lower)

**potrs** - Solve using Cholesky factorization

Solves :math:`A x = b` using factorization from potrf

**potri** - Inverse using Cholesky factorization

Computes :math:`A^{-1}` using factorization from potrf

**pocon** - Condition number estimate for positive definite

**poequ** - Equilibration for positive definite

**pbtrf** - Band Cholesky factorization

For positive definite band matrices

**pbtrs** - Band triangular solve

**pbcon** - Band condition estimate

Symmetric/Hermitian Indefinite
-------------------------------

**sysv / hesv** - Symmetric/Hermitian solve (simple driver)

Bunch-Kaufman factorization (:math:`A = U D U^T` or :math:`A = L D L^T`) then solve

**sysvx / hesvx** - Expert driver with condition estimates

**sytrf / hetrf** - Bunch-Kaufman factorization

Computes :math:`A = U D U^T` or :math:`A = L D L^T` with D block diagonal

**sytrs / hetrs** - Solve using symmetric factorization

**sytri / hetri** - Inverse using symmetric factorization

**sycon / hecon** - Condition estimate for symmetric/Hermitian

**syrfs / herfs** - Iterative refinement

**sytrs2 / hetrs2** - Solve with rook pivoting

Triangular Systems
------------------

**trtrs** - Triangular solve

Solves :math:`op(A) x = b` where A is triangular

**trtri** - Triangular inverse

Computes :math:`A^{-1}` where A is triangular

**trcon** - Triangular condition estimate

Band Matrices
-------------

**gbtrf** - Band LU factorization

**gbtrs** - Band triangular solve

**gbcon** - Band condition estimate

Tridiagonal Systems
-------------------

**gtsv** - Tridiagonal solve (simple driver)

**gtsvx** - Tridiagonal solve (expert driver)

**gttrf** - Tridiagonal LU factorization

**gttrs** - Tridiagonal solve using factorization

**gtcon** - Tridiagonal condition estimate

Symmetric Positive Definite Tridiagonal
----------------------------------------

**ptsv** - SPD tridiagonal solve

**ptsvx** - SPD tridiagonal expert driver

**pttrf** - SPD tridiagonal factorization

**pttrs** - SPD tridiagonal solve

**ptcon** - SPD tridiagonal condition estimate

See Also
--------

- :doc:`least_squares` - Overdetermined/underdetermined systems
- :doc:`util` - Enumerations and error handling
