Auxiliary Functions
===================

Helper routines for matrix operations, norms, and transformations.

Matrix Norms
------------

**lange** - General matrix norm

Computes matrix norms:
- One: Maximum absolute column sum :math:`\max_j \sum_i |a_{ij}|`
- Inf: Maximum absolute row sum :math:`\max_i \sum_j |a_{ij}|`
- Fro: Frobenius norm :math:`\sqrt{\sum_{i,j} |a_{ij}|^2}`
- Max: Maximum absolute element :math:`\max_{i,j} |a_{ij}|`

**lansy / lanhe** - Symmetric/Hermitian matrix norm

**lantr** - Triangular matrix norm

**lansb / lanhb** - Symmetric/Hermitian band matrix norm

**langt** - Tridiagonal matrix norm

**lanhs** - Hessenberg matrix norm

**lanst** - Symmetric tridiagonal norm

**lanhp / lansp** - Packed symmetric/Hermitian norm

Condition Number Estimation
----------------------------

**gecon** - General matrix condition number

Estimates :math:`\kappa_1(A)` or :math:`\kappa_\infty(A)` using LU factorization.

**pocon** - Positive definite condition number

Uses Cholesky factorization.

**sycon / hecon** - Symmetric/Hermitian condition number

Uses Bunch-Kaufman factorization.

**trcon** - Triangular condition number

**pbcon** - Band positive definite condition number

**gbcon** - General band condition number

**gtcon** - Tridiagonal condition number

**ptcon** - Symmetric positive definite tridiagonal

Equilibration and Scaling
--------------------------

**geequ** - General matrix equilibration

Computes row and column scaling factors to equilibrate matrix.

**poequ** - Positive definite equilibration

**syequb / heequb** - Symmetric/Hermitian equilibration with Bunch-Kaufman

**gbequ** - General band equilibration

**pbequ** - Positive definite band equilibration

**lascl** - Scale matrix by scalar

Multiplies matrix by :math:`cfrom/cto` with overflow protection.

Matrix Initialization and Copying
----------------------------------

**laset** - Initialize matrix

Sets matrix elements to constants:
- Diagonal to β
- Off-diagonal to α

**lacpy** - Copy matrix or submatrix

Copies selected triangle or full matrix.

**laswp** - Row permutation

Applies row interchanges (from pivoting).

Matrix Addition
---------------

**geadd** - General matrix addition

Computes :math:`B = \alpha A + \beta B`.

Householder Transformations
----------------------------

**larfg** - Generate Householder reflector

Generates elementary reflector H such that:
:math:`H \begin{bmatrix} \alpha \\ x \end{bmatrix} = \begin{bmatrix} \beta \\ 0 \end{bmatrix}`

**larf** - Apply Householder reflector

Applies :math:`H` to matrix from left or right.

**larfb** - Apply block of Householder reflectors

More efficient than repeated larf calls.

**larft** - Form triangular factor of block reflector

Computes T for efficient block application.

**larfx** - Apply optimized small reflector

Specialized for small vectors (≤ 10 elements).

Givens Rotations
----------------

**lartg** - Generate Givens rotation

Computes rotation that zeros second element:
:math:`\begin{bmatrix} c & s \\ -\bar{s} & c \end{bmatrix} \begin{bmatrix} f \\ g \end{bmatrix} = \begin{bmatrix} r \\ 0 \end{bmatrix}`

**lartgp** - Generate Givens with positive diagonal

**lartgs** - Generate Givens for SVD

**lasr** - Apply sequence of Givens rotations

**rot** - Apply single Givens rotation (BLAS-like)

Matrix Multiplication (Triangular)
-----------------------------------

**lauum** - U^H U or L L^H multiplication

Computes :math:`U^H U` or :math:`L L^H` for triangular matrix.
Used in computing matrix inverse from Cholesky factorization.

Special Matrix Operations
--------------------------

**lagge** - Generate random general matrix

**lagsy / laghe** - Generate random symmetric/Hermitian

**lagtr** - Random triangular matrix

**latms** - Generate test matrices with specified properties

**lapy2** - :math:`\sqrt{x^2 + y^2}` with overflow protection

**lapy3** - :math:`\sqrt{x^2 + y^2 + z^2}` with overflow protection

**lamch** - Machine parameters (epsilon, safe minimum, etc.)

**lamc3** - Force storage of a + b (compiler optimization barrier)

**ladiv** - Complex division with overflow protection

Permutations and Sorting
-------------------------

**lapmt** - Column permutation

**lapmr** - Row permutation

**lasrt** - Sort numbers

**lasorte** - Sort eigenvalues with eigenvectors

Workspace Queries
-----------------

Most LAPACK routines support workspace queries:
- Call with `lwork = -1` or `lrwork = -1`
- Optimal workspace size returned in first element of work array
- Enables dynamic workspace allocation

Example:

.. code-block:: cpp

    // Query optimal workspace
    int64_t lwork = -1;
    double work_query;
    lapack::geqrf(m, n, A, lda, tau, &work_query, lwork);
    
    lwork = (int64_t)work_query;
    std::vector<double> work(lwork);
    
    // Actual computation
    lapack::geqrf(m, n, A, lda, tau, work.data(), lwork);

Error Checking and Diagnostics
-------------------------------

**ilaenv** - Environment parameters

Returns algorithm-specific parameters (block sizes, etc.).

**xerbla** - Error handler

Called when invalid parameter detected (not typically called directly by users).

See Also
--------

- :doc:`util` - Error handling and enumerations
- :doc:`flops` - Performance counting
