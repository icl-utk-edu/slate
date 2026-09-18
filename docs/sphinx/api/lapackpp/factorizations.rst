Matrix Factorizations
=====================

LAPACK++ provides comprehensive matrix factorization routines for decomposing matrices into products of simpler matrices with special structure.

LU Factorization
----------------

**getrf** - LU factorization with partial pivoting: :math:`A = PLU`

**getrf2** - Recursive LU with partial pivoting

**gbtrf** - LU factorization of band matrix

**getri** - Compute matrix inverse from LU factorization

**getrs** - Solve system using LU factorization

Cholesky Factorization
-----------------------

**potrf** - Cholesky factorization: :math:`A = LL^H` or :math:`A = U^HU`

**potrf2** - Recursive Cholesky factorization

**pbtrf** - Cholesky factorization of band matrix

**pftrf** - Cholesky factorization in RFP format

**potri** - Compute inverse from Cholesky factorization

**potrs** - Solve system using Cholesky factorization

**pstrf** - Cholesky with pivoting for semidefinite matrices

QR Factorization
----------------

**geqrf** - QR factorization: :math:`A = QR`

**geqp3** - QR with column pivoting: :math:`AP = QR`

**geqrt** - QR using recursive algorithm

**orgqr** - Generate orthogonal Q from geqrf

**ormqr** - Multiply by orthogonal Q from geqrf

**tpqrt** - QR of triangular-pentagonal matrix

LQ Factorization
----------------

**gelqf** - LQ factorization: :math:`A = LQ`

**gelqt** - LQ using recursive algorithm

**orglq** - Generate orthogonal Q from gelqf

**ormlq** - Multiply by orthogonal Q from gelqf

QL Factorization
----------------

**geqlf** - QL factorization

**orgql** - Generate orthogonal Q from geqlf

**ormql** - Multiply by orthogonal Q from geqlf

RQ Factorization
----------------

**gerqf** - RQ factorization

**orgrq** - Generate orthogonal Q from gerqf

**ormrq** - Multiply by orthogonal Q from gerqf

Symmetric/Hermitian Factorizations
-----------------------------------

**sytrf**, **hetrf** - Bunch-Kaufman factorization: :math:`A = UDU^T` or :math:`LDL^T`

**sytri**, **hetri** - Compute inverse from Bunch-Kaufman

**sytrs**, **hetrs** - Solve using Bunch-Kaufman

**sptrf**, **hptrf** - Bunch-Kaufman for packed storage

Triangular Factorizations
--------------------------

**trtri** - Compute inverse of triangular matrix

**trtrs** - Solve triangular system

Bidiagonal Reduction
--------------------

**gebrd** - Reduce general matrix to bidiagonal form

**gbbrd** - Reduce band matrix to bidiagonal

**orgbr** - Generate orthogonal matrices from gebrd

**ormbr** - Multiply by orthogonal matrices from gebrd

Tridiagonal Reduction
---------------------

**sytrd**, **hetrd** - Reduce symmetric/Hermitian to tridiagonal

**sbtrd**, **hbtrd** - Reduce banded to tridiagonal

**orgtr**, **ungtr** - Generate orthogonal matrix from ∗ytrd

**ormtr**, **unmtr** - Multiply by orthogonal matrix from ∗ytrd

Hessenberg Reduction
--------------------

**gehrd** - Reduce general matrix to upper Hessenberg form

**orghr**, **unghr** - Generate orthogonal matrix from gehrd

**ormhr**, **unmhr** - Multiply by orthogonal matrix from gehrd

Example Usage
-------------

LU factorization and solve:

.. code-block:: cpp

   int64_t n = 100, nrhs = 5;
   std::vector<double> A(n * n);
   std::vector<double> B(n * nrhs);
   std::vector<int64_t> ipiv(n);
   
   // Factor: A = P*L*U
   int64_t info = lapack::getrf(n, n, A.data(), n, ipiv.data());
   
   if (info == 0) {
       // Solve A*X = B using factors
       info = lapack::getrs(lapack::Op::NoTrans, n, nrhs,
                            A.data(), n, ipiv.data(), 
                            B.data(), n);
       // Solution in B
   }

Cholesky factorization:

.. code-block:: cpp

   int64_t n = 100;
   std::vector<double> A(n * n);  // Positive definite
   
   // Factor: A = L*L^T (lower triangle)
   int64_t info = lapack::potrf(lapack::Uplo::Lower, n, 
                                 A.data(), n);
   
   if (info == 0) {
       // L stored in lower triangle of A
       // Can now use potrs for solving, or potri for inverse
   } else if (info > 0) {
       // A(info,info) not positive definite
   }

QR factorization:

.. code-block:: cpp

   int64_t m = 200, n = 100;
   std::vector<double> A(m * n);
   std::vector<double> tau(std::min(m, n));
   
   // Factor: A = Q*R
   int64_t info = lapack::geqrf(m, n, A.data(), m, tau.data());
   
   if (info == 0) {
       // R stored in upper triangle of A
       // Elementary reflectors stored below diagonal
       // tau contains scalar factors
       
       // Generate explicit Q matrix
       std::vector<double> Q = A;  // Copy
       lapack::orgqr(m, m, n, Q.data(), m, tau.data());
       // Q now contains explicit orthogonal matrix
   }

QR with column pivoting (rank-revealing):

.. code-block:: cpp

   int64_t m = 200, n = 100;
   std::vector<double> A(m * n);
   std::vector<double> tau(std::min(m, n));
   std::vector<int64_t> jpvt(n, 0);  // Initialize to 0
   
   // Factor: A*P = Q*R with column pivoting
   int64_t info = lapack::geqp3(m, n, A.data(), m, 
                                 jpvt.data(), tau.data());
   
   if (info == 0) {
       // jpvt[j] indicates A(:,jpvt[j]) is j-th column of Q*R
       // Can estimate rank by examining diagonal of R
       
       double tol = 1e-10;
       int64_t rank = 0;
       for (int64_t i = 0; i < std::min(m, n); ++i) {
           if (std::abs(A[i + i*m]) > tol) {
               rank++;
           }
       }
   }

Symmetric indefinite factorization (Bunch-Kaufman):

.. code-block:: cpp

   int64_t n = 100;
   std::vector<double> A(n * n);  // Symmetric indefinite
   std::vector<int64_t> ipiv(n);
   
   // Factor: A = U*D*U^T or L*D*L^T
   int64_t info = lapack::sytrf(lapack::Uplo::Lower, n,
                                 A.data(), n, ipiv.data());
   
   if (info == 0) {
       // Factors stored in A and ipiv
       // Can use sytrs to solve systems
   }

Reduce to bidiagonal (for SVD):

.. code-block:: cpp

   int64_t m = 200, n = 100;
   std::vector<double> A(m * n);
   std::vector<double> d(std::min(m, n));      // Diagonal
   std::vector<double> e(std::min(m, n) - 1);  // Off-diagonal
   std::vector<double> tauq(std::min(m, n));
   std::vector<double> taup(std::min(m, n));
   
   // Reduce A to bidiagonal form: A = U*B*V^T
   int64_t info = lapack::gebrd(m, n, A.data(), m,
                                 d.data(), e.data(),
                                 tauq.data(), taup.data());
   
   if (info == 0) {
       // B has diagonal d and superdiagonal e
       // Can generate U and V with orgbr/ormbr
   }

Reduce symmetric to tridiagonal (for eigenvalues):

.. code-block:: cpp

   int64_t n = 100;
   std::vector<double> A(n * n);  // Symmetric
   std::vector<double> d(n);      // Diagonal
   std::vector<double> e(n - 1);  // Off-diagonal
   std::vector<double> tau(n - 1);
   
   // Reduce A to tridiagonal: A = Q*T*Q^T
   int64_t info = lapack::sytrd(lapack::Uplo::Lower, n,
                                 A.data(), n, 
                                 d.data(), e.data(), tau.data());
   
   if (info == 0) {
       // Tridiagonal T has diagonal d and off-diagonal e
       // Generate orthogonal Q with orgtr if needed
   }

Reduce to Hessenberg (for eigenvalues):

.. code-block:: cpp

   int64_t n = 100;
   std::vector<double> A(n * n);
   std::vector<double> tau(n - 1);
   
   // Reduce A to Hessenberg: A = Q*H*Q^T
   int64_t info = lapack::gehrd(n, 1, n, A.data(), n, tau.data());
   
   if (info == 0) {
       // Upper Hessenberg H stored in A
       // Generate Q with orghr if needed
   }

Factorization Algorithms
-------------------------

**LU (getrf):**

- Uses partial (row) pivoting for stability
- Complexity: O(n³)
- Backward stable: :math:`(L U - P A) / ||A|| = O(\epsilon)`
- Growth factor can be large but rarely occurs in practice

**Cholesky (potrf):**

- No pivoting needed (matrix positive definite)
- ~2x faster than LU
- Complexity: O(n³/3)
- Backward stable for positive definite matrices
- Fails if matrix not positive definite

**QR (geqrf):**

- Uses Householder reflectors
- Complexity: O(2mn² - 2n³/3) for m >= n
- Excellent numerical stability
- Minimum-norm solution for rank-deficient systems

**QR with pivoting (geqp3):**

- Reveals numerical rank
- Can handle rank-deficiency
- Slightly slower than geqrf
- Essential for rank-revealing decompositions

**Bunch-Kaufman (sytrf/hetrf):**

- For symmetric/Hermitian indefinite matrices
- Uses 1x1 and 2x2 pivots
- More complex than Cholesky
- Required when matrix not positive definite

Storage Efficiency
------------------

**Packed storage:** sptrf, hptrf, pptrf save ~50% memory for triangular/symmetric

**RFP format:** pftrf uses rectangular full packed format (cache-friendly)

**Band storage:** gbtrf, pbtrf exploit band structure

**In-place:** Most factorizations overwrite input matrix

Performance Characteristics
---------------------------

Relative costs for n = 1000:

- **Cholesky (potrf)**: 1.0x (fastest for SPD)
- **LU (getrf)**: 1.5x
- **QR (geqrf)**: 2.0x  
- **Bunch-Kaufman (sytrf)**: 2.5x
- **QR with pivoting (geqp3)**: 3.0x

Blocking and Level-3 BLAS
--------------------------

All factorizations use blocked algorithms calling Level-3 BLAS for efficiency:

- Panel factorization (Level-2)
- Trailing matrix update (Level-3 gemm, syrk, etc.)
- Block size typically 32-256 depending on architecture

Choosing a Factorization
-------------------------

**For linear systems:**

- Positive definite → Cholesky (potrf)
- General → LU (getrf)
- Symmetric indefinite → Bunch-Kaufman (sytrf)
- Overdetermined → QR (geqrf) or SVD
- Underdetermined → LQ (gelqf) or SVD

**For least squares:**

- Full rank → QR (geqrf + ormqr)
- Rank-deficient → QR with pivoting (geqp3) or SVD

**For eigenvalue problems:**

- Symmetric/Hermitian → Tridiagonal (sytrd) → syev
- General → Hessenberg (gehrd) → geev

**For SVD:**

- Bidiagonal (gebrd) → gesvd/gesdd

See Also
--------

- :doc:`linear_systems` - Using factorizations to solve systems
- :doc:`eigenvalues` - Using reductions for eigenproblems  
- :doc:`svd` - Using bidiagonal reduction for SVD
