SLATE Operations
================

This chapter describes the mathematical operations available in SLATE, organized by category.


Execution Options
-----------------

All SLATE routines accept an optional ``Options`` argument to control execution:

.. code-block:: cpp

    slate::Options opts = {
        {slate::Option::Target, slate::Target::Devices},
        {slate::Option::Lookahead, 2},
    };
    slate::multiply(alpha, A, B, beta, C, opts);


Common Options
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Option
     - Description
     - Default
   * - ``Target``
     - Execution target: ``HostTask``, ``HostNest``, ``HostBatch``, ``Devices``
     - ``HostTask``
   * - ``Lookahead``
     - Panels to process ahead of trailing matrix
     - 1
   * - ``InnerBlocking``
     - Inner blocking size for panels
     - 16
   * - ``MaxPanelThreads``
     - Maximum threads for panel operations
     - OMP_NUM_THREADS/2
   * - ``PivotThreshold``
     - Threshold for LU pivoting (0 to 1)
     - 1.0
   * - ``MethodLU``
     - LU pivoting: ``PartialPiv``, ``CALU``, ``NoPiv``
     - ``PartialPiv``


Matrix Norms
------------

SLATE computes matrix norms for all matrix types:

.. code-block:: cpp

    slate::Matrix<double> A(m, n, nb, p, q, MPI_COMM_WORLD);
    
    double norm_one = slate::norm(slate::Norm::One, A);  // ||A||_1
    double norm_inf = slate::norm(slate::Norm::Inf, A);  // ||A||_∞
    double norm_fro = slate::norm(slate::Norm::Fro, A);  // ||A||_F
    double norm_max = slate::norm(slate::Norm::Max, A);  // max|a_ij|

Works with: ``Matrix``, ``SymmetricMatrix``, ``HermitianMatrix``, ``TriangularMatrix``, band matrices.


Level 3 BLAS Operations
-----------------------

Matrix Multiply (gemm)
~~~~~~~~~~~~~~~~~~~~~~

General matrix-matrix multiplication :math:`C = \alpha A B + \beta C`:

.. code-block:: cpp

    // Simplified API
    slate::multiply(alpha, A, B, beta, C);
    
    // Traditional API
    slate::gemm(alpha, A, B, beta, C);
    
    // With transposed matrices
    auto AT = slate::transpose(A);
    auto BH = slate::conj_transpose(B);
    slate::multiply(alpha, AT, BH, beta, C);


Hermitian/Symmetric Matrix Multiply (hemm/symm)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When A is Hermitian or symmetric:

.. code-block:: cpp

    // C = alpha * A * B + beta * C (A on left)
    slate::multiply(alpha, A_hermitian, B, beta, C);
    slate::hemm(slate::Side::Left, alpha, A_hermitian, B, beta, C);
    
    // C = alpha * B * A + beta * C (A on right)
    slate::multiply(alpha, B, A_hermitian, beta, C);
    slate::hemm(slate::Side::Right, alpha, A_hermitian, B, beta, C);


Rank-k Update (herk/syrk)
~~~~~~~~~~~~~~~~~~~~~~~~~

Hermitian/symmetric rank-k update :math:`C = \alpha A A^H + \beta C`:

.. code-block:: cpp

    // Hermitian: C = alpha * A * A^H + beta * C
    slate::rank_k_update(alpha, A, beta, C_hermitian);
    slate::herk(alpha, A, beta, C_hermitian);
    
    // Symmetric: C = alpha * A * A^T + beta * C
    slate::rank_k_update(alpha, A, beta, C_symmetric);
    slate::syrk(alpha, A, beta, C_symmetric);


Rank-2k Update (her2k/syr2k)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:`C = \alpha A B^H + \bar{\alpha} B A^H + \beta C`:

.. code-block:: cpp

    // Hermitian
    slate::rank_2k_update(alpha, A, B, beta, C_hermitian);
    slate::her2k(alpha, A, B, beta, C_hermitian);
    
    // Symmetric: C = alpha * A * B^T + alpha * B * A^T + beta * C
    slate::rank_2k_update(alpha, A, B, beta, C_symmetric);
    slate::syr2k(alpha, A, B, beta, C_symmetric);


Triangular Multiply (trmm)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Multiply by triangular matrix:

.. code-block:: cpp

    // B = alpha * A * B (A on left)
    slate::triangular_multiply(alpha, A_triangular, B);
    slate::trmm(slate::Side::Left, alpha, A_triangular, B);
    
    // B = alpha * B * A (A on right)
    slate::triangular_multiply(alpha, B, A_triangular);
    slate::trmm(slate::Side::Right, alpha, A_triangular, B);


Triangular Solve (trsm)
~~~~~~~~~~~~~~~~~~~~~~~

Solve :math:`AX = \alpha B`:

.. code-block:: cpp

    // Solve A * X = alpha * B (A on left)
    slate::triangular_solve(alpha, A_triangular, B);
    slate::trsm(slate::Side::Left, alpha, A_triangular, B);
    
    // Solve X * A = alpha * B (A on right)
    slate::triangular_solve(alpha, B, A_triangular);
    slate::trsm(slate::Side::Right, alpha, A_triangular, B);


Linear Systems
--------------

LU Factorization (General Matrices)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solve :math:`AX = B` for general non-symmetric matrices:

.. code-block:: cpp

    slate::Matrix<double> A(n, n, nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<double> B(n, nrhs, nb, p, q, MPI_COMM_WORLD);
    
    // Driver: Factor and solve
    slate::lu_solve(A, B);
    slate::gesv(A, pivots, B);  // traditional
    
    // Computational: Factor only
    slate::Pivots pivots;
    slate::lu_factor(A, pivots);
    slate::getrf(A, pivots);  // traditional
    
    // Solve with existing factorization
    slate::lu_solve_using_factor(A, pivots, B);
    slate::getrs(A, pivots, B);  // traditional
    
    // Compute inverse
    slate::lu_inverse_using_factor(A, pivots);
    slate::getri(A, pivots);  // traditional


LU Pivoting Options
~~~~~~~~~~~~~~~~~~~

Control pivoting behavior:

.. code-block:: cpp

    slate::Options opts = {
        // Pivoting method
        {slate::Option::MethodLU, slate::MethodLU::PartialPiv},  // default
        // {slate::Option::MethodLU, slate::MethodLU::CALU},      // tournament
        // {slate::Option::MethodLU, slate::MethodLU::NoPiv},     // no pivoting
        
        // Threshold for partial pivoting (0 to 1)
        {slate::Option::PivotThreshold, 0.5},  // reduce row exchanges
    };
    slate::lu_solve(A, B, opts);


Cholesky Factorization (Positive Definite)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solve :math:`AX = B` for Hermitian/symmetric positive definite matrices:

.. code-block:: cpp

    slate::HermitianMatrix<double> A(slate::Uplo::Lower, n, nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<double> B(n, nrhs, nb, p, q, MPI_COMM_WORLD);
    
    // Driver: Factor A = L*L^H and solve
    slate::chol_solve(A, B);
    slate::posv(A, B);  // traditional
    
    // Computational: Factor only
    slate::chol_factor(A);
    slate::potrf(A);  // traditional
    
    // Solve with existing factorization
    slate::chol_solve_using_factor(A, B);
    slate::potrs(A, B);  // traditional
    
    // Compute inverse
    slate::chol_inverse_using_factor(A);
    slate::potri(A);  // traditional


Indefinite Solve (Aasen's Algorithm)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solve :math:`AX = B` for Hermitian/symmetric indefinite matrices using Aasen's algorithm:

.. code-block:: cpp

    slate::HermitianMatrix<double> A(slate::Uplo::Lower, n, nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<double> B(n, nrhs, nb, p, q, MPI_COMM_WORLD);
    
    // Driver: Factor A = L*T*L^H and solve
    slate::indefinite_solve(A, B);
    
    // Traditional API with workspaces
    slate::Matrix<double> H(n, n, nb, p, q, MPI_COMM_WORLD);
    slate::BandMatrix<double> T(n, n, nb, nb, nb, p, q, MPI_COMM_WORLD);
    slate::Pivots pivots, pivots2;
    slate::hesv(A, pivots, T, pivots2, H, B);


Band Matrix Solve
~~~~~~~~~~~~~~~~~

.. code-block:: cpp

    slate::BandMatrix<double> Ab(m, n, kl, ku, nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<double> B(n, nrhs, nb, p, q, MPI_COMM_WORLD);
    
    slate::lu_solve(Ab, B);
    
    // Traditional
    slate::Pivots pivots;
    slate::gbsv(Ab, pivots, B);


Mixed-Precision Iterative Refinement
------------------------------------

Factor in lower precision, refine in higher precision:

.. code-block:: cpp

    slate::Matrix<double> A(n, n, nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<double> B(n, nrhs, nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<double> X(n, nrhs, nb, p, q, MPI_COMM_WORLD);
    int iters = 0;
    
    // LU with iterative refinement
    slate::gesv_mixed(A, pivots, B, X, iters);
    
    // LU with GMRES refinement (single RHS)
    slate::Matrix<double> B1(n, 1, nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<double> X1(n, 1, nb, p, q, MPI_COMM_WORLD);
    slate::gesv_mixed_gmres(A, pivots, B1, X1, iters);
    
    // Cholesky variants
    slate::posv_mixed(A_hermitian, B, X, iters);
    slate::posv_mixed_gmres(A_hermitian, B1, X1, iters);


Least Squares
-------------

Solve overdetermined (:math:`m \geq n`) or underdetermined (:math:`m < n`) systems:

.. code-block:: cpp

    int64_t max_mn = std::max(m, n);
    slate::Matrix<double> A(m, n, nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<double> BX(max_mn, nrhs, nb, p, q, MPI_COMM_WORLD);
    
    // Overdetermined (m >= n): minimize ||AX - B||_2
    auto B = BX;                              // Input: m rows
    auto X = BX.slice(0, n-1, 0, nrhs-1);     // Output: n rows
    slate::least_squares_solve(A, BX);
    
    // Underdetermined (m < n): minimize ||X||_2 subject to AX = B
    auto AH = slate::conj_transpose(A);
    auto B_under = BX.slice(0, n-1, 0, nrhs-1);
    auto X_under = BX;
    slate::least_squares_solve(AH, BX);


QR and LQ Factorizations
------------------------

.. code-block:: cpp

    slate::Matrix<double> A(m, n, nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<double> C(m, k, nb, p, q, MPI_COMM_WORLD);
    slate::TriangularFactors<double> T;
    
    // QR: A = QR
    slate::qr_factor(A, T);
    slate::geqrf(A, T);  // traditional
    
    // Multiply by Q: C = Q*C or C = Q^H*C
    slate::qr_multiply_by_q(slate::Side::Left, slate::Op::NoTrans, A, T, C);
    slate::gemqr(slate::Side::Left, slate::Op::NoTrans, A, T, C);
    
    // LQ: A = LQ
    slate::lq_factor(A, T);
    slate::gelqf(A, T);  // traditional
    
    // Multiply by Q from LQ
    slate::lq_multiply_by_q(slate::Side::Right, slate::Op::NoTrans, A, T, C);
    slate::gemlq(slate::Side::Right, slate::Op::NoTrans, A, T, C);


Eigenvalue Problems
-------------------

Hermitian/Symmetric Eigenvalues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute eigenvalues and optionally eigenvectors:

.. code-block:: cpp

    slate::HermitianMatrix<double> A(slate::Uplo::Lower, n, nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<double> Z(n, n, nb, p, q, MPI_COMM_WORLD);
    std::vector<double> Lambda(n);
    
    // Eigenvalues only
    slate::eig_vals(A, Lambda);
    slate::eig(A, Lambda);
    slate::heev(A, Lambda);  // traditional
    
    // Eigenvalues and eigenvectors: A = Z * Lambda * Z^H
    slate::eig(A, Lambda, Z);
    slate::heev(A, Lambda, Z);  // traditional


Eigenvalue Methods
~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

    slate::Options opts = {
        // QR iteration (default for values only)
        {slate::Option::MethodEig, slate::MethodEig::QR},
        // Divide and conquer (default for vectors)
        {slate::Option::MethodEig, slate::MethodEig::DC},
    };
    slate::eig(A, Lambda, Z, opts);


Generalized Eigenvalue Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solve :math:`Ax = \lambda Bx` for positive definite B:

.. code-block:: cpp

    slate::HermitianMatrix<double> A(slate::Uplo::Lower, n, nb, p, q, MPI_COMM_WORLD);
    slate::HermitianMatrix<double> B(slate::Uplo::Lower, n, nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<double> Z(n, n, nb, p, q, MPI_COMM_WORLD);
    std::vector<double> Lambda(n);
    
    // Type 1: A*x = lambda*B*x
    slate::eig(1, A, B, Lambda, Z);
    slate::hegv(1, A, B, Lambda, Z);
    
    // Type 2: A*B*x = lambda*x
    slate::eig(2, A, B, Lambda, Z);
    
    // Type 3: B*A*x = lambda*x
    slate::eig(3, A, B, Lambda, Z);


Singular Value Decomposition
----------------------------

Compute :math:`A = U \Sigma V^H`:

.. code-block:: cpp

    int64_t min_mn = std::min(m, n);
    slate::Matrix<double> A(m, n, nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<double> U(m, min_mn, nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<double> VH(min_mn, n, nb, p, q, MPI_COMM_WORLD);
    std::vector<double> Sigma(min_mn);
    
    // Singular values only
    slate::svd_vals(A, Sigma);
    slate::svd(A, Sigma);
    
    // With singular vectors
    slate::svd(A, Sigma, U, VH);
    
    // Only U vectors
    slate::Matrix<double> Vempty;
    slate::svd(A, Sigma, U, Vempty);
    
    // Only V^H vectors
    slate::Matrix<double> Uempty;
    slate::svd(A, Sigma, Uempty, VH);


Auxiliary Operations
--------------------

Matrix Add
~~~~~~~~~~

.. code-block:: cpp

    // B = alpha*A + beta*B
    slate::add(alpha, A, beta, B);


Matrix Copy
~~~~~~~~~~~

.. code-block:: cpp

    // B = A
    slate::copy(A, B);
    
    // With precision conversion
    slate::Matrix<double> A_double(...);
    slate::Matrix<float> A_float(...);
    slate::copy(A_double, A_float);  // double -> float


Matrix Set
~~~~~~~~~~

.. code-block:: cpp

    // Set off-diagonal to alpha, diagonal to beta
    slate::set(alpha, beta, A);


Matrix Scale
~~~~~~~~~~~~

.. code-block:: cpp

    // A = alpha * A
    slate::scale(alpha, A);
    
    // Row and column scaling (equilibration)
    // A = diag(R) * A * diag(C)
    slate::scale_row_col(equed, R, C, A);


Condition Number Estimate
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

    // After factorization
    double A_norm = slate::norm(slate::Norm::One, A);
    double rcond = slate::lu_condest_using_factor(slate::Norm::One, A, A_norm);
    // kappa = 1 / rcond


Matrix Print
~~~~~~~~~~~~

.. code-block:: cpp

    slate::Options opts = {
        {slate::Option::PrintWidth, 10},
        {slate::Option::PrintPrecision, 4},
        {slate::Option::PrintVerbose, 2},  // corner elements
    };
    slate::print("A", A, opts);


Naming Conventions
------------------

SLATE provides two naming schemes:

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - Simplified API
     - Traditional API
     - Operation
   * - ``multiply``
     - ``gemm``
     - General matrix multiply
   * - ``lu_solve``
     - ``gesv``
     - LU solve
   * - ``lu_factor``
     - ``getrf``
     - LU factorization
   * - ``chol_solve``
     - ``posv``
     - Cholesky solve
   * - ``chol_factor``
     - ``potrf``
     - Cholesky factorization
   * - ``triangular_solve``
     - ``trsm``
     - Triangular solve
   * - ``least_squares_solve``
     - ``gels``
     - Least squares
   * - ``eig``
     - ``heev``/``syev``
     - Eigenvalues
   * - ``svd``
     - ``gesvd``
     - SVD

The simplified API uses descriptive names and automatically selects the appropriate variant 
based on matrix types.
