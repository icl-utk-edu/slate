Example 07: Linear Systems (Cholesky)
=====================================

This example demonstrates solving symmetric/Hermitian positive definite linear systems using Cholesky factorization.

Key Concepts
------------

1.  **Cholesky Solve**: Using ``slate::chol_solve`` (posv) for a one-step solution of :math:`AX=B` where :math:`A` is positive definite.
2.  **Explicit Factorization**: Separating factorization (``chol_factor``/``potrf``) and solve (``chol_solve_using_factor``/``potrs``).
3.  **Matrix Inversion**: Computing :math:`A^{-1}` using ``chol_inverse_using_factor`` (potri).
4.  **Mixed Precision**: Using iterative refinement (``posv_mixed``).
5.  **Condition Number**: Estimating the condition number of a Hermitian positive definite matrix.

C++ Example
-----------


**Cholesky Solve (Lines 38-40)**

.. code-block:: cpp

    slate::chol_solve( A, B );  // simplified API
    slate::posv( A, B );        // traditional API

Solves :math:`Ax=B` for symmetric/Hermitian positive definite `A`.

- Requires `A` to be defined as `HermitianMatrix` or `SymmetricMatrix`.
- `A` is overwritten by the Cholesky factor :math:`L` (if `Uplo::Lower`) or :math:`U` (if `Uplo::Upper`).
- `B` is overwritten by the solution.
- Cholesky is roughly twice as fast as LU factorization for applicable matrices.

**Mixed Precision (Lines 80-81)**

.. code-block:: cpp

    slate::posv_mixed( A, B, X, iters );

Similar to the LU case, this routine factors `A` in lower precision and iteratively refines the solution `X` to high precision. It requires positive definiteness.

**Explicit Factorization (Lines 106-111)**

.. code-block:: cpp

    slate::chol_factor( A );
    slate::chol_solve_using_factor( A, B );

1.  `chol_factor` (potrf): Computes :math:`A = LL^H`.
2.  `chol_solve_using_factor` (potrs): Solves using the factors.

**Inversion (Lines 134-139)**

.. code-block:: cpp

    slate::chol_factor( A );
    slate::chol_inverse_using_factor( A );

Computes :math:`A^{-1}` for a positive definite matrix.

1.  Factorize.
2.  Call `chol_inverse_using_factor` (potri). `A` is overwritten by the inverse.

**Condition Number (Lines 165-171)**

.. code-block:: cpp

    real_t A_norm = slate::norm( slate::Norm::One, A );
    slate::chol_factor( A );
    real_t rcond = slate::chol_rcondest_using_factor( slate::Norm::One, A, A_norm );

Standard condition number estimation flow: Norm -> Factor -> Estimate.

.. literalinclude:: ../../../examples/ex07_linear_system_cholesky.cc
   :language: cpp
   :linenos:
