Example 06: Linear Systems (LU)
===============================

This example demonstrates solving linear systems :math:`Ax=B` using LU factorization.

Key Concepts
------------

1.  **Simple Solve**: Using ``slate::lu_solve`` (gesv) for a one-step solution.
2.  **Explicit Factorization**: Separating factorization (``lu_factor``/``getrf``) and solve (``lu_solve_using_factor``/``getrs``).
3.  **Matrix Inversion**: Computing :math:`A^{-1}` using ``lu_inverse_using_factor`` (getri).
4.  **Mixed Precision**: Using iterative refinement to solve systems with lower-precision factorization.
5.  **Condition Number**: Estimating the condition number of the matrix.

C++ Example
-----------


**Standard LU Solve (Lines 38-41)**

.. code-block:: cpp

    slate::lu_solve( A, B );        // simplified API
    slate::gesv( A, pivots, B );    // traditional API

The simplest way to solve :math:`Ax=B`.

- `A` is overwritten by its LU factors.
- `B` is overwritten by the solution :math:`X`.
- `pivots` (in the traditional API) stores the pivot indices found during factorization. `lu_solve` manages this internally if you don't need the pivots later.

**Mixed Precision Iterative Refinement (Lines 82-83)**

.. code-block:: cpp

    slate::gesv_mixed( A, pivots, B, X, iters );

Mixed precision solvers can provide a significant speedup by doing the expensive factorization in lower precision (e.g., float) and then refining the solution to high precision (e.g., double) using the original matrix.

- `A`, `B`, `X` are high precision (e.g., double).
- The internal factorization happens in low precision (e.g., float).
- `iters` returns the number of refinement iterations performed.

**Explicit Factorization and Solve (Lines 113-118)**

.. code-block:: cpp

    slate::lu_factor( A, pivots );
    slate::lu_solve_using_factor( A, pivots, B );

Sometimes you need to solve for multiple right-hand sides that arrive at different times, or you want to reuse the factors.

1.  `lu_factor` (getrf): Computes :math:`PA = LU`.
2.  `lu_solve_using_factor` (getrs): Solves :math:`Ax=B` using the pre-computed factors and pivots.

**Matrix Inversion (Lines 142-147)**

.. code-block:: cpp

    slate::lu_factor( A, pivots );
    slate::lu_inverse_using_factor( A, pivots );

Computes the inverse of a matrix in-place.

1.  Factorize the matrix.
2.  Call `lu_inverse_using_factor` (getri). `A` is overwritten by :math:`A^{-1}`.

**Condition Number Estimation (Lines 173-179)**

.. code-block:: cpp

    real_t A_norm = slate::norm( slate::Norm::One, A );
    slate::lu_factor( A, pivots );
    real_t rcond = slate::lu_rcondest_using_factor( slate::Norm::One, A, A_norm );

Estimates the reciprocal condition number :math:`1/\kappa(A)`.

1.  Compute the norm of the original matrix **before** factorization.
2.  Factorize the matrix.
3.  Call `lu_rcondest_using_factor`. This estimates :math:`\|A^{-1}\|` cheaply using the factors and combines it with the provided :math:`\|A\|`.


.. literalinclude:: ../../../examples/ex06_linear_system_lu.cc
   :language: cpp
   :linenos:


C API Example
-------------

.. literalinclude:: ../../../examples/c_api/ex06_linear_system_lu.c
   :language: c
   :linenos:
