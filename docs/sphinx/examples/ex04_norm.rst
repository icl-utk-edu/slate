Example 04: Matrix Norms
========================

This example demonstrates how to compute various matrix norms.

Key Concepts
------------

1.  **Norm Types**:
    *   ``Norm::One``: Maximum column sum.
    *   ``Norm::Inf``: Maximum row sum.
    *   ``Norm::Max``: Maximum absolute element.
    *   ``Norm::Fro``: Frobenius norm (square root of sum of squares).
2.  **Polymorphism**: The ``slate::norm`` function works for all matrix types (General, Symmetric, Hermitian, Triangular, Trapezoid).

C++ Example
-----------

**General Matrix Norms (Lines 27-41)**

.. code-block:: cpp

    slate::Matrix<scalar_type> A( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    // ... insert tiles and fill data ...

    real_type A_norm_one = slate::norm( slate::Norm::One, A );
    real_type A_norm_inf = slate::norm( slate::Norm::Inf, A );
    // ...

Computes norms for a standard general matrix `A`.

- `Norm::One` (1-norm): Maximum absolute column sum. :math:`\max_j \sum_i |a_{ij}|`.
- `Norm::Inf` (Infinity-norm): Maximum absolute row sum. :math:`\max_i \sum_j |a_{ij}|`.
- `Norm::Max` (Max-norm): Maximum absolute value of any element. :math:`\max_{i,j} |a_{ij}|`.
- `Norm::Fro` (Frobenius-norm): Square root of the sum of squares of all elements. :math:`\sqrt{\sum_{i,j} |a_{ij}|^2}`.

**Symmetric Matrix Norms (Lines 59-73)**

.. code-block:: cpp

    slate::SymmetricMatrix<scalar_type> S( ... );
    // ...
    real_type S_norm_one = slate::norm( slate::Norm::One, S );

`slate::norm` automatically handles symmetric matrices. It understands that elements in the unreferenced triangle are equal to their symmetric counterparts. For a symmetric matrix, the 1-norm and Infinity-norm are equal.

**Hermitian Matrix Norms (Lines 88-96)**

Similar to symmetric matrices, but for Hermitian matrices (where symmetric elements are complex conjugates). The logic for norms accounts for the magnitudes properly.

**Triangular and Trapezoid Norms (Lines 111-143)**

.. code-block:: cpp

    slate::TriangularMatrix<scalar_type> T( ... );
    real_type T_norm_one = slate::norm( slate::Norm::One, T );

For triangular (and trapezoid) matrices, `slate::norm` respects the structure. Elements in the empty part of the matrix (e.g., upper triangle for a Lower triangular matrix) are treated as zero and do not contribute to the norm.

.. literalinclude:: ../../../examples/ex04_norm.cc
   :language: cpp
   :linenos:
