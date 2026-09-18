Example 02: Matrix Type Conversion
==================================

This example demonstrates how to create different matrix views (Trapezoid, Triangular, Symmetric, Hermitian) from a general Matrix.

Key Concepts
------------

1.  **Shallow Copies**: Creating views of existing data without copying the elements.
2.  **Matrix Types**:
    *   ``TrapezoidMatrix``: Lower or Upper trapezoid.
    *   ``TriangularMatrix``: Square lower or upper triangle.
    *   ``SymmetricMatrix``: Symmetric matrix (where :math:`A_{ji} = A_{ij}`).
    *   ``HermitianMatrix``: Hermitian matrix (where :math:`A_{ji} = \bar{A}_{ij}`).
3.  **Slicing**: Creating a square slice of a general matrix to fit triangular requirements.

C++ Example
-----------

**General Matrix Creation (Lines 26-29)**

.. code-block:: cpp

    slate::Matrix<scalar_type>
        A( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );

We start with a standard, general `m` by `n` matrix `A`. This holds the underlying data.

**Trapezoid View (Lines 31-34)**

.. code-block:: cpp

    slate::TrapezoidMatrix<scalar_type>
        Lz( slate::Uplo::Lower, slate::Diag::Unit, A );

We create a `TrapezoidMatrix` named `Lz` from `A`.

- This is a **shallow copy**. `Lz` points to the same data tiles as `A`.
- `Uplo::Lower` specifies we are interested in the lower trapezoidal part.
- `Diag::Unit` specifies that the diagonal elements are implicitly assumed to be 1.0 (they are not accessed/modified).

**Slicing for Square Requirements (Lines 37-39)**

.. code-block:: cpp

    int64_t min_mn = std::min( m, n );
    auto A_square = A.slice( 0, min_mn-1, 0, min_mn-1 );

Triangular, Symmetric, and Hermitian matrices **must be square**. If `A` is rectangular (m != n), we cannot directly convert it to these types. We use `slice` to create a square view `A_square` of the top-left portion of `A`.

**Triangular Views (Lines 41-48)**

.. code-block:: cpp

    slate::TriangularMatrix<scalar_type>
        L( slate::Uplo::Lower, slate::Diag::Unit, A_square );

    slate::TriangularMatrix<scalar_type>
        U( slate::Uplo::Upper, slate::Diag::NonUnit, A_square );

Here we create Lower (`L`) and Upper (`U`) triangular views.

- `L` effectively sees only the lower triangle of `A_square`.
- `U` sees the upper triangle.
- These are used for operations like triangular solves (TRSM) or Cholesky factorization.

**Symmetric and Hermitian Views (Lines 50-57)**

.. code-block:: cpp

    slate::SymmetricMatrix<scalar_type>
        S( slate::Uplo::Upper, A_square );

    slate::HermitianMatrix<scalar_type>
        H( slate::Uplo::Upper, A_square );

- `S` represents a symmetric matrix where :math:`A_{ji} = A_{ij}`. Only the upper triangle is stored/referenced; the lower triangle is implicitly defined by symmetry.
- `H` represents a Hermitian matrix where :math:`A_{ji} = \bar{A}_{ij}`.
- These are crucial for optimized solvers (like Cholesky or LDLT) that exploit symmetry to save computation and storage.


.. literalinclude:: ../../../examples/ex02_conversion.cc
   :language: cpp
   :linenos:
