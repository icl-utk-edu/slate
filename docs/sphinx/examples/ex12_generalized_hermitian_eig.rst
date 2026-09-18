Example 12: Generalized Hermitian Eigenvalues
=============================================

This example demonstrates solving generalized Hermitian eigenvalue problems.

Key Concepts
------------

1.  **Problem Types**:
    *   Type 1: :math:`Ax = \lambda Bx`
    *   Type 2: :math:`ABx = \lambda x`
    *   Type 3: :math:`BAx = \lambda x`
2.  **Positive Definite B**: The matrix B must be Hermitian positive definite.
3.  **API Usage**: Using ``slate::eig_vals``, ``slate::eig``, and ``slate::hegv`` with the type parameter.

C++ Example
-----------



**Problem Setup (Lines 32-37)**

.. code-block:: cpp

    slate::HermitianMatrix<scalar_type> A( ... ), B( ... );
    slate::Matrix<scalar_type> Z( ... );
    std::vector<real_t> Lambda( n );

- `A`: Hermitian matrix.
- `B`: Hermitian **positive definite** matrix.
- `Z`: Eigenvectors.
- `Lambda`: Eigenvalues.

**Type 1: Ax = lambda Bx (Lines 50-64)**

.. code-block:: cpp

    slate::eig_vals( 1, A, B, Lambda );
    slate::eig( 1, A, B, Lambda );
    slate::hegv( 1, A, B, Lambda );

Solves :math:`Ax = \lambda Bx`.

- `A` is overwritten.
- `B` is overwritten by its Cholesky factor.

**Type 2: ABx = lambda x (Lines 74-123)**

.. code-block:: cpp

    slate::eig( 2, A, B, Lambda, Z );

Solves :math:`ABx = \lambda x`.

**Type 3: BAx = lambda x (Lines 84-129)**

.. code-block:: cpp

    slate::eig( 3, A, B, Lambda, Z );

Solves :math:`BAx = \lambda x`.

Note that for all types, `B` must be positive definite because the algorithms internally perform a Cholesky factorization of `B` to transform the generalized problem into a standard eigenvalue problem.

.. literalinclude:: ../../../examples/ex12_generalized_hermitian_eig.cc
   :language: cpp
   :linenos:
