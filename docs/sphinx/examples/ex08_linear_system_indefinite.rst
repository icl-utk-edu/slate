Example 08: Linear Systems (Indefinite)
=======================================

This example demonstrates solving symmetric/Hermitian indefinite linear systems using Aasen's algorithm.

Key Concepts
------------

1.  **Indefinite Solve**: Using ``slate::indefinite_solve`` (hesv/sysv) for symmetric/Hermitian matrices that are not necessarily positive definite.
2.  **Aasen's Algorithm**: A factorization method :math:`A = L T L^H` where :math:`T` is tridiagonal.
3.  **Explicit Factorization**: Using ``indefinite_factor`` (hetrf) and ``indefinite_solve_using_factor`` (hetrs).
4.  **Workspace**: Allocating necessary workspace matrices (BandMatrix T, Matrix H).

C++ Example
-----------

**Indefinite Solve (Lines 41-50)**

.. code-block:: cpp

    slate::indefinite_solve( A, B );  // simplified API

    // traditional API with workspace setup
    slate::Matrix<scalar_type>     H( ... );
    slate::BandMatrix<scalar_type> T( ... );
    slate::Pivots pivots, pivots2;
    slate::hesv( A, pivots, T, pivots2, H, B );

Solves :math:`Ax=B` where `A` is symmetric/Hermitian but not positive definite.

- The **simplified API** (`indefinite_solve`) automatically handles the allocation of the auxiliary workspaces `T` and `H` and pivot vectors.
- The **traditional API** (`hesv` for Hermitian, `sysv` for Symmetric) requires you to pre-allocate:

  - `T`: A band matrix to store the tridiagonal factor.
  - `H`: A matrix for internal workspace.
  - `pivots`, `pivots2`: Vectors to store pivot information.

**Explicit Factorization (Lines 78-83)**

.. code-block:: cpp

    slate::indefinite_factor( A, pivots, T, pivots2, H );
    slate::indefinite_solve_using_factor( A, pivots, T, pivots2, B );

Separates the factorization (Aasen's algorithm) from the solve.

- `indefinite_factor` (hetrf): Computes the :math:`LTL^H` factorization.
- `indefinite_solve_using_factor` (hetrs): Solves the system using the factors.
- Requires managing the workspaces `T` and `H` explicitly even in the simplified API wrapper if you want to keep the factors.

.. literalinclude:: ../../../examples/ex08_linear_system_indefinite.cc
   :language: cpp
   :linenos:

