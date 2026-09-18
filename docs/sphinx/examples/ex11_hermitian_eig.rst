Example 11: Hermitian Eigenvalue Problems
=========================================

This example demonstrates computing eigenvalues and eigenvectors for Hermitian (or symmetric) matrices.

Key Concepts
------------

1.  **Eigenvalues Only**: Computing just the eigenvalues :math:`\lambda` using ``eig_vals`` (heev).
2.  **Eigenvectors**: Computing eigenvalues and eigenvectors :math:`Z` such that :math:`A Z = Z \Lambda`.
3.  **Simplified vs Traditional**: Using ``slate::eig`` vs ``slate::heev``.

C++ Example
-----------


**Setup (Lines 28-32)**

.. code-block:: cpp

    slate::HermitianMatrix<scalar_type> A( ... );
    slate::Matrix<scalar_type> Z( ... );
    std::vector<real_t> Lambda( n );

- `A`: Input Hermitian matrix `n` by `n`.
- `Z`: Matrix to store eigenvectors. Must be `n` by `n`.
- `Lambda`: Vector to store eigenvalues. Must be size `n`.

**Eigenvalues Only (Lines 43-49)**

.. code-block:: cpp

    slate::eig_vals( A, Lambda );  // simplified API
    // or
    slate::eig( A, Lambda );       // simplified API
    // or
    slate::heev( A, Lambda );      // traditional API

Computes only the eigenvalues. `A` is overwritten by the tridiagonal factors (if `heev` logic is followed, though conceptually `A` is destroyed). `Lambda` contains the eigenvalues in ascending order.

- Note: `heev` stands for Hermitian EigenValues.

**Eigenvalues and Eigenvectors (Lines 63-68)**

.. code-block:: cpp

    slate::eig( A, Lambda, Z );    // simplified API
    // or
    slate::heev( A, Lambda, Z );   // traditional API

Computes both eigenvalues and eigenvectors.

- `Z` is overwritten with the eigenvectors.
- The columns of `Z` correspond to the eigenvalues in `Lambda`.


.. literalinclude:: ../../../examples/ex11_hermitian_eig.cc
   :language: cpp
   :linenos:

