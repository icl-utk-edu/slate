Example 10: Singular Value Decomposition (SVD)
==============================================

This example demonstrates computing the SVD of a matrix :math:`A = U \Sigma V^H`.

Key Concepts
------------

1.  **Singular Values Only**: Computing just :math:`\Sigma` using ``svd_vals``.
2.  **Full SVD**: Computing :math:`\Sigma`, :math:`U`, and :math:`V^H` using ``svd`` (gesvd).
3.  **Partial Vectors**: Computing only :math:`U` or only :math:`V^H`.

C++ Example
-----------

**Setup (Lines 29-30)**

.. code-block:: cpp

    slate::Matrix<scalar_type> A( m, n, nb, ... );
    std::vector<real_t> Sigma( min_mn );

- `A`: Input matrix `m` by `n`.
- `Sigma`: Vector to store the singular values. Size must be at least `min(m, n)`.

**Singular Values Only (Lines 41-46)**

.. code-block:: cpp

    slate::svd_vals( A, Sigma );
    // or
    slate::svd( A, Sigma );

If you only need the singular values (not the vectors `U` and `V`), use `svd_vals` or `svd` without matrix arguments. This is significantly faster than computing vectors.

**Singular Vectors (Lines 57-70)**

.. code-block:: cpp

    slate::Matrix<scalar_type>  U( m, min_mn, nb, ... );
    slate::Matrix<scalar_type> VH( min_mn, n, nb, ... );

    slate::svd( A, Sigma, U, VH );

To compute vectors:

- `U`: Left singular vectors. Dimensions `m` by `min(m, n)`.
- `VH`: Right singular vectors (transposed). Dimensions `min(m, n)` by `n`.
- Note: This example computes the **reduced** SVD.

**Partial Vectors (Lines 75-80)**

.. code-block:: cpp

    slate::Matrix<scalar_type> Uempty, Vempty;
    slate::svd( A, Sigma, U, Vempty );  // only U
    slate::svd( A, Sigma, Uempty, VH ); // only V^H

You can compute just `U` or just `VH` by passing an empty matrix placeholder for the unwanted component. This saves computation time.

.. literalinclude:: ../../../examples/ex10_svd.cc
   :language: cpp
   :linenos:
