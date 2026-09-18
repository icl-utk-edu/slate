slate
=====

This section provides complete API documentation for SLATE functions, classes, and enumerations.

SLATE provides two naming schemes:

1. **Simplified API**: Descriptive names (e.g., ``lu_solve``, ``multiply``)
2. **Traditional API**: BLAS/LAPACK-style names (e.g., ``gesv``, ``gemm``)

All routines are in the ``slate::`` namespace.


Quick Reference
---------------

.. list-table:: Linear Systems
   :header-rows: 1
   :widths: 30 30 40

   * - Simplified
     - Traditional
     - Operation
   * - ``lu_solve``
     - ``gesv``
     - Solve :math:`AX = B` (general)
   * - ``lu_factor``
     - ``getrf``
     - Factor :math:`PA = LU`
   * - ``chol_solve``
     - ``posv``
     - Solve :math:`AX = B` (positive definite)
   * - ``chol_factor``
     - ``potrf``
     - Factor :math:`A = LL^H`
   * - ``indefinite_solve``
     - ``hesv/sysv``
     - Solve :math:`AX = B` (indefinite)

.. list-table:: BLAS Operations
   :header-rows: 1
   :widths: 30 30 40

   * - Simplified
     - Traditional
     - Operation
   * - ``multiply``
     - ``gemm``
     - :math:`C = \alpha AB + \beta C`
   * - ``multiply``
     - ``hemm/symm``
     - Hermitian/symmetric multiply
   * - ``rank_k_update``
     - ``herk/syrk``
     - :math:`C = \alpha AA^H + \beta C`
   * - ``triangular_solve``
     - ``trsm``
     - Solve :math:`AX = \alpha B`
   * - ``norm``
     - ``lange``
     - Matrix norms

.. list-table:: Decompositions
   :header-rows: 1
   :widths: 30 30 40

   * - Simplified
     - Traditional
     - Operation
   * - ``qr_factor``
     - ``geqrf``
     - Factor :math:`A = QR`
   * - ``lq_factor``
     - ``gelqf``
     - Factor :math:`A = LQ`
   * - ``eig``
     - ``heev/syev``
     - Eigenvalues :math:`A = Z\Lambda Z^H`
   * - ``svd``
     - ``gesvd``
     - SVD :math:`A = U\Sigma V^H`


Contents
--------

.. toctree::
   :maxdepth: 2
   :titlesonly:

   blas
   linear_systems
   least_squares
   eigenvalues
   svd
   matrices_api
   enums


Data Types
----------

SLATE routines are templated on scalar type:

- ``float`` - Single precision real
- ``double`` - Double precision real  
- ``std::complex<float>`` - Single precision complex
- ``std::complex<double>`` - Double precision complex


Common Parameters
-----------------

Most SLATE routines share common parameter patterns:

.. cpp:function:: template <typename scalar_t> \
                  void routine(Matrix<scalar_t>& A, Options const& opts = {})

Parameters:
    - **A** -- Input/output matrix
    - **opts** -- Optional execution parameters

Options include:

.. code-block:: cpp

    slate::Options opts = {
        {slate::Option::Target, slate::Target::Devices},
        {slate::Option::Lookahead, 2},
    };


Error Handling
--------------

SLATE throws exceptions for errors:

.. code-block:: cpp

    try {
        slate::lu_solve(A, B);
    }
    catch (slate::Exception& e) {
        std::cerr << "SLATE error: " << e.what() << std::endl;
    }


Header Files
------------

Main header (includes everything):

.. code-block:: cpp

    #include <slate/slate.hh>

Individual headers:

.. code-block:: cpp

    #include <slate/Matrix.hh>
    #include <slate/HermitianMatrix.hh>
    #include <slate/TriangularMatrix.hh>
    // etc.


C and Fortran APIs
------------------

SLATE provides C and Fortran bindings with type-specific suffixes:

- ``_r32`` - float
- ``_r64`` - double
- ``_c32`` - complex<float>
- ``_c64`` - complex<double>

Example:

.. code-block:: c

    // C API
    slate_lu_solve_r64(...);
    slate_multiply_c64(...);


See Also
--------

- :doc:`../../user_guide/operations` - User guide with examples
- :doc:`../../user_guide/matrices` - Matrix creation and manipulation
- `Online API Reference <https://icl.bitbucket.io/slate/>`_ - Doxygen-generated documentation


