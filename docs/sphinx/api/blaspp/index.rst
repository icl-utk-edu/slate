blaspp
======

BLAS++ provides a modern C++ interface to the Basic Linear Algebra Subroutines (BLAS), supporting both CPU and GPU execution.

Features
--------

- **Type-generic templates**: Single API for ``float``, ``double``, ``std::complex<float>``, ``std::complex<double>``
- **Multiple backends**: Reference C++, vendor BLAS (MKL, OpenBLAS), GPU (cuBLAS, rocBLAS, SYCL)
- **Modern C++**: C++11/14 features, strong typing, ``std::complex``
- **Performance counters**: Optional PAPI integration
- **Device support**: Asynchronous GPU operations

Organization
------------

BLAS++ operations are organized by level:

- **Level 1**: Vector-vector operations (axpy, dot, nrm2, scal, etc.)
- **Level 2**: Matrix-vector operations (gemv, ger, trmv, etc.)
- **Level 3**: Matrix-matrix operations (gemm, trmm, herk, etc.)

Contents
--------

.. toctree::
   :maxdepth: 2
   :titlesonly:

   level1
   level2
   level3
   util
   device
   performance


Quick Reference
---------------

Level 1 BLAS (Vector-Vector)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Function
     - Operation
   * - ``asum``
     - Sum of absolute values: :math:`\sum |x_i|`
   * - ``axpy``
     - Vector plus scaled vector: :math:`y = \alpha x + y`
   * - ``copy``
     - Copy vector: :math:`y = x`
   * - ``dot``
     - Dot product: :math:`x^T y` (conjugate for complex)
   * - ``dotu``
     - Dot product unconjugated: :math:`x^T y`
   * - ``iamax``
     - Index of max absolute value
   * - ``nrm2``
     - Euclidean norm: :math:`\|x\|_2`
   * - ``scal``
     - Scale vector: :math:`x = \alpha x`
   * - ``swap``
     - Swap vectors: :math:`x \leftrightarrow y`
   * - ``rot``
     - Apply plane rotation
   * - ``rotg``
     - Generate plane rotation
   * - ``rotm``
     - Apply modified plane rotation
   * - ``rotmg``
     - Generate modified plane rotation


Level 2 BLAS (Matrix-Vector)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Function
     - Operation
   * - ``gemv``
     - General matrix-vector multiply: :math:`y = \alpha Ax + \beta y`
   * - ``ger``
     - General rank-1 update: :math:`A = \alpha xy^T + A`
   * - ``geru``
     - General rank-1 update unconjugated
   * - ``hemv``
     - Hermitian matrix-vector multiply
   * - ``her``
     - Hermitian rank-1 update
   * - ``her2``
     - Hermitian rank-2 update
   * - ``symv``
     - Symmetric matrix-vector multiply
   * - ``syr``
     - Symmetric rank-1 update
   * - ``syr2``
     - Symmetric rank-2 update
   * - ``trmv``
     - Triangular matrix-vector multiply
   * - ``trsv``
     - Triangular solve: :math:`x = A^{-1}x`


Level 3 BLAS (Matrix-Matrix)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Function
     - Operation
   * - ``gemm``
     - General matrix multiply: :math:`C = \alpha AB + \beta C`
   * - ``hemm``
     - Hermitian matrix multiply
   * - ``herk``
     - Hermitian rank-k update: :math:`C = \alpha AA^H + \beta C`
   * - ``her2k``
     - Hermitian rank-2k update
   * - ``symm``
     - Symmetric matrix multiply
   * - ``syrk``
     - Symmetric rank-k update
   * - ``syr2k``
     - Symmetric rank-2k update
   * - ``trmm``
     - Triangular matrix multiply
   * - ``trsm``
     - Triangular solve: :math:`X = \alpha A^{-1}B`


Basic Usage
-----------

CPU (Host) Operations
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

    #include <blas.hh>
    
    // Matrix-matrix multiply: C = alpha*A*B + beta*C
    blas::gemm(
        blas::Layout::ColMajor,
        blas::Op::NoTrans, blas::Op::NoTrans,
        m, n, k,
        alpha, A, lda,
               B, ldb,
        beta,  C, ldc);
    
    // Vector operations
    blas::axpy(n, alpha, x, incx, y, incy);  // y = alpha*x + y
    double norm = blas::nrm2(n, x, incx);     // ||x||_2


GPU (Device) Operations
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

    #include <blas.hh>
    
    // Create device queue
    blas::Queue queue(device_id);
    
    // Device matrix multiply (d_A, d_B, d_C are device pointers)
    blas::gemm(
        blas::Layout::ColMajor,
        blas::Op::NoTrans, blas::Op::NoTrans,
        m, n, k,
        alpha, d_A, lda,
               d_B, ldb,
        beta,  d_C, ldc,
        queue);
    
    // Wait for completion
    queue.sync();


Common Parameters
-----------------

Layout
~~~~~~

Specifies matrix storage order:

- ``Layout::ColMajor`` - Column-major (Fortran-style)
- ``Layout::RowMajor`` - Row-major (C-style)

Op
~~

Specifies transpose operation:

- ``Op::NoTrans`` - No transpose: :math:`A`
- ``Op::Trans`` - Transpose: :math:`A^T`
- ``Op::ConjTrans`` - Conjugate transpose: :math:`A^H`

Uplo
~~~~

Specifies triangular/symmetric matrix part:

- ``Uplo::Upper`` - Upper triangle
- ``Uplo::Lower`` - Lower triangle

Diag
~~~~

Specifies diagonal type for triangular matrices:

- ``Diag::NonUnit`` - Diagonal elements are arbitrary
- ``Diag::Unit`` - Diagonal elements are 1

Side
~~~~

Specifies matrix position in operation:

- ``Side::Left`` - Matrix on left: :math:`AB`
- ``Side::Right`` - Matrix on right: :math:`BA`


Data Types
----------

BLAS++ functions are templated on scalar type:

- ``float`` - Single precision real
- ``double`` - Double precision real
- ``std::complex<float>`` - Single precision complex
- ``std::complex<double>`` - Double precision complex


Header Files
------------

Main header (includes everything):

.. code-block:: cpp

    #include <blas.hh>

Individual operation headers:

.. code-block:: cpp

    #include <blas/gemm.hh>
    #include <blas/axpy.hh>
    // etc.

Utility headers:

.. code-block:: cpp

    #include <blas/util.hh>      // Enumerations, error handling
    #include <blas/device.hh>    // Device queue, memory management
    #include <blas/counter.hh>   // Performance counters (PAPI)
    #include <blas/flops.hh>     // FLOP counting


Error Handling
--------------

BLAS++ uses exceptions for errors:

.. code-block:: cpp

    try {
        blas::gemm(...);
    }
    catch (blas::Error& e) {
        std::cerr << "BLAS++ error: " << e.what() << std::endl;
    }


See Also
--------

- `BLAS++ GitHub Repository <https://github.com/icl-utk-edu/blaspp>`_
- `BLAS Reference <https://www.netlib.org/blas/>`_
