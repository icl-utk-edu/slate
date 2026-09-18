Utilities and Types
===================

Core enumerations, error handling, and type utilities for BLAS++.

Enumerations
------------

**Layout** - Matrix storage layout:
    - ``ColMajor`` - Column-major storage (Fortran style)
    - ``RowMajor`` - Row-major storage (C style)

**Op** - Matrix transpose operation:
    - ``NoTrans`` - No transpose: :math:`op(A) = A`
    - ``Trans`` - Transpose: :math:`op(A) = A^T`
    - ``ConjTrans`` - Conjugate transpose: :math:`op(A) = A^H`

**Uplo** - Upper or lower triangle:
    - ``Upper`` - Upper triangle
    - ``Lower`` - Lower triangle
    - ``General`` - General (full) matrix

**Diag** - Diagonal type:
    - ``NonUnit`` - Non-unit diagonal
    - ``Unit`` - Unit diagonal (all 1's)

**Side** - Matrix multiplication side:
    - ``Left`` - Matrix on left: :math:`AB`
    - ``Right`` - Matrix on right: :math:`BA`

Error Handling
--------------

.. doxygenclass:: blas::Error
   :project: BLASPP
   :members:

Type Traits and Safety Functions
----------------------------------

See ``blaspp/include/blas/util.hh`` for:

- ``is_complex<T>`` - Check if type is complex
- ``scalar_type_traits<T...>`` - Deduce scalar type from multiple arguments
- ``real_type_traits<T>`` - Get real type from scalar or complex type
- ``safe_min<T>()``, ``safe_max<T>()`` - Safe numerical bounds
- ``root_min<T>()``, ``root_max<T>()`` - Safe bounds for square root operations
- ``ceildiv(x, y)`` - Integer ceiling division
