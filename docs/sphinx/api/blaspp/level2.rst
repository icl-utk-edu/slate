Level 2 BLAS (Matrix-Vector Operations)
========================================

Level 2 BLAS perform matrix-vector operations with O(n²) complexity.

All operations are templated on scalar type (float, double, std::complex<float>, std::complex<double>) and support both CPU and GPU execution via the optional Queue parameter.

Operations
----------

**gemv** - General matrix-vector multiply: :math:`y = \alpha op(A)x + \beta y`

**ger** - General rank-1 update (conjugated): :math:`A = \alpha x y^H + A`

**geru** - General rank-1 update (unconjugated): :math:`A = \alpha x y^T + A`

**hemv** - Hermitian matrix-vector multiply: :math:`y = \alpha A x + \beta y`

**her** - Hermitian rank-1 update: :math:`A = \alpha x x^H + A`

**her2** - Hermitian rank-2 update: :math:`A = \alpha x y^H + \overline{\alpha} y x^H + A`

**symv** - Symmetric matrix-vector multiply: :math:`y = \alpha A x + \beta y`

**syr** - Symmetric rank-1 update: :math:`A = \alpha x x^T + A`

**syr2** - Symmetric rank-2 update: :math:`A = \alpha x y^T + \alpha y x^T + A`

**trmv** - Triangular matrix-vector multiply: :math:`x = op(A)x`

**trsv** - Triangular solve: :math:`op(A)x = b`

All functions are defined in the ``blas`` namespace and documented in individual header files under ``include/blas/``.
