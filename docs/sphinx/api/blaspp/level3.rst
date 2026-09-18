Level 3 BLAS (Matrix-Matrix Operations)
========================================

Level 3 BLAS perform matrix-matrix operations with O(n³) complexity.

All operations are templated on scalar type (float, double, std::complex<float>, std::complex<double>) and support both CPU and GPU execution via the optional Queue parameter.

Operations
----------

**gemm** - General matrix-matrix multiply: :math:`C = \alpha op(A) op(B) + \beta C`

**hemm** - Hermitian matrix-matrix multiply: :math:`C = \alpha A B + \beta C` or :math:`C = \alpha B A + \beta C`

**herk** - Hermitian rank-k update: :math:`C = \alpha A A^H + \beta C`

**her2k** - Hermitian rank-2k update: :math:`C = \alpha A B^H + \overline{\alpha} B A^H + \beta C`

**symm** - Symmetric matrix-matrix multiply: :math:`C = \alpha A B + \beta C` or :math:`C = \alpha B A + \beta C`

**syrk** - Symmetric rank-k update: :math:`C = \alpha A A^T + \beta C`

**syr2k** - Symmetric rank-2k update: :math:`C = \alpha A B^T + \alpha B A^T + \beta C`

**trmm** - Triangular matrix-matrix multiply: :math:`B = \alpha op(A) B` or :math:`B = \alpha B op(A)`

**trsm** - Triangular solve: :math:`op(A) X = \alpha B` or :math:`X op(A) = \alpha B`

All functions are defined in the ``blas`` namespace and documented in individual header files under ``include/blas/``.
