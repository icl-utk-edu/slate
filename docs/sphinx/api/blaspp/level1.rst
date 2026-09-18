Level 1 BLAS (Vector Operations)
=================================

Level 1 BLAS perform vector-vector operations with O(n) complexity.

All operations are templated on scalar type (float, double, std::complex<float>, std::complex<double>) and support both CPU and GPU execution via the optional Queue parameter.

Operations
----------

**asum** - Sum of absolute values: :math:`\sum |x_i|`

**axpy** - Scale and add vectors: :math:`y = \alpha x + y`

**copy** - Copy vector: :math:`y = x`

**dot** - Dot product (conjugated): :math:`x^H y`

**dotu** - Dot product (unconjugated): :math:`x^T y`

**iamax** - Index of maximum absolute value

**nrm2** - Euclidean norm: :math:`\|x\|_2`

**rot** - Apply plane rotation

**rotg** - Generate plane rotation

**rotm** - Apply modified Givens rotation

**rotmg** - Generate modified Givens rotation

**scal** - Scale vector: :math:`x = \alpha x`

**swap** - Swap vectors: :math:`x \leftrightarrow y`

All functions are defined in the ``blas`` namespace and documented in individual header files under ``include/blas/``.
