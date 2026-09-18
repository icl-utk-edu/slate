lapackpp
========

LAPACK++ provides a modern C++11 interface to LAPACK (Linear Algebra PACKage).

**Categories:**

.. toctree::
   :maxdepth: 2

   util
   device
   flops
   factorizations
   linear_systems
   least_squares
   eigenvalues
   svd
   auxiliary

Overview
--------

LAPACK++ delivers comprehensive linear algebra functionality:

- **Type Safety**: C++ templates for float, double, complex<float>, complex<double>
- **Error Handling**: Exception-based via lapack::Error
- **GPU Support**: CUDA, ROCm/HIP, and SYCL backends
- **BLAS++ Integration**: Consistent API design
- **400+ Routines**: Complete coverage of standard LAPACK

Key Features
------------

- Solves linear systems: gesv, posv, sysv, gels
- Eigenvalue problems: syev, geev, gges
- Singular value decomposition: gesvd, gesdd, gesvdx
- Matrix factorizations: getrf, potrf, sytrf, geqrf
- Condition numbers and norms: gecon, lange, lansy

See Also
--------

- `LAPACK Documentation <https://www.netlib.org/lapack/>`_
- :doc:`../blaspp/index` - BLAS++ Documentation
