2021.05.02
  - CMake: fix include paths with HIP for Spack

2021.05.01
  - CMake: fix library paths for Spack

2021.05.00
  - HIP/ROCm support
  - Improved performance (BLAS, Cholesky, LU, etc.)
  - Improved testers, matrix generation
  - More robust CUDA & HIP kernels, allow larger nb
  - CMake fixes

2020.10.00
  - Initial release. Functionality:
    - Level 3 BLAS
    - Matrix norms
    - LU, Cholesky, symmetric indefinite linear system solvers
    - Hermitian and generalized Hermitian eigenvalues (values only; vectors coming)
    - SVD (values only; vectors coming)
    - Makefile, CMake, and Spack build options
    - CUDA support
