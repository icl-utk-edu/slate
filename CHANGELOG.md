2022.05.00
  - Improved performance, including:
    LU, Cholesky, QR, mixed-precision LU and Cholesky, trsm, hemm, gemm,
    eigenvalues
  - Added LU threshold pivoting
  - Added scale, add, print
  - Added row-major MPI grid order; fixes ScaLAPACK API
  - Included HIP sources in repo, to eliminate build requirement of hipify-perl
  - Fixed OpenMP issues
  - Fixed QR with low-rank local blocks
  - Added C API in CMake
  - Rewrote testers to use less memory and reduce ScaLAPACK dependency
  - Use fast residual test for BLAS routines

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
