2022.07.00
  - Improved performance of QR factorization on GPUs by moving panel to GPU:
    5.5x faster on tall-skinny problem
  - Added Cholesky QR `cholqr`; added as option in least squares solver, `gels`
  - Added GPU implementation of `gemmA`, used when n is small (e.g., n <= nb)
  - Added row and column scaling, `scale_row_col`
  - Added print of individual tile
  - Removed use of life counter in `gemm`, `herk`
  - Removed setting MKL threads, which is no longer needed
  - Removed `SLATE_NO_{HIP, CUDA}` macros in favor of
    `BLAS_HAVE_{CUBLAS, ROCBLAS}` macros from BLAS++
  - Introduced `tile` namespace

2022.06.00
  - Fixed algorithm selection (issue #41)
  - Fixed set for triangular, trapezoid, symmetric, Hermitian matrices (tzset)
  - Fixed ScaLAPACK pdsgesv wrapper (issue #42)
  - Fixed norm for general band matrix (gbnorm)
  - Added macro for OpenMP `default(none)`; by default empty since it
    causes unpredictable errors for some compilers or libraries

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
