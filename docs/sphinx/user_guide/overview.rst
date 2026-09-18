Overview
========

This guide provides comprehensive documentation for using SLATE (Software for Linear Algebra 
Targeting Exascale). SLATE provides fundamental dense linear algebra capabilities for 
high-performance computing systems with multi-core processors and hardware accelerators.

SLATE is a library providing dense linear algebra capabilities for high-performance systems 
supporting large-scale distributed-nodes with accelerators. SLATE provides coverage of existing 
ScaLAPACK functionality, including:

- **Parallel BLAS**: Matrix-matrix multiply, triangular solve, and more
- **Linear Systems**: LU (general), Cholesky (positive definite), Aasen (indefinite)
- **Least Squares**: QR and LQ factorizations
- **Eigenvalue Problems**: Hermitian/symmetric eigenvalues, generalized eigenvalues
- **Singular Value Decomposition (SVD)**

SLATE is designed to be a replacement for ScaLAPACK, with superior performance and scalability 
in distributed-memory environments with multi-core processors and hardware accelerators.

Design Goals
~~~~~~~~~~~~

SLATE fulfills the following design goals:

**Target Modern HPC Hardware**
    Large number of nodes with multi-core processors and hardware accelerators (NVIDIA, AMD, Intel GPUs)

**Portable High Performance**
    Relies on vendor-optimized BLAS, batched BLAS, LAPACK, MPI and OpenMP

**Scalability**
    Uses 2D block-cyclic distribution, communication-avoiding algorithms, and modern parallel 
    programming approaches

**Productivity**
    Intuitive SPMD programming model with simple matrix abstractions

**Maintainability**
    Modern C++ with templates, overloading, and minimal code

**Easy Transition**
    Native support for ScaLAPACK 2D block-cyclic layout with backwards-compatible API


Software Requirements
~~~~~~~~~~~~~~~~~~~~~

SLATE requires:

- **Compiler**: C++17 compatible compiler (GCC 7+, Clang 5+, Intel 19+)
- **MPI**: MPI-3.0+ with ``MPI_THREAD_MULTIPLE`` support
- **OpenMP**: OpenMP 4.5+
- **BLAS/LAPACK**: Vendor-optimized libraries (Intel MKL, OpenBLAS, etc.)
- **Optional**: CUDA (NVIDIA), ROCm (AMD), or SYCL (Intel) for GPU support

SLATE also depends on:

- `BLAS++ <https://github.com/icl-utk-edu/blaspp>`_: C++ wrappers for BLAS
- `LAPACK++ <https://github.com/icl-utk-edu/lapackpp>`_: C++ wrappers for LAPACK

Project Resources
~~~~~~~~~~~~~~~~~

- **Website**: https://icl.utk.edu/slate/
- **Source Code**: https://github.com/icl-utk-edu/slate/
- **Issue Tracker**: https://github.com/icl-utk-edu/slate/issues/
- **User Forum**: https://groups.google.com/a/icl.utk.edu/g/slate-user
