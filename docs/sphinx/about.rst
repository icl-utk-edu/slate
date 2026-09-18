About SLATE
===========

SLATE (Software for Linear Algebra Targeting Exascale) is a modern C++ linear algebra library for distributed-memory systems with GPU acceleration. 

Key Features
------------

- **Distributed Memory**: MPI-based parallelism across nodes
- **GPU Acceleration**: CUDA, ROCm, and OpenMP target support
- **Tile-Based**: Modern tiled algorithms for better performance
- **Standards Compatible**: BLAS, LAPACK, and ScaLAPACK API compatibility
- **High Performance**: Optimized for modern architectures

Project Information
-------------------

SLATE is developed at the University of Tennessee's Innovative Computing Laboratory (ICL).

- **Website**: https://icl.utk.edu/slate/
- **Repository**: https://github.com/icl-utk-edu/slate
- **License**: BSD 3-Clause

Version
-------

SLATE version information can be obtained programmatically:

.. code-block:: cpp

   #include <slate/slate.hh>
   
   int version = slate::version();
   const char* id = slate::id();
