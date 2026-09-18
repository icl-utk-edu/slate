Installation
============

SLATE can be built from source using Makefile, CMake, or installed via Spack. This guide covers 
all three methods.


Prerequisites
-------------

SLATE requires:

- **C++ Compiler**: C++17 support (GCC 7+, Clang 5+, Intel 19+)
- **MPI**: MPI-3.0+ with ``MPI_THREAD_MULTIPLE`` support
- **OpenMP**: OpenMP 4.5+
- **BLAS/LAPACK**: Optimized implementation (MKL, OpenBLAS, ESSL, LibSci, etc.)

Optional for GPU acceleration:

- **NVIDIA GPUs**: CUDA toolkit
- **AMD GPUs**: ROCm
- **Intel GPUs**: oneAPI with SYCL/DPC++


Downloading Source
------------------

Clone the repository with submodules:

.. code-block:: bash

    # HTTPS
    git clone --recursive https://github.com/icl-utk-edu/slate.git
    
    # SSH
    git clone --recursive git@github.com:icl-utk-edu/slate.git
    
    cd slate


Makefile Build
--------------

The Makefile build auto-detects compilers and dependencies from your environment.

Basic Configuration
~~~~~~~~~~~~~~~~~~~

Create a ``make.inc`` file with your configuration:

.. code-block:: make

    # make.inc
    CXX = mpicxx
    FC  = mpif90
    blas = openblas

Build and Install
~~~~~~~~~~~~~~~~~

.. code-block:: bash

    make lib           # Build libraries
    make tester        # Build tester
    make check         # Run sanity tests
    make install prefix=/usr/local  # Install


Configuration Options
~~~~~~~~~~~~~~~~~~~~~

Common ``make.inc`` options:

.. code-block:: make

    # Compiler settings
    CXX = mpicxx
    FC  = mpif90
    
    # BLAS library: openblas, mkl, essl, accelerate
    blas = mkl
    
    # GPU support: cuda, hip, sycl, or leave empty
    gpu_backend = cuda
    
    # Build type
    # Use 'make static' for static libraries
    
    # Additional flags
    CXXFLAGS += -O3
    LDFLAGS  += -L/path/to/lib


CMake Build
-----------

CMake provides more control over configuration.

Basic Build
~~~~~~~~~~~

.. code-block:: bash

    mkdir build && cd build
    
    export CXX=g++
    export FC=gfortran
    
    cmake -D blas=openblas \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          ..
    
    make lib           # Build libraries
    make tester        # Build tester
    make check         # Run sanity tests
    make install       # Install


CMake Options
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Option
     - Description
     - Default
   * - ``blas``
     - BLAS library (openblas, mkl, essl, etc.)
     - auto-detect
   * - ``gpu_backend``
     - GPU backend (cuda, hip, sycl, none)
     - auto-detect
   * - ``BUILD_SHARED_LIBS``
     - Build shared libraries
     - ON
   * - ``CMAKE_INSTALL_PREFIX``
     - Installation directory
     - /usr/local
   * - ``CMAKE_BUILD_TYPE``
     - Build type (Release, Debug)
     - Release


Example with MKL and CUDA
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    cmake -D blas=mkl \
          -D gpu_backend=cuda \
          -D CMAKE_CUDA_ARCHITECTURES=70 \
          -D CMAKE_INSTALL_PREFIX=$HOME/local \
          ..


Spack Installation
------------------

Spack is the recommended method for HPC systems.

Install Spack
~~~~~~~~~~~~~

If you don't have Spack:

.. code-block:: bash

    git clone https://github.com/spack/spack.git
    source spack/share/spack/setup-env.sh
    spack compiler find


Install SLATE
~~~~~~~~~~~~~

.. code-block:: bash

    # View available options
    spack info slate
    
    # Install with defaults
    spack install slate
    
    # With specific compiler
    spack install slate %gcc@11.3.1
    
    # With OpenBLAS, without CUDA
    spack install slate %gcc@11.3.1 ^openblas ~cuda
    
    # With CUDA support
    spack install slate %gcc@11.3.1 +cuda cuda_arch=70


Loading SLATE
~~~~~~~~~~~~~

.. code-block:: bash

    spack load slate
    # or
    module load slate  # if configured


GPU-Aware MPI
-------------

For optimal GPU performance, enable GPU-aware MPI:

.. code-block:: bash

    # Enable SLATE's GPU-aware MPI
    export SLATE_GPU_AWARE_MPI=1
    
    # For Cray MPI (Frontier, Perlmutter)
    export MPICH_GPU_SUPPORT_ENABLED=1

Check your HPC center's documentation for additional flags.


Verifying Installation
----------------------

Run the basic test suite:

.. code-block:: bash

    export OMP_NUM_THREADS=8
    
    # Single process test
    ./test/tester gemm
    
    # Multi-process test (4 MPI ranks)
    mpirun -n 4 ./test/tester gemm
    
    # With Slurm
    srun --nodes=4 --ntasks=16 --cpus-per-task=${OMP_NUM_THREADS} \
         ./test/tester gemm

Successful output:

.. code-block:: text

    gemm dtype=d, origin=h, target=t, transA=n, transB=n, m=100, n=100, k=100, ...
    time=0.00123, gflop/s=1.63, ref_time=..., ref_gflop/s=..., error=1.2e-15, okay


Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**MPI_THREAD_MULTIPLE not supported**

Some MPI implementations require configuration. For OpenMPI:

.. code-block:: bash

    mpirun --mca mpi_thread_multiple 1 -n 4 ./program

**BLAS library not found**

Set library paths:

.. code-block:: bash

    export LIBRARY_PATH=/path/to/blas/lib:$LIBRARY_PATH
    export LD_LIBRARY_PATH=/path/to/blas/lib:$LD_LIBRARY_PATH


**CUDA not detected**

Ensure ``nvcc`` is in your PATH:

.. code-block:: bash

    export PATH=/usr/local/cuda/bin:$PATH


Getting Help
~~~~~~~~~~~~

- **Issues**: https://github.com/icl-utk-edu/slate/issues/
- **User Forum**: https://groups.google.com/a/icl.utk.edu/g/slate-user
- **BLAS++ Issues**: https://github.com/icl-utk-edu/blaspp/issues/
- **LAPACK++ Issues**: https://github.com/icl-utk-edu/lapackpp/issues/


Module Environment (HPC Systems)
--------------------------------

Many HPC systems provide SLATE as a module:

.. code-block:: bash

    module avail slate
    module load slate

Check with your HPC support desk for available versions and configurations.
