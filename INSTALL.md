SLATE Installation Notes
================================================================================

[TOC]

Synopsis
--------------------------------------------------------------------------------

Checkout or download. SLATE uses git modules, which require an update step:

    git clone --recursive https://bitbucket.org/icl/slate

or

    git clone https://bitbucket.org/icl/slate
    git submodule update --init

If you have an existing git repository and pull updates, you may need to
also update submodules, if they changed:

    git pull
    git submodule update

Or download the release tar file, which includes BLAS++ and LAPACK++, from
[downloads](https://bitbucket.org/icl/slate/downloads/).

--------------------------------------------------------------------------------

Configure and compile the SLATE library and its tester,
then install the headers and library. This will also compile
BLAS++, LAPACK++, and TestSweeper.

**Option 1: Makefile**

    # create make.inc file, for example:
    CXX  = mpicxx    # MPI compiler wrappers recommended
    FC   = mpif90
    blas = openblas

Compile and install:

    make && make install

**Option 2: CMake**

    export CXX=g++      # or your preferred C++ compiler
    export FC=gfortran  # or your preferred Fortran compiler
    mkdir build && cd build
    cmake -Dblas=openblas ..
    make && make install


Environment variables (Makefile and CMake)
--------------------------------------------------------------------------------

Standard environment variables affect both Makefile (configure.py) and CMake.
These include:

    CXX                 C++ compiler
    CXXFLAGS            C++ compiler flags
    FC                  Fortran compiler
    FCFLAGS             Fortran compiler flags
    LDFLAGS             linker flags
    CPATH               compiler include search path
    LIBRARY_PATH        compile-time library search path
    LD_LIBRARY_PATH     runtime library search path
    DYLD_LIBRARY_PATH   runtime library search path on macOS


Options (Makefile and CMake)
--------------------------------------------------------------------------------

See the BLAS++ [INSTALL.md](https://bitbucket.org/icl/blaspp/src/master/INSTALL.md)
for its options, which include:

(Note: SLATE's Makefile uses 1 or 0 instead of yes or no. CMake can use either.)

    blas
        BLAS libraries to search for. One or more of:
        auto            search for all libraries (default)
        * libsci        Cray LibSci
        * mkl           Intel MKL
        * essl          IBM ESSL
        * openblas      OpenBLAS
        accelerate      Apple Accelerate framework
        acml            AMD ACML (deprecated)
        generic         generic -lblas
        * SLATE's Makefile currently supports only libsci, mkl, essl,
        openblas (lowercase).
        SLATE's CMake supports all libraries.

    blas_int
        BLAS integer size to search for. One or more of:
        auto            search for both sizes (default)
        int             32-bit int (LP64 model)
        * int64         64-bit int (ILP64 model)
        * int64 is not currently supported in SLATE

    blas_threaded
        Whether to search for multi-threaded or sequential BLAS.
        Currently applies to Intel MKL and IBM ESSL. One of:
        1               multi-threaded BLAS
        0               sequential BLAS (default in SLATE)

    blas_fortran
        Fortran interface to use. Currently applies only to Intel MKL.
        One or more of:
        ifort           use Intel ifort interfaces (e.g., libmkl_intel_lp64)
        gfortran        use GNU gfortran interfaces (e.g., libmkl_gf_lp64)

    fortran_mangling
        (Makefile only; CMake always searches all manglings)
        BLAS and LAPACK are written in Fortran, which has a
        compiler-specific name mangling scheme: routine DGEMM is called
        dgemm_, dgemm, or DGEMM in the library. One or more of:
        auto            search all manglings (default)
        add_            add _ to names  (dgemm_)
        lower           lowercase names (dgemm)
        upper           uppercase names (DGEMM)

    BLAS_LIBRARIES [CMake only]
        Specify the exact BLAS libraries, overriding the built-in search. E.g.,
        cmake -DBLAS_LIBRARIES='-lopenblas' ..

    gpu_backend
        auto            auto-detect CUDA or HIP/ROCm (default)
        cuda            build with CUDA support
        hip             build with HIP/ROCm support
        none            do not build with GPU backend

    color
        Whether to use ANSI colors in output. One of:
        auto            uses color if output is a TTY
                        (default with Makefile; not support with CMake)
        yes             (default with CMake)
        no

See the LAPACK++ [INSTALL.md](https://bitbucket.org/icl/lapackpp/src/master/INSTALL.md)
for its options, which include:

    lapack [CMake only]
        LAPACK libraries to search for.
        LAPACK is often included in the BLAS library (e.g., -lopenblas contains both),
        so there is usually no need to specify this. One or more of:
        auto            search for all libraries (default)
        generic         generic -llapack

    LAPACK_LIBRARIES [CMake only]
        Specify the exact LAPACK libraries, overriding the built-in search.
        Again, there is usually no need to specify this. E.g.,
        cmake -DLAPACK_LIBRARIES='-lopenblas' ..

SLATE specific options include:

    mkl_blacs [Makefile only]
        openmpi         Open MPI BLACS in SLATE's testers.
        intelmpi        Intel MPI BLACS in SLATE's testers (default).

    SCALAPACK_LIBRARIES [Makefile and CMake]
        For SLATE's testers, specify the exact ScaLAPACK libraries to
        use, overriding the built-in search, or set to `none` to build
        testers without ScaLAPACK.
        With MKL, by default it uses
            -lmkl_scalapack_lp64 -lmkl_blacs_openmpi_lp64
        or  -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64;
        with LibSci via Cray's CC compiler wrapper, no library is needed;
        otherwise it uses -lscalapack.

With Makefile, options are specified as environment variables or on the
command line using `option=value` syntax, such as:

    python configure.py blas=mkl

With CMake, options are specified on the command line using
`-Doption=value` syntax (not as environment variables), such as:

    cmake -Dblas=mkl ..


Makefile Installation
--------------------------------------------------------------------------------

Available targets:

    make           - compiles the library and tester; also configures and
                     compiles BLAS++, LAPACK++, and TestSweeper libraries.
    make lib       - compiles the library (lib/libslate.so)
    make tester    - compiles test/tester
    make check     - run basic checks using tester
    make docs      - generates documentation in docs/html/index.html
    make install   - installs the library and headers to ${prefix}
    make uninstall - remove installed library and headers from ${prefix}
    make clean     - deletes object (*.o) and library (*.a, *.so) files
    make distclean - also deletes dependency files (*.d) and
                     cleans BLAS++, LAPACK++, and TestSweeper.

### Makefile options

Besides the Environment variables and Options listed above, additional
options include:

    static
        0                   build shared libraries (libslate.so) (default)
        1                   build static libraries (libslate.a)

    mpi
        The Makefile will detect mpi from the MPI compiler wrapper name
        (e.g., mpicxx). To compile using MPI without the MPI compiler
        wrapper, set one of:
        mpi = 1             link with `-lmpi`
        mpi = spectrum      link with `-lmpi_ibm`
        mpi = cray          using Cray compiler wrappers (CXX=CC, FC=ftn)

    cuda_arch
        By default, SLATE uses nvcc's default architecture.
        To use a different architecture, set `cuda_arch` to one or more of:
        `kepler maxwell pascal volta turing ampere hopper sm_XY`
        where XY is a valid CUDA architecture (see `nvcc -h | grep sm_`),
        separated by space.

    hip_arch
        By default, SLATE uses hipcc's default architecture.
        To use a different architecture, set `hip_arch` to one or more of:
        `gfx900` or `mi25`  for AMD Radeon Instinct MI25 / Vega 10
        `gfx906` or `mi50`  for AMD Radeon Instinct MI50 / Vega 20
        `gfx908` or `mi100` for AMD Instinct MI100
        `gfx90a` or `mi200` for AMD Instinct MI200 series (MI250)
        or other valid HIP architecture, separated by space.
        See https://llvm.org/docs/AMDGPUUsage.html

    openmp
        SLATE will compile with OpenMP by default. To compile without
        OpenMP, set `openmp = 0`.

    c_api
        Whether to build C API. Python is required. One of:
        1                   build C API
        0                   don't build C API

    fortran_api
        Whether to build Fortran 2003 API. Requires c_api. One of:
        1                   build Fortran API
        0                   don't build Fortran API

With Makefile, creating a `make.inc` file with the necesary options is
recommended, to ensure the same options are used by all `make` commands.
It is recommended to use MPI compiler wrappers such as mpicxx and
mpif90. For instance:

    # make.inc
    CXX  = mpicxx
    FC   = mpif90
    blas = openblas

Alternatively, options can be specified as environment variables or on the
command line using `option=value` syntax, such as:

    export CXX=mpicxx
    export FC=mpif90
    make blas=mkl

Then compile. Possible targets include:

    make lib        # build libraries
    make test       # build tester
    make install    # install libraries

Running `make` by itself will compile `make lib; make test`.

    cd examples
    make

CMake Installation
--------------------------------------------------------------------------------

The CMake script enforces an out-of-source build. Create a build
directory under the SLATE root directory:

    cd /path/to/slate
    mkdir build && cd build
    cmake [-DCMAKE_INSTALL_PREFIX=/path/to/install] [options] ..
    make
    make install

SLATE uses the
[BLAS++](https://bitbucket.org/icl/blaspp),
[LAPACK++](https://bitbucket.org/icl/lapackpp), and
[TestSweeper](https://bitbucket.org/icl/testsweeper) libraries.
These are generally checked out as git submodules in the slate directory,
so the user does not have to install them beforehand. If CMake finds already
installed versions, it will use those instead of compiling new versions.


### CMake Options

Besides the Environment variables and Options listed above, additional
options include:

    use_cuda [deprecated; use gpu_backend]
    use_hip  [deprecated; use gpu_backend]

    CMAKE_CUDA_ARCHITECTURES
        CUDA architectures, as semi-colon delimited list of 2-digit numbers.
        Each number can take optional `-real` or `-virtual` suffix.
        Default is `60`, for Pascal architecture. For description, see:
        https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
        For other architectures, CMAKE_CUDA_ARCHITECTURES **should be defined**.
        For example, `-DCMAKE_CUDA_ARCHITECTURES=70` should be added for
        the Volta architecture.

    use_openmp
        Whether to use OpenMP, if available. One of:
        yes (default)
        no

    build_tests
        Whether to build test suite (test/tester).
        Requires ScaLAPACK unless SCALAPACK_LIBRARIES=none. One of:
        yes (default)
        no

    c_api
        Whether to build C API. Python is required. One of:
        yes
        no (default)

BLAS++ options include:

    use_cmake_find_blas
        Whether to use CMake's FindBLAS, instead of BLAS++ search. One of:
        yes
        no (default)
        If BLA_VENDOR is set, it automatically uses CMake's FindBLAS.

    BLA_VENDOR
        Use CMake's FindBLAS, instead of BLAS++ search. For values, see:
        https://cmake.org/cmake/help/latest/module/FindBLAS.html

LAPACK++ options include:

    use_cmake_find_lapack
        Whether to use CMake's FindLAPACK, instead of LAPACK++ search.
        Again, as LAPACK is often included in the BLAS library,
        there is usually no need to specify this. One of:
        yes
        no (default)
        If BLA_VENDOR is set, it automatically uses CMake's FindLAPACK.

Standard CMake options include:

    BUILD_SHARED_LIBS
        Whether to build as a static or shared library. One of:
        yes             shared library (default)
        no              static library

    CMAKE_INSTALL_PREFIX (alias prefix)
        Where to install, default /opt/slate.
        Headers go   in ${prefix}/include,
        library goes in ${prefix}/lib

    CMAKE_PREFIX_PATH
        Where to look for CMake packages such as BLAS++ and TestSweeper.

    CMAKE_BUILD_TYPE
        Type of build. One of:
        [empty]         default compiler optimization          (no flags)
        Debug           no optimization, with asserts          (-O0 -g)
        Release         optimized, no asserts, no debug info   (-O3 -DNDEBUG)
        RelWithDebInfo  optimized, no asserts, with debug info (-O2 -DNDEBUG -g)
        MinSizeRel      Release, but optimized for size        (-Os -DNDEBUG)

    CMAKE_MESSAGE_LOG_LEVEL (alias log)
        Level of messages to report. In ascending order:
        FATAL_ERROR, SEND_ERROR, WARNING, AUTHOR_WARNING, DEPRECATION,
        NOTICE, STATUS, VERBOSE, DEBUG, TRACE.
        Particularly, DEBUG or TRACE gives useful information.

With CMake, options are specified on the command line using
`-Doption=value` syntax (not as environment variables), such as:

    # in build directory
    cmake -Dblas=mkl -Dbuild_tests=no -DCMAKE_INSTALL_PREFIX=/usr/local ..

Alternatively, use the `ccmake` text-based interface or the CMake app GUI.

    # in build directory
    ccmake ..
    # Type 'c' to configure, then 'g' to generate Makefile

To re-configure CMake, you may need to delete CMake's cache:

    # in build directory
    rm CMakeCache.txt
    # or
    rm -rf *
    cmake [options] ..

To debug the build, set `VERBOSE`:

    # in build directory, after running cmake
    make VERBOSE=1
