SLATE Examples
================================================================================

These are designed as minimal, standalone examples to demonstrate
how to include, call, and link with SLATE.

These examples are used in the
[SLATE tutorial presentation](https://bitbucket.org/icl/slate/downloads/slate-tutorial.pdf)

These examples are also referenced in the
[SLATE Users' Guide]((https://www.icl.utk.edu/publications/swan-010)

For compiling, there are two options:

## Option 1: Makefile

Build & install SLATE (see slate/INSTALL.md):

    slate>  make
    slate>  make install prefix=${cwd}/install

Build examples:

    slate>  cd examples
    slate/examples>  make
    slate/examples>  make run
    slate/examples>  make run_scalapack

Installation puts the SLATE, BLAS++, and LAPACK++ headers all in the
same include directory, and libraries all in the same lib directory,
which simplifies compiling the examples. The above installs it in a
subdirectory of the slate source.

If needed, in the examples directory you can create a `make.inc` file
that sets `slate_dir` to where SLATE was installed; it is set to
`../install` by default to match the directions above. The scalapack
example also needs `scalapack_libs` set to your system's ScaLAPACK
libraries. For instance:

    # make.inc file
    slate_dir = /path/to/slate
    scalapack_libs = -L/path/to/scalapack -lscalapack -lgfortran

With Intel MKL, `scalapack_libs` should be some variant of this:

    scalapack_libs = -L${MKLROOT}/lib/intel64 -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64

depending on your MPI library (Intel MPI or Open MPI). Normally use the
`lp64` variants, unless `blas_int = int64` when SLATE was compiled, then
use the `ilp64` variants.

The examples Makefile detects whether SLATE was compiled with CUDA or
not, and links with CUDA if needed.

You may need to change the CXX compiler in the Makefile. It is set to `mpicxx`.

Examples of SLATE's C and Fortran APIs are in the c_api and fortran
directories. They use the same `make.inc` file and can be compiled
similarly.


## Option 2: CMake

Build & install SLATE (see slate/INSTALL.md):

    slate>  mkdir build && cd build
    slate/build>  cmake -DCMAKE_INSTALL_PREFIX=../install ..
    slate/build>  make
    slate/build>  make install

Build examples:

    slate/build>  cd ../examples
    slate/examples>  mkdir build && cd build
    slate/examples/build>  cmake -DCMAKE_PREFIX_PATH=${cwd}/../../install ..
    slate/examples/build>  make
    slate/examples/build>  make test

CTest output is in `Testing/Temporary/LastTest.log`.

CMake needs to find the SLATE, BLAS++, and LAPACK++ installations by
setting the `CMAKE_PREFIX_PATH` to the absolute path to install
directory. The above installs it in a subdirectory of the slate source.

todo: The scalapack example is not yet done in CMake.

todo: The c_api and fortran examples are not yet done in CMake.
