SLATE Examples
================================================================================

These are designed as minimal, standalone examples to demonstrate
how to include, call, and link with SLATE.

These examples are used in the
[SLATE tutorial presentation](https://bitbucket.org/icl/slate/downloads/2023-02-ecp-slate-tutorial.pdf)

These examples are also referenced in the
[SLATE Users' Guide]((https://www.icl.utk.edu/publications/swan-010)

For compiling, there are two options:

## Option 1: Makefile

Build & install SLATE (see slate/INSTALL.md).
This installs it into a sub-directory of the SLATE source.

    slate>  make
    slate>  make install prefix=install
    slate>  export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:`pwd`/install

Build examples:

    slate>  cd examples
    slate/examples>  make
    slate/examples>  make check

Installation puts the SLATE, BLAS++, and LAPACK++ headers all in the
same include directory, and libraries all in the same lib directory,
which simplifies compiling the examples. The Makefile queries pkg-config
for all settings.

todo: update the C and Fortran API examples.

Examples of SLATE's C and Fortran APIs are in the c_api and fortran
directories.


## Option 2: CMake

Build & install SLATE (see slate/INSTALL.md).
This installs it into a sub-directory of the SLATE source.

    slate>  mkdir build && cd build
    slate/build>  cmake -DCMAKE_INSTALL_PREFIX=../install ..
    slate/build>  make
    slate/build>  make install

Build examples:

    slate/build>  cd ../examples
    slate/examples>  mkdir build && cd build
    slate/examples/build>  cmake -DCMAKE_PREFIX_PATH=`pwd`/../../install ..
    slate/examples/build>  make
    slate/examples/build>  make test

CTest output is in `Testing/Temporary/LastTest.log`.

CMake needs to find the SLATE, BLAS++, and LAPACK++ installations by
setting the `CMAKE_PREFIX_PATH` to the absolute path to the install
directory.

todo: The scalapack example is not yet done in CMake.

todo: The c_api and fortran examples are not yet done in CMake.
