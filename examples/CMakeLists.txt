# CXX compiler must match the one used to compiler BLAS++.
# Set it in your environment.

cmake_minimum_required( VERSION 3.8 )

project(
    slate_example
    LANGUAGES CXX
)

#-------------------------------------------------------------------------------
# Enforce out-of-source build
string( TOLOWER "${CMAKE_CURRENT_SOURCE_DIR}" source_dir )
string( TOLOWER "${CMAKE_CURRENT_BINARY_DIR}" binary_dir )
if ("${source_dir}" STREQUAL "${binary_dir}")
    message( FATAL_ERROR
    "Compiling with CMake requires an out-of-source build. To proceed:
    rm -rf CMakeCache.txt CMakeFiles/   # delete files in ${CMAKE_CURRENT_SOURCE_DIR}
    mkdir build
    cd build
    cmake ..
    make" )
endif()

#-------------------------------------------------------------------------------
# todo: fix slate package so it loads blaspp, lapackpp, mpi.
set( MPI_DETERMINE_LIBRARY_VERSION true )
find_package( MPI COMPONENTS CXX )
find_package( blaspp    REQUIRED )
find_package( lapackpp  REQUIRED )
find_package( slate     REQUIRED )

set(
    src_list
    ex01_matrix.cc
    ex02_conversion.cc
    ex03_submatrix.cc
    ex04_norm.cc
    ex05_blas.cc
    ex06_linear_system_lu.cc
    ex07_linear_system_cholesky.cc
    ex08_linear_system_indefinite.cc
    ex09_least_squares.cc
    ex10_svd.cc
    ex11_hermitian_eig.cc
    ex12_generalized_hermitian_eig.cc
    ex13_non_uniform_block_size.cc
    # ex14_scalapack_gemm.cc  # requires -lscalapack
)

enable_testing()

# Get precisions to test. See .github/workflows/test.sh
set( test_args $ENV{test_args} )
if (NOT test_args)
    set( test_args "s d c z" )
endif()
string( REPLACE " " ";" test_args ${test_args} ) # convert to list

foreach (src IN LISTS src_list)
    string( REPLACE ".cc" ""     exe "${src}" )
    string( REPLACE ".cc" ".txt" txt "${src}" )
    message( "add_executable( ${exe}, ${src} )" )

    add_executable( "${exe}" "${src}" )
    target_link_libraries( "${exe}" slate )

    list( APPEND txt_list "${txt}" )
    list( APPEND exe_list "${exe}" )

    # ctest can't redirect the output of each test to a file.
    # It saves all the output to Testing/Temporary/LastTest.log
    # Using `make test ARGS="-V"` prints output to terminal,
    # with test ID prefixed.
    add_test( NAME "${exe}" COMMAND mpirun -np 4 ./${exe} ${test_args} )
endforeach()

set_tests_properties( ${exe_list} PROPERTIES RUN_SERIAL true )
