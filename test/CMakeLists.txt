cmake_minimum_required(VERSION 3.8)

project(slate_test CXX CUDA)

file (GLOB SLATE_TEST_SOURCES
    CONFIGURE_DEPENDS *.cc)

add_executable(slate_test ${SLATE_TEST_SOURCES})

# Check if user supplied LIBTEST_DIR on the command line
# Set -Dlibtest_DIR=/path/to/libtest
if(libtest_DIR)
    message("Found libtest_DIR")
else()
    find_path(libtest_DIR libtestConfig.cmake
        PATHS
        ${CMAKE_INSTALL_PREFIX}/test
        )
    message("libtest_DIR: " ${libtest_DIR})
endif()

find_package(libtest)

if (0)
set(SCALAPACK_VERBOSE ON)
set(BLA_VENDOR CACHE STRING "Intel")
set_property(CACHE BLA_VENDOR PROPERTY STRINGS Intel)
find_package(scalapack)
find_package(SCALAPACK)

endif()

target_link_libraries(slate_test
    slate
    libtest
)

if(SLATE_WITH_MKL)
    target_link_libraries(slate_test
        mkl_scalapack_lp64
        mkl_blacs_intelmpi_lp64
    )
else()
    target_link_libraries(slate_test
        scalapack
    )
endif()

