cmake_minimum_required(VERSION 3.8)

project(tester CXX CUDA)

file (GLOB SLATE_TEST_SOURCES
    CONFIGURE_DEPENDS *.cc)

add_executable(slate_test ${SLATE_TEST_SOURCES})

# Check if user supplied testsweeper_DIR on the command line
# Set -Dlibtest_DIR=/path/to/testsweeper
if(testsweeper_DIR)
    message("Found testsweeper_DIR")
else()
    find_path(testsweeper_DIR testsweeperConfig.cmake
        PATHS
        ${CMAKE_INSTALL_PREFIX}/test
        )
    message("testsweeper_DIR: " ${testsweeper_DIR})
endif()

find_package(testsweeper)

if (0)
set(SCALAPACK_VERBOSE ON)
set(BLA_VENDOR CACHE STRING "Intel")
set_property(CACHE BLA_VENDOR PROPERTY STRINGS Intel)
find_package(scalapack)
find_package(SCALAPACK)

endif()

target_link_libraries(tester
    slate
    testsweeper
)

if(SLATE_WITH_MKL)
    target_link_libraries(tester
        mkl_scalapack_lp64
        mkl_blacs_intelmpi_lp64
    )
else()
    target_link_libraries(tester
        scalapack
    )
endif()

