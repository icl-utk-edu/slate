# Can't build testers if ScaLAPACK or TestSweeper are not found.
message( "" )
message( "---------------------------------------- ScaLAPACK" )

# User or Spack can give us SCALAPACK_LIBRARIES;
# otherwise, try to figure it out to match BLAS and MPI.
# todo: more robust FindScaLAPACK package; several exist.
if (NOT DEFINED SCALAPACK_LIBRARIES)
    if ("${blaspp_defines}" MATCHES "HAVE_MKL")
        if ("${blaspp_defines}" MATCHES "BLAS_ILP64")
            set( int_ "ilp64" )
        else()
            set( int_ "lp64" )
        endif()
        if (MPI_CXX_LIBRARY_VERSION_STRING MATCHES "Intel|MPICH")
            # Intel MPI, based on MPICH
            set( SCALAPACK_LIBRARIES "mkl_scalapack_${int_};mkl_blacs_intelmpi_${int_}" )
        else()
            # Open MPI
            set( SCALAPACK_LIBRARIES "mkl_scalapack_${int_};mkl_blacs_openmpi_${int_}" )
        endif()
    else()
        set( SCALAPACK_LIBRARIES "scalapack" )
    endif()
endif()
if (SCALAPACK_LIBRARIES STREQUAL "none")
    message( STATUS "Not using ScaLAPACK" )
else()
    message( STATUS "Using ScaLAPACK library: ${SCALAPACK_LIBRARIES}" )
endif()

# Search for TestSweeper library, if not already included (e.g., in SLATE).
message( "" )
message( "---------------------------------------- TestSweeper" )
message( STATUS "Checking for TestSweeper library" )
if (NOT TARGET testsweeper)
    find_package( testsweeper QUIET )
    if (testsweeper_FOUND)
        message( "   Found TestSweeper library: ${testsweeper_DIR}" )

    elseif (EXISTS "${CMAKE_SOURCE_DIR}/testsweeper/CMakeLists.txt")
        add_subdirectory( "${CMAKE_SOURCE_DIR}/testsweeper"
                          "${CMAKE_BINARY_DIR}/testsweeper" )

    else()
        set( url "https://bitbucket.org/icl/testsweeper" )
        message( "" )
        message( "---------- TestSweeper" )
        message( STATUS "Fetching TestSweeper from ${url}" )
        include( FetchContent )
        FetchContent_Declare( testsweeper GIT_REPOSITORY "${url}" )
        FetchContent_MakeAvailable( testsweeper )
        message( "---------- TestSweeper done" )
        message( "" )
    endif()
else()
    message( "   TestSweeper already included" )
endif()
message( STATUS "TestSweeper done" )
message( "" )

#-------------------------------------------------------------------------------
# In future, if these *.f are eliminated, may still need dummy.f so CMake links
# with Fortran (e.g., -lgfortran), which may be needed for ScaLAPACK.
# What is a better solution?
file(
    GLOB tester_src
    CONFIGURE_DEPENDS
    *.cc
)
if (NOT SCALAPACK_LIBRARIES STREQUAL "none")
    file(
        GLOB tester_f_src
        CONFIGURE_DEPENDS
        *.f
    )
endif()

set( tester "${slate_}tester" )
add_executable(
    ${tester}
    ${tester_src}
    ${tester_f_src}
)

# Use and export -std=c++17.
# CMake inexplicably allows gnu++17 or "decay" to c++11 or 14; prohibit those.
target_compile_features( ${tester} PUBLIC cxx_std_17 )
set_target_properties(
    ${tester} PROPERTIES
    CXX_STANDARD_REQUIRED true  # prohibit < c++17
    CXX_EXTENSIONS false        # prohibit gnu++17
)

target_link_libraries(
    ${tester}
    testsweeper
    slate
)

if (NOT SCALAPACK_LIBRARIES STREQUAL "none")
    target_compile_definitions( ${tester} PUBLIC "SLATE_HAVE_SCALAPACK" )
    target_link_libraries( ${tester} ${SCALAPACK_LIBRARIES} )
endif()

# todo: remove internal includes from testers (e.g., test_posv.cc).
target_include_directories(
    ${tester}
    PRIVATE
        "${CMAKE_SOURCE_DIR}/src"
)

#-------------------------------------------------------------------------------
# Copy run_tests script to build directory.
add_custom_command(
    TARGET ${tester} POST_BUILD
    COMMAND
        cp ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.py
           ${CMAKE_CURRENT_BINARY_DIR}/run_tests.py
)

if (slate_is_project)
    # Add 'make check' target.
    add_custom_target(
        "check"
        COMMAND
            python run_tests.py --quick gesv posv gels heev gesvd
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()
