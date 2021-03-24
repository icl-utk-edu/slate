# CMake script for SLATE library
# repo: http://bitbucket.org/icl/slate
# Requires BLAS++ library from
#     http://bitbucket.org/icl/blaspp
# Requires LAPACK++ library from
#     http://bitbucket.org/icl/lapackpp
# Tests require TestSweeper library from
#     http://bitbucket.org/icl/testsweeper

cmake_minimum_required( VERSION 3.15 )
# 3.1  target_compile_features
# 3.8  target_compile_features( cxx_std_11 )
# 3.14 install( LIBRARY DESTINATION lib ) default
# 3.15 $<$COMPILE_LANG_AND_ID  # optional
# 3.15 message DEBUG, string REPEAT

project(
    slate
    VERSION 2020.10.00
    LANGUAGES CXX Fortran
)

# When built as a sub-project, add a namespace to make targets unique,
# e.g., `make tester` becomes `make slate_tester`.
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set( slate_is_project true )
    set( slate_ "" )
else()
    set( slate_is_project false )
    set( slate_ "slate_" )
endif()

#-------------------------------------------------------------------------------
# Options
if (slate_is_project)
    set( log "" CACHE STRING "Shorthand for CMAKE_MESSAGE_LOG_LEVEL" )
    set_property( CACHE log PROPERTY STRINGS
                  FATAL_ERROR SEND_ERROR WARNING AUTHOR_WARNING DEPRECATION
                  NOTICE STATUS VERBOSE DEBUG TRACE )
    if (log)
        set( CMAKE_MESSAGE_LOG_LEVEL "${log}" )
    endif()
endif()

option( BUILD_SHARED_LIBS "Build shared libraries" true )
option( build_tests "Build test suite" "${slate_is_project}" )
option( color "Use ANSI color output" true )
option( use_cuda "Use CUDA, if available" true )
option( use_mpi "Use MPI, if available" true )
option( use_openmp "Use OpenMP, if available" true )
# todo: option( c_api "Build C API" false )
# todo: option( fortran_api "Build Fortran API. Requires C API." false )

# Default prefix=/opt/slate
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT
    AND slate_is_project)

    set( prefix "/opt/slate" CACHE PATH "Shorthand for CMAKE_INSTALL_PREFIX" )
    set( CMAKE_INSTALL_PREFIX "${prefix}"
         CACHE PATH
         "Install path prefix, prepended onto install directories."
         FORCE
    )
    message( STATUS "Setting CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}" )
    # Append the new CMAKE_INSTALL_PREFIX, since CMake appended the old value.
    # This helps find TestSweeper.
    list( APPEND CMAKE_SYSTEM_PREFIX_PATH ${CMAKE_INSTALL_PREFIX} )
else()
    message( STATUS "Using CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}" )
endif()

# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if ("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif()

# Provide menu of options. (Why doesn't CMake do this?)
set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
              None Debug Release RelWithDebInfo MinSizeRel )

#-----------------------------------
# BLAS options
# SLATE defaults to single-threaded BLAS.
set( blas_threaded "false" CACHE STRING
     "Multi-threaded BLAS? (Passed to BLAS++.)" )
set_property(
    CACHE blas_threaded PROPERTY STRINGS
    "auto" "true" "false" )

message( DEBUG "Settings:
CMAKE_INSTALL_PREFIX   = ${CMAKE_INSTALL_PREFIX}
CMAKE_BUILD_TYPE       = ${CMAKE_BUILD_TYPE}
BUILD_SHARED_LIBS      = ${BUILD_SHARED_LIBS}
build_tests            = ${build_tests}
color                  = ${color}
use_cuda               = ${use_cuda}
use_mpi                = ${use_mpi}
use_openmp             = ${use_openmp}
c_api                  = ${c_api}
fortran_api            = ${fortran_api}
slate_is_project       = ${slate_is_project}
slate_                 = ${slate_}
" )

#-------------------------------------------------------------------------------
# Enforce out-of-source build
string( TOLOWER "${CMAKE_CURRENT_SOURCE_DIR}" source_dir )
string( TOLOWER "${CMAKE_CURRENT_BINARY_DIR}" binary_dir )
if ("${source_dir}" STREQUAL "${binary_dir}")
    message( FATAL_ERROR
    "Compiling SLATE with CMake requires an out-of-source build. To proceed:
    rm -rf CMakeCache.txt CMakeFiles/   # delete files in ${CMAKE_CURRENT_SOURCE_DIR}
    mkdir build
    cd build
    cmake ..
    make" )
endif()

#-------------------------------------------------------------------------------
# Build library.

# todo: these need generation
# src/c_api/*.cc
# src/fortran_api/*.f90

file(
    GLOB libslate_src
    CONFIGURE_DEPENDS  # glob at build time
    src/*.cc
    src/*.f
    src/aux/*.cc
    src/core/*.cc
    src/internal/*.cc
    src/work/*.cc
)
message( DEBUG "libslate_src = ${libslate_src}" )

add_library(
    slate
    ${libslate_src}
)

#--------------------
# lapack_api
file(
    GLOB lapack_api_src
    CONFIGURE_DEPENDS  # glob at build time
    lapack_api/*.cc
)
message( DEBUG "lapack_api_src = ${lapack_api_src}" )

add_library(
    slate_lapack_api
    ${lapack_api_src}
)
target_link_libraries( slate_lapack_api PUBLIC slate )

#--------------------
# scalapack_api
# todo: requires ScaLAPACK
# file(
#     GLOB scalapack_api_src
#     CONFIGURE_DEPENDS  # glob at build time
#     scalapack_api/*.cc
# )
# # todo: getri not finished.
# list( FILTER scalapack_api_src EXCLUDE REGEX "getri" )
# message( DEBUG "scalapack_api_src = ${scalapack_api_src}" )
#
# add_library(
#     slate_scalapack_api
#     ${scalapack_api_src}
# )
# target_link_libraries( slate_scalapack_api PUBLIC slate )
# target_link_libraries( slate_scalapack_api PUBLIC scalapack )

#-------------------------------------------------------------------------------
# CUDA support.
if (NOT use_cuda)
    message( STATUS "User has requested to NOT use CUDA" )
else()
    include( CheckLanguage )
    check_language( CUDA )
endif()
if (CMAKE_CUDA_COMPILER)
    enable_language( CUDA )

    if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
        # CUDA architectures required in CMake 3.18
        # Fermi   20
        # Kepler  30 35 37
        # Maxwell 50 52 53
        # Pascal  60 61 62
        # Volta   70 72
        # Turing  75
        # Ampere  80 86
        # Also -real and -virtual suffixes.
        set( CMAKE_CUDA_ARCHITECTURES "60" CACHE STRING
             "CUDA architectures, as semi-colon separated list of 2-digit numbers, e.g., 60;70;80 for Pascal, Volta, Ampere" )
        set_property( CACHE CMAKE_CUDA_ARCHITECTURES PROPERTY STRINGS
                      30 50 60 70 75 80 )
    endif()

    message( STATUS "Building CUDA kernels." )
    file(
        GLOB libslate_cuda_src
        CONFIGURE_DEPENDS  # glob at build time
        src/cuda/*.cu
    )
    target_sources(
        slate
        PRIVATE
        ${libslate_cuda_src}
    )

    # CMake 3.17 adds find_package( CUDAToolkit )
    # with CUDA::cudart and CUDA::cublas targets.
    # For compatibility with older CMake, for now do search ourselves.
    find_library( cudart_lib cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} )
    find_library( cublas_lib cublas ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} )

    # Some platforms need these to be public libraries.
    target_link_libraries( slate PUBLIC "${cudart_lib}" "${cublas_lib}" )
    target_include_directories(
        slate PUBLIC "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" )
else()
    message( STATUS "No CUDA support in SLATE" )
    target_sources(
        slate
        PRIVATE
        src/stubs/cublas_stubs.cc
        src/stubs/cuda_stubs.cc
    )
    target_compile_definitions( slate PUBLIC "-DSLATE_NO_CUDA" )
endif()

# Include directory.
# During build it's {source}/include; after install it's {prefix}/include.
target_include_directories(
    slate
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# OpenMP support.
if (NOT use_openmp)
    message( STATUS "User has requested to NOT use OpenMP" )
else()
    find_package( OpenMP )
endif()
if (OpenMP_CXX_FOUND)
    target_link_libraries( slate PUBLIC "OpenMP::OpenMP_CXX" )
else()
    message( STATUS "No OpenMP support in SLATE" )
    target_sources(
        slate
        PRIVATE
        src/stubs/openmp_stubs.cc
    )
endif()

# MPI support.
# CXX means MPI C API being usable from C++, not the MPI-2 C++ API.
if (NOT use_mpi)
    message( STATUS "User has requested to NOT use MPI" )
else()
    set( MPI_DETERMINE_LIBRARY_VERSION true )
    find_package( MPI COMPONENTS CXX )
endif()
if (MPI_FOUND)
    target_link_libraries( slate PUBLIC MPI::MPI_CXX )

    # Remove Open MPI flags that are incompatible with nvcc (-fexceptions).
    get_target_property( mpi_cxx_defines MPI::MPI_CXX INTERFACE_COMPILE_OPTIONS )
    message( DEBUG "mpi_cxx_defines = '${mpi_cxx_defines}'" )
    list( REMOVE_ITEM mpi_cxx_defines "-fexceptions" )
    message( DEBUG "mpi_cxx_defines = '${mpi_cxx_defines}'" )
    set_target_properties( MPI::MPI_CXX PROPERTIES INTERFACE_COMPILE_OPTIONS
                           "${mpi_cxx_defines}" )
else()
    message( STATUS "No MPI support in SLATE" )
    target_sources(
        slate
        PRIVATE
        src/stubs/mpi_stubs.cc
    )
    target_compile_definitions( slate PUBLIC "-DSLATE_NO_MPI" )
endif()

# Get git commit id.
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process( COMMAND git rev-parse --short HEAD
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                     OUTPUT_VARIABLE slate_id )
    string( STRIP "${slate_id}" slate_id )
    message( STATUS "slate_id = ${slate_id}" )
    target_compile_definitions(
        slate PRIVATE SLATE_ID="${slate_id}" )
endif()

# Use and export -std=c++17; don't allow -std=gnu++17 extensions.
# The `target_compile_features( slate PUBLIC cxx_std_17 )` syntax
# doesn't seem to allow setting CUDA_STANDARD separately, and slightly
# older CMake (3.17) doesn't recognize CUDA_STANDARD=17.
set_target_properties(
    slate PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 11
    CXX_STANDARD_REQUIRED true
    CXX_EXTENSIONS false
)

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.15)
    # Conditionally add -Wall. See CMake tutorial.
    set( gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU>" )
    target_compile_options(
        slate PRIVATE "$<${gcc_like_cxx}:$<BUILD_INTERFACE:-Wall>>" )
endif()

#-------------------------------------------------------------------------------
# Search for BLAS library, if not already included.
message( "" )
message( "---------------------------------------- BLAS++" )
if (NOT TARGET blaspp)
    find_package( blaspp QUIET )
    if (blaspp_FOUND)
        message( "   Found BLAS++: ${blaspp_DIR}" )
    elseif (EXISTS "${CMAKE_SOURCE_DIR}/blaspp/CMakeLists.txt")
        set( build_tests_save "${build_tests}" )
        set( build_tests "false" )

        add_subdirectory( "blaspp" )

        set( build_tests "${build_tests_save}" )
        set( blaspp_DIR "${CMAKE_BINARY_DIR}/blaspp" )
    else()
        message( FATAL_ERROR "blaspp/CMakeLists.txt doesn't exist. Use:\n"
                 "    git submodule update --init\n"
                 "to checkout submodules." )
    endif()
else()
    message( "   BLAS++ already included" )
endif()
message( STATUS "BLAS++ done" )

# Search for LAPACK library, if not already included.
message( "" )
message( "---------------------------------------- LAPACK++" )
if (NOT TARGET lapackpp)
    find_package( lapackpp QUIET )
    if (lapackpp_FOUND)
        message( "   Found LAPACK++: ${lapackpp_DIR}" )

    elseif (EXISTS "${CMAKE_SOURCE_DIR}/lapackpp/CMakeLists.txt")
        set( build_tests_save "${build_tests}" )
        set( build_tests "false" )

        add_subdirectory( "lapackpp" )

        set( build_tests "${build_tests_save}" )
        set( lapackpp_DIR "${CMAKE_BINARY_DIR}/lapackpp" )

    else()
        message( FATAL_ERROR "lapackpp/CMakeLists.txt doesn't exist. Use:\n"
                 "    git submodule update --init\n"
                 "to checkout submodules." )
    endif()
else()
    message( "   LAPACK++ already included" )
endif()
message( STATUS "LAPACK++ done" )

target_link_libraries( slate PUBLIC blaspp lapackpp )

if ("${blaspp_defines}" MATCHES "HAVE_MKL")
    target_compile_definitions( slate PUBLIC "-DSLATE_WITH_MKL" )
endif()

#-------------------------------------------------------------------------------
if (build_tests)
    add_subdirectory( test )
endif()

#-------------------------------------------------------------------------------
# Add 'make lib' target.
if (slate_is_project)
    add_custom_target( lib DEPENDS slate )
endif()

#-------------------------------------------------------------------------------
# GNU Filesystem Conventions
include( GNUInstallDirs )
set( install_configdir "${CMAKE_INSTALL_LIBDIR}/slate" )

# Install library and add to <package>Targets.cmake
install(
    TARGETS slate slate_lapack_api
    EXPORT slateTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)

# Install header files
# todo: also Fortran API .mod file
install(
    # / copies contents, not directory itself
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING REGEX "\\.(h|hh)$"
)

# Install <package>Targets.cmake
install(
    EXPORT slateTargets
    DESTINATION "${install_configdir}"
)

# Also export <package>Targets.cmake in build directory
export(
    EXPORT slateTargets
    FILE "slateTargets.cmake"
)

# Install <package>Config.cmake and <package>ConfigVersion.cmake,
# to enable find_package( <package> ).
include( CMakePackageConfigHelpers )
configure_package_config_file(
    "slateConfig.cmake.in"
    "slateConfig.cmake"
    INSTALL_DESTINATION "${install_configdir}"
)
write_basic_package_version_file(
    "slateConfigVersion.cmake"
    VERSION "${slate_VERSION}"
    COMPATIBILITY AnyNewerVersion
)
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/slateConfig.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/slateConfigVersion.cmake"
    DESTINATION "${install_configdir}"
)
