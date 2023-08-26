# Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
#
# CMake script for SLATE library. See INSTALL.md for directions.

cmake_minimum_required( VERSION 3.18 )
# 3.1  target_compile_features
# 3.8  target_compile_features( cxx_std_17 )
# 3.14 install( LIBRARY DESTINATION lib ) default
# 3.15 $<$COMPILE_LANG_AND_ID  # optional
# 3.15 message DEBUG, string REPEAT
# 3.17 find_package( CUDAToolkit )
# 3.18 CMAKE_CUDA_ARCHITECTURES

project(
    slate
    VERSION 2023.08.25
    LANGUAGES CXX Fortran
)

include( CheckCXXCompilerFlag )

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
option( use_mpi "Use MPI, if available" true )
option( use_openmp "Use OpenMP, if available" true )
option( c_api "Build C API" false )
# todo: option( fortran_api "Build Fortran API. Requires C API." false )

set( gpu_backend "auto" CACHE STRING "GPU backend to use" )
set_property( CACHE gpu_backend PROPERTY STRINGS
              auto cuda hip sycl none )

# After setting $color.
include( "cmake/util.cmake" )

# Recognize CTest's BUILD_TESTING flag. (Quotes required.)
if (NOT "${BUILD_TESTING}" STREQUAL "")
    set( build_tests "${BUILD_TESTING}" )
endif()

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
list( FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
      "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" is_system_dir )
if ("${is_system_dir}" STREQUAL "-1")
    list( APPEND CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" )
endif()

# Provide menu of options. (Why doesn't CMake do this?)
set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
              None Debug Release RelWithDebInfo MinSizeRel )

# CUDA architectures required in CMake 3.18
# Fermi   20        # CUDA 3.2 to 8
# Kepler  30 35 37  # CUDA 5 to 10; sm_35 and sm_37 deprecated in CUDA 11
# Maxwell 50 52 53  # CUDA 6 to 11
# Pascal  60 61 62  # CUDA >= 8
# Volta   70 72     # CUDA >= 9
# Turing  75        # CUDA >= 10
# Ampere  80 86 87  # CUDA >= 11.1
# Hopper  90        # CUDA >= 12
# Also -real and -virtual suffixes.
set( CMAKE_CUDA_ARCHITECTURES "60" CACHE STRING
     "CUDA architectures, as semi-colon separated list of numbers, e.g., 60;70;80 for Pascal, Volta, Ampere" )
set_property( CACHE CMAKE_CUDA_ARCHITECTURES PROPERTY STRINGS
              30 35 50 60 70 75 80 90 )

#-----------------------------------
message( DEBUG "Settings:
CMAKE_INSTALL_PREFIX   = ${CMAKE_INSTALL_PREFIX}
CMAKE_BUILD_TYPE       = ${CMAKE_BUILD_TYPE}
BUILD_SHARED_LIBS      = ${BUILD_SHARED_LIBS}
build_tests            = ${build_tests}
color                  = ${color}
gpu_backend            = ${gpu_backend}
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
# src/fortran_api/*.f90

message( "---------------------------------------- C API: c_api = ${c_api}" )
if (c_api)
    find_package(Python COMPONENTS Interpreter)
    set( cmd ${Python_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/c_api/generate_util.py
        ${CMAKE_CURRENT_SOURCE_DIR}/include/slate/c_api/types.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/slate/c_api/util.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/src/c_api/util.cc )
    execute_process( COMMAND ${cmd} )

    set( cmd ${Python_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/c_api/generate_matrix.py
        ${CMAKE_CURRENT_SOURCE_DIR}/include/slate/Tile.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/include/slate/c_api/matrix.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/c_api/matrix.cc )
    execute_process( COMMAND ${cmd} )

    set( cmd ${Python_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/c_api/generate_wrappers.py
        ${CMAKE_CURRENT_SOURCE_DIR}/src/c_api/wrappers.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/include/slate/c_api/wrappers.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/c_api/wrappers_precisions.cc )
    execute_process( COMMAND ${cmd} )

    file(
        GLOB c_api_src
        CONFIGURE_DEPENDS  # glob at build time
        src/c_api/util.cc
        src/c_api/matrix.cc
        src/c_api/wrappers.cc
        src/c_api/wrappers_precisions.cc
        )
endif()

file(
    GLOB libslate_src
    CONFIGURE_DEPENDS  # glob at build time
    src/*.cc
    src/*.f
    src/auxiliary/*.cc
    src/core/*.cc
    src/device/*.cc
    src/internal/*.cc
    src/work/*.cc
    ${c_api_src}
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

# Use and export -std=c++17.
# CMake inexplicably allows gnu++17 or "decay" to c++11 or 14; prohibit those.
# target_compile_features( slate PUBLIC cuda_std_11 ) doesn't seem to work:
# CUDA nvcc gets -std=c++17, which requires CUDA >= 11.0.
target_compile_features( slate PUBLIC cxx_std_17 )
set_target_properties(
    slate PROPERTIES
    CUDA_STANDARD 11
    CXX_STANDARD_REQUIRED true  # prohibit < c++17
    CXX_EXTENSIONS false        # prohibit gnu++17
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.15)
    # Conditionally add -Wall. See CMake tutorial.
    set( gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU>" )
    target_compile_options(
        slate PRIVATE "$<${gcc_like_cxx}:$<BUILD_INTERFACE:-Wall>>" )
endif()

#-------------------------------------------------------------------------------
# OpenMP support.
# Needed before BLAS++.
message( "" )
message( "---------------------------------------- OpenMP: use_openmp = ${use_openmp}" )
set( slate_use_openmp false )  # output in slateConfig.cmake.in
if (NOT use_openmp)
    message( STATUS "User has requested to NOT use OpenMP" )
else()
    find_package( OpenMP )
    if (OpenMP_CXX_FOUND)
        target_link_libraries( slate PUBLIC "OpenMP::OpenMP_CXX" )
        set( slate_use_openmp true )
    else()
        message( STATUS "No OpenMP support in SLATE" )
        target_sources(
            slate
            PRIVATE
            src/stubs/openmp_stubs.cc
        )
    endif()
endif()

#-------------------------------------------------------------------------------
# MPI support.
# CXX means MPI C API being usable from C++, not the MPI-2 C++ API.
message( "" )
message( "---------------------------------------- MPI: use_mpi = ${use_mpi}" )
set( slate_use_mpi false )  # output in slateConfig.cmake.in
if (NOT use_mpi)
    message( STATUS "User has requested to NOT use MPI" )
else()
    set( MPI_DETERMINE_LIBRARY_VERSION true )
    find_package( MPI COMPONENTS CXX )
endif()
if (MPI_FOUND)
    get_target_property( mpi_lib MPI::MPI_CXX INTERFACE_LINK_LIBRARIES )
    message( STATUS "Found MPI: ${mpi_lib}" )

    target_link_libraries( slate PUBLIC MPI::MPI_CXX )
    set( slate_use_mpi true )

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

#-------------------------------------------------------------------------------
# SYCL support.
message( "" )
message( "---------------------------------------- oneMKL-SYCL: gpu_backend = ${gpu_backend}" )
set( slate_use_sycl false )  # output in slateConfig.cmake.in
if (gpu_backend MATCHES "^(sycl|auto)$")
    message( STATUS "${bold}Looking for SYCL${not_bold} (gpu_backend = ${gpu_backend})" )
    if (TARGET MKL::MKL_DPCPP) # Search for MKL only if not already been found
        set( MKL_FOUND true )
    endif()
    if (NOT MKL_FOUND) # Search for MKL only if not already been found
        if (gpu_backend STREQUAL "sycl")
            find_package( MKL CONFIG REQUIRED HINTS "$ENV{MKL_ROOT}" )
        else()
            find_package( MKL CONFIG QUIET HINTS "$ENV{MKL_ROOT}" )
        endif()
    endif()
    # message(STATUS "Available targets: ${MKL_IMPORTED_TARGETS}")

    # Check if compiler supports the SYCL flag
    check_cxx_compiler_flag( "-fsycl" FSYCL_SUPPORT )

    # If oneMKL is found and the compiler supports SYCL then
    # enable oneMKL-SYCL-device support
    if (MKL_FOUND AND FSYCL_SUPPORT)
        set( gpu_backend "sycl" )
        set( slate_use_sycl true )

        # Avoid "comparison with NaN" warnings from the SYCL
        # Intel-IntelLLVM compiler while compiling omptarget offload
        # routines. (the compiler uses fast floating point mode by
        # default).
        target_compile_options( slate PRIVATE "-fp-model=precise" )

        # -Wno-unused-command-line-argument avoids
        # icpx warning: -Wl,-rpath,...: 'linker' input unused.
        #
        # -Wno-c99-extensions avoids
        # icpx warning: '_Complex' is a C99 extension.
        #
        # -Wno-pass-failed avoids (on src/omptarget/device_transpose.cc)
        # icpx warning: loop not vectorized.
        #
        target_compile_options(
            slate PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-command-line-argument>"
            "$<$<COMPILE_LANGUAGE:CXX>:-Wno-c99-extensions>"
            "$<$<COMPILE_LANGUAGE:CXX>:-Wno-pass-failed>" )

        # Intel icpx options for OpenMP offload.
        target_compile_options( slate PRIVATE "-fopenmp-targets=spir64" )
        target_link_options(    slate PRIVATE "-fopenmp-targets=spir64" )

        file(
            GLOB libslate_omptarget_src
            CONFIGURE_DEPENDS  # glob at build time
            src/omptarget/*.cc
        )
        target_sources(
            slate
            PRIVATE
            ${libslate_omptarget_src}
        )

        target_link_libraries( slate PUBLIC -lmkl_sycl -lsycl -lOpenCL )
        message( STATUS "Building SYCL support" )
    elseif (gpu_backend STREQUAL "sycl")
        message( FATAL_ERROR "${red}SYCL compiler not found${plain}" )
    else()
        message( STATUS "${red}No SYCL support: oneMKL or SYCL compiler not found${plain}" )
    endif()
else()
    message( STATUS "${red}No SYCL support: gpu_backend = ${gpu_backend}${plain}" )
endif()

#-------------------------------------------------------------------------------
# BLAS options
# Set blas_threaded support (single-threaded, multi-threaded)
# Note: oneMKL requires multi-threaded BLAS.
if (slate_use_sycl AND MKL_FOUND)
    # oneMKL requires using threaded-blas
    set( blas_threaded "true" CACHE STRING
         "Multi-threaded BLAS? (Passed to BLAS++.)" )
else()
    # otherwise use sequential-blas
    set( blas_threaded "false" CACHE STRING
         "Multi-threaded BLAS? (Passed to BLAS++.)" )
endif()
 set_property(
     CACHE blas_threaded PROPERTY STRINGS
     "auto" "true" "false" )
message( DEBUG "blas_threaded = ${blas_threaded}" )

#-------------------------------------------------------------------------------
# Search for BLAS library, if not already included.
# Needed before ROCm, for include paths.
message( "" )
message( "---------------------------------------- BLAS++" )
if (NOT TARGET blaspp)
    find_package( blaspp QUIET )
    if (blaspp_FOUND)
        message( STATUS "   Found BLAS++: ${blaspp_DIR}" )
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
    message( STATUS "   BLAS++ already included" )
endif()
message( STATUS "BLAS++ done" )

#-------------------------------------------------------------------------------
# Search for LAPACK library, if not already included.
message( "" )
message( "---------------------------------------- LAPACK++" )
if (NOT TARGET lapackpp)
    find_package( lapackpp QUIET )
    if (lapackpp_FOUND)
        message( STATUS "   Found LAPACK++: ${lapackpp_DIR}" )

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
    message( STATUS "   LAPACK++ already included" )
endif()
message( STATUS "LAPACK++ done" )

target_link_libraries( slate PUBLIC blaspp lapackpp )

#-------------------------------------------------------------------------------
# CUDA support.
message( "" )
message( "---------------------------------------- GPU support" )

set( slate_use_cuda false )  # output in slateConfig.cmake.in
if (gpu_backend MATCHES "^(auto|cuda)$")
    message( STATUS "${bold}Looking for CUDA${not_bold} (gpu_backend = ${gpu_backend})" )
    if (gpu_backend STREQUAL "cuda")
        find_package( CUDAToolkit REQUIRED )
    else()
        find_package( CUDAToolkit QUIET )
    endif()
    if (CUDAToolkit_FOUND)
        set( gpu_backend "cuda" )
        set( slate_use_cuda true )

        # Some platforms need these to be public libraries.
        target_link_libraries(
            slate PUBLIC CUDA::cudart CUDA::cublas )

        enable_language( CUDA )
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

        message( STATUS "${blue}Building CUDA support${plain}" )
    else()
        message( STATUS "${red}No CUDA support: CUDA not found${plain}" )
    endif()
else()
    message( STATUS "${red}No CUDA support: gpu_backend = ${gpu_backend}${plain}" )
endif()

#-------------------------------------------------------------------------------
# HIP/ROCm support.
message( "" )
set( slate_use_hip false )  # output in slateConfig.cmake.in
if (gpu_backend MATCHES "^(auto|hip)$")
    message( STATUS "${bold}Looking for HIP/ROCm${not_bold} (gpu_backend = ${gpu_backend})" )
    if (gpu_backend STREQUAL "hip")
        find_package( HIP REQUIRED )
    else()
        find_package( HIP QUIET )
    endif()
endif()
if (HIP_FOUND)
    message( STATUS "Building HIP/ROCm support in SLATE" )

    # In their official docs, they have 'hip' and 'HIP' as different packages,
    # which both need to be found
    # https://github.com/ROCm-Developer-Tools/HIP/issues/1029
    find_package( hip REQUIRED )
    find_package( rocblas REQUIRED )
    set( slate_use_hip true )
    set( gpu_backend "hip" )

    # For ROCm >= 3.5, wipe hip-clang specific interface options which are propagated
    set_target_properties( hip::device PROPERTIES INTERFACE_COMPILE_OPTIONS "-fPIC" )
    set_target_properties( hip::device PROPERTIES INTERFACE_LINK_LIBRARIES "hip::host" )

    # First, get the HIP source's base files (which are CUDA sources, needing
    # to be hipified)
    file(
        GLOB libslate_hip_src_cuda
        CONFIGURE_DEPENDS  # glob at build time
        src/cuda/*.cu
        src/cuda/*.cuh
    )
    message( DEBUG "libslate_hip_src_cuda ${libslate_hip_src_cuda}" )

    # Make the object directory
    file( MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/src/hip" )

    # Generate BLAS++ and LAPACK++ include flags
    get_target_property( blaspp_inc   blaspp   INTERFACE_INCLUDE_DIRECTORIES )
    get_target_property( lapackpp_inc lapackpp INTERFACE_INCLUDE_DIRECTORIES )
    get_target_property( rocblas_inc roc::rocblas INTERFACE_INCLUDE_DIRECTORIES )
    set( dirs ${blaspp_inc} )
    list( APPEND dirs ${lapackpp_inc} )
    list( APPEND dirs ${rocblas_inc} )
    list( REMOVE_DUPLICATES dirs )
    set( inc_ "" )
    foreach (dir IN LISTS dirs)
        list( APPEND inc_ $<$<BOOL:${dir}>:-I${dir}> )
    endforeach()
    message( DEBUG "blaspp_inc   = '${blaspp_inc}'" )
    message( DEBUG "lapackpp_inc = '${lapackpp_inc}'" )
    message( DEBUG "rocblas_inc  = '${rocblas_inc}'" )
    message( DEBUG "dirs = '${dirs}'" )
    message( DEBUG "inc_ = '${inc_}'" )

    # Generate definitions list
    get_target_property( defs_ slate COMPILE_DEFINITIONS )
    set( Ddefs_ "" )
    foreach (def_ ${defs_})
        list( APPEND Ddefs_ "-D${def_}" )
    endforeach()
    message( DEBUG "defs_  = '${defs_}'" )
    message( DEBUG "Ddefs_ = '${Ddefs_}'" )

    # Iterate over CUDA files
    foreach (src_cu ${libslate_hip_src_cuda})
        # Get name without directory, and replace extension.
        get_filename_component( name "${src_cu}" NAME )
        set( src_cu_relative  "src/cuda/${name}" )
        string( REGEX REPLACE ".cu$"  ".hip.cc" name "${name}"  )
        string( REGEX REPLACE ".cuh$" ".hip.hh" name "${name}" )
        set( src_hip "${CMAKE_SOURCE_DIR}/src/hip/${name}"   )
        set( obj_hip "src/hip/${name}.o" )  # in binary dir

        message( DEBUG "src_cu_relative ${src_cu_relative}" )
        message( DEBUG "src_cu  ${src_cu}" )
        message( DEBUG "src_hip ${src_hip}" )
        message( DEBUG "obj_hip ${obj_hip}" )
        message( DEBUG "" )

        # Automatically generate HIP source from CUDA source.
        # As in the Makefile, this applies the given build rule ($cmd)
        # only if the md5 sums of the target's dependency ($src_cu.md5)
        # doesn't match that stored in the target's dep file
        # ($src_hip.dep). If the target ($src_hip) is already up-to-date
        # based on md5 sums, its timestamp is updated so make will
        # recognize it as up-to-date. Otherwise, the target is built and
        # its dep file updated. Instead of depending on the src file,
        # the target depends on the md5 file of the src file.
        string(
            CONCAT cmd
            "if [ -e ${src_hip} ]"
            "   && diff ${src_cu}.md5 ${src_hip}.dep > /dev/null 2>&1; then"
            "    echo '${src_hip} is up-to-date based on md5sum.';"
            "    touch ${src_hip};"
            "else"
            "    echo '${src_hip} is out-of-date based on md5sum.';"
            "    echo 'hipify-perl ${src_cu} > ${src_hip}';"
            "          hipify-perl ${src_cu} > ${src_hip};"
            "    sed -i -e 's/\.cuh/.hip.hh/g' ${src_hip};"
            "    cp ${src_cu}.md5 ${src_hip}.dep;"
            "fi"
        )
        message( DEBUG "cmd <${cmd}>" )

        message( "cu  ${src_cu}\n => ${src_hip}" )
        add_custom_command(
            OUTPUT   "${src_hip}"
            DEPENDS  "${src_cu}.md5"
            VERBATIM
            COMMAND  sh -c "${cmd}"
        )

        message( "md5 ${src_cu}\n => ${src_cu}.md5\n in ${CMAKE_SOURCE_DIR}" )
        add_custom_command(
            OUTPUT   "${src_cu}.md5"
            DEPENDS  "${src_cu}"
            VERBATIM
            # md5sum needs ${src_cu_relative} to be relative to
            # CMAKE_SOURCE_DIR to match .dep files.
            COMMAND  md5sum "${src_cu_relative}" > "${src_cu}.md5"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        )

        # Compile hip.cc source (not hip.hh headers).
        if (${src_hip} MATCHES ".hip.cc$")
            message( "hcc ${src_hip}\n && src/hip/device_util.hip.hh\n => ${obj_hip}" )
            add_custom_command(
                OUTPUT  "${obj_hip}"
                DEPENDS "${src_hip}"
                        "src/hip/device_util.hip.hh"
                VERBATIM
                COMMAND pwd
                COMMAND hipcc ${Ddefs_} ${inc_}
                        "-I${CMAKE_SOURCE_DIR}/include"
                        -fPIC -c -o ${obj_hip} ${src_hip}
            )

            # Add the .o as a 'source' (even though it is already compiled)
            target_sources(
                slate
                PRIVATE
                "${obj_hip}"
            )
        endif()
    endforeach()
else()
    message( STATUS "${red}No HIP/ROCm support: gpu_backend = ${gpu_backend}${plain}" )
endif()

#-------------------------------------------------------------------------------
# Add 'make lib' target.
if (slate_is_project)
    add_custom_target( lib DEPENDS slate )
endif()

#-------------------------------------------------------------------------------
if (build_tests)
    add_subdirectory( test )
    add_subdirectory( unit_test )
endif()

#-------------------------------------------------------------------------------
# Install rules.
# GNU Filesystem Conventions
include( GNUInstallDirs )
if (WIN32)
    set( install_configdir "slate" )
else()
    set( install_configdir "${CMAKE_INSTALL_LIBDIR}/cmake/slate" )
endif()

# Install library and add to <package>Targets.cmake
install(
    TARGETS slate slate_lapack_api
    EXPORT slateTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
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
