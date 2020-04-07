cmake_minimum_required(VERSION 3.8)

project(SLATE CXX CUDA)
#project(SLATE)

# Do not build the testers by default
option(BUILD_SLATE_TESTS "Build SLATE tests - has more reqs" OFF)

file (GLOB SLATE_SOURCES
    CONFIGURE_DEPENDS  # forces cmake to check sources
    #src/stubs/*.cc
    src/*.cc
    src/core/*.cc
    src/aux/*.cc
    src/internal/*.cc
    src/cuda/*.cu
)

add_library(slate ${SLATE_SOURCES})

set_target_properties (slate PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_include_directories(slate
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        include/slate
        include/slate/internal
        src
        src/internal
    )

find_package(MPI REQUIRED COMPONENTS CXX)

find_package(blaspp REQUIRED)
if(WIP) # work in progress
find_package(blaspp QUIET)
if(NOT blaspp_FOUND)
    message ("  ******  blaspp not found!")
    set (BLASPP_BUILD_TESTS)
    add_subdirectory(blaspp)
else ()
    find_package(blaspp REQUIRED)
endif()
endif()

get_target_property(blaspp_links blaspp INTERFACE_LINK_LIBRARIES)
get_target_property(blaspp_compile_definitions blaspp COMPILE_DEFINITIONS)
get_target_property(blaspp_interface_compile_definitions blaspp INTERFACE_COMPILE_DEFINITIONS)

if ("${blaspp_links}" MATCHES "OpenMP")
    message ("  OpenMP found and linking")
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(slate PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()

#message ("blaspp_compile_definitions ${blaspp_compile_definitions}")
#message ("blaspp_interface_compile_definitions ${blaspp_interface_compile_definitions}")

if ("${blaspp_interface_compile_definitions}" MATCHES "MKL")
    message ("  MKL found - setting define")
    target_compile_definitions(slate PUBLIC SLATE_WITH_MKL)
    set(SLATE_WITH_MKL "TRUE")
endif()

# Set -Dlapackpp_DIR=/path/to/lapackpp or -DLAPACKPP_DIR=/path/to/lapackpp
if(LAPACKPP_DIR)
    message("using user supplied LAPACKPP_DIR: ${LAPACKPP_DIR}")
    #set(LAPACKPP_DIR "${LAPACKPP_DIR}")
else()
    find_path(lapackpp_DIR lapackppConfig.cmake
        PATHS
            ${CMAKE_INSTALL_PREFIX}/lapackpp
    )
    #message("lapackpp_DIR: " ${lapackpp_DIR})
endif()

find_package(lapackpp REQUIRED)
if(WIP) # work in progress
find_package(lapackpp QUIET)
if(NOT lapackpp_FOUND)
    message ("  ******  lapackpp not found!")
    unset (LAPACKPP_BUILD_TESTS)
    add_subdirectory(lapackpp)
else ()
    find_package(lapackpp REQUIRED)
endif()
endif()

# remove mpi flags that are incompatible with nvcc
get_target_property(mpi_cxx_interface_compile_definitions MPI::MPI_CXX INTERFACE_COMPILE_OPTIONS)
#message ("mpi_cxx_interface_compile_definitions: ${mpi_cxx_interface_compile_definitions}")
list (REMOVE_ITEM mpi_cxx_interface_compile_definitions "-fexceptions")
#message ("mpi_cxx_interface_compile_definitions: ${mpi_cxx_interface_compile_definitions}")
set_target_properties(MPI::MPI_CXX PROPERTIES INTERFACE_COMPILE_OPTIONS "${mpi_cxx_interface_compile_definitions}")

target_link_libraries(slate
    PUBLIC
        MPI::MPI_CXX
        blaspp
        lapackpp
        cuda
        cublas
)

# dependency information
if(SLATE_DEBUG)
    include(CMakePrintHelpers)
    cmake_print_properties(TARGETS MPI::MPI_CXX PROPERTIES
        INTERFACE_INCLUDE_DIRS INTERFACE_LINK_LIBRARIES INTERFACE_COMPILE_DEFINITIONS INTERFACE_COMPILE_OPTIONS)
    cmake_print_properties(TARGETS blaspp PROPERTIES
        INTERFACE_INCLUDE_DIRS INTERFACE_LINK_LIBRARIES COMPILE_DEFINITIONS INTERFACE_COMPILE_DEFINITIONS)
    cmake_print_properties(TARGETS lapackpp PROPERTIES
        INTERFACE_INCLUDE_DIRS INTERFACE_LINK_LIBRARIES COMPILE_DEFINITIONS INTERFACE_COMPILE_DEFINITIONS)
    cmake_print_properties(TARGETS slate PROPERTIES
        INTERFACE_INCLUDE_DIRS INTERFACE_LINK_LIBRARIES COMPILE_DEFINITIONS INTERFACE_COMPILE_DEFINITIONS INTERFACE_COMPILE_OPTIONS)
endif()

if (BUILD_SLATE_TESTS)
    add_subdirectory(test)
endif()

install (TARGETS slate
    EXPORT slateTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include
)

export(EXPORT slateTargets
    FILE slateTargets.cmake
)

export (PACKAGE slate)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/slateConfig.cmake.in
    ${CMAKE_CURRENT_SOURCE_DIR}/slateConfig.cmake
    COPYONLY
)


install (EXPORT slateTargets
    FILE slateTargets.cmake
    DESTINATION lib/slate
)

install(
    DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h*"
)

install (
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/slateConfig.cmake
    DESTINATION lib/slate
)