#-------------------------------------------------------------------------------
file(
    GLOB unit_src
    CONFIGURE_DEPENDS
    test_*.cc
)
if (NOT ${c_api})
    list( FILTER unit_src EXCLUDE REGEX "c_api" )
endif()

# Use -std=c++17 on all unit testers.
# CMake inexplicably allows gnu++17 or "decay" to c++11 or 14; prohibit those.
set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED true )  # prohibit < c++17
set( CMAKE_CXX_EXTENSIONS false )        # prohibit gnu++17

foreach (src IN LISTS unit_src)
    get_filename_component( tester "${src}" NAME_WE )
    add_executable( ${tester} ${src} unit_test.cc )
    target_include_directories( ${tester} PRIVATE "${CMAKE_SOURCE_DIR}/src" )
    target_link_libraries( ${tester} slate testsweeper )
endforeach()

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
        unit_check
        COMMAND
            python3 run_tests.py
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
    add_dependencies( check unit_check )
endif()
