# Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

if (color)
    string( ASCII 27 Esc )
    set( ansi_reset    "${Esc}[0m"  )
    set( bold          "${Esc}[1m"  )
    set( not_bold      "${Esc}[22m" )  # "normal"
    set( italic        "${Esc}[3m"  )
    set( not_italic    "${Esc}[23m" )

    set( black         "${Esc}[30m" )
    set( red           "${Esc}[31m" )
    set( green         "${Esc}[32m" )
    set( yellow        "${Esc}[33m" )
    set( blue          "${Esc}[34m" )
    set( magenta       "${Esc}[35m" )
    set( cyan          "${Esc}[36m" )
    set( gray          "${Esc}[37m" )
    set( default_color "${Esc}[39m" )
    set( plain         "${Esc}[39m" )
endif()

#-------------------------------------------------------------------------------
# pad_string( input length output_variable )
# Adds spaces to input up to length and saves to output_variable.
#
function( pad_string input length output_variable )
    string( LENGTH "${input}" len )
    math( EXPR pad_len "${length} - ${len}" )
    if (pad_len LESS 0)
        set( pad_len 0 )
    endif()
    string( REPEAT " " ${pad_len} pad )
    set( ${output_variable} "${input}${pad}" PARENT_SCOPE )
endfunction()

#-------------------------------------------------------------------------------
# debug_try_compile( msg compile_result compile_output )
# Prints compile_result at log level DEBUG (5);
#        compile_output at log level TRACE (6).
#
function( debug_try_compile msg compile_result compile_output )
    message( DEBUG "${msg}: compile_result '${compile_result}'" )
    message( TRACE "compile_output: <<<\n${compile_output}>>>" )
endfunction()

#-------------------------------------------------------------------------------
# debug_try_run( msg compile_result run_result compile_output run_output )
# Prints {compile,run}_result at debug DEBUG (5);
#        {compile,run}_output at debug TRACE (6).
#
function( debug_try_run msg compile_result compile_output run_result run_output )
    message( DEBUG "${msg}: compile_result '${compile_result}', run_result '${run_result}'" )
    message( TRACE "compile_output: '''\n${compile_output}'''" )
    message( TRACE "run_output: '''\n${run_output}'''" )
endfunction()

#-------------------------------------------------------------------------------
# assert( condition )
# Aborts if condition is not true.
function( assert var )
    if (NOT ${var})
        message( FATAL_ERROR "\n${red}Assertion failed: ${var} (value is '${${var}}')${default_color}\n" )
    endif()
endfunction()
