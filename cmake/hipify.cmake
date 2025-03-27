# Copyright (c) 2017-2025, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#-------------------------------------------------------------------------------
# Runs hipify to generate HIP code from CUDA code, if the CUDA code has changed.
# Tracks changes using md5sum of the CUDA code.
# Assumes CUDA files end in cuda/*.cu or cuda/*.cuh,
# and creates HIP files ending in hip/*.hip or hip/*.hip.hh.
#
# hipify( src/cuda/foo.cu src/cuda/foo.cuh hip_src ) generates:
#
#   src/cuda/foo.cu.md5     # output of `m5dsum src/cuda/foo.cu`
#   src/hip/foo.hip         # output of hipify-perl src/cuda/foo.cu
#   src/hip/foo.hip.dep     # copy of foo.cu.md5
#
#   src/cuda/foo.cuh.md5    # output of `m5dsum src/cuda/foo.cuh`
#   src/hip/foo.hip.hh      # output of hipify-perl src/cuda/foo.cuh
#   src/hip/foo.hip.hh.dep  # copy of foo.cuh.md5
#
# And will set hip_src = src/hip/foo.hip.
# It excludes headers from hip_src.
#
function( hipify cuda_src output_hip_src )

    message( DEBUG "hipify: cuda_src='${cuda_src}'" )

    # Convert CUDA files to HIP files, if not up-to-date.
    foreach (file_cu ${cuda_src})
        # CMake needs file_cu to be abs path.
        if (NOT file_cu MATCHES "^/")
            set( file_cu "${CMAKE_SOURCE_DIR}/${file_cu}" )
        endif()

        string( REGEX REPLACE [[/cuda/([a-zA-Z0-9_]+)\.cu$]]  [[/hip/\1.hip]]    file_hip "${file_cu}" )
        string( REGEX REPLACE [[/cuda/([a-zA-Z0-9_]+)\.cuh$]] [[/hip/\1.hip.hh]] file_hip "${file_hip}" )
        if ("${file_hip}" MATCHES [[\.hip$]])
            list( APPEND hip_src "${file_hip}" )
        endif()

        # CMake needs abs paths, but the md5sum below needs
        # a relative path for the .cu file.
        string( REGEX REPLACE "^${CMAKE_SOURCE_DIR}/" "" file_cu_relative "${file_cu}" )

        message( DEBUG "file_cu_relative  ${file_cu_relative}" )
        message( DEBUG "file_cu  ${file_cu}" )
        message( DEBUG "file_hip ${file_hip}" )

        # Automatically generate HIP source from CUDA source.
        # As in the Makefile, this applies the given build rule ($cmd)
        # only if the md5 sums of the target's dependency ($file_cu.md5)
        # doesn't match that stored in the target's dep file
        # ($file_hip.dep). If the target ($file_hip) is already up-to-date
        # based on md5 sums, its timestamp is updated so make will
        # recognize it as up-to-date. Otherwise, the target is built and
        # its dep file updated. Instead of depending on the src file,
        # the target depends on the md5 file of the src file.
        string(
            CONCAT cmd
            "if [ -e ${file_hip} ]"
            "   && diff ${file_cu}.md5 ${file_hip}.dep > /dev/null 2>&1; then"
            "    echo '${file_hip} is up-to-date based on md5sum.';"
            "    touch ${file_hip};"
            "else"
            "    echo '${file_hip} is out-of-date based on md5sum.';"
            "    echo 'hipify-perl ${file_cu} > ${file_hip}';"
            "          hipify-perl ${file_cu} > ${file_hip};"
            "    sed -i -e 's/\.cuh/.hip.hh/g' ${file_hip};"
            "    cp ${file_cu}.md5 ${file_hip}.dep;"
            "fi"
        )
        message( DEBUG "cmd <${cmd}>" )

        message( DEBUG "cu  ${file_cu}\n"
                       " => ${file_hip}" )
        add_custom_command(
            OUTPUT   "${file_hip}"
            DEPENDS  "${file_cu}.md5"
            VERBATIM
            COMMAND  sh -c "${cmd}"
        )

        message( DEBUG "md5 ${file_cu}\n"
                       " => ${file_cu}.md5\n in ${CMAKE_SOURCE_DIR}" )
        add_custom_command(
            OUTPUT   "${file_cu}.md5"
            DEPENDS  "${file_cu}"
            VERBATIM
            # md5sum needs ${file_cu_relative} to be relative to
            # CMAKE_SOURCE_DIR so output matches .dep files.
            COMMAND  md5sum "${file_cu_relative}" > "${file_cu}.md5"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        )
        message( DEBUG "" )
    endforeach()

    set( ${output_hip_src} "${hip_src}" PARENT_SCOPE )
    message( DEBUG "hipify: hip_src='${hip_src}'" )

endfunction()
