#!/bin/bash -e

maker=$1
gpu=$2

if [ "${maker}" = "cmake" ]; then
    rm -rf build
    mkdir -p build
fi

mydir=`dirname $0`
source $mydir/setup.sh

git submodule update --init

if [ "${maker}" = "make" ]; then
    section "======================================== Make"
    make echo
fi

if [ "${maker}" = "cmake" ]; then
    section "======================================== CMake"
    module load cmake
    cmake -Dcolor=no -DCMAKE_CXX_FLAGS="-Werror" \
          -DCMAKE_INSTALL_PREFIX=${top}/install \
          ..
fi

