#!/bin/bash -x

maker=$1
device=$2

mydir=$(dirname $0)
source ${mydir}/setup_env.sh

# Instead of exiting on the first failed test (bash -e),
# run all the tests and accumulate failures into $err.
err=0

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0

print "======================================== Unit tests"
cd unit_test

./run_tests.py --timeout 300 --xml ${top}/report-unit-${maker}.xml
(( err += $? ))

print "======================================== Tests"
cd ../test

target="d"
origin="d"
if [ "$device" = "cpu" ]; then
    target="t"
    origin="s"
    tests=""
else
    tests=""
fi

if [ "$maker" = "cmake" ]; then
   # Limit cmake to running a minimal sanity test.
   tests="potrf"
fi

./run_tests.py --timeout 1200 --origin ${origin} --target ${target} --quick \
               --xml ${top}/report-${maker}.xml ${tests}
(( err += $? ))

print "======================================== Smoke tests"
cd ${top}/examples

if [ "${maker}" = "make" ]; then
    export PKG_CONFIG_PATH+=:${top}/install/lib/pkgconfig
    make clean || exit 20

elif [ "${maker}" = "cmake" ]; then
    rm -rf build && mkdir build && cd build
    cmake "-DCMAKE_PREFIX_PATH=${top}/install" .. || exit 30
fi

# Makefile or CMakeLists.txt picks up ${test_args}.
if [ "${device}" = "gpu_intel" ]; then
    # Our Intel GPU supports only single precision.
    export test_args="s c"
else
    export test_args="s d c z"
fi

# ARGS=-V causes CTest to print output. Makefile doesn't use it.
make -j8 || exit 40
make test ARGS=-V
(( err += $? ))

print "======================================== Check HIP files are up-to-date"
if [ "${maker}" = "make" ]; then
    cd ${top}
    touch src/cuda/*.cu
    if make hipify | grep "out-of-date"; then
        print "HIP files are out-of-date with CUDA files."
        print "Run 'make hipify' and commit changes."
        print "Run 'touch src/cuda/*.cu' first if needed to force hipify."
        err=30
    fi
fi

print "======================================== Finished test"
exit ${err}
