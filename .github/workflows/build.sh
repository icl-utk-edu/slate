#!/bin/bash -x

maker=$1
device=$2

mydir=$(dirname $0)
source ${mydir}/setup_env.sh

print "======================================== Build"
make -j8 || exit 10

print "======================================== Install"
make -j8 install || exit 11
ls -R ${top}/install

print "======================================== Verify build"
ldd_result=$(ldd test/tester) || exit 12
echo "${ldd_result}"

# Verify that tester linked with cublas or rocblas as intended.
if [ "${device}" = "gpu_nvidia" ]; then
    echo "${ldd_result}" | grep cublas || exit 13

elif [ "${device}" = "gpu_amd" ]; then
    echo "${ldd_result}" | grep rocblas || exit 14

else
    # CPU-only not linked with cublas or rocblas.
    echo "${ldd_result}" | grep -P "cublas|rocblas" && exit 15
fi

print "======================================== Finished build"
exit 0
