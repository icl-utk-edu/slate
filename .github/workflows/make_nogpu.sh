#!/bin/bash -e

maker=make
gpu=nogpu

shopt -s expand_aliases
set -x
source /etc/profile
export top=`pwd`

rm -rf blaspp lapackpp testsweeper

print_section() {
    builtin echo "$*"
    date
    case "$save_flags" in
        (*x*)  set -x
    esac
}
alias section='{ save_flags="$-"; set +x; } 2> /dev/null; print_section'

module load gcc@7.3.0
module load intel-mkl

section "======================================== maker=$maker"

# For simplicity, create make.inc regardless of ${maker}
export color=no

cat > make.inc << END
CXX    = mpicxx
FC     = gfortran
blas   = mkl
prefix = ${top}/install
END


section "======================================== Load openmpi"
module load openmpi
export OMPI_CXX=${CXX}

section "======================================== Configure NoGPU"
echo "gpu_backend = none" >> make.inc


git submodule update --init

section "======================================== make"
make -j8

section "======================================== make install"
make -j8 install
find ${top}/install


section "======================================== Verify tester"
ldd test/tester

section "======================================== Unit tests"
export OMP_NUM_THREADS=8
cd unit_test
./run_tests.py --xml ${top}/report-unit-${maker}.xml

section "======================================== Tests"
cd ../test
# Temporarily limiting tests to potrf to save time
./run_tests.py --origin s --target t,d --quick --ref n --xml ${top}/report-${maker}.xml potrf

