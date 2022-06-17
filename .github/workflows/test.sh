#!/bin/bash -e

maker=$1
gpu=$2

mydir=`dirname $0`
source $mydir/setup.sh

section "======================================== Verify tester"
ldd test/tester

section "======================================== Unit tests"
export OMP_NUM_THREADS=8
cd unit_test
./run_tests.py --xml ${top}/report-unit-${maker}.xml

section "======================================== Tests"
cd ../test
./run_tests.py --origin s --target t,d --quick --ref n --xml ${top}/report-${maker}.xml

