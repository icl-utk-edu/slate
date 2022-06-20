#!/bin/bash -e

maker=$1
device=$2

mydir=`dirname $0`
source $mydir/setup.sh

section "======================================== make"
make -j8

section "======================================== make install"
make -j8 install
ls -R ${top}/install

