#!/bin/bash -e

maker=$1
gpu=$2

mydir=`dirname $0`
source $mydir/setup.sh

ldd test/tester


