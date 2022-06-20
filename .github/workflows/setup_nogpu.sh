#!/bin/bash -e

section "======================================== Load openmpi"
module load openmpi
export OMPI_CXX=${CXX}
echo "mkl_blacs = openmpi" >> make.inc

section "======================================== Configure NoGPU"
echo "gpu_backend = none" >> make.inc

