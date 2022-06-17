#!/bin/bash -e

section "======================================== Load openmpi"
module load openmpi
export OMPI_CXX=${CXX}

section "======================================== Configure NoGPU"
echo "gpu_backend = none" >> make.inc

