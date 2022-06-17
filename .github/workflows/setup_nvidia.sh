#!/bin/bash -e

section "======================================== Load openmpi"
module load openmpi
export OMPI_CXX=${CXX}

section "======================================== Configure NVIDIA"
cat >> make.inc << END
CXXFLAGS  = -Werror
CXXFLAGS += -Dslate_omp_default_none='default(none)'
mkl_blacs = openmpi
cuda_arch = volta
gpu_backend = cuda
END

# Load CUDA. LD_LIBRARY_PATH set by Spack.
module load cuda
export CPATH=${CPATH}:${CUDA_HOME}/include
export LIBRARY_PATH=${LIBRARY_PATH}:${CUDA_HOME}/lib64

which nvcc

