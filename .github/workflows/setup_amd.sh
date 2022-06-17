#!/bin/bash -e

section "======================================== Load intel-mpi"
module load intel-mpi
export FI_PROVIDER=tcp

section "======================================== Configure AMD"
cat >> make.inc << END
mkl_blacs = intelmpi
gpu_backend = hip
END

# Load ROCm/HIP.
# Some hip utilities require /usr/sbin/lsmod
rocm=/opt/rocm
export PATH=${PATH}:$rocm/bin:/usr/sbin
export CPATH=${CPATH}:$rocm/include
export LIBRARY_PATH=${LIBRARY_PATH}:$rocm/lib:$rocm/lib64
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$rocm/lib:$rocm/lib64

# HIP headers have many errors; reduce noise.
perl -pi -e 's/-pedantic//' GNUmakefile

which hipcc

