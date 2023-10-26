#!/bin/sh

# !!!   Lines between `//---------- begin label`          !!!
# !!!             and `//---------- end label`            !!!
# !!!   are included in the SLATE Users' Guide.           !!!

set -v

# //---------- begin script
# Locations of SLATE, BLAS++, LAPACK++ install or build directories.
export SLATE_ROOT=/path/to/slate
export BLASPP_ROOT=${SLATE_ROOT}/blaspp      # or ${SLATE_ROOT}, if installed
export LAPACKPP_ROOT=${SLATE_ROOT}/lapackpp  # or ${SLATE_ROOT}, if installed
# export CUDA_HOME=/usr/local/cuda           # wherever CUDA is installed
# export ROCM_PATH=/opt/rocm                 # wherever ROCm is installed

# Compile the example.
mpicxx -fopenmp -c slate_lu.cc
       -I${SLATE_ROOT}/include \
       -I${BLASPP_ROOT}/include \
       -I${LAPACKPP_ROOT}/include
       # -I${CUDA_HOME}/include         # For CUDA
       # -I${ROCM_PATH}/include         # For ROCm

mpicxx -fopenmp -o slate_lu slate_lu.o \
       -L${SLATE_ROOT}/lib    -Wl,-rpath,${SLATE_ROOT}/lib \
       -L${BLASPP_ROOT}/lib   -Wl,-rpath,${BLASPP_ROOT}/lib \
       -L${LAPACKPP_ROOT}/lib -Wl,-rpath,${LAPACKPP_ROOT}/lib \
       -lslate -llapackpp -lblaspp

       # For CUDA, may need to add:
       # -L${CUDA_HOME}/lib64 -Wl,-rpath,${CUDA_HOME}/lib64 \
       # -lcusolver -lcublas -lcudart

       # For ROCm, may need to add:
       # -L${ROCM_PATH}/lib -Wl,-rpath,${ROCM_PATH}/lib \
       # -lrocsolver -lrocblas -lamdhip64

# Run the slate_lu executable.
mpirun -n 4 ./slate_lu

# Output from the run will be something like the following:
# lu_solve n 5000, nb 256, p-by-q 2-by-2, residual 8.41e-20, tol 2.22e-16, time 7.65e-01 sec,
#   pass
# //---------- end script

set +v
