#!/bin/bash

#-------------------------------------------------------------------------------
# Functions

# Suppress echo (-x) output of commands executed with `quiet`.
# Useful for sourcing files, loading modules, spack, etc.
# set +x, set -x are not echo'd.
quiet() {
    { set +x; } 2> /dev/null;
    $@;
    set -x
}

# `print` is like `echo`, but suppresses output of the command itself.
# https://superuser.com/a/1141026
echo_and_restore() {
    builtin echo "$*"
    date
    case "${save_flags}" in
        (*x*)  set -x
    esac
}
alias print='{ save_flags="$-"; set +x; } 2> /dev/null; echo_and_restore'


#-------------------------------------------------------------------------------
quiet source /etc/profile

hostname && pwd
export top=$(pwd)

shopt -s expand_aliases

quiet module load intel-oneapi-mkl
print "MKLROOT=${MKLROOT}"

quiet module load python
quiet which python
quiet which python3
python  --version
python3 --version

quiet module load pkgconf
quiet which pkg-config

# CMake finds CUDA in /usr/local/cuda, so need to explicitly set gpu_backend.
export gpu_backend=none
export color=no
# Don't use `export CXXFLAGS` here because then that
# is exported to BLAS++ and it inherits the -D defines.

# For simplicity, create make.inc regardless of ${maker}
cat > make.inc << END
CXXFLAGS = -Werror -Dslate_omp_default_none='default(none)'
CXX      = mpicxx
CC       = mpicc
FC       = mpif90
blas     = mkl
prefix   = ${top}/install
md5sum   = md5sum
END

#----------------------------------------------------------------- Compiler
if [ "${device}" = "gpu_intel" ]; then
    print "======================================== Load Intel oneAPI compiler"
    quiet module load intel-oneapi-compilers
else
    print "======================================== Load GNU compiler"
    quiet module load gcc@11.3
fi
print "---------------------------------------- Verify compiler"
print "CXX = $CXX"
print "CC  = $CC"
print "FC  = $FC"
${CXX} --version
${CC}  --version
${FC}  --version

#----------------------------------------------------------------- MPI
# Test Open MPI with CPU and CUDA.
# Test Intel MPI with ROCm and SYCL.
# Note: Open MPI hides SYCL devices, at least in our current CI.
if [ "${device}" = "cpu" -o "${device}" = "gpu_nvidia" ]; then
    print "======================================== Load Open MPI"
    quiet module load openmpi
    export OMPI_CXX=${CXX}
    export OMPI_CC=${CC}
    export OMPI_FC=${FC}

    echo "mkl_blacs = openmpi" >> make.inc
else
    print "======================================== Load Intel oneAPI MPI"
    quiet module load intel-oneapi-mpi
    export FI_PROVIDER=tcp
    export I_MPI_CXX=${CXX}
    export I_MPI_CC=${CC}
    export I_MPI_FC=${FC}

    echo "mkl_blacs = intelmpi" >> make.inc
fi
print "---------------------------------------- Verify MPI"
quiet which mpicxx
quiet which mpif90
mpicxx --version
mpif90 --version

#----------------------------------------------------------------- GPU
if [ "${device}" = "gpu_nvidia" ]; then
    print "======================================== Load CUDA"
    quiet module load cuda
    print "CUDA_HOME=${CUDA_HOME}"
    export PATH=${PATH}:${CUDA_HOME}/bin
    export gpu_backend=cuda
    quiet which nvcc
    nvcc --version

    echo "cuda_arch = volta" >> make.inc

elif [ "${device}" = "gpu_amd" ]; then
    print "======================================== Load ROCm"
    export ROCM_PATH=/opt/rocm
    # Some hip utilities require /usr/sbin/lsmod
    export PATH=${PATH}:${ROCM_PATH}/bin:/usr/sbin
    export gpu_backend=hip
    quiet which hipcc
    hipcc --version

    if [ -e ${ROCM_PATH}/lib/rocblas/library ]; then
        # ROCm 5.2
        export ROCBLAS_TENSILE_LIBPATH=${ROCM_PATH}/lib/rocblas/library
    elif [ -e ${ROCM_PATH}/rocblas/lib/library ]; then
        # ROCm 5.1
        export ROCBLAS_TENSILE_LIBPATH=${ROCM_PATH}/rocblas/lib/library
    fi

elif [ "${device}" = "gpu_intel" ]; then
    # Intel oneAPI SYCL compiler loaded above
    export gpu_backend=sycl
    echo "LIBS += -lifcore" >> make.inc
fi

echo "gpu_backend = ${gpu_backend}" >> make.inc

#----------------------------------------------------------------- CMake
if [ "${maker}" = "cmake" ]; then
    print "======================================== Load cmake"
    quiet module load cmake
    quiet which cmake
    cmake --version
    cd build
fi
