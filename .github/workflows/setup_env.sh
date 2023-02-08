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


print "======================================== Load compiler"
quiet module load gcc@7.3.0
quiet which g++
g++ --version

quiet module load intel-oneapi-mkl
echo "MKLROOT=${MKLROOT}"

#quiet module load python  # no module
quiet which python
quiet which python3
python  --version
python3 --version

quiet module load pkgconf
quiet which pkg-config

# CMake will find CUDA in /usr/local/cuda, so need to explicitly set
# gpu_backend.
export gpu_backend=none
export color=no

# For simplicity, create make.inc regardless of ${maker}
cat > make.inc << END
CXX    = mpicxx
FC     = mpif90
blas   = mkl
prefix = ${top}/install
md5sum = md5sum
END

if [ "${device}" = "cpu" -o "${device}" = "gpu_nvidia" ]; then
    print "======================================== Load Open MPI"
    quiet module load openmpi
    export OMPI_CXX=${CXX}

    cat >> make.inc << END
CXXFLAGS  = -Werror -Dslate_omp_default_none='default(none)'
mkl_blacs = openmpi
END
else
    print "======================================== Load Intel MPI"
    quiet module load intel-mpi
    export FI_PROVIDER=tcp

    # AMD has header warnings, so don't use -Werror.
    cat >> make.inc << END
mkl_blacs = intelmpi
END
fi
print "======================================== Verify MPI"
quiet which mpicxx
quiet which mpif90
mpicxx --version
mpif90 --version

if [ "${device}" = "gpu_nvidia" ]; then
    print "======================================== Load CUDA"
    quiet module load cuda
    echo "CUDA_HOME=${CUDA_HOME}"
    export PATH=${PATH}:${CUDA_HOME}/bin
    export CPATH=${CPATH}:${CUDA_HOME}/include
    export LIBRARY_PATH=${LIBRARY_PATH}:${CUDA_HOME}/lib64
    export gpu_backend=cuda
    quiet which nvcc
    nvcc --version

    echo "cuda_arch = volta" >> make.inc

elif [ "${device}" = "gpu_amd" ]; then
    print "======================================== Load ROCm"
    export ROCM_HOME=/opt/rocm
    # Some hip utilities require /usr/sbin/lsmod
    export PATH=${PATH}:${ROCM_HOME}/bin:/usr/sbin
    export CPATH=${CPATH}:${ROCM_HOME}/include
    export LIBRARY_PATH=${LIBRARY_PATH}:${ROCM_HOME}/lib:${ROCM_HOME}/lib64
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ROCM_HOME}/lib:${ROCM_HOME}/lib64
    export gpu_backend=hip
    quiet which hipcc
    hipcc --version

    if [ -e ${ROCM_HOME}/lib/rocblas/library ]; then
        # ROCm 5.2
        export ROCBLAS_TENSILE_LIBPATH=${ROCM_HOME}/lib/rocblas/library
    elif [ -e ${ROCM_HOME}/rocblas/lib/library ]; then
        # ROCm 5.1
        export ROCBLAS_TENSILE_LIBPATH=${ROCM_HOME}/rocblas/lib/library
    fi

    # HIP headers have many errors; reduce noise.
    perl -pi -e 's/-pedantic//' GNUmakefile
fi

echo "gpu_backend = ${gpu_backend}" >> make.inc

if [ "${maker}" = "cmake" ]; then
    print "======================================== Load cmake"
    quiet module load cmake
    quiet which cmake
    cmake --version
    cd build
fi

quiet module list
