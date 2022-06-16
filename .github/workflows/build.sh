#!/bin/bash -e 

maker=$1
gpu=$2
echo Starting maker=$maker gpu=$gpu

source /etc/profile

export top=`pwd`

git submodule update --init

echo "======================================== load compiler"
date

module load gcc@7.3.0
module load intel-mkl

if [[ "${gpu}" = "none" ]]; then
    module load openmpi
    export OMPI_CXX=${CXX}
fi

echo "======================================== load CUDA or ROCm"
# Load CUDA.
if [ "${gpu}" = "nvidia" ]; then
    module load openmpi
    export OMPI_CXX=${CXX}

    echo "CXXFLAGS  = -Werror" >> make.inc
    echo "CXXFLAGS += -Dslate_omp_default_none='default(none)'" >> make.inc
    echo "mkl_blacs = openmpi" >> make.inc
    echo "cuda_arch = kepler"  >> make.inc
    echo "gpu_backend = cuda"  >> make.inc

    # Load CUDA. 
    export CUDA_HOME=/usr/local/cuda/
    export PATH=${PATH}:${CUDA_HOME}/bin
    export CPATH=${CPATH}:${CUDA_HOME}/include
    export LIBRARY_PATH=${LIBRARY_PATH}:${CUDA_HOME}/lib64
    which nvcc
    nvcc --version
fi

# Load HIP.
if [ "${gpu}" = "amd" ]; then
    module load intel-mpi
    export FI_PROVIDER=tcp

    echo "mkl_blacs = intelmpi" >> make.inc
    echo "gpu_backend = hip"    >> make.inc

    # Load ROCm/HIP.
    export PATH=${PATH}:/opt/rocm/bin
    export CPATH=${CPATH}:/opt/rocm/include
    export LIBRARY_PATH=${LIBRARY_PATH}:/opt/rocm/lib:/opt/rocm/lib64
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/rocm/lib:/opt/rocm/lib64
    which hipcc
    hipcc --version

    # HIP headers have many errors; reduce noise.
    perl -pi -e 's/-pedantic//' GNUmakefile
fi

echo "======================================== verify dependencies"
# Check what is loaded.
module list
echo "MKLROOT ${MKLROOT}"
which mpicxx
which mpif90
mpicxx --version
mpif90 --version

echo "======================================== env"
env

echo "========================================"
date

# For simplicity, create make.inc regardless of ${maker}
export color=no
cat > make.inc << END
CXX    = mpicxx
FC     = mpif90
blas   = mkl
prefix = ${top}/install
END

echo "======================================== setup build"
date
echo "maker ${maker}"
export color=no
rm -rf ${top}/install
if [ "${maker}" = "make" ]; then
    make echo
fi
if [ "${maker}" = "cmake" ]; then
    module load cmake
    which cmake
    cmake --version

    mkdir build && cd build
    cmake -Dcolor=no -DCMAKE_CXX_FLAGS="-Werror" \
          -DCMAKE_INSTALL_PREFIX=${top}/install ..
fi

echo "======================================== build"
date
make -j8 || make -j1

echo "======================================== install"
date
make -j8 install
ls -R ${top}/install

echo "======================================== verify build"
date
ldd test/tester

echo "======================================== run tests"
date
export OMP_NUM_THREADS=8
cd unit_test
./run_tests.py --xml ${top}/report-unit-${maker}.xml
cd ..

echo "========================================"
date
cd test
if [ "${maker}" = "cmake" ]; then
    # only sanity check with cmake build
    export tests=potrf
fi
./run_tests.py --origin s --target t,d --quick --ref n --xml ${top}/report-${maker}.xml ${tests}

date

