pipeline {

agent none
options {
    // Required to clean before build
    skipDefaultCheckout( true )
}

// cron syntax: minute hour day-of-month month day-of-week
// run hourly
triggers { pollSCM 'H * * * *' }

stages {
    //======================================================================
    stage('Parallel Build') {
        matrix {
            axes {
                axis {
                    name 'maker'
                    values 'make', 'cmake'
                }
                axis {
                    name 'host'
                    values 'dopamine', 'gpu_nvidia'
                }
            } // axes
            stages {
                stage('Build') {
                    agent { label "${host}" }

                    //----------------------------------------------------------
                    steps {
                        cleanWs()
                        checkout scm
                        sh '''
#!/bin/sh

set +e  # errors are not fatal (e.g., Spack sometimes has spurious failures)
set -x  # echo commands

date
hostname && pwd
export top=`pwd`

date
git submodule update --init

# Suppress echo (-x) output of commands executed with `run`. Useful for Spack.
# set +x, set -x are not echo'd.
run() {
    { set +x; } 2> /dev/null;
    $@;
    set -x
}

# Suppress echo (-x) output of `print` commands. https://superuser.com/a/1141026
# aliasing `echo` causes issues with spack_setup, so use `print` instead.
echo_and_restore() {
    builtin echo "$*"
    case "$save_flags" in
        (*x*)  set -x
    esac
}
alias print='{ save_flags="$-"; set +x; } 2> /dev/null; echo_and_restore'

date
run module load python
run which python
run which python3

run module avail gcc
set +x  # not verbose; run() doesn't seem to work with ||.
echo "module load gcc/7.5.0 || module load gcc/7.3.0"
      module load gcc/7.5.0 || module load gcc/7.3.0
set -x  # verbose
run which g++
g++ --version

run module load intel-oneapi-mkl
echo "MKLROOT=${MKLROOT}"

# hipcc needs /usr/sbin/lsmod
export PATH=${PATH}:/usr/sbin

print "========================================"
date
print "maker ${maker}"

# For simplicity, create make.inc regardless of ${maker}
export color=no
cat > make.inc << END
CXX    = mpicxx
FC     = mpif90
blas   = mkl
prefix = ${top}/install
md5sum = md5sum
END

print "========================================"
# Run CUDA, OpenMPI tests.
if [ "${host}" = "gpu_nvidia" ]; then
    run module load openmpi/4
    export OMPI_CXX=${CXX}
    run which mpicxx
    mpicxx --version

    echo "CXXFLAGS  = -Werror" >> make.inc
    echo "CXXFLAGS += -Dslate_omp_default_none='default(none)'" >> make.inc
    echo "mkl_blacs = openmpi" >> make.inc
    echo "cuda_arch = sm_35"   >> make.inc  # kepler sm_35 works in CUDA 11
    echo "gpu_backend = cuda"  >> make.inc

    # Load CUDA. LD_LIBRARY_PATH set by Spack.
    run module avail cuda
    # CUDA 11.8 seems to require linking with -lcublasLt; stick with 11.4 or 11.6.
    # Error "provided PTX was compiled with an unsupported toolchain" with
    # 11.6 indicates driver is older; nvidia-smi suggests driver is from 11.4.
    run module load cuda/11.4
    run which nvcc
    nvcc --version

    export CPATH=${CPATH}:${CUDA_HOME}/include
    export LIBRARY_PATH=${LIBRARY_PATH}:${CUDA_HOME}/lib64
fi

# Run HIP, Intel MPI tests.
if [ "${host}" = "dopamine" ]; then
    run module load intel-mpi
    export FI_PROVIDER=tcp
    run which mpicxx
    mpicxx --version

    #echo "CXXFLAGS  = -Werror"  >> make.inc  # HIP headers have many errors; ignore.
    echo "mkl_blacs = intelmpi" >> make.inc
    echo "gpu_backend = hip"    >> make.inc

    # Load ROCm/HIP.
    export PATH=${PATH}:/opt/rocm/bin
    export CPATH=${CPATH}:/opt/rocm/include
    export LIBRARY_PATH=${LIBRARY_PATH}:/opt/rocm/lib:/opt/rocm/lib64
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/rocm/lib:/opt/rocm/lib64
    export ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library/

    # HIP headers have many errors; reduce noise.
    perl -pi -e 's/-pedantic//' GNUmakefile

    # See the files that are opened by the executable.
    # Used for debugging the ROCm library.
    # extra_test_args=( -t \\\" strace -e trace=open ./tester \\\" )
fi

if [ "${maker}" = "make" ]; then
    print "========================================"
    make echo
fi

if [ "${maker}" = "cmake" ]; then
    print "========================================"
    run module load cmake
    run which cmake
    cmake --version

    rm -rf build && mkdir build && cd build
    cmake -Dcolor=no -DCMAKE_CXX_FLAGS="-Werror" \
          -DCMAKE_INSTALL_PREFIX=${top}/install \
          ..
fi

print "========================================"
# Check what is loaded.
module list

which mpicxx
which mpif90
mpicxx --version
mpif90 --version

which nvcc
nvcc --version

which hipcc
hipcc --version

echo "MKLROOT ${MKLROOT}"

print "========================================"
env

print "========================================"
date
make -j8

print "========================================"
date
make -j8 install
ls -R ${top}/install

print "========================================"
ldd test/tester

print "========================================"
date
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0

cd unit_test
./run_tests.py --timeout 300 --xml ${top}/report-unit-${maker}.xml
cd ..

print "========================================"
date
cd test
if [ "${maker}" = "cmake" ]; then
    # only sanity check with cmake build
    export tests=potrf
fi
eval ./run_tests.py --origin s --target t,d --quick --ref n \
    "${extra_test_args[@]}" \
    --timeout 1200 \
    --xml ${top}/report-${maker}.xml ${tests} \
    2> ${top}/report-${maker}.txt
cd ..

print "========================================"
if [ "${maker}" = "make" ]; then
    touch src/cuda/*.cu
    if make hipify | grep "out-of-date"; then
        print "HIP files are out-of-date with CUDA files."
        print "Run 'make hipify' and commit changes."
        print "Run 'touch src/cuda/*.cu' first if needed to force hipify."
        exit 1
    fi
fi

date
'''
                    } // steps

                    //----------------------------------------------------------
                    post {
                        failure {
                            mail to: 'slate-test@icl.utk.edu',
                                subject: "${currentBuild.fullDisplayName} >> ${STAGE_NAME} >> ${host} failed",
                                body: "See more at ${env.BUILD_URL}"
                        }
                        always {
                            junit '*.xml'
                        }
                    } // post

                } // stage(Build)
            } // stages
        } // matrix
    } // stage(Parallel Build)
} // stages

} // pipeline
