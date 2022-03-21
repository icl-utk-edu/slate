pipeline {

agent none
options {
    // Required to clean before build
    skipDefaultCheckout( true )
}
triggers { pollSCM 'H/10 * * * *' }
stages {
    //======================================================================
    stage('Parallel Build') {
        matrix {
            axes {
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
run source /home/jenkins/spack_setup
run sload gcc@7.3.0
run spack compiler find
run sload intel-mkl

print "========================================"
date
cat > make.inc << END
CXX  = mpicxx
FC   = mpif90
blas = mkl
END

print "========================================"
# Run CUDA, OpenMPI tests.
if [ "${host}" = "gpu_nvidia" ]; then
    run sload openmpi%gcc@7.3.0
    export OMPI_CXX=${CXX}

    echo "CXXFLAGS  = -Werror" >> make.inc
    echo "mkl_blacs = openmpi" >> make.inc
    echo "cuda_arch = kepler"  >> make.inc
    echo "gpu_backend = cuda"  >> make.inc

    # Load CUDA. LD_LIBRARY_PATH set by Spack.
    run sload cuda@10.2.89
    export CPATH=${CPATH}:${CUDA_HOME}/include
    export LIBRARY_PATH=${LIBRARY_PATH}:${CUDA_HOME}/lib64
fi

# Run HIP, Intel MPI tests.
if [ "${host}" = "gpu_amd" ]; then
    run sload intel-mpi
    export FI_PROVIDER=tcp

    #echo "CXXFLAGS  = -Werror"  >> make.inc  # HIP headers have many errors; ignore.
    echo "mkl_blacs = intelmpi" >> make.inc
    echo "gpu_backend = hip"    >> make.inc

    # Load ROCm/HIP.
    export PATH=${PATH}:/opt/rocm/bin
    export CPATH=${CPATH}:/opt/rocm/include
    export LIBRARY_PATH=${LIBRARY_PATH}:/opt/rocm/lib:/opt/rocm/lib64
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/rocm/lib:/opt/rocm/lib64

    # HIP headers have many errors; reduce noise.
    perl -pi -e 's/-pedantic//' GNUmakefile
fi

export color=no

print "========================================"
# Check what is loaded.
run spack find --loaded

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
make distclean

print "========================================"
make echo

print "========================================"
date
make -j8

print "========================================"
date
make -j8 install prefix=${top}/install
ls -R ${top}/install

print "========================================"
ldd test/tester

print "========================================"
date
export OMP_NUM_THREADS=8
cd ${top}/unit_test
./run_tests.py --xml ../report_unit.xml

print "========================================"
date
cd ${top}/test
./run_tests.py --origin s --target t,d --quick --ref n --xml ${top}/report_test.xml

date
'''
                    } // steps

                    //----------------------------------------------------------
                    post {
                        failure {
                            mail to: 'slate-dev@icl.utk.edu',
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
