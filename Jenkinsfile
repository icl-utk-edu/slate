pipeline {

agent none
triggers { pollSCM 'H/10 * * * *' }
stages {
    //======================================================================
    stage('Parallel Build') {
        matrix {
            axes {
                axis {
                    name 'host'
                    values 'caffeine', 'lips'
                }
            } // axes
            stages {
                stage('Build') {
                    agent { node "${host}.icl.utk.edu" }

                    //----------------------------------------------------------
                    steps {
                        sh '''
#!/bin/sh +x
date
hostname && pwd
export top=`pwd`

date
git submodule update --init

# Suppress trace output of commands executed with `run`. Useful for Spack.
run() {
    { set +x; } 2> /dev/null;
    $@;
    set -x
}

# Suppress trace output of `print` commands. https://superuser.com/a/1141026
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
# Run CUDA, OpenMPI tests on lips.
if [ "${host}" = "lips" ]; then
    run sload openmpi%gcc@7.3.0
    export OMPI_CXX=${CXX}

    echo "CXXFLAGS  = -Werror" >> make.inc
    echo "mkl_blacs = openmpi" >> make.inc
    echo "cuda_arch = kepler"  >> make.inc

    # Load CUDA. LD_LIBRARY_PATH set by Spack.
    run sload cuda@10.2.89
    export CPATH=${CPATH}:${CUDA_HOME}/include
    export LIBRARY_PATH=${LIBRARY_PATH}:${CUDA_HOME}/lib64
fi

# Run HIP, Intel MPI tests on caffeine.
if [ "${host}" = "caffeine" ]; then
    sload intel-mpi
    export FI_PROVIDER=tcp

    #echo "CXXFLAGS  = -Werror"  >> make.inc  # HIP headers have many errors; ignore.
    echo "mkl_blacs = intelmpi" >> make.inc
    echo "hip_arch  = gfx900"   >> make.inc  # MI25 / Vega 10

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
./run_tests.py --quick --ref n --xml ${top}/report_test.xml

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
