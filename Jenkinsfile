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
hostname && pwd
export top=`pwd`

git submodule update --init

source /home/jenkins/spack_setup
sload gcc@7.3.0
spack compiler find
sload intel-mkl

#========================================
cat > make.inc << END
CXX  = mpicxx
FC   = mpif90
blas = mkl
END

echo "========================================"
# Run CUDA, OpenMPI tests on lips.
if [ "${host}" = "lips" ]; then
    sload openmpi%gcc@7.3.0
    export OMPI_CXX=${CXX}

    echo "CXXFLAGS  = -Werror" >> make.inc
    echo "mkl_blacs = openmpi" >> make.inc
    echo "cuda_arch = kepler"  >> make.inc

    # Load CUDA. LD_LIBRARY_PATH set by Spack.
    sload cuda@10.2.89
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

#========================================
env

echo "========================================"
make distclean

echo "========================================"
make echo

echo "========================================"
make -j8

echo "========================================"
make -j8 install prefix=${top}/install
ls -R ${top}/install

echo "========================================"
ldd test/tester

echo "========================================"
export OMP_NUM_THREADS=8
cd ${top}/unit_test
./run_tests.py --xml ../report_unit.xml

echo "========================================"
cd ${top}/test
./run_tests.py --ref n --xml ${top}/report_test.xml
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
