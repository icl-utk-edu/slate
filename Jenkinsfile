pipeline {
    agent none
    triggers { pollSCM 'H/10 * * * *' }
    stages {
        //======================================================================
        stage('Parallel Build') {
            parallel {
                //--------------------------------------------------------------
                stage('Build - Caffeine (gcc 7.3, HIP, MKL, Intel MPI)') {
                    agent { node 'caffeine.icl.utk.edu' }
                    steps {
                        sh '''
                        #!/bin/sh +x
                        echo "SLATE Building"
                        hostname && pwd

                        git submodule update --init

                        source /home/jenkins/spack_setup
                        sload gcc@7.3.0
                        spack compiler find
                        sload intel-mkl
                        sload intel-mpi

                        # load ROCm/HIP
                        export PATH=${PATH}:/opt/rocm/bin
                        export CPATH=${CPATH}:/opt/rocm/include
                        export LIBRARY_PATH=${LIBRARY_PATH}:/opt/rocm/lib:/opt/rocm/lib64
                        export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/rocm/lib:/opt/rocm/lib64

                        export color=no

                        #========================================
                        env

                        #========================================
                        cat > make.inc << END
                        CXX       = mpicxx
                        FC        = mpif90
                        # CXXFLAGS  = -Werror  # HIP headers have many errors.
                        blas      = mkl
                        # openmp=1 by default
END

                        # HIP headers have many errors; reduce noise.
                        perl -pi -e 's/-pedantic//' GNUmakefile

                        echo "========================================"
                        make distclean
                        echo "========================================"
                        make echo
                        echo "========================================"
                        make -j4
                        echo "========================================"
                        ldd test/tester
                        '''
                    } // steps
                    post {
                        failure {
                            mail to: 'slate-dev@icl.utk.edu',
                                subject: "${currentBuild.fullDisplayName} Caffeine build failed",
                                body: "See more at ${env.BUILD_URL}"
                        }
                    } // post
                } // stage(Build - Caffeine)

                //--------------------------------------------------------------
                stage('Build - Lips (gcc 7.3, CUDA, MKL, Open MPI)') {
                    agent { node 'lips.icl.utk.edu' }
                    steps {
                        sh '''
                        #!/bin/sh +x
                        echo "SLATE Building"
                        hostname && pwd

                        git submodule update --init

                        source /home/jenkins/spack_setup
                        sload gcc@7.3.0
                        spack compiler find
                        sload cuda@10.2.89
                        sload intel-mkl
                        sload openmpi%gcc@7.3.0

                        # Load CUDA. LD_LIBRARY_PATH already set.
                        export CPATH=${CPATH}:${CUDA_HOME}/include
                        export LIBRARY_PATH=${LIBRARY_PATH}:${CUDA_HOME}/lib64

                        export color=no
                        export OMPI_CXX=${CXX}

                        #========================================
                        env

                        #========================================
                        cat > make.inc << END
                        CXX       = mpicxx
                        FC        = mpif90
                        CXXFLAGS  = -Werror
                        blas      = mkl
                        mkl_blacs = openmpi
                        cuda_arch = kepler
                        # openmp=1 by default
END

                        echo "========================================"
                        make distclean
                        echo "========================================"
                        make echo
                        echo "========================================"
                        make -j4
                        echo "========================================"
                        ldd test/tester
                        '''
                    } // steps
                    post {
                        failure {
                            mail to: 'slate-dev@icl.utk.edu',
                                subject: "${currentBuild.fullDisplayName} Lips build failed",
                                body: "See more at ${env.BUILD_URL}"
                        }
                    } // post
                } // stage(Build - Lips)
            } // parallel
        } // stage(Parallel Build)

        //======================================================================
        stage('Parallel Test') {
            parallel {
                //--------------------------------------------------------------
                stage('Test - Caffeine') {
                    agent { node 'caffeine.icl.utk.edu' }
                    steps {
                        sh '''
                        #!/bin/sh +x
                        echo "SLATE Testing"
                        hostname && pwd

                        source /home/jenkins/spack_setup
                        sload gcc@7.3.0
                        spack compiler find
                        sload intel-mkl
                        sload intel-mpi

                        # load ROCm/HIP
                        export PATH=${PATH}:/opt/rocm/bin
                        export CPATH=${CPATH}:/opt/rocm/include
                        export LIBRARY_PATH=${LIBRARY_PATH}:/opt/rocm/lib:/opt/rocm/lib64
                        export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/rocm/lib:/opt/rocm/lib64

                        export FI_PROVIDER=tcp

                        cd unit_test
                        ./run_tests.py --xml ../report_unit.xml
                        cd ..

                        cd test
                        ./run_tests.py --ref n --xml ../report_test.xml
                        cd ..
                        '''
                    } // steps
                    post {
                        failure {
                            mail to: 'slate-dev@icl.utk.edu',
                                subject: "${currentBuild.fullDisplayName} Caffeine test failed",
                                body: "See more at ${env.BUILD_URL}"
                        }
                        always {
                            junit '*.xml'
                        }
                    } // post
                } // stage(Test - Caffeine)

                //--------------------------------------------------------------
                stage('Test - Lips') {
                    agent { node 'lips.icl.utk.edu' }
                    steps {
                        sh '''
                        #!/bin/sh +x
                        echo "SLATE Testing"
                        hostname && pwd

                        source /home/jenkins/spack_setup
                        sload gcc@7.3.0
                        spack compiler find
                        sload cuda@10.2.89
                        sload intel-mkl
                        sload openmpi%gcc@7.3.0

                        cd unit_test
                        ./run_tests.py --xml ../report_unit.xml
                        cd ..

                        cd test
                        ./run_tests.py --ref n --xml ../report.xml
                        cd ..
                        '''
                    } // steps
                    post {
                        failure {
                            mail to: 'slate-dev@icl.utk.edu',
                                subject: "${currentBuild.fullDisplayName} Lips test failed",
                                body: "See more at ${env.BUILD_URL}"
                        }
                        always {
                            junit '*.xml'
                        }
                    } // post
                } // stage(Test - Lips)
            } // parallel
        } // stage(Parallel Test)
    } // stages
} // pipeline
