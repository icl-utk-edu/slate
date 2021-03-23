pipeline {
    agent none
    triggers { cron ('H H(4-5) * * *') }
    stages {
        //======================================================================
        stage('Parallel Build') {
            parallel {
                //--------------------------------------------------------------
                stage('Build - Caffeine (gcc 6.4, CUDA, MKL, Intel MPI)') {
                    agent { node 'caffeine.icl.utk.edu' }
                    steps {
                        sh '''
                        #!/bin/sh +x
                        echo "SLATE Building"
                        hostname && pwd

                        git submodule update --init

                        source /home/jenkins/spack_setup
                        sload gcc@6.4.0
                        sload cuda@10.2.89
                        sload intel-mkl
                        sload intel-mpi

                        export color=no

                        cat > make.inc << END
                        CXX       = mpicxx
                        FC        = mpif90
                        CXXFLAGS  = -Werror
                        blas      = mkl
                        cuda_arch = pascal
                        # openmp=1 by default
END

                        echo "========================================"
                        make distclean
                        echo "========================================"
                        ls -R
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
                stage('Build - Lips (gcc 6.4, CUDA, MKL, Open MPI)') {
                    agent { node 'lips.icl.utk.edu' }
                    steps {
                        sh '''
                        #!/bin/sh +x
                        echo "SLATE Building"
                        hostname && pwd

                        git submodule update --init

                        source /home/jenkins/spack_setup
                        sload gcc@6.4.0
                        spack compiler find
                        sload cuda@10.2.89
                        sload intel-mkl
                        sload openmpi%gcc@6.4.0

                        export color=no
                        export OMPI_CXX=${CXX}

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
                        ls -R
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
                        sload gcc@6.4.0
                        sload cuda@10.2.89
                        sload intel-mkl
                        sload intel-mpi

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
                        sload gcc@6.4.0
                        spack compiler find
                        sload cuda@10.2.89
                        sload intel-mkl
                        sload openmpi%gcc@6.4.0

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
