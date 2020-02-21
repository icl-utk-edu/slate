pipeline {
    agent none
    triggers { cron ('H H(0-2) * * *') }
        stages {
            stage('Parallel Build') {
                parallel {
                    stage('Build - Caffeine') {
                        agent { node 'caffeine.icl.utk.edu' }
                        steps {
                            sh '''
                                #!/bin/sh +x
                                echo "SLATE Jenkinsfile"
                                hostname && pwd

                                source /home/jmfinney/spack/share/spack/setup-env.sh
                                spack load gcc@6.4.0
                                spack load cuda
                                spack load intel-mkl
                                spack load intel-mpi

                                cat > make.inc << "END"
                                mpi=1
                                mkl=1
                                cuda_arch=pascal
                                openmp=1
END

                                cd testsweeper
                                make config CXX=mpicxx
                                # disable color output so JUnit recognizes the XML even if there's an error
                                sed -i '/CXXFLAGS/s/$/ -DNO_COLOR/' make.inc
                                make
                                cd ..

                                cd blaspp
                                make config CXX=mpicxx
                                make -j4
                                cd ..

                                cd lapackpp
                                make config CXX=mpicxx
                                make -j4

                                # add Netlib LAPACKE
                                export LAPACKDIR=$LAPACK_DIR
                                sed -i 's/-lmkl_gf_lp64/-L${LAPACKDIR} -llapacke -lmkl_gf_lp64/g' make.inc
                                cd ..

                                make -j4 CXX=mpicxx
                                '''
                        }
                    }
                    stage('Build - Lips') {
                        agent { node 'lips.icl.utk.edu' }
                        steps {
                            sh '''
                                #!/bin/sh +x
                                hostname && pwd
                                #rm -rf *

                                #hg clone https://bitbucket.org/icl/slate
                                #cd slate
                                source /home/jmfinney/spack/share/spack/setup-env.sh
                                spack load gcc@6.4.0
                                spack load cuda
                                spack load intel-mkl
                                spack load openmpi^gcc@6.4.0

                                cat > make.inc <<-END
                                mpi=1
                                mkl=1
                                cuda=1
                                openmp=1
                                mkl_blacs=openmpi
                                CXXFLAGS = -DNO_COLOR
                                CXX=mpicxx

END

                                export OMPI_CXX=${CXX}

                                make -j4
                                '''
                        }
                    }
                }
            }
            stage('Parallel Test') {
                parallel {
            stage ('Test - Caffeine') {
                agent { node 'caffeine.icl.utk.edu' }
                steps {
                    sh '''
                        #!/bin/sh +x
                        echo "SLATE Test Phase"
                        hostname && pwd

                        source /home/jmfinney/spack/share/spack/setup-env.sh
                        spack load gcc@6.4.0
                        spack load cuda
                        spack load intel-mkl
                        spack load intel-mpi

                        export FI_PROVIDER=tcp

                        #cd unit_test
                        #./run_tests.py --xml report_unit.xml
                        cd test
                        ./run_tests.py --xml report_integration.xml
                        '''
                }
                post {
                    always {
                        junit '*/*.xml'
                    }
                }
            }
            stage ('Test - Lips') {
                agent { node 'lips.icl.utk.edu' }
                steps {
                    sh '''
                        #!/bin/sh +x
                        echo "SLATE Test Phase"
                        hostname && pwd
                        #cd slate

                        source /home/jmfinney/spack/share/spack/setup-env.sh
                        spack load gcc@6.4.0
                        spack load cuda
                        spack load intel-mkl
                        spack load openmpi^gcc@6.4.0

                        cd test
                        ./run_tests.py --xml report_integration.xml
                        '''
                }
                post {
                    always {
                        junit 'test/*.xml'
                    }
                }
            }
                }
            }
        }
}
