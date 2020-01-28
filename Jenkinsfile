pipeline {
    agent none
        triggers {
            pollSCM ('H/10 * * * *')
            cron ('H H(0-2) * * *')
        }
        stages {
            stage('Parallel Build') {
                parallel {
                    stage('Build - Master') {
                        agent { label 'cpu_intel && gpu_amd' }
                        steps {
                            sh '''
                                #!/bin/sh +x
                                echo "SLATE Jenkinsfile"
                                hostname && pwd

                                source /home/jmfinney/spack/share/spack/setup-env.sh
                                spack load gcc
                                spack load intel-mkl
                                spack load intel-mpi

                                cat > make.inc << "END"
                                mpi=1
                                mkl=1
                                cuda_arch=pascal
                                openmp=1
END

                                cd libtest
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
                        post {
                            success {
                                slackSend channel: '#slate_ci',
                                          color: 'good',
                                          message: "SLATE Non-GPU - pipeline ${currentBuild.fullDisplayName} completed successfully."
                            }
                            unsuccessful {
                                // notify users when the Pipeline fails
                                mail to: 'slate-dev@cl.utk.edu',
                                subject: "Failed Pipeline: ${currentBuild.fullDisplayName}",
                                body: "Something is wrong with ${env.BUILD_URL}"
                            }
                        }
                    }
                    stage('Build - gpu_nvidia') {
                        agent { label 'gpu_nvidia' }
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
                        post {
                            success {
                                slackSend channel: '#slate_ci',
                                          color: 'good',
                                          message: "SLATE Nvidia - pipeline ${currentBuild.fullDisplayName} completed successfully."
                            }
                            unsuccessful {
                                // notify users when the Pipeline fails
                                mail to: 'slate-dev@cl.utk.edu',
                                subject: "Failed Pipeline: ${currentBuild.fullDisplayName}",
                                body: "Something is wrong with ${env.BUILD_URL}"
                            }
                        }
                    }
                }
            }
            stage('Parallel Test') {
                parallel {
            stage ('Test - Master') {
                agent { label 'master' }
                steps {
                    sh '''
                        #!/bin/sh +x
                        echo "SLATE Test Phase"
                        hostname && pwd

                        source /opt/spack/share/spack/setup-env.sh
                        spack load gcc
                        spack load cuda
                        spack load intel-mkl
                        spack load intel-mpi

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
                    success {
                        slackSend channel: '#slate_ci',
                            color: 'good',
                            message: "SLATE Nvidia - pipeline ${currentBuild.fullDisplayName} completed successfully."
                    }
                    unsuccessful {
                        // notify users when the Pipeline fails
                        mail to: 'slate-dev@cl.utk.edu',
                        subject: "Failed Pipeline: ${currentBuild.fullDisplayName}",
                        body: "Something is wrong with ${env.BUILD_URL}"
                    }
                }
            }
            stage ('Test - gpu_nvidia') {
                agent { label 'gpu_nvidia' }
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
                    success {
                        slackSend channel: '#slate_ci',
                            color: 'good',
                            message: "SLATE Nvidia - pipeline ${currentBuild.fullDisplayName} completed successfully."
                    }
                    unsuccessful {
                        // notify users when the Pipeline fails
                        mail to: 'slate-dev@cl.utk.edu',
                        subject: "Failed Pipeline: ${currentBuild.fullDisplayName}",
                        body: "Something is wrong with ${env.BUILD_URL}"
                    }
                }
            }
                }
            }
        }
}
