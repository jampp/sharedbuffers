@Library("jampp-shared-libraries@v2.14.3") _

pipeline {
    agent {
        node {
            label "spot-large"
        }
    }

    environment {
        PYTHONTEST_IMAGE_VERSION = "3.6.0-python3.11"
        REQUIRES_SDIST = "true"
    }

    stages {
        stage("Run CI") {
            steps {
                script {
                    docker.image("docker.jampp.com/pythontest-image-builder:${PYTHONTEST_IMAGE_VERSION}").inside(
                        """\
                        -v ${WORKSPACE}:/src \
                        -e PYTHON_PRE_DEPENDENCIES=Cython \
                        -e REQUIRES_SDIST=true \
                        -e REQUIRES_BUILD=true \
                        -e FORCE_SESSION_ROLLBACK_UNITTEST \
                        """
                    ) {
                        sh 'pip install Cython==0.29.36'
                        sh """
                        """.stripIndent()
                        sh script: '/docker-entrypoint.sh pytest_coverage', returnStatus: true
                    }
                }

                junit "output.xml"
                cobertura autoUpdateHealth: false,
                autoUpdateStability: false,
                coberturaReportFile: "coverage.xml",
                failNoReports: true,
                failUnhealthy: false,
                failUnstable: false,
                maxNumberOfBuilds: 0,
                onlyStable: false,
                sourceEncoding: "ASCII",
                zoomCoverageChart: false,
                lineCoverageTargets: '80, 0, 0',
                methodCoverageTargets: '80, 0, 0',
                conditionalCoverageTargets: '80, 0, 0'

                archiveArtifacts "dist/sharedbuffers-*.tar.gz"
            }
        }
    }
}