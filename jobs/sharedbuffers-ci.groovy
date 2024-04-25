@Library("jampp-shared-libraries@v2.14.3") _

pipeline {
    agent {
        node {
            label "spot-small"
        }
    }

    environment {
        PYTHONTEST_IMAGE_VERSION = "3.5.0"
        REQUIRES_SDIST = "true"
    }
    when {
        expression{
            return hasChangesIn.run("sharedbuffers")
        }
    }
    axes {
        axis {
            name 'PYTHON_VERSION'
            values 'python3.11'
        }
    }

    stages {
        stage("Run CI") {
            steps {
                script {
                    docker.image("docker.jampp.com/sharedbuffers:${PYTHONTEST_IMAGE_VERSION}-${PYTHON_VERSION}").inside(
                        """\
                        -v ${WORKSPACE}:/src \
                        -e REQUIRES_SDIST \
                        """
                    ) {
                        sh "/docker-entrypoint.sh pytest_coverage"
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