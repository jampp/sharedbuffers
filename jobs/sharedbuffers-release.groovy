@Library('jampp-shared-libraries@v2.14.3') _

def slackResponse = ''
script {
    slackResponse = slackSendNotification.notifyBuildStart()
}

pipeline {
    agent { node { label 'on-demand-large' } }
    stages {
        stage('Check tags & package version') {
            steps{
                script {
                    PKG_VERSION = readFile("VERSION").trim()

                    if (params.TEST_VERSION) {
                        PKG_VERSION += "-test"
                    }

                    GIT_REF= sh(returnStdout: true, script: "git show-ref --tags ${PKG_VERSION} || echo ''").trim()

                    if (GIT_REF != "") {
                        error "Tag ${PKG_VERSION} exists: ${GIT_REF}"
                    }
                }
            }
        }
        stage('Publish') {
            steps {
                script {
                    docker.image("python:3.11-alpine3.18").inside('-u root:root'){
                        sh '''
                        cat << EOF > ~/.pypirc
                        [distutils]
                        index-servers = jampp
                        [jampp]
                        repository: http://pypi.jampp.com/pypi
                        username: ''
                        password: ''
                        EOF
                        '''.stripIndent()
                        sh '''
                        apk --no-cache add musl-dev linux-headers g++
                        pip install numpy
                        python setup.py build sdist bdist_egg upload -r jampp
                        '''
                    }
                    withCredentialedGit.run(credential: 'github', setGlobally: false, {
                        sh "git tag ${PKG_VERSION} && git push --tags"
                    })
                }
            }
        }
    }

    post {
        always {
            script {
                slackSendNotification.notifyResult(slackResponse)
            }
        }
    }
}