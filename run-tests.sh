#!/bin/sh

set -e

VIRTUALENV_DIR="${VIRTUALENV_DIR:-venv}"

if [ "${#}" -gt 0 ] && [ "${1}" == "virtualenv" ]; then

  [ ! -d ./${VIRTUALENV_DIR} ] && virtualenv ${VIRTUALENV_DIR}
  . ${VIRTUALENV_DIR}/bin/activate

fi

PYTHON_DEFAULT=$(dirname `which python`)
PYTHON_PREFIX=${PYTHON_PREFIX:-${PYTHON_DEFAULT}/}

echo Python bin: ${PYTHON_PREFIX}
${PYTHON_PREFIX}python --version

${PYTHON_PREFIX}pip install -r requirements.txt

${PYTHON_PREFIX}python setup.py test

