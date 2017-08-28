#!/bin/sh

set -e

PYTHON_DEFAULT=$(dirname `which python`)
PYTHON_PREFIX=${PYTHON_PREFIX:-${PYTHON_DEFAULT}/}

echo Python bin: ${PYTHON_PREFIX}
${PYTHON_PREFIX}python --version

${PYTHON_PREFIX}pip install -r requirements.txt

${PYTHON_PREFIX}python setup.py test

