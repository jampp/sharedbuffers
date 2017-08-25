#!/bin/sh

set -e

PATH=/usr/local/bin:$PATH

python --version

pip install -q -r requirements.txt

python setup.py test

