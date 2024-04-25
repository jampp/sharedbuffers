ARG PYTHONTEST_VERSION=3.5.0
ARG PY_VERSION=python3.11
FROM docker.jampp.com/pythontest-image-builder:$PYTHONTEST_VERSION-$PY_VERSION AS sharedbuffers

USER root

RUN apt-get update && apt-get install -y pkg-config && apt-get clean all

COPY requirements.txt .
COPY requirements-dev.txt .

USER $USER

RUN pip install --user "Cython<3" && pip cache purge
RUN pip install --user -r requirements.txt && pip install --user -r requirements-dev.txt && pip cache purge