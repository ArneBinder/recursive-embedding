FROM ubuntu:16.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        python-dev \
        python-pip \
        unixodbc \
        unixodbc-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && pip install --no-cache-dir setuptools
RUN pip install --no-cache-dir spacy && python -m spacy download en_core_web_md && python -m spacy link en_core_web_md en

ARG ROOT=/root/recursive-embedding
ARG TMP=/tmp/sick
RUN mkdir "$TMP"
RUN mkdir "$ROOT"
VOLUME /root/corpora_out

########################################################################################################################

ARG OWN_LOCATION=docker/create-corpus/sick

COPY "$OWN_LOCATION"/requirements.txt "$ROOT"/requirements.txt
RUN pip install --no-cache-dir -r "$ROOT"/requirements.txt

COPY "$OWN_LOCATION"/data /root/corpora_in/SICK

COPY src "$ROOT"/src
WORKDIR "$ROOT"/src

#CMD bash

