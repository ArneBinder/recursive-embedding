FROM ubuntu:16.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        gosu \
        python-dev \
        python-pip \
        unixodbc \
        unixodbc-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && pip install --no-cache-dir setuptools
RUN pip install --no-cache-dir spacy && python -m spacy download en_core_web_md && python -m spacy link en_core_web_md en

ARG PROJECT_ROOT=/root/recursive-embedding
RUN mkdir "$PROJECT_ROOT"
RUN chmod 777 /root
COPY docker/set-user-with-folder.sh /root/set-user-with-folder.sh
RUN chmod +x /root/set-user-with-folder.sh

########################################################################################################################

ARG OWN_LOCATION=docker/preprocessing

COPY "$OWN_LOCATION"/requirements.txt "$PROJECT_ROOT"/requirements.txt
RUN pip install --no-cache-dir -r "$PROJECT_ROOT"/requirements.txt

COPY "$OWN_LOCATION"/scripts /root/scripts
COPY src "$PROJECT_ROOT"/src
WORKDIR "$PROJECT_ROOT"/src

#CMD bash

