#FROM joernhees/virtuoso:latest
FROM ubuntu:16.04
#FROM python:2.7-alpine

ARG OWN_LOCATION=docker/create-corpus/dbpedia-nif
ARG TMP=/tmp/dbpedia-nif
ARG ROOT=/root/recursive-embedding

RUN mkdir "$TMP"
RUN mkdir "$ROOT"

# see for connection issues:
# https://stackoverflow.com/questions/24319662/from-inside-of-a-docker-container-how-do-i-connect-to-the-localhost-of-the-mach

# get virtuoso ODBC driver
# RUN curl -O http://packages.comsode.eu/pool/main/v/virtuoso-opensource/libvirtodbc0_7.2.4_amd64.deb
COPY "$OWN_LOCATION"/libvirtodbc0_7.2.4_amd64.deb "$TMP"/libvirtodbc0_7.2.4_amd64.deb

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

RUN dpkg -i "$TMP"/libvirtodbc0_7.2.4_amd64.deb

COPY requirements.txt "$ROOT"/requirements.txt
RUN pip install --no-cache-dir -r "$ROOT"/requirements.txt
RUN python -m spacy download en_core_web_md

COPY src "$ROOT"/src
COPY "$OWN_LOCATION"/etc_odbc.ini /etc/odbc.ini

VOLUME /root/corpora_out

WORKDIR "$ROOT"/src

#CMD bash

