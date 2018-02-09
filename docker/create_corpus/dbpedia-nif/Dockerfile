#FROM joernhees/virtuoso:latest
FROM ubuntu:16.04
#FROM python:2.7-alpine

ARG OWN_LOCATION=docker/create_corpus/dbpedia-nif
ARG TMP=/tmp/dbpedia-nif
ARG ROOT=/root/recursive-embedding

RUN mkdir "$TMP"
RUN mkdir "$ROOT"

# get virtuoso ODBC driver
# RUN curl -O http://packages.comsode.eu/pool/main/v/virtuoso-opensource/libvirtodbc0_7.2.4_amd64.deb
COPY "$OWN_LOCATION"/libvirtodbc0_7.2.4_amd64.deb "$TMP"/libvirtodbc0_7.2.4_amd64.deb

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        #libiodbc2 \
        python-dev \
        python-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && pip install --no-cache-dir setuptools

# download & compile & install unixODBC
#RUN cd "$TMP" \
#    && curl -O 'ftp://ftp.unixodbc.org/pub/unixODBC/unixODBC-2.3.5.tar.gz' \
#    && tar -xz -f unixODBC-2.3.5.tar.gz \
#    && cd unixODBC-2.3.5 \
#    && ./configure \
#    && make \
#    && make install
    #&& rm "$TMP"/unixODBC-2.3.5.tar.gz \
    #&& rm -rf "$TMP"/unixODBC-2.3.5

#ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/usr/local/lib"

# TODO: move to other apt-get
RUN apt-get update && apt-get install unixodbc unixodbc-dev -y

RUN dpkg -i "$TMP"/libvirtodbc0_7.2.4_amd64.deb

#RUN -O https://github.com/openlink/iODBC/releases/download/v3.52.12/libiodbc-3.52.12.tar.gz

COPY requirements.txt "$ROOT"/requirements.txt
RUN pip install --no-cache-dir -r "$ROOT"/requirements.txt
#RUN python -m spacy download en_core_web_md

# debug
#RUN apt-get update && apt-get install iodbc -y

COPY src "$ROOT"/src

#ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/usr/local/lib"
#COPY "$OWN_LOCATION"/odbc.ini /usr/local/etc/odbc.ini
COPY "$OWN_LOCATION"/odbc.ini /root/.odbc.ini
COPY "$OWN_LOCATION"/odbcinst.ini /root/.odbcinst.ini
COPY "$OWN_LOCATION"/etc_odbcinst.ini /etc/odbcinst.ini
COPY "$OWN_LOCATION"/etc_odbc.ini /etc/odbc.ini

VOLUME /root/corpora_out

WORKDIR "$ROOT"/src

#CMD bash

