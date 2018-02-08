FROM joernhees/virtuoso:latest
#FROM ubuntu:16.04
#FROM python:2.7-alpine

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        python-dev \
        python-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install --no-cache-dir setuptools

# install unixODBC
RUN curl -O 'ftp://ftp.unixodbc.org/pub/unixODBC/unixODBC-2.3.5.tar.gz' \
    && tar -xz -f unixODBC-2.3.5.tar.gz \
    && cd unixODBC-2.3.5 \
    && ./configure \
    && make \
    && make install \
    && rm /unixODBC-2.3.5.tar.gz \
    && rm -rf /unixODBC-2.3.5

RUN mkdir /recursive-embedding
COPY requirements.txt recursive-embedding/requirements.txt
RUN pip install --no-cache-dir -r /recursive-embedding/requirements.txt


#COPY requirements.txt recursive-embedding/requirements.txt
#RUN pip --no-cache-dir install -r recursive-embedding/requirements.txt
#RUN python -m spacy download en_core_web_md
#COPY src recursive-embedding/src
#VOLUME /root/corpora_in /root/corpora_out
#WORKDIR /root/recursive-embedding/src

#CMD bash

