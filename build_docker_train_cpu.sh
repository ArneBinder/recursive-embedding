#!/bin/sh

echo Building tensorflowfold_conda:pre
docker build -t tensorflowfold_conda:pre . -f docker/tensorflow_fold/tensorflowfold_conda_pre/Dockerfile

echo Building tensorflowfold_conda_cpu:tf1.3
docker build -t tensorflowfold_conda_cpu:tf1.3 . -f docker/tensorflow_fold/tensorflowfold_conda_tf1.3_mkl/Dockerfile
