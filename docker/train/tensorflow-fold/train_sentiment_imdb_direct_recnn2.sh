#!/bin/sh

USE_GPUS="$1"
echo "USE_GPUS=$USE_GPUS"
shift

ENV_GENERAL="train-settings/general/gpu-train-dev.env"
## use next argument as gpu nbr, if available
if [ -n "$1" ]; then
    ENV_GENERAL="$1"
    shift
fi
echo "ENV_GENERAL=$ENV_GENERAL"


## IMDB sentiment prediction

## DIRECT

# RECNN
./train.sh "$USE_GPUS" SENTIMENT/IMDB/DIRECT/RECNN2 train-settings/task/sentiment.env train-settings/model/recnn.env train-settings/dataset/corenlp/imdb-direct.env "$ENV_GENERAL"
