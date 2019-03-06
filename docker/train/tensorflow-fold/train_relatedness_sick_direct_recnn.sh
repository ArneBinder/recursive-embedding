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


## SICK relatedness prediction

## DIRECT

# RECNN
./train.sh "$USE_GPUS" RELATEDNESS/SICK/DIRECT/RECNN train-settings/task/relatedness.env train-settings/model/recnn.env train-settings/dataset/corenlp/sick-direct.env "$ENV_GENERAL"