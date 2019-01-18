#!/bin/sh

USE_GPUS="$1"
echo "USE_GPUS=$USE_GPUS"
shift

ENV_GENERAL="train-settings/general/gpu-search.env"
## use next argument as gpu nbr, if available
if [ -n "$1" ]; then
    ENV_GENERAL="$1"
    shift
fi
echo "ENV_GENERAL=$ENV_GENERAL"



# TACRED relation extraction
./train.sh "$USE_GPUS" RE/TACRED/DIRECT/RECNN train-settings/task/re-tacred.env train-settings/model/recnn.env train-settings/dataset/corenlp/tacred-direct.env "$ENV_GENERAL"

