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


## TACRED relation extraction via LM

## DIRECT + add_heads=EDG
./train.sh "$USE_GPUS" RE/TACRED/DIRECT/LM train-settings/task/re-semeval.env train-settings/model/ml.env train-settings/dataset/corenlp/tacred-direct.env train-settings/specific/add-heads-EDG.env "$ENV_GENERAL"

## EDGES
./train.sh "$USE_GPUS" RE/TACRED/EDGES/LM train-settings/task/re-semeval.env train-settings/model/ml.env train-settings/dataset/corenlp/tacred-edges.env "$ENV_GENERAL"
