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


## SICK entailment prediction

## DIRECT

# RECNN
./train.sh "$USE_GPUS" ENTAILMENT/SICK/DIRECT/RECNN train-settings/task/entailment.env train-settings/model/recnn.env train-settings/dataset/corenlp/sick-direct.env "$ENV_GENERAL"

# BOW
./train.sh "$USE_GPUS" ENTAILMENT/SICK/DIRECT/RNN train-settings/task/entailment.env train-settings/model/bow.env train-settings/dataset/corenlp/sick-direct.env "$ENV_GENERAL"

# RNN
./train.sh "$USE_GPUS" ENTAILMENT/SICK/DIRECT/BOW train-settings/task/entailment.env train-settings/model/rnn.env train-settings/dataset/corenlp/sick-direct.env "$ENV_GENERAL"

## EDGES

# RECNN
./train.sh "$USE_GPUS" ENTAILMENT/SICK/EDGES/RECNN train-settings/task/entailment.env train-settings/model/recnn.env train-settings/dataset/corenlp/sick-edges.env "$ENV_GENERAL"
