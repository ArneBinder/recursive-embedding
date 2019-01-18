#!/bin/sh

USE_GPUS="$1"
echo "USE_GPUS=$USE_GPUS"
shift

ENV_GENERAL="train-settings/general/gpu-check-data.env"
## use next argument as gpu nbr, if available
if [ -n "$1" ]; then
    ENV_GENERAL="$1"
    shift
fi
echo "ENV_GENERAL=$ENV_GENERAL"

## RECNN

# SEMEVAL relation extraction
./train.sh "$USE_GPUS" RE/SEMEVAL2010T8/SPAN/RECNN train-settings/task/re-semeval.env train-settings/model/recnn.env train-settings/dataset/corenlp/semeval-span.env "$ENV_GENERAL"

# TACRED relation extraction
./train.sh "$USE_GPUS" RE/TACRED/SPAN/RECNN train-settings/task/re-tacred.env train-settings/model/recnn.env train-settings/dataset/corenlp/tacred-span.env "$ENV_GENERAL"


## BOW

# SEMEVAL relation extraction
./train.sh "$USE_GPUS" RE/SEMEVAL2010T8/SPAN/BOW train-settings/task/re-semeval.env train-settings/model/bow.env train-settings/dataset/corenlp/semeval-span.env "$ENV_GENERAL"

# TACRED relation extraction
./train.sh "$USE_GPUS" RE/TACRED/SPAN/BOW train-settings/task/re-tacred.env train-settings/model/bow.env train-settings/dataset/corenlp/tacred-span.env "$ENV_GENERAL"


## RNN -> not necessary, BOW creates flat data