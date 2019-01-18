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

# SICK relatedness
./train.sh "$USE_GPUS" RELATEDNESS/SICK/EDGES/RECNN train-settings/task/relatedness.env train-settings/model/recnn.env train-settings/dataset/corenlp/sick-edges.env "$ENV_GENERAL"

# SICK entailment
./train.sh "$USE_GPUS" ENTAILMENT/SICK/EDGES/RECNN train-settings/task/entailment.env train-settings/model/recnn.env train-settings/dataset/corenlp/sick-edges.env "$ENV_GENERAL"

# SEMEVAL relation extraction
./train.sh "$USE_GPUS" RE/SEMEVAL2010T8/EDGES/RECNN train-settings/task/re-semeval.env train-settings/model/recnn.env train-settings/dataset/corenlp/semeval-edges.env "$ENV_GENERAL"

# TACRED relation extraction
./train.sh "$USE_GPUS" RE/TACRED/EDGES/RECNN train-settings/task/re-tacred.env train-settings/model/recnn.env train-settings/dataset/corenlp/tacred-edges.env "$ENV_GENERAL"

# IMDB sentiment
./train.sh "$USE_GPUS" SENTIMENT/IMDB/EDGES/RECNN train-settings/task/sentiment.env train-settings/model/recnn.env train-settings/dataset/corenlp/imdb-edges.env "$ENV_GENERAL"
