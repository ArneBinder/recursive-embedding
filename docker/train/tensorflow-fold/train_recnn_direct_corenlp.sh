#!/bin/sh

USE_GPUS="$1"
echo "USE_GPUS=$USE_GPUS"

# SICK relatedness
./train.sh "$USE_GPUS" RELATEDNESS/SICK/DIRECT/RECNN train-settings/general/gpu.env train-settings/model/recnn.env train-settings/task/relatedness.env train-settings/dataset/corenlp/sick-direct.env

# SICK entailment
./train.sh "$USE_GPUS" ENTAILMENT/SICK/DIRECT/RECNN train-settings/general/gpu.env train-settings/model/recnn.env train-settings/task/entailment.env train-settings/dataset/corenlp/sick-direct.env

# SEMEVAL relation extraction
./train.sh "$USE_GPUS" RE/SEMEVAL2010T8/DIRECT/RECNN train-settings/general/gpu.env train-settings/model/recnn.env train-settings/task/re-semeval.env train-settings/dataset/corenlp/semeval-direct.env

# TACRED relation extraction
./train.sh "$USE_GPUS" RE/TACRED/DIRECT/RECNN train-settings/general/gpu.env train-settings/model/recnn.env train-settings/task/re-tacred.env train-settings/dataset/corenlp/tacred-direct.env

# IMDB sentiment
./train.sh "$USE_GPUS" SENTIMENT/IMDB/DIRECT/RECNN train-settings/general/gpu.env train-settings/model/recnn.env train-settings/task/sentiment.env train-settings/dataset/corenlp/imdb-direct.env
