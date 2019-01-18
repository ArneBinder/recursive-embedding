#!/bin/sh

USE_GPUS="$1"
echo "USE_GPUS=$USE_GPUS"

# SEMEVAL relation extraction
./train.sh "$USE_GPUS" RE/SEMEVAL2010T8/SPAN/RNN train-settings/general/gpu.env train-settings/task/re-semeval.env train-settings/model/rnn.env train-settings/dataset/corenlp/semeval-span.env

# TACRED relation extraction
./train.sh "$USE_GPUS" RE/TACRED/SPAN/RNN train-settings/general/gpu.env train-settings/task/re-tacred.env train-settings/model/rnn.env train-settings/dataset/corenlp/tacred-span.env
