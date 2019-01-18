#!/bin/sh

USE_GPUS="$1"
echo "USE_GPUS=$USE_GPUS"

## TACRED relation extraction on DIRECT data

# RECNN
./train.sh "$USE_GPUS" RE/TACRED/DIRECT/RECNN train-settings/general/gpu-train-dev.env train-settings/task/re-tacred.env train-settings/model/recnn.env train-settings/dataset/corenlp/tacred-span.env

# BOW
./train.sh "$USE_GPUS" RE/TACRED/DIRECT/RNN train-settings/general/gpu-train-dev.env train-settings/task/re-tacred.env train-settings/model/bow.env train-settings/dataset/corenlp/tacred-span.env

# RNN
./train.sh "$USE_GPUS" RE/TACRED/DIRECT/BOW train-settings/general/gpu-train-dev.env train-settings/task/re-tacred.env train-settings/model/rnn.env train-settings/dataset/corenlp/tacred-span.env
