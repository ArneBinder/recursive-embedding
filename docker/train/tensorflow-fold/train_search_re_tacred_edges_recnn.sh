#!/bin/sh

USE_GPUS="$1"
echo "USE_GPUS=$USE_GPUS"

# TACRED relation extraction
./train.sh "$USE_GPUS" RE/TACRED/EDGES/RECNN train-settings/general/gpu-search.env train-settings/model/recnn.env train-settings/task/re-tacred.env train-settings/dataset/corenlp/tacred-edges.env
