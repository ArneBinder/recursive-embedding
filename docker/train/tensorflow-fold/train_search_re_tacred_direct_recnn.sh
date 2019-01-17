#!/bin/sh

USE_GPUS="$1"

# TACRED relation extraction
./train.sh "$USE_GPUS" RE/TACRED/DIRECT/RECNN train-settings/general/gpu-search.env train-settings/model/recnn.env train-settings/task/re-tacred.env train-settings/dataset/corenlp/tacred-direct.env
