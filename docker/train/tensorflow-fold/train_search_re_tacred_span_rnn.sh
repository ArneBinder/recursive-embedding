#!/bin/sh

USE_GPUS="$1"

# TACRED relation extraction
./train.sh "$USE_GPUS" RE/TACRED/SPAN/RNN train-settings/general/gpu-search.env train-settings/model/rnn.env train-settings/task/re-tacred.env train-settings/dataset/corenlp/tacred-span.env
