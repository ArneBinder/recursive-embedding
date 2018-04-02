#!/bin/bash

## add variables from .env file
my_dir="$(dirname "$0")"
source "$my_dir/.env"

array=( TRAIN_DATA TRAIN_LOGDIR HOST_PORT_NOTEBOOK HOST_PORT_TENSORBOARD LIMIT_CPUS CPU_SET NV_GPU )
echo ".env vars:"
for i in "${array[@]}"
do
	if [ -z "${!i}" ]; then
        echo "   ATTENTION: $i is NOT SET"
    else
        echo "   $i=${!i}"
    fi
done


if [ -n "$NV_GPU" ]; then
    COMMAND="NV_GPU=$NV_GPU nvidia-docker"
else
    COMMAND=docker
fi

echo execute command: $COMMAND

$COMMAND run \
    -v $HOST_TRAIN:/root/train \
    -v $HOST_CORPORA_OUT:/root/corpora_out \
    -p $HOST_PORT_NOTEBOOK:8888 \
    tensorflowfold:tf1_3_cpu_mkl \
    /root/recursive-embedding/set-user-with-folder.sh /root/train \
    python train_fold.py \
        --train_data_path=/root/corpora_out/$TRAIN_DATA \
        --logdir=/root/train/$TRAIN_LOGDIR \
        --model_type=tuple \
        --dev_file_index=0 \
        --batch_size=10 \
        --tree_embedder=TreeEmbedding_HTU_reduceSUM_mapGRU \
        --learning_rate=0.003 \
        --optimizer=AdamOptimizer \
        --early_stop_queue=0 \
        --root_fc_size=0 \
        --leaf_fc_size=300 \
        --state_size=150 \
        --keep_prob=0.9 \
        --init_only=False \
        --concat_mode=tree \
        --max_depth=10 \
        --context=0