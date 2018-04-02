#!/usr/bin/env bash

## change this!
#HOST_CORPORA_OUT=/home/abinder/corpora
#HOST_TRAIN=/home/abinder/train
HOST_CORPORA_OUT=/mnt/DATA/ML/data/corpora_out
HOST_TRAIN=/mnt/DATA/ML/training

## general resource limitations
LIMIT_CPUS=4.0
CPU_SET=0-3
#NV_GPU=0

## general settings
HOST_PORT_NOTEBOOK=8887
HOST_PORT_TENSORBOARD=6005

## training settings
TRAIN_DATA=DBPEDIANIF/100/merged/forest
TRAIN_LOGDIR=tf_new/supervised/log/DEBUG/DBPEDIANIF/FC1000

echo TRAIN_DATA=$TRAIN_DATA
echo TRAIN_LOGDIR=$TRAIN_LOGDIR
echo HOST_PORT_NOTEBOOK=$HOST_PORT_NOTEBOOK
echo HOST_PORT_TENSORBOARD=$HOST_PORT_TENSORBOARD
echo LIMIT_CPUS=$LIMIT_CPUS
echo CPU_SET=$CPU_SET
echo NV_GPU=$NV_GPU

if [ -n "$NV_GPU" ]; then
    COMMAND="NV_GPU=$NV_GPU nvidia-docker"
else
    COMMAND=docker
fi

echo COMMAND=$COMMAND

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