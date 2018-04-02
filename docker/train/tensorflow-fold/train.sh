#!/bin/bash

## add variables from .env file
my_dir="$(dirname "$0")"
project_root_dir="$my_dir/../../.."
echo "MY_DIR=$my_dir"
echo "PROJECT_ROOT_DIR=$project_root_dir"
source "$my_dir/.env"

## check variables content
array=( HOST_CORPORA_OUT HOST_TRAIN TRAIN_DATA TRAIN_LOGDIR HOST_PORT_NOTEBOOK HOST_PORT_TENSORBOARD LIMIT_CPUS CPU_SET NV_GPU )
echo ".env vars:"
for i in "${array[@]}"
do
	if [ -z "${!i}" ]; then
        echo "   ATTENTION: $i is NOT SET"
    else
        echo "   $i=${!i}"
    fi
done


## set command and docker image depending on if GPUs are configured as available
if [ -n "$NV_GPU" ]; then
    DOCKERFILE="Dockerfile.tf1_3_gpu"
    IMAGE="tensorflowfold:tf1_3_gpu"
    COMMAND="NV_GPU=$NV_GPU nvidia-docker"
else
    DOCKERFILE="Dockerfile.tf1_3_gpu"
    IMAGE="tensorflowfold:tf1_3_cpu_mkl"
    COMMAND="docker"
fi

echo "execute COMMAND: $COMMAND @IMAGE: $IMAGE"


## build docker image, if it does not exist
if [ -z $(docker images "$IMAGE" -q) ]; then
    echo image: "$IMAGE" not found, build it
    docker build -f "$my_dir/$DOCKERFILE" -t "$IMAGE" "$project_root_dir"
else
    echo use available image: "$IMAGE"
fi


## start training
$COMMAND run \
    -v $HOST_TRAIN:/root/train \
    -v $HOST_CORPORA_OUT:/root/corpora_out \
    -p $HOST_PORT_NOTEBOOK:8888 \
    $IMAGE \
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