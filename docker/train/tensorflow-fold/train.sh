#!/bin/bash

## add variables from .env file
MY_DIR="$(dirname "$0")"
project_root_dir="$MY_DIR/../../.."
echo "MY_DIR=$MY_DIR"
echo "PROJECT_ROOT_DIR=$project_root_dir"
source "$MY_DIR/.env"

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
    COMMAND="NV_GPU=$NV_GPU nvidia-docker run \
    --env-file $MY_DIR/.env \
    -v $HOST_TRAIN:/root/train \
    -v $HOST_CORPORA_OUT:/root/corpora_out \
    -p $HOST_PORT_NOTEBOOK:8888 \
    $IMAGE"
else
    DOCKERFILE="Dockerfile.tf1_3_cpu_mkl"
    IMAGE="tensorflowfold:tf1_3_cpu_mkl"
    COMMAND="docker run \
    --env-file $MY_DIR/.env \
    -v $HOST_TRAIN:/root/train \
    -v $HOST_CORPORA_OUT:/root/corpora_out \
    -p $HOST_PORT_NOTEBOOK:8888 \
    $IMAGE"
fi

echo "execute COMMAND: '$COMMAND' @IMAGE: $IMAGE"


## build docker image, if it does not exist
if [ -z $(docker images "$IMAGE" -q) ] || [ "$DONT_REBUILD_IMAGE" != 1 ]; then
    echo image: "$IMAGE not found, build it with $DOCKERFILE"
    docker build \
        -f "$MY_DIR/$DOCKERFILE" \
        --build-arg OWN_LOCATION="$MY_DIR" \
        --build-arg PROJECT_ROOT=/root/recursive-embedding \
        -t "$IMAGE" "$project_root_dir"
else
    echo use available image: "$IMAGE"
fi


## start training
#$COMMAND run \
#    --env-file "$MY_DIR/.env" \
#    -v $HOST_TRAIN:/root/train \
#    -v $HOST_CORPORA_OUT:/root/corpora_out \
#    -p $HOST_PORT_NOTEBOOK:8888 \
#    $IMAGE