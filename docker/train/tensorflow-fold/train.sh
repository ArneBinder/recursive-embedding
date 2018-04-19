#!/bin/bash

## get current directory
HOST_SCRIPT_DIR="$(dirname "$0")"
cd "$HOST_SCRIPT_DIR"
HOST_SCRIPT_DIR="$(pwd)"
echo "HOST_SCRIPT_DIR=$HOST_SCRIPT_DIR"

## project root is three folders above SCRIPT_DIR
cd "$HOST_SCRIPT_DIR/../../.."
HOST_PROJECT_ROOT_DIR="$(pwd)"
echo "HOST_PROJECT_ROOT_DIR=$HOST_PROJECT_ROOT_DIR"

## add variables from .env file
source "$HOST_SCRIPT_DIR/.env"

## check variables content
array=( HOST_CORPORA_OUT HOST_TRAIN TRAIN_DATA TRAIN_LOGDIR HOST_PORT_NOTEBOOK HOST_PORT_TENSORBOARD MEM_LIMIT CPU_SET NV_GPU MODEL_TYPE DEV_FILE_INDEX BATCH_SIZE BATCH_ITER TREE_EMBEDDER LEARNING_RATE OPTIMIZER EARLY_STOP_QUEUE ROOT_FC_SIZES LEAF_FC_SIZE FC_SIZES STATE_SIZE KEEP_PROB INIT_ONLY CONCAT_MODE MAX_DEPTH CONTEXT NEG_SAMPLES NO_FIXED_VECS TRAIN_FILES )
echo ".env vars:"
for i in "${array[@]}"
do
	if [ -z "${!i}" ]; then
        echo "ATTENTION: $i is NOT SET"
    else
        echo "   $i=${!i}"
    fi
done


## set command and docker image depending on if GPUs are configured as available
if [ -n "$NV_GPU" ]; then
    DOCKERFILE="Dockerfile.tf1_3_gpu"
    IMAGE="tensorflowfold:tf1_3_gpu"
    export NV_GPU="$NV_GPU"
    DOCKER="nvidia-docker"
else
    DOCKERFILE="Dockerfile.tf1_3_cpu_mkl"
    IMAGE="tensorflowfold:tf1_3_cpu_mkl"
    DOCKER="docker"
fi

echo "execute COMMAND: '$DOCKER' @IMAGE: $IMAGE"

DOCKER_PROJECT_ROOT=/root/recursive-embedding

## build docker image, if it does not exist
if [ -z $(docker images "$IMAGE" -q) ] || [ "$REBUILD_IMAGE" == 1 ]; then
    echo image: "$IMAGE not found, build it with $DOCKERFILE"
    docker build \
        -f "$HOST_SCRIPT_DIR/$DOCKERFILE" \
        --build-arg OWN_LOCATION="$HOST_SCRIPT_DIR" \
        --build-arg PROJECT_ROOT="$DOCKER_PROJECT_ROOT" \
        -t "$IMAGE" "$HOST_PROJECT_ROOT_DIR"
else
    echo use available image: "$IMAGE"
fi


## start training
$DOCKER run -it \
    --cpuset-cpus "$CPU_SET" \
    --env-file "$HOST_SCRIPT_DIR/.env" \
    --memory-swap "$MEM_LIMIT" \
    --memory "$MEM_LIMIT" \
    -v "$HOST_TRAIN:/root/train" \
    -v "$HOST_CORPORA_OUT:/root/corpora_out" \
    -v "$HOST_PROJECT_ROOT_DIR/src:$DOCKER_PROJECT_ROOT/src" \
    -p "$HOST_PORT_NOTEBOOK:8888" \
    $IMAGE \
        --train_data_path=/root/corpora_out/$TRAIN_DATA \
        --logdir=/root/train/$TRAIN_LOGDIR \
        --model_type=$MODEL_TYPE \
        --dev_file_index=$DEV_FILE_INDEX \
        --batch_size=$BATCH_SIZE \
        --tree_embedder=$TREE_EMBEDDER \
        --learning_rate=$LEARNING_RATE \
        --optimizer=$OPTIMIZER \
        --early_stop_queue=$EARLY_STOP_QUEUE \
        --root_fc_sizes=$ROOT_FC_SIZES \
        --leaf_fc_size=$LEAF_FC_SIZE \
        --fc_sizes=$FC_SIZES \
        --state_size=$STATE_SIZE \
        --keep_prob=$KEEP_PROB \
        --init_only=$INIT_ONLY \
        --concat_mode=$CONCAT_MODE \
        --max_depth=$MAX_DEPTH \
        --context=$CONTEXT \
        --neg_samples=$NEG_SAMPLES \
        --train_files=$TRAIN_FILES \
        --no_fixed_vecs=$NO_FIXED_VECS \
        --batch_iter=$BATCH_ITER
