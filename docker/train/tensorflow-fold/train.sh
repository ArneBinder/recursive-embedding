#!/bin/bash

NVIDIA_VISIBLE_DEVICES=0
## use first argument as gpu nbr, if available
if [ -n "$1" ]; then
    NVIDIA_VISIBLE_DEVICES="$1"
    shift
fi

## use second argument as logdir, if available
if [ -n "$1" ]; then
    TRAIN_LOGDIR="$1"
    export TRAIN_LOGDIR
    echo "TRAIN_LOGDIR: $TRAIN_LOGDIR"
    shift
fi

## use remaining arguments as environment file names, if available

## enable exporting all variables that are defined from now on
set -a
while (( "$#" )); do
    echo "use environment variables from: $1"
    source "$1"
    shift
done
## disable automatic variable export
set +a

## set cpu_set
NBR_CPUS=4
echo "NBR_CPUS: $NBR_CPUS"
CPU_SET=$(($NVIDIA_VISIBLE_DEVICES * $NBR_CPUS))-$(($NVIDIA_VISIBLE_DEVICES * $NBR_CPUS + $NBR_CPUS - 1))

export CPU_SET
echo "CPU_SET: $CPU_SET"
export NVIDIA_VISIBLE_DEVICES
echo "NVIDIA_VISIBLE_DEVICES: $NVIDIA_VISIBLE_DEVICES"

PROJECT_NAME=${TRAIN_LOGDIR//\//_}
CONTAINER_NAME=${PROJECT_NAME}_gpu${NVIDIA_VISIBLE_DEVICES}
export CONTAINER_NAME
## create log dir (not TRAIN_LOGDIR!) if it does nto exist
mkdir -p "$HOST_TRAIN/logs"
LOG_FN="$HOST_TRAIN/logs/$CONTAINER_NAME.log"

echo "log to: $LOG_FN"
echo "container_name: $CONTAINER_NAME"

docker-compose -p "$CONTAINER_NAME" up train-fold > "$LOG_FN"
