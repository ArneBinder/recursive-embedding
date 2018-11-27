#!/bin/bash

NVIDIA_VISIBLE_DEVICES=0
## use first argument as gpu nbr, if available
if [ -n "$1" ]; then
    NVIDIA_VISIBLE_DEVICES="$1"
    shift
fi

## enable exporting all variables that are defined from now on
set -a
## use remaining arguments as environment file names, if available
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

LOG_FN="$HOST_TRAIN/$PROJECT_NAME"_gpu"$NVIDIA_VISIBLE_DEVICES.log"

echo "log to: $LOG_FN"
echo "container_name: train_gpu$NVIDIA_VISIBLE_DEVICES"_"$PROJECT_NAME"

docker-compose -p "$PROJECT_NAME""_gpu$NVIDIA_VISIBLE_DEVICES" up train-fold > "$LOG_FN"
