#!/bin/bash

## use first argument as environment file name, if available
if [ -n "$1" ]; then
    ENV_FN="$1"
else
    ENV_FN=.env
fi
echo "use environment variables from: $ENV_FN"
source "$ENV_FN"
# copy to .env for docker-compose
cp "$ENV_FN" .env
# copy used .env file into logdir
cp "$ENV_FN" "$HOST_TRAIN/$TRAIN_LOGDIR/"

## use second argument as gpu nbr (and adjust used cpus), if available
if [ -n "$2" ]; then
    NVIDIA_VISIBLE_DEVICES="$2"
    echo "NVIDIA_VISIBLE_DEVICES: $NVIDIA_VISIBLE_DEVICES"
    cpu_count=4
    CPU_SET=$(($2 * $cpu_count))-$(($2 * $cpu_count + $cpu_count - 1))
    echo "CPU_SET: $CPU_SET"
fi

## use second argument as log output file name, if available
#if [ -n "$2" ]; then
#    LOG_FN="$2"
#else
LOG_FN="$HOST_TRAIN/$PROJECT_NAME.log"
#fi

echo "log to: $LOG_FN"
echo "container_name: train_gpu$NVIDIA_VISIBLE_DEVICES"_"$PROJECT_NAME"

docker-compose -p "$PROJECT_NAME" up train-fold > "$LOG_FN"
