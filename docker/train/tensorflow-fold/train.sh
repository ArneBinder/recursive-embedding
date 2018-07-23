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

NVIDIA_VISIBLE_DEVICES=0
## use second argument as gpu nbr, if available
if [ -n "$2" ]; then
    NVIDIA_VISIBLE_DEVICES="$2"
fi

NBR_CPUS=4
## use third argument as cpu nbr, if available
if [ -n "$3" ]; then
    NBR_CPUS="$3"
fi

echo "NBR_CPUS: $NBR_CPUS"
export NVIDIA_VISIBLE_DEVICES
echo "NVIDIA_VISIBLE_DEVICES: $NVIDIA_VISIBLE_DEVICES"
export CPU_SET=$(($2 * $NBR_CPUS))-$(($2 * $NBR_CPUS + $NBR_CPUS - 1))
echo "CPU_SET: $CPU_SET"

LOG_FN="$HOST_TRAIN/$PROJECT_NAME.log"

echo "log to: $LOG_FN"
echo "container_name: train_gpu$NVIDIA_VISIBLE_DEVICES"_"$PROJECT_NAME"

docker-compose -p "$PROJECT_NAME" up train-fold > "$LOG_FN"
