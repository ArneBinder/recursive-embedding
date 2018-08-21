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


## use third argument as cpu set, if available
if [ -n "$3" ]; then
    CPU_SET="$3"
else
    NBR_CPUS=4
    echo "NBR_CPUS: $NBR_CPUS"
    CPU_SET=$(($2 * $NBR_CPUS))-$(($2 * $NBR_CPUS + $NBR_CPUS - 1))
fi

export CPU_SET
echo "CPU_SET: $CPU_SET"
export NVIDIA_VISIBLE_DEVICES
echo "NVIDIA_VISIBLE_DEVICES: $NVIDIA_VISIBLE_DEVICES"

LOG_FN="$HOST_TRAIN/$PROJECT_NAME"_gpu"$NVIDIA_VISIBLE_DEVICES.log"

echo "log to: $LOG_FN"
echo "container_name: train_gpu$NVIDIA_VISIBLE_DEVICES"_"$PROJECT_NAME"

docker-compose -p "$PROJECT_NAME""_gpu$NVIDIA_VISIBLE_DEVICES" up train-fold > "$LOG_FN"
