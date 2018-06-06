#!/bin/bash

## use first argument as environment file name, if available
if [ -n "$1" ]; then
    ENV_FN="$1"
else
    ENV_FN=.env
fi
echo "use environment variables from: $ENV_FN"
source "$ENV_FN"

## use second argument as log output file name, if available
if [ -n "$2" ]; then
    LOG_FN="$2"
else
    LOG_FN="$HOST_TRAIN/train$NVIDIA_VISIBLE_DEVICES"_"$PROJECT_NAME.log"
fi
echo "log to: $LOG_FN"

docker-compose -p "$PROJECT_NAME" up train-fold > "$LOG_FN"
