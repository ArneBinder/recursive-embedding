#!/bin/sh

# Continue all training runs in the directory gieven as first argument. The remaining arguments are passed to "pyton train_fold.py".
# call this like:
# sh continue_all.sh /home/arne/ML_local/tf/supervised/log/SA/600 2> train_fold-err.log 1> train_fold.log

path=$1
shift
echo path: $path
echo arguments: $*

find $path -name "checkpoint" -exec sh -c 'python train_fold.py --logdir_continue=$(dirname "{}")'" $*" \;


