#!/bin/sh

#echo Building tensorflowfold_conda:pre
#docker build -t tensorflowfold_conda:pre . -f docker/tensorflow_fold/tensorflowfold_conda_pre/Dockerfile

echo Building tensorflowfold_cpu:tf1.3
docker build -t tensorflowfold_cpu:tf1.3 . -f docker/tensorflow_fold/tensorflowfold_conda_tf1.3_mkl/Dockerfile

## start
## docker run -v HOST_PATH:CONTAINER_PATH tensorflowfold_cpu:tf1.3 --train_data_path=/mnt/DATA/ML/data/corpora_out/corpora/DBPEDIANIF/100/merged/forest --model_type=tuple --dev_file_index=0 --batch_size=10 --tree_embedder=TreeEmbedding_HTU_reduceSUM_mapGRU --learning_rate=0.003 --optimizer=AdamOptimizer --early_stop_queue=0 --auto_restore=False --root_fc_size=0 --leaf_fc_size=300 --state_size=150 --keep_prob=0.9 --logdir=/mnt/DATA/ML/training/tf/supervised/log/DEBUG/DBPEDIANIF/1000X_new --init_only=False --concat_mode=tree --max_depth=10 --context=0
## works, but executes as root:
# docker run -v /home/abinder/train:/root/train -v /home/abinder/corpora:/root/corpora_out tensorflowfold_cpu:tf1.3 --train_data_path=/root/corpora_out/DBPEDIANIF/100/merged/forest --model_type=tuple --dev_file_index=0 --batch_size=10 --tree_embedder=TreeEmbedding_HTU_reduceSUM_mapGRU --learning_rate=0.003 --optimizer=AdamOptimizer --early_stop_queue=0 --auto_restore=False --root_fc_size=0 --leaf_fc_size=300 --state_size=150 --keep_prob=0.9 --logdir=/root/train/tf/supervised/log/DEBUG/DBPEDIANIF/FC1000 --init_only=False --concat_mode=tree --max_depth=10 --context=0
