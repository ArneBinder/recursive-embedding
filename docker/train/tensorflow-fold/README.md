# Train one or multiple rec-emb models with rec-emb data

This is configured by several environment files located in [train-settings](train-settings). [train-settings/general](train-settings/general) provide major entry points.

## HOW TO train

1. Install:
 * docker
 * docker compose

2. Clone this repo and switch into this folder:
```bash
git clone https://github.com/ArneBinder/recursive-embedding.git
cd recursive-embedding/docker/train/tensorflow-fold
```

3. Set execution permission of train scripts: `chmod +x *.sh`

4. Adapt parameters in (at least one) environment file(s) in folder `train-settings/general`. One of them will be used for training.

5. Start training process by executing: `./<TRAIN_SCRIPT> <GPU_ID> <GENERAL_TRAIN_SETTING_FILE>`, e.g. `./train_entailment_sick_direct_recnn.sh 0 train-settings/general/cpu-train-dev.env`

NOTE:
 * When training is started the first time, it needs quite a long time (up to 2h!) to start because tensorflow fold and tensorflow have to be compiled for the machine.
 * The `<GPU_ID>` will not be used for cpu settings, but is still required as placeholder.
 * DONT use `train.sh` as `<TRAIN_SCRIPT>`. It is called by the specific train setting scripts and sets an appropriate container name and saves the logging output `$HOST_TRAIN/logs/$PROJECT_NAME.log` (see `.env` files in [train-settings/general](train-settings/general)).

## Post-Processing

In general, settings are configured to create a `scores.tsv` file located per train execution. Note that some train scripts execute several settings. All `scores.tsv` for one kind of task (relatedness prediction, recognizing textual entailment, sentiment analysis, or relation extraction) can be collect in a final tsv with help of [/src/eval_merge_results.py](/src/eval_merge_results.py). It requires the python packages `numpy` and `plac`. When training for all runs is finished, call it like:

```bash
cd PATH/TO/HOST_TRAIN
python /PATH/TO/recursive_embedding/src/eval_merge_results.py TASK/PARENT/FOLDER
```
where `TASK/PARENT/FOLDER` is the relative path to the parent of the directories containing `scores.tsv` files (arbitrary nested) you are interested in. The merged result is written to `TASK_PARENT_FOLDER.tsv`.

NOTE: You can use the flag `-e` to automatically calculate F1 macro scores for the RE task with the official SemEval 2010 Task 8 evaluation script.

