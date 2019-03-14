# Train one or multiple rec-emb models with rec-emb data

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

4. Adapt parameters in (at least one) environment files in folder `train-settings/general`. One of them will be used for training.

5. Start training process by executing: `./<TRAIN_SCRIPT> <GPU_ID> <GENERAL_TRAIN_SETTING_FILE>`, e.g. `./train_entailment_sick_direct_recnn.sh 0 train-settings/general/cpu-train-dev.env`

NOTE:
 * The `GPU_ID` will not be used for cpu settings, but is still required as placeholder.
 * DONT use `train.sh` as `TRAIN_SCRIPT`. It is called by the specific train setting scripts and sets an appropriate container name and saves the logging output `$HOST_TRAIN/logs/$PROJECT_NAME.log` (see `.env` file).
