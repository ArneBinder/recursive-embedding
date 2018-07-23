# Train a rec-emb model with rec-emb data

Copy the `docker/train/tensorflow-fold/.env.dev` to `docker/train/tensorflow-fold/.env` and adapt its parameters for
your needs. Start training by executing:
```bash
cd docker/train/tensorflow-fold && docker-compose up train-fold-cpu
```

Or start via helper script `train.sh`:
```bash
# set execution permission
chmod +x docker/train/tensorflow-fold/train.sh
cd docker/train/tensorflow-fold && train.sh <ENV_FILE> <NVIDIA_VISIBLE_DEVICES> <NBR_CPUS>
```
`ENV_FILE` (default: `.env`): path to .env file (will be copied into log directory for later reference)
`NVIDIA_VISIBLE_DEVICES` (default: `0`): gpu devices that will be used by tensorflow
`NBR_CPUS` (default: `4`): use the first `NBR_CPUS` for data loading etc.

`train.sh` sets an appropriate container name and saves the logging output `$HOST_TRAIN/$PROJECT_NAME.log` (see `.env` file)


