# Matryoshka Updates

Run experiments with

```bash
NUM_GPUS=2
ENV_NAME=fep

./slurm/run_fep_a100.sh $NUM_GPUS ./src/experiments/run_pretraining.sh
```

For a local single-node run, use the same experiment script:

```bash
NUM_GPUS=1 ENV_NAME=fep ./src/experiments/run_pretraining.sh
```
