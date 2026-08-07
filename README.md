# Recurrent-depth distillation

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

`src/configs/pretraining-config.yaml` selects `recurrent_distillation_trainer`.
Each batch samples a student depth uniformly from 1 through
`2 * mean_recurrence - 1`, runs a detached teacher at the configured depth
multiplier, and combines next-token cross-entropy with temperature-scaled
teacher-to-student KL divergence.

The recurrent-distillation evaluator records a pre-training generation baseline
and evaluates the default reference depths (1, 2, 4, 8, 16, and 32 loops) plus
the maximum possible teacher depth. It logs loss and token accuracy at every
depth, `accuracy_delta@r_to_2r_loops`, shallow before/after improvements, and
whether the model at the original teacher depth beats that baseline.
