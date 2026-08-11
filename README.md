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
`2 * mean_recurrence - 1`, adds a uniformly sampled teacher-depth offset from
the configured inclusive range, and combines next-token cross-entropy with
temperature-scaled teacher-to-student KL divergence. The teacher is a separate,
frozen copy of the model whose weights are updated after every student optimizer
step as `teacher = teacher_momentum * teacher + (1 - teacher_momentum) * student`.
`teacher_momentum` must be in `[0, 1)`; setting it to `0` keeps the teacher
weights equal to the student weights, while values close to `1` produce a
slower-moving teacher. The pretraining config uses `0.999`. The default offsets
of 2 through 6 give teacher depths from 3 through 13 when `mean_recurrence` is
4.

The recurrent-distillation evaluator is independent of the training-time
teacher schedule. It evaluates the default reference depths of 1, 2, 4, and 8
loops, or the explicitly configured evaluation depths.
