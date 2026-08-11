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
`2 * mean_recurrence - 1`. With `teacher_depth_mode: additive`, it adds a
uniformly sampled teacher-depth offset from the configured inclusive range.
With `teacher_depth_mode: multiplicative`, it instead computes
`teacher_depth = teacher_depth_multiplier * student_depth`; the multiplier must
be an integer of at least 2. The loss combines next-token cross-entropy with
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

## Width–Depth μP

The recurrent stack implements the parameterization from
[Width–Depth μP](https://github.com/ML-GSAI/Width-Depth-muP). Enable it with
`depth_alpha_enabled: 1`, set `depth_multiplier` to the nominal effective depth
divided by the base effective depth, and choose `depth_alpha_exp` in `[0.5, 1]`.
The reference experiments use `depth_alpha_exp: 1.0`.

For a run whose mean recurrence and layer count match the base model,
`depth_multiplier` is `1`. If only `mean_recurrence` grows from 4 to 8, use
`depth_multiplier: 2`; if `num_layers` also grows from 4 to 8, use `4`.

For a sampled forward with depth `l`, nominal mean recurrence `L`, configured
depth multiplier `D`, and exponent `alpha`, every recurrent attention and
feed-forward update is scaled by `(D * l / L)^(-alpha)`. Student and teacher
therefore use their own intended depths. The optimizer applies the corresponding
`D^(alpha - 1)` learning-rate correction to recurrent parameters and the
reference Adam epsilon corrections. Embeddings, pre/post blocks, and the readout
remain outside the depth learning-rate correction. Setting
`depth_alpha_enabled: 0` retains width-only μP.
