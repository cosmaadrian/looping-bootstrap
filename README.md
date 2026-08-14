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

## Related work
https://arxiv.org/pdf/2303.01469
https://arxiv.org/pdf/2410.11081 <-- close to us

## TODOs

## Improvements:
- [X] Use LoopFormer trick: add a time-conditioned embedding at each layer.
- [ ] Log or check to see if teacher ends up in a loop or in a fixed point attractor

## Test-Time Training

- [ ] Semi-Supervised learning (i.e., self distillation) on questions requiring algorithmic steps (incorporating reasoning inside parameters)
- [ ] From initial embeddings, add some noise K times, (i.e., K different starting positions)
- [ ] From each of the K starting positions, do 2N loops, average logits, sharpen (this is the teacher)
- [ ] Train a student with N loops on the average logits. Something about agreement at multiple depth levels --> this allows us to check for divergences (or convergences or answers)
- [ ] One nice thing about looping transformers is that we can decode at each recurrence level without problems. We can then leverage that.