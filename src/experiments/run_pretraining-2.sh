#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" # use the NLP cache
export SLURM_JOB_ID="${SLURM_JOB_ID:-local}"
export ENV_NAME="${ENV_NAME:-fep}"
export NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"

###########################################################################
###########################################################################
###########################################################################

cd "src/"
ARGS=(--config_file ./configs/pretraining-config.yaml --group test --env "$ENV_NAME" --debug 0 --name test-pretrain-justloops --recurrent_distillation 0)

######################################################################################################
##################################### VANILLA #######################################################
#####################################################################################################

uv run torchrun --standalone --nnodes 1 --nproc-per-node "$NUM_GPUS" main.py "${ARGS[@]}"
