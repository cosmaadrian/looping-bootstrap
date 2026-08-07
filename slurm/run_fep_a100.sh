#!/usr/bin/env bash
set -euo pipefail

REQUESTED_NUM_GPUS="${1:?Usage: $0 NUM_GPUS SCRIPT}"
SCRIPT="${2:?Usage: $0 NUM_GPUS SCRIPT}"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

if ! [[ "$REQUESTED_NUM_GPUS" =~ ^[0-9]+$ ]] || [ "$REQUESTED_NUM_GPUS" -lt 1 ]; then
    echo "NUM_GPUS must be a positive integer; got '$REQUESTED_NUM_GPUS'" >&2
    exit 2
fi

export NUM_GPUS="$REQUESTED_NUM_GPUS"
export ENV_NAME="${ENV_NAME:-fep}"
# export HF_HOME="${HF_HOME:-/export/projects/nlp/.cache}"

: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT in the environment or .env}"
: "${SLURM_TIMELIMIT:?Set SLURM_TIMELIMIT in the environment or .env}"

sbatch --export=ALL,NUM_GPUS="$NUM_GPUS",ENV_NAME="$ENV_NAME" --gres="gpu:${NUM_GPUS}" -A "$SLURM_ACCOUNT" -p dgxa100 --time="$SLURM_TIMELIMIT" --ntasks 1 --cpus-per-task 8 --mem-per-cpu 8G "$SCRIPT"
# sbatch --gres gpu:$NUM_GPUS -A $SLURM_ACCOUNT -p h200 --time=$SLURM_TIMELIMIT --ntasks 1 --cpus-per-task 8 --mem-per-cpu 8G $SCRIPT
