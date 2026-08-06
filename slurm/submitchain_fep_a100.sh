#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./submit_sequence.sh NUM_JOBS NUM_GPUS /path/to/generate_sequences.sh
#
# Example:
#   ./submit_sequence.sh 20 4 ./generate_sequences.sh

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

export HF_HOME="${HF_HOME:-/export/projects/nlp/.cache}"
export ENV_NAME="${ENV_NAME:-fep}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-student}"

: "${SLURM_TIMELIMIT:?Set SLURM_TIMELIMIT in the environment or .env}"

NUM_JOBS="${1:?Usage: $0 NUM_JOBS NUM_GPUS SCRIPT}"
NUM_GPUS="${2:?Usage: $0 NUM_JOBS NUM_GPUS SCRIPT}"
SCRIPT="${3:?Usage: $0 NUM_JOBS NUM_GPUS SCRIPT}"

if ! [[ "$NUM_JOBS" =~ ^[0-9]+$ ]] || [ "$NUM_JOBS" -lt 1 ]; then
    echo "NUM_JOBS must be a positive integer; got '$NUM_JOBS'" >&2
    exit 2
fi

if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [ "$NUM_GPUS" -lt 1 ]; then
    echo "NUM_GPUS must be a positive integer; got '$NUM_GPUS'" >&2
    exit 2
fi

SCRIPT_BASENAME="$(basename "$SCRIPT")"
SCRIPT_NAME="${SCRIPT_BASENAME%.*}"

PARTITION="dgxa100"
NTASKS="1"
CPUS_PER_TASK="8"
MEM_PER_CPU="8G"

prev_job_id=""

for i in $(seq 1 "$NUM_JOBS"); do
    job_name="${SCRIPT_NAME}_${i}"

    if [[ -z "$prev_job_id" ]]; then
        job_id="$(
            sbatch --parsable \
                --job-name="$job_name" \
                --export=ALL,NUM_GPUS="$NUM_GPUS",ENV_NAME="$ENV_NAME",HF_HOME="$HF_HOME" \
                --gres="gpu:${NUM_GPUS}" \
                -p "$PARTITION" \
                --time="$SLURM_TIMELIMIT" \
                -A "$SLURM_ACCOUNT" \
                --ntasks="$NTASKS" \
                --cpus-per-task="$CPUS_PER_TASK" \
                --mem-per-cpu="$MEM_PER_CPU" \
                "$SCRIPT"
        )"
    else
        job_id="$(
            sbatch --parsable \
                --job-name="$job_name" \
                --dependency="afterany:${prev_job_id}" \
                --export=ALL,NUM_GPUS="$NUM_GPUS",ENV_NAME="$ENV_NAME",HF_HOME="$HF_HOME" \
                --gres="gpu:${NUM_GPUS}" \
                -p "$PARTITION" \
                --time="$SLURM_TIMELIMIT" \
                -A "$SLURM_ACCOUNT" \
                --ntasks="$NTASKS" \
                --cpus-per-task="$CPUS_PER_TASK" \
                --mem-per-cpu="$MEM_PER_CPU" \
                "$SCRIPT"
        )"
    fi

    echo "Submitted job $i/$NUM_JOBS: $job_id"
    prev_job_id="$job_id"
done
