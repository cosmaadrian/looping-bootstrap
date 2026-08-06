#!/usr/bin/env bash
set -euo pipefail

root="$1"

to_cancel=("$root")
seen=" $root "

while true; do
    found=0

    # List pending jobs with their remaining dependencies
    while read -r jobid deps; do
        [[ -z "${jobid:-}" || "$deps" == "(null)" ]] && continue

        for parent in "${to_cancel[@]}"; do
            if [[ "$deps" =~ (^|[^0-9])$parent([^0-9]|$) ]]; then
                if [[ "$seen" != *" $jobid "* ]]; then
                    to_cancel+=("$jobid")
                    seen+=" $jobid "
                    found=1
                fi
            fi
        done
    done < <(squeue -h -u "$USER" -t PENDING -o "%A %E")

    [[ "$found" -eq 0 ]] && break
done

printf '%s\n' "${to_cancel[@]}"
scancel "${to_cancel[@]}"
