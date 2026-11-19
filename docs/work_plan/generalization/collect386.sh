#!/usr/bin/env bash
# Collect only completed exports; replay with no GPU and without writing to workers.
set -euo pipefail
: "${PRIMARY_HOST:?}" "${REPLICA_HOST:?}" "${REMOTE_RUN:?}" "${LOCAL_FIT:?}" "${PYTHON:?}"
declare -a ssh_opts=(-o BatchMode=yes -o ConnectTimeout=8)
mkdir -p "$LOCAL_FIT"
deadline=$((SECONDS+3600))
for seed in 19 7 43; do
    host=$REPLICA_HOST
    if [[ "$seed" == 19 ]]; then host=$PRIMARY_HOST; fi
    until ssh "${ssh_opts[@]}" "$host" test -s "$REMOTE_RUN/$seed/RESULT.json"; do
        if (( SECONDS >= deadline )); then echo "Timed out waiting for seed $seed" >&2; exit 1; fi
        status=$(ssh "${ssh_opts[@]}" "$host" systemctl --user is-active advantage-probe386.service || true)
        if [[ "$status" == failed || "$status" == inactive ]]; then
            ssh "${ssh_opts[@]}" "$host" test -s "$REMOTE_RUN/$seed/RESULT.json" || exit 1
        fi
        sleep 5
    done
    rsync -a --checksum "$host:$REMOTE_RUN/$seed" "$LOCAL_FIT/"
done
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "$PYTHON" docs/work_plan/generalization/advantage_account386.py --fit "$LOCAL_FIT" "$@"
