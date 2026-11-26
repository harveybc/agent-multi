#!/usr/bin/env bash
# Run as a user service, independent of the interactive session.
set -euo pipefail
root=/var/tmp/critic489
here=$(cd "$(dirname "$0")" && pwd)
script="$here/role489_critic_factorial.py"
remote=${1:?supply the second worker SSH alias}
runner=(env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
  PYTHONPATH=/var/tmp/critic369-runtime-81bcbe36:/var/tmp/selector-024-20261104
  STRATEGY_ENGINE_ROOT=/var/tmp/sac70-warm-v3-20261108/engine /var/tmp/discrete-sac-probe-venv/bin/python)
mkdir -p "$root/weeks"
if [[ ! -f "$root/frozen/spec.json" ]]; then "${runner[@]}" "$script" prepare; fi
remote_start() {
  ssh -o ConnectTimeout=4 "$remote" mkdir -p "$root/weeks" &&
  rsync -a /var/tmp/critic369-runtime-81bcbe36 "$remote:/var/tmp/" &&
  rsync -a "$root/frozen" "$script" "$remote:$root/" &&
  ssh "$remote" systemd-run --user --collect --unit=critic489-shard "${runner[@]}" "$root/role489_critic_factorial.py" worker --part remote --workers 12
}
if remote_start; then
  "${runner[@]}" "$script" worker --part local --workers 8
  # Collect atomic per-week outputs. Recover unfinished weeks locally on a
  # host failure or after ten minutes with no newly completed remote week.
  last=-1; stale=0
  while (( stale < 40 )); do
    timeout 25 rsync -a --include='*.json' --exclude='*' "$remote:$root/weeks/" "$root/weeks/" || break
    if ! ssh -o ConnectTimeout=4 "$remote" systemctl --user is-active --quiet critic489-shard; then
      timeout 25 rsync -a --include='*.json' --exclude='*' "$remote:$root/weeks/" "$root/weeks/" || true
      break
    fi
    count=$(find "$root/weeks" -name '*.json' | wc -l)
    if (( count == last )); then stale=$((stale+1)); else stale=0; fi
    last=$count; sleep 15
  done
fi
"${runner[@]}" "$script" worker --part all --workers 10
"${runner[@]}" "$script" analyze
