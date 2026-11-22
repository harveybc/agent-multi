#!/usr/bin/env bash
# Launch an admitted research arm on the selected host; never start a second writer.
set -euo pipefail
index=$1 years=$2 seed=$3
root=/var/tmp/sac399-series
me=$(nvidia-smi -i "$index" --query-gpu=uuid --format=csv,noheader)
exec systemd-run --user --unit="sac399-${years}y-s${seed}" \
  --working-directory="$root/runtime" --property=MemoryMax=5G --property=Nice=5 \
  --setenv="CUDA_VISIBLE_DEVICES=$me" --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=MKL_NUM_THREADS=1 --setenv=OMP_NUM_THREADS=1 --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  --setenv=PYTHONPATH=.:/var/tmp/selector-024-20261104 --setenv=PYTHONUNBUFFERED=1 \
  --setenv=STRATEGY_ENGINE_ROOT=/var/tmp/sac70-warm-v3-20261108/engine \
  /var/tmp/discrete-sac-probe-venv/bin/python -m tools.series399 --root "$root" --years "$years" --seed "$seed"
