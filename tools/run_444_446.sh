#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=.:tools
py="${PYTHON_BIN:-/var/tmp/discrete-sac-probe-venv/bin/python}"
for stage in role444_weekly_swaps role445_weekly_transfer role446_transfer_gate; do
    "$py" "tools/${stage}.py"
done
