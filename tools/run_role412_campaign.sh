#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /var/tmp/role408-runtime
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="/var/tmp/role408-runtime:/var/tmp/selector-024-20261104"
export STRATEGY_ENGINE_ROOT=/var/tmp/sac70-warm-v3-20261108/engine
exec /var/tmp/discrete-sac-probe-venv/bin/python "$script_dir/role412_campaign.py"
