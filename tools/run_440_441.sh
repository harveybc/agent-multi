#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=.:tools
python_bin="${PYTHON_BIN:-/var/tmp/discrete-sac-probe-venv/bin/python}"
"$python_bin" tools/role440_reentry.py
"$python_bin" tools/role441_shadow_reentry.py
"$python_bin" tools/role442_activity_placebo.py
"$python_bin" tools/role443_weekly_refit.py
