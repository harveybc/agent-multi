#!/usr/bin/env bash
set -euo pipefail
src="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /var/tmp/role413-pinned-runtime
export PYTHONPATH="$PWD:/var/tmp/selector-024-20261104"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export STRATEGY_ENGINE_ROOT=/var/tmp/sac70-warm-v3-20261108/engine
py=/var/tmp/discrete-sac-probe-venv/bin/python
while [[ ! -f /var/tmp/role421-sell-value/DONE.json ]] && systemctl --user is-active --quiet role421-sell-value.service; do
    sleep 5
done
"$py" "$src/role421_sell_value.py"
"$py" "$src/role422_sell_cross.py"
