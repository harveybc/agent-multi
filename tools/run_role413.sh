#!/usr/bin/env bash
set -euo pipefail
src="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runtime=/var/tmp/role413-pinned-runtime
if [[ ! -d "$runtime" ]]; then
    tmp="$(mktemp -d /var/tmp/role413-init.XXXXXX)"
    cp -a /var/tmp/role408-runtime/. "$tmp"
    cp "$src/role404_fit.py" "$tmp/tools/role404_fit.py"
    mv -T "$tmp" "$runtime"
fi
cmp "$src/role404_fit.py" "$runtime/tools/role404_fit.py"
cd "$runtime"
export PYTHONPATH="$runtime:/var/tmp/selector-024-20261104"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export STRATEGY_ENGINE_ROOT=/var/tmp/sac70-warm-v3-20261108/engine
exec /var/tmp/discrete-sac-probe-venv/bin/python "$src/role413_sequence.py"
