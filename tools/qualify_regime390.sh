#!/usr/bin/env bash
# Offline CPU-only sequence. All inputs/outputs are explicit, no broker or workers.
set -euo pipefail
if [[ $# != 7 ]]; then
    echo "usage: bash $0 PYTHON EMPTY_OUT SPEC FACTOR BUSINESS BUNDLE PARENT" >&2
    exit 2
fi
py=$1 out=$2 spec=$3 factor=$4 business=$5 bundle=$6 parent=$7
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export STRATEGY_ENGINE_ROOT="$parent/engine" PYTHONPATH=".:${PYTHONPATH:-}"
"$py" -m pytest -q tests/unit/test_regime390.py
"$py" -m tools.regime390 --out "$out" --spec "$spec" --factor "$factor" --business "$business" --recipe "$bundle/RECIPE.json"
"$py" -m tools.native390 --root "$out" --spec "$spec" --parent "$parent" --bundle "$bundle"
"$py" -m tools.inspect_regime390_inputs --root "$out" --spec "$spec"
"$py" -m tools.transfer390 --root "$out" --business "$business" --spec "$spec"
"$py" -m tools.cohorts390 --screen "$out" --out "$out/banks" --spec "$spec" --factor "$factor" --business "$business" --recipe "$bundle/RECIPE.json"
