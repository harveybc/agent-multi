#!/usr/bin/env bash
# CPU-only; diagnostic artifacts are not an authorization to train a policy.
set -euo pipefail
: "${PYTHON:?}" "${BUSINESS:?}" "${SPEC:?}" "${CLOCK:?}" "${FACTOR:?}" "${RECIPES:?}" "${OUT:?}" "${ENGINE:?}" "${ZIGZAG_SOURCE:?}"
export CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2
mkdir "$OUT"
"$PYTHON" -m tools.scenario393 --catalog "$BUSINESS/real_catalog.parquet" --bank "$BUSINESS/19.npz" \
  --clock "$CLOCK" --out "$OUT/macro"
"$PYTHON" -m tools.decoder393_data --spec "$SPEC" --bank "$BUSINESS/19.npz" \
  --profiles "$OUT/macro/TRAIN_PROFILES.parquet" --out "$OUT/data" \
  --recipes "$RECIPES/61/bundle/RECIPE.json" "$RECIPES/71/bundle/RECIPE.json"
for width in 61 71; do
  for variant in plain bridge; do
    args=()
    if [[ "$variant" == bridge ]]; then args+=(--bridge); fi
    root="$OUT/$width-$variant"
    "$PYTHON" -m tools.probe393_diffusion "$OUT/data/$width/EXAMPLES.npz" "$root" \
      --epochs 12 --objective velocity "${args[@]}"
    for seed in 41 73; do
      for start in 0 -1; do
        "$PYTHON" -m tools.rollout393 --recipe "$RECIPES/$width/bundle/RECIPE.json" \
          --spec "$SPEC" --bank "$BUSINESS/19.npz" --factor "$FACTOR" \
          --script "$OUT/macro/SCRIPT.parquet" --checkpoint "$root/PROBE.pt" \
          --out "$root/path-$seed-$start" --seed "$seed" --start-row "$start"
      done
    done
  done
done
"$PYTHON" -m tools.score393 --root "$OUT" --engine "$ENGINE" --source "$ZIGZAG_SOURCE" --bank "$BUSINESS/19.npz"
