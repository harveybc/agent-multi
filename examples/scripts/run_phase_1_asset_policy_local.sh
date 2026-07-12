#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROFILE="${1:-smoke}"
MACHINE="${2:-omega}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE="$ROOT/examples/config/phase_1_asset_policy/phase_1_asset_policy_solusdt_4h_sac_config.json"
OVERLAY="$ROOT/configs/runtime/${MACHINE}.json"

case "$PROFILE" in
  baseline)
    CONFIG="$BASE"
    EXTRA=()
    ;;
  smoke)
    CONFIG="$ROOT/examples/config/phase_1_asset_policy/optimization/phase_1_asset_policy_solusdt_4h_sac_smoke_optimization_config.json"
    EXTRA=(--base_config "$BASE")
    ;;
  optimization)
    CONFIG="$ROOT/examples/config/phase_1_asset_policy/optimization/phase_1_asset_policy_solusdt_4h_sac_optimization_config.json"
    EXTRA=()
    ;;
  inference)
    CONFIG="$ROOT/examples/config/phase_1_asset_policy/inference/phase_1_asset_policy_solusdt_4h_sac_inference_config.json"
    EXTRA=(--base_config "$BASE")
    ;;
  *)
    echo "usage: $0 {baseline|smoke|optimization|inference} {omega|dragon|gamma_5070ti|gamma_5090}" >&2
    exit 2
    ;;
esac

if [[ ! -f "$OVERLAY" ]]; then
  echo "runtime overlay does not exist: $OVERLAY" >&2
  exit 2
fi

cd "$ROOT"
"$PYTHON_BIN" examples/scripts/validate_phase_1_asset_policy.py \
  --runtime-overlay "$OVERLAY"

ARGS=(
  -m app.main
  "${EXTRA[@]}"
  --load_config "$CONFIG"
  --runtime_overlay "$OVERLAY"
)
if [[ "${OPTIMIZATION_RESUME:-0}" == "1" ]]; then
  ARGS+=(--optimization_resume true)
fi
exec "$PYTHON_BIN" "${ARGS[@]}"
