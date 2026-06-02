#!/usr/bin/env bash
set -u

AGENT="/home/harveybc/Documents/GitHub/agent-multi"
FD="/home/harveybc/Documents/GitHub/financial-data"
PY="/home/harveybc/anaconda3/envs/tensorflow/bin/python"
STATE_FILE="experiments/stage3x_sac_smoke_plan/stage3x_sac_smoke_dispatch_state.json"

while true; do
  cd "$AGENT" || exit 2
  remaining="$("$PY" - <<'PYCODE'
import json
from pathlib import Path

state = json.loads(Path("experiments/stage3x_sac_smoke_plan/stage3x_sac_smoke_dispatch_state.json").read_text())
print(sum(1 for task in state.get("tasks", []) if task.get("status") in {"pending", "running"}))
PYCODE
)"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) remaining=${remaining}"
  if [ "$remaining" = "0" ]; then
    break
  fi
  sleep 120
done

cd "$FD" || exit 2
"$PY" _scripts/workers/stage3x_sac_smoke_result_synthesis_worker.py
"$PY" _scripts/workers/stage3x_absurdity_guard_worker.py
"$PY" _scripts/workers/stage3x_sac_smoke_request_worker.py
