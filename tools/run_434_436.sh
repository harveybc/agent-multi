#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py=${CAMPAIGN_PYTHON:-/var/tmp/discrete-sac-probe-venv/bin/python}
if [[ ! -s /var/tmp/role434-monitor/DONE.json ]]; then "$py" tools/role434_campaign.py; fi
if ! "$py" -c 'import json,sys; d=json.load(open("/var/tmp/role434-monitor/DONE.json")); assert d["stages"]==18; sys.exit(0 if d["qualified"]==0 else 2)'; then
    echo "434 prerequisite not met; no probe launch" >&2
    exit 1
fi
if [[ ! -s /var/tmp/role435-within-week/DONE.json ]]; then "$py" tools/role435_campaign.py; fi
"$py" tools/role436_gate_replay.py
