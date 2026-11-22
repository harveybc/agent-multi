#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
py=${CAMPAIGN_PYTHON:-/var/tmp/discrete-sac-probe-venv/bin/python}
receipt=/var/tmp/role429-sell-ranking/DONE.json
if [[ ! -f "$receipt" ]]; then
    "$py" tools/role429_campaign.py
fi
"$py" tools/role429_report.py
if "$py" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["pairs"]==6; sys.exit(0 if d["decision"]=="reject_for_integration" else 2)' "$receipt"; then
    "$py" tools/role430_sell_swaps.py
else
    rc=$?
    if [[ "$rc" != 2 ]]; then exit "$rc"; fi
    echo "Review integration qualification; no automatic model promotion."
fi
