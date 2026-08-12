#!/usr/bin/env bash
# Screen-gate verification for the P1LR DECISION unit (finding 233 /
# finding 226). Called by ExecStartPre of p1lr-decision@.service BEFORE
# any training process starts, so a decision worker can never launch on
# an absent, malformed, foreign or non-viable gate — and can never
# silently fall back to a screen run.
#
#   usage: p1lr_decision_gate_check.sh <screen_verdict.json> [contract.json]
#
# Verifies, in order:
#   1. the gate file EXISTS and parses as JSON;
#   2. schema  == agent_multi.p1_difficulty_lr_screen_verdict.v1;
#   3. outcome == SCREEN_VIABLE_REGION (the ONLY outcome that authorizes
#      the decision budget);
#   4. gates.replica_terminal_loads is boolean true (finding 225);
#   5. the gate's contract_sha256 equals sha256(contract file), so the
#      gate belongs to the contract this unit runs.
#
# Exit codes follow the runner's EXIT_CLASS contract:
#   0  gate verified
#   4  REFUSED_* — a configuration refusal, never retried by systemd
#      (p1lr-decision@.service sets RestartPreventExitStatus=4)
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Documents/GitHub/agent-multi}"
GATE="${1:-${P1LR_SCREEN_GATE:-}}"
CONTRACT="${2:-$REPO_DIR/examples/config/phase_3_eth_sac_dynamics/p1_difficulty_lr_factorial_v1.json}"

if [[ -z "$GATE" ]]; then
    echo "REFUSED_SCREEN_GATE_MISSING: no screen-gate path supplied;" \
         "decision mode requires a pinned verified gate" >&2
    exit 4
fi

PY="${P1LR_PYTHON:-$HOME/anaconda3/envs/trading-stack/bin/python}"
if [[ ! -x "$PY" ]]; then
    PY="$(command -v python3 || true)"
fi
if [[ -z "$PY" ]]; then
    echo "REFUSED_NO_PYTHON: no interpreter available to verify the gate" >&2
    exit 4
fi

"$PY" - "$GATE" "$CONTRACT" <<'PYEOF' || exit 4
import hashlib
import json
import sys

SCHEMA = "agent_multi.p1_difficulty_lr_screen_verdict.v1"
gate_path, contract_path = sys.argv[1], sys.argv[2]
refusals = []

try:
    with open(gate_path, "rb") as fh:
        gate = json.loads(fh.read().decode("utf-8"))
except FileNotFoundError:
    refusals.append(f"REFUSED_SCREEN_GATE_MISSING: {gate_path} does not exist")
    gate = None
except (OSError, ValueError) as exc:
    refusals.append(f"REFUSED_SCREEN_GATE_UNREADABLE: {type(exc).__name__}")
    gate = None

if isinstance(gate, dict):
    if gate.get("schema") != SCHEMA:
        refusals.append(
            f"REFUSED_SCREEN_GATE_SCHEMA: {gate.get('schema')!r} != {SCHEMA!r}")
    if gate.get("outcome") != "SCREEN_VIABLE_REGION":
        refusals.append(
            f"REFUSED_SCREEN_NOT_VIABLE: outcome {gate.get('outcome')!r} — "
            "only SCREEN_VIABLE_REGION authorizes the decision budget")
    if (gate.get("gates") or {}).get("replica_terminal_loads") is not True:
        refusals.append(
            "REFUSED_REPLICA_PROOF_MISSING: gates.replica_terminal_loads is "
            "not boolean true (finding 225)")
    try:
        digest = hashlib.sha256(open(contract_path, "rb").read()).hexdigest()
    except OSError as exc:
        digest = None
        refusals.append(f"REFUSED_CONTRACT_UNREADABLE: {type(exc).__name__}")
    if digest is not None and gate.get("contract_sha256") != digest:
        refusals.append(
            "REFUSED_SCREEN_GATE_FOREIGN: the gate binds contract sha "
            f"{gate.get('contract_sha256')} but this unit runs {digest}")
elif gate is not None:
    refusals.append("REFUSED_SCREEN_GATE_SHAPE: gate is not a JSON object")

payload = {
    "check": "p1lr_decision_screen_gate",
    "gate_path": gate_path,
    "contract_path": contract_path,
    "outcome": (gate or {}).get("outcome") if isinstance(gate, dict) else None,
    "verified": not refusals,
    "refusals": refusals,
}
print(json.dumps(payload, sort_keys=True))
sys.exit(0 if not refusals else 1)
PYEOF

echo "p1lr decision screen gate VERIFIED: $GATE"
