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
#
# HOW SYSTEMD CONSUMES THIS (R3, order 2026-09-11). The unit calls this
# script from ExecCondition=, NOT ExecStartPre=. The difference is not
# cosmetic: RestartPreventExitStatus= is compared against the exit code
# of the MAIN process, and when ExecStartPre fails the main process
# never runs, so ExecMainStatus stays 0 and Restart=on-failure retries
# a configuration refusal every RestartSec until the start limit
# freezes it. ExecCondition treats any exit in 1..254 as "this unit
# must not run now": the remaining commands are skipped, the unit is
# NOT marked failed, and no restart is scheduled. Exit 255 or a signal
# remains a genuine harness failure and stays distinguishable.
#
# The typed outcome is not lost in the skip: every invocation writes a
# durable record under P1LR_GATE_STATE_DIR naming the refusal class, so
# a skipped unit is always accompanied by the exact reason.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Documents/GitHub/agent-multi}"
GATE="${1:-${P1LR_SCREEN_GATE:-}}"
CONTRACT="${2:-$REPO_DIR/examples/config/phase_3_eth_sac_dynamics/p1_difficulty_lr_factorial_v1.json}"

if [[ -z "$GATE" ]]; then
    echo "REFUSED_SCREEN_GATE_MISSING: no screen-gate path supplied;" \
         "decision mode requires a pinned verified gate" >&2
    exit 4
fi

STATE_DIR="${P1LR_GATE_STATE_DIR:-$HOME/.local/state/agent-multi/p1lr-gate}"
INSTANCE="${P1LR_GATE_INSTANCE:-unknown}"
mkdir -p "$STATE_DIR"
RECORD="$STATE_DIR/p1lr-decision@$INSTANCE.gate.json"

PY="${P1LR_PYTHON:-$HOME/anaconda3/envs/trading-stack/bin/python}"
if [[ ! -x "$PY" ]]; then
    PY="$(command -v python3 || true)"
fi
if [[ -z "$PY" ]]; then
    echo "REFUSED_NO_PYTHON: no interpreter available to verify the gate" >&2
    exit 4
fi

"$PY" - "$GATE" "$CONTRACT" "$RECORD" <<'PYEOF' || exit 4
import hashlib
import json
import sys

SCHEMA = "agent_multi.p1_difficulty_lr_screen_verdict.v1"
gate_path, contract_path, record_path = sys.argv[1:4]
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
# The skip must never be silent: whatever systemd does with the exit
# code, the typed outcome lands in a durable record beside it.
payload["schema"] = "agent_multi.p1lr_decision_gate_check.v1"
payload["refusal_classes"] = sorted({r.split(":", 1)[0] for r in refusals})
payload["disposition"] = (
    "GATE_VERIFIED" if not refusals else "REFUSED_NOT_EXECUTABLE")
try:
    import datetime as _dt
    payload["observed_at"] = (_dt.datetime.now(_dt.timezone.utc)
                              .replace(microsecond=0).isoformat()
                              .replace("+00:00", "Z"))
    with open(record_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
        fh.write("\n")
except OSError as exc:
    # A record we cannot write is reported, never swallowed — but it
    # does not turn a verified gate into a refusal.
    print(json.dumps({"gate_record_error": type(exc).__name__}))

print(json.dumps(payload, sort_keys=True))
sys.exit(0 if not refusals else 1)
PYEOF

echo "p1lr decision screen gate VERIFIED: $GATE"
