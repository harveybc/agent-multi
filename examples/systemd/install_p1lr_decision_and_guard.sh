#!/usr/bin/env bash
# Install the rootless P1LR DECISION unit and the 15-minute idle-guard
# service+timer on ONE host (findings 229 + 233).
#
# REVIEWED OPERATOR STEP — this script is shipped, never auto-executed,
# and it deliberately does NOT enable or start anything. It copies the
# unit files, creates the per-host environment templates, reloads the
# user manager and PRINTS the exact enable commands for the owner to
# run. Enabling a unit is an owner decision, not a side effect of a
# code review.
#
#   bash examples/systemd/install_p1lr_decision_and_guard.sh
#
# It is rootless (systemd --user) and idempotent. It never touches a
# system unit, never installs packages and never stops a running
# process — a decision run launched directly (nohup) keeps running; the
# unit takes over at the next explicit start.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Documents/GitHub/agent-multi}"
UNIT_DIR="$HOME/.config/systemd/user"
CONF_DIR="$HOME/.config/agent-multi"
GUARD_ENV="$CONF_DIR/p1lr-idle-guard.env"
GUARD_STATE="$HOME/.local/state/agent-multi/p1lr-idle-guard"

# Seeds assigned to THIS host, from the contract (never hardcoded).
CONTRACT="$REPO_DIR/examples/config/phase_3_eth_sac_dynamics/p1_difficulty_lr_factorial_v1.json"
PY="${P1LR_PYTHON:-$HOME/anaconda3/envs/trading-stack/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python3)"

mkdir -p "$UNIT_DIR" "$CONF_DIR" "$GUARD_STATE"

install -m 0644 "$REPO_DIR/examples/systemd/p1lr-decision@.service" "$UNIT_DIR/"
install -m 0644 "$REPO_DIR/examples/systemd/p1lr-idle-guard.service" "$UNIT_DIR/"
install -m 0644 "$REPO_DIR/examples/systemd/p1lr-idle-guard.timer" "$UNIT_DIR/"
chmod +x "$REPO_DIR/examples/systemd/p1lr_decision_gate_check.sh"

# Preserve an existing per-host guard tuning file; create a commented
# template otherwise. The unit's own default is --mode decision.
if [[ ! -f "$GUARD_ENV" ]]; then
    cat > "$GUARD_ENV" <<'EOF'
# P1LR idle guard, per-host overrides (finding 233).
# The unit default is P1LR_GUARD_MODE=decision — the DECISION root is
# the live one. Uncomment to guard the screen root instead:
# P1LR_GUARD_MODE=screen
# Extra flags, e.g. --idle-after-seconds 1800:
# P1LR_GUARD_EXTRA_ARGS=
EOF
fi

# Verify the pinned screen gate now, so a broken decision unit is found
# at install time and not at 03:00 on a restart.
GATE="${P1LR_SCREEN_GATE:-$HOME/.local/share/agent-multi/p1lr_collection_cd823e2b_20260812/screen_verdict.json}"
if bash "$REPO_DIR/examples/systemd/p1lr_decision_gate_check.sh" "$GATE"; then
    gate_state="VERIFIED"
else
    gate_state="REFUSED (fix before enabling p1lr-decision@)"
fi

systemctl --user daemon-reload

seeds="$("$PY" - "$CONTRACT" <<'PYEOF'
import json, socket, sys
contract = json.load(open(sys.argv[1]))
host = socket.gethostname()
print(" ".join(seed for seed, a in (contract.get("assignments") or {}).items()
                if isinstance(a, dict) and a.get("hostname") == host))
PYEOF
)"

echo
echo "installed (NOT enabled) on $(hostname):"
echo "  $UNIT_DIR/p1lr-decision@.service"
echo "  $UNIT_DIR/p1lr-idle-guard.service"
echo "  $UNIT_DIR/p1lr-idle-guard.timer"
echo "  screen gate: $gate_state"
echo "  guard env:   $GUARD_ENV (mode default: decision)"
echo
echo "OWNER enable commands for this host:"
echo "  systemctl --user enable --now p1lr-idle-guard.timer"
for seed in $seeds; do
    echo "  systemctl --user enable --now p1lr-decision@${seed}.service" \
         "   # only when the direct nohup worker for seed ${seed} has stopped"
done
echo
echo "verify:"
echo "  systemctl --user list-timers p1lr-idle-guard.timer --no-pager"
echo "  systemctl --user status p1lr-idle-guard.service --no-pager | tail -20"
echo "  $PY $REPO_DIR/tools/multifront_status.py --p1lr-mode decision --no-l1"
