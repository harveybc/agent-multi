#!/usr/bin/env bash
# Install the identity-specific P1LR v2 supervision on ONE host (WO4,
# finding AUD-GEN-20260815-250).
#
# REVIEWED OPERATOR STEP — this script is shipped, never auto-executed,
# and it deliberately ENABLES NOTHING and STARTS NOTHING. It copies the
# unit template, the 20-v2-identity.conf drop-in, the per-seed v2
# identity files and the READ-ONLY lease-gate control bundle, verifies
# the v2 screen gate and the transition lease READ-ONLY, reloads the
# user manager and PRINTS the exact next-safe-boundary commands for the
# operator.
#
# Finding AUD-GEN-20260816-256: the per-seed identity files carry the
# suffix .env.conf, NOT .env — the repository's credential ignore rules
# (*.env / .env / .env.*) silently swallowed the earlier .env payload, so
# this glob used to match nothing at all. Non-secret by construction:
# seed, hostname, GPU UUID, mode, contract path + sha256, chain id,
# experiment and output root. No token, key or account fact.
#
# Finding AUD-GEN-20260816-261: the lease gate is a restart admission
# gate and therefore must not execute from the mutable canonical
# checkout. It is installed READ-ONLY (0444 in a 0555 directory) under a
# CONTENT-ADDRESSED path named after CONTROL_MANIFEST.sha256's own
# digest, which the drop-in pins literally and re-verifies at every unit
# start.
#
# ABSOLUTE RULE (order 2026-08-15 WO4): the live v2 decision PIDs
# (chain cdf30aebf585385b, nohup from the immutable runtime worktree)
# continue UNTOUCHED. No systemctl start/restart of a seed's unit while
# that seed's matching v2 PID is alive. Enabling a unit WITHOUT --now
# is the only arming step this script even prints for the pre-boundary
# window: it starts nothing and reconstructs v2 after a reboot.
#
#   bash examples/systemd/install_p1lr_v2_identity_supervision.sh
#
# Rootless (systemd --user), idempotent, never installs packages and
# never stops a running process.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Documents/GitHub/agent-multi}"
UNIT_DIR="$HOME/.config/systemd/user"
DROPIN_DIR="$UNIT_DIR/p1lr-decision@.service.d"
ENV_DIR="$HOME/.config/agent-multi/p1lr-v2"
CONTROL_ROOT="$HOME/.local/lib/agent-multi/p1lr-control"
RUNTIME_DIR="${P1LR_V2_RUNTIME_DIR:-$HOME/Documents/GitHub/.runtime/agent-multi-p1lr-v2-924910fe}"
GATE="${P1LR_V2_SCREEN_GATE:-$HOME/.local/share/agent-multi/p1lr_v2_collections_20260815/screen_14e7ce82/screen_verdict.json}"
GUARD_ENV="$HOME/.config/agent-multi/p1lr-idle-guard.env"

PY="${P1LR_PYTHON:-$HOME/anaconda3/envs/trading-stack/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python3)"
SHA256SUM="${P1LR_SHA256SUM:-/usr/bin/sha256sum}"
[[ -x "$SHA256SUM" ]] || { echo "FATAL: $SHA256SUM absent; the drop-in verifies the control bundle with it" >&2; exit 2; }

mkdir -p "$UNIT_DIR" "$DROPIN_DIR" "$ENV_DIR" "$CONTROL_ROOT" "$(dirname "$GUARD_ENV")"

install -m 0644 "$REPO_DIR/examples/systemd/p1lr-decision@.service" "$UNIT_DIR/"
install -m 0644 "$REPO_DIR/examples/systemd/p1lr-decision@.service.d/20-v2-identity.conf" "$DROPIN_DIR/"
install -m 0644 "$REPO_DIR"/examples/config/phase_3_eth_sac_dynamics/p1lr_env_v2/seed*.env.conf "$ENV_DIR/"
install -m 0644 "$REPO_DIR/examples/systemd/p1lr-idle-guard.service" "$UNIT_DIR/"
install -m 0644 "$REPO_DIR/examples/systemd/p1lr-idle-guard.timer" "$UNIT_DIR/"
chmod +x "$REPO_DIR/examples/systemd/p1lr_decision_gate_check.sh" || true

# ── READ-ONLY lease-gate control bundle (finding AUD-GEN-20260816-261) ──
# Content-addressed: the directory name IS the sha256 of the manifest
# that pins every member, and the SAME literal is already baked into the
# installed drop-in. An update lands BESIDE its predecessor, never over
# it, so a rollback is a unit edit and never a file rewrite.
MANIFEST_SRC="$REPO_DIR/examples/systemd/p1lr-control/CONTROL_MANIFEST.sha256"
MANIFEST_SHA="$("$SHA256SUM" "$MANIFEST_SRC" | cut -d' ' -f1)"
CONTROL_DIR="$CONTROL_ROOT/$MANIFEST_SHA"
PINNED_SHA="$(sed -n "s#.*p1lr-control/\([0-9a-f]\{64\}\)/CONTROL_MANIFEST.sha256.*#\1#p" "$DROPIN_DIR/20-v2-identity.conf" | head -1)"
if [[ "$PINNED_SHA" != "$MANIFEST_SHA" ]]; then
    echo "FATAL: drop-in pins control manifest $PINNED_SHA but the shipped manifest hashes to $MANIFEST_SHA." >&2
    echo "       Regenerate both: $PY $REPO_DIR/tools/p1lr_identity_supervision.py materialize" >&2
    exit 2
fi
# Idempotent under 0555: make the tree writable, refresh, seal again.
if [[ -d "$CONTROL_DIR" ]]; then
    chmod -R u+w "$CONTROL_DIR"
fi
mkdir -p "$CONTROL_DIR/tools"
install -m 0444 "$MANIFEST_SRC" "$CONTROL_DIR/"
while read -r _digest member; do
    install -m 0444 "$REPO_DIR/$member" "$CONTROL_DIR/$member"
done < "$MANIFEST_SRC"
chmod 0555 "$CONTROL_DIR/tools" "$CONTROL_DIR"
if (cd "$CONTROL_DIR" && "$SHA256SUM" --status -c CONTROL_MANIFEST.sha256); then
    control_state="VERIFIED (read-only, content-addressed $MANIFEST_SHA)"
else
    echo "FATAL: the installed control bundle does not match its manifest." >&2
    exit 2
fi

# A stale runtime pin from an older campaign would fight the identity
# drop-in (20- wins on reset directives, but the leftover deserves a
# reviewed removal, not a silent one).
if [[ -f "$DROPIN_DIR/10-runtime-pin.conf" ]]; then
    echo "NOTE: $DROPIN_DIR/10-runtime-pin.conf exists (older runtime pin)."
    echo "      20-v2-identity.conf overrides it; review and remove it"
    echo "      explicitly when convenient."
fi

# Guard binding: the idle guard must supervise the V2 contract, not the
# v1 default. Preserve an existing reviewed file; propose otherwise.
GUARD_LINES="P1LR_GUARD_MODE=decision
P1LR_GUARD_EXTRA_ARGS=--contract $REPO_DIR/examples/config/phase_3_eth_sac_dynamics/p1_difficulty_lr_factorial_v2.json"
if [[ ! -f "$GUARD_ENV" ]]; then
    printf '%s\n' "$GUARD_LINES" > "$GUARD_ENV"
    guard_env_state="written ($GUARD_ENV)"
else
    printf '%s\n' "$GUARD_LINES" > "$GUARD_ENV.v2-proposed"
    guard_env_state="EXISTS — review $GUARD_ENV.v2-proposed and merge"
fi

# READ-ONLY verification: the v2 gate against the v2 contract, and the
# durable transition lease for the v2 chain. Neither touches a process.
if bash "$RUNTIME_DIR/examples/systemd/p1lr_decision_gate_check.sh" \
        "$GATE" "$RUNTIME_DIR/examples/config/phase_3_eth_sac_dynamics/p1_difficulty_lr_factorial_v2.json"; then
    gate_state="VERIFIED"
else
    gate_state="REFUSED (fix before the next boundary)"
fi

ENV_ONE="$(ls "$ENV_DIR"/seed*.env.conf | head -1)"
CHAIN="$(sed -n 's/^P1LR_EXPECTED_CHAIN_ID=//p' "$ENV_ONE")"
EXPERIMENT="$(sed -n 's/^P1LR_EXPERIMENT=//p' "$ENV_ONE")"
# Dogfood the pinned control copy, exactly as the unit will.
if "$PY" "$CONTROL_DIR/tools/p1lr_identity_supervision.py" lease-gate \
        --expected-chain-id "$CHAIN" --experiment "$EXPERIMENT" \
        --mode decision >/dev/null; then
    lease_state="PASS (chain $CHAIN rules the durable transition)"
else
    lease_state="REFUSED (the durable queue does not prove chain $CHAIN)"
fi

systemctl --user daemon-reload

seeds="$("$PY" - "$REPO_DIR/examples/config/phase_3_eth_sac_dynamics/p1_difficulty_lr_factorial_v2.json" <<'PYEOF'
import json, socket, sys
contract = json.load(open(sys.argv[1]))
host = socket.gethostname()
print(" ".join(seed for seed, a in (contract.get("assignments") or {}).items()
               if isinstance(a, dict) and a.get("hostname") == host))
PYEOF
)"

echo
echo "installed (NOTHING enabled, NOTHING started) on $(hostname):"
echo "  $UNIT_DIR/p1lr-decision@.service"
echo "  $DROPIN_DIR/20-v2-identity.conf   (v2 identity pin, all seeds)"
echo "  $ENV_DIR/seed*.env.conf           (v2 contract+sha, chain, CUDA UUID)"
echo "  $CONTROL_DIR                      (read-only lease-gate control bundle)"
echo "  guard env: $guard_env_state"
echo "  v2 screen gate: $gate_state"
echo "  control bundle: $control_state"
echo "  transition lease: $lease_state"
echo
echo "NEXT SAFE BOUNDARY commands for this host (operator decision;"
echo "NEVER while the seed's matching v2 PID is alive — check with"
echo "  pgrep -af 'p1_difficulty_lr_factorial.py --seed <seed> --mode decision'):"
for seed in $seeds; do
    echo "  systemctl --user enable p1lr-decision@${seed}.service   # arms reboot reconstruction; starts NOTHING"
    echo "  systemctl --user start  p1lr-decision@${seed}.service   # ONLY once seed ${seed}'s v2 PID has exited"
done
echo "  systemctl --user enable --now p1lr-idle-guard.timer   # guard only; touches no worker"
echo
echo "verify (read-only):"
echo "  $PY $REPO_DIR/tools/p1lr_identity_supervision.py verify-control"
echo "  $PY $REPO_DIR/tools/p1lr_identity_supervision.py plan --with-deploy-commands"
echo "  $PY $REPO_DIR/tools/p1lr_idle_guard.py --mode decision --contract $REPO_DIR/examples/config/phase_3_eth_sac_dynamics/p1_difficulty_lr_factorial_v2.json --dry-run"
