#!/usr/bin/env bash
# Pin decision-unit restarts to one immutable experiment source revision.
#
# Usage:
#   bash examples/systemd/pin_p1lr_decision_runtime.sh <git-revision>
#
# The canonical checkout may continue receiving docs/audit commits. The
# systemd decision unit is redirected to a detached, clean worktree so a
# restart derives the same experiment identity as the original process.
set -euo pipefail

[[ $# -eq 1 ]] || {
    echo "usage: $0 <git-revision>" >&2
    exit 2
}

REPO_DIR="${REPO_DIR:-$HOME/Documents/GitHub/agent-multi}"
RUNTIME_ROOT="${P1LR_RUNTIME_ROOT:-$HOME/Documents/GitHub/.runtime}"
REVISION="$1"
FULL_REVISION="$(git -C "$REPO_DIR" rev-parse --verify "${REVISION}^{commit}")"
RUNTIME_DIR="$RUNTIME_ROOT/agent-multi-p1lr-${FULL_REVISION:0:16}"
DROPIN_DIR="$HOME/.config/systemd/user/p1lr-decision@.service.d"
DROPIN="$DROPIN_DIR/10-runtime-pin.conf"

mkdir -p "$RUNTIME_ROOT" "$DROPIN_DIR"
if [[ -e "$RUNTIME_DIR" ]]; then
    [[ "$(git -C "$RUNTIME_DIR" rev-parse HEAD)" == "$FULL_REVISION" ]] || {
        echo "REFUSED: existing runtime worktree has a different revision" >&2
        exit 4
    }
    [[ -z "$(git -C "$RUNTIME_DIR" status --porcelain)" ]] || {
        echo "REFUSED: runtime worktree is dirty" >&2
        exit 4
    }
else
    git -C "$REPO_DIR" worktree add --detach "$RUNTIME_DIR" "$FULL_REVISION"
fi

cat >"$DROPIN" <<EOF
[Service]
WorkingDirectory=$RUNTIME_DIR
EnvironmentFile=
EnvironmentFile=$RUNTIME_DIR/examples/config/phase_3_eth_sac_dynamics/p1lr_env/seed%i.env
EnvironmentFile=-%h/.config/agent-multi/p1lr-decision@%i.env
ExecStartPre=
ExecStartPre=$RUNTIME_DIR/examples/systemd/p1lr_decision_gate_check.sh \${P1LR_SCREEN_GATE}
EOF

systemctl --user daemon-reload

PY="${P1LR_PYTHON:-$HOME/anaconda3/envs/trading-stack/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python3)"
identity="$({
    cd "$RUNTIME_DIR"
    "$PY" tools/p1_difficulty_lr_factorial.py --preflight
} | "$PY" -c 'import json,sys; x=json.load(sys.stdin); assert x["outcome"] == "PREFLIGHT_PASS", x; print(x["modes"]["decision"]["experiment_identity"])')"

printf 'runtime_revision=%s\nruntime_dir=%s\ndecision_identity=%s\ndropin=%s\n' \
    "$FULL_REVISION" "$RUNTIME_DIR" "$identity" "$DROPIN"
