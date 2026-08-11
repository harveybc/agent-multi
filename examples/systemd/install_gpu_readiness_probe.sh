#!/usr/bin/env bash
# Install the rootless GPU readiness probe user units on ONE host.
#
# REVIEWED OPERATOR STEP — this script is shipped, never auto-executed.
# Run it manually per worker host (omega, dragon, gamma) after review:
#
#   bash examples/systemd/install_gpu_readiness_probe.sh
#
# It is rootless (systemd --user) and idempotent. It does NOT install
# kernel/driver packages and does NOT touch any system unit.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Documents/GitHub/agent-multi}"
UNIT_DIR="$HOME/.config/systemd/user"
ENV_FILE="$HOME/.config/agent-multi/gpu-readiness.env"

mkdir -p "$UNIT_DIR" "$(dirname "$ENV_FILE")" \
    "$HOME/.local/state/agent-multi/gpu-readiness"

cp "$REPO_DIR/examples/systemd/gpu-readiness-probe.service" "$UNIT_DIR/"
cp "$REPO_DIR/examples/systemd/gpu-readiness-probe.timer" "$UNIT_DIR/"

# Preserve an existing per-host tuning file; create an empty template
# otherwise (disk budget flags are per-host operator decisions).
if [[ ! -f "$ENV_FILE" ]]; then
    cat > "$ENV_FILE" <<'EOF'
# Extra flags for tools/gpu_readiness_probe.py on this host, e.g.:
# GPU_READINESS_EXTRA_ARGS=--output-fs /home/harveybc --expected-artifact-bytes 20000000000
GPU_READINESS_EXTRA_ARGS=
EOF
fi

systemctl --user daemon-reload
systemctl --user enable --now gpu-readiness-probe.timer

echo "installed: gpu-readiness-probe.timer (rootless, 10 min cadence)"
systemctl --user list-timers gpu-readiness-probe.timer --no-pager || true
