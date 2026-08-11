#!/usr/bin/env bash
# Fleet dispatch for the bounded M0->L1 mechanism ladder (finding 220,
# order WP3). Four training arms, one per GPU, launched CONCURRENTLY
# through tools/m0_l1_mechanism_ladder.py, which holds the
# authoritative per-arm exclusive_claim flocks under the diagnostic
# identity and enforces the contract hostname + GPU-UUID binding
# (CUDA_VISIBLE_DEVICES must EQUAL the assignment — WP13, fail
# closed). This script mirrors dispatch_l1_factorial_fleet.sh: a
# host-level `flock -n` around each launch, full detachment (</dev/null
# on both sides), pids recorded in files, never from racing shells. A
# second dispatch is safe: the runner returns ALREADY_RUNNING /
# ALREADY_COMPLETE instead of becoming a second writer.
#
# Per-arm env files carry the GPU-UUID bindings:
#   examples/config/phase_3_eth_sac_dynamics/ladder_env/<ARM>.env
#
# D1_EVALUATOR_ONLY is NOT dispatched here: it runs CPU-side after
# D0 terminates —
#   python tools/m0_l1_d1_evaluator.py --d0-record \
#     <output_root>/<diag_id>/D0_M0_EXACT/ladder_arm_record.json
#
# LAUNCH GATE (order WP3): dispatch only after the socket-free suite
# passes. This script requires the explicit word `launch`.
set -euo pipefail

REPO_REMOTE=/home/harveybc/Documents/GitHub/agent-multi
PY=/home/harveybc/anaconda3/envs/trading-stack/bin/python
ENVDIR_REL=examples/config/phase_3_eth_sac_dynamics/ladder_env
RUNDIR=/home/harveybc/.local/share/agent-multi/m0_l1_mechanism_ladder_20260810_v1
LOGDIR=$RUNDIR/logs
EXTRA_ARGS="${LADDER_EXTRA_ARGS:-}"

if [[ "${1:-}" != "launch" ]]; then
  echo "usage: $0 launch"
  echo "  launches the four training arms on their assigned hosts:"
  echo "    D0_M0_EXACT        omega  GPU-612d1e0c-33de-d5cc-56eb-06c0ae424326"
  echo "    D2_BOUNDARY_ONLY   dragon GPU-a8bd1b2c-26c4-f3a9-0fc0-fc3dfc6780f9"
  echo "    D3_COST_PROTECTION gamma  GPU-b77fc3ad-db77-b648-dc15-ec79b65e2519"
  echo "    D4_FULL_L1         gamma  GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8"
  echo "  gate: socket-free tests must pass first (order WP3 §3.5)."
  exit 2
fi

launch_local() {
  local arm="$1" repo="$2"
  mkdir -p "$LOGDIR"
  echo "== $(hostname) arm $arm (local)"
  cd "$repo"
  # shellcheck disable=SC1090
  set -a; source "$repo/$ENVDIR_REL/$arm.env"; set +a
  flock -n "$LOGDIR/dispatch.$arm.hostflock" \
    nohup $PY tools/m0_l1_mechanism_ladder.py --arm "$arm" $EXTRA_ARGS \
    > "$LOGDIR/$arm.runner.log" 2>&1 < /dev/null &
  echo $! > "$LOGDIR/$arm.pid"
  echo "  pid $(cat "$LOGDIR/$arm.pid")"
}

remote_launch() {
  local host="$1" arm="$2"
  echo "== $host arm $arm"
  ssh -o BatchMode=yes "$host" "mkdir -p $LOGDIR && cd $REPO_REMOTE && \
    set -a && source $REPO_REMOTE/$ENVDIR_REL/$arm.env && set +a && \
    flock -n $LOGDIR/dispatch.$arm.hostflock \
    nohup $PY tools/m0_l1_mechanism_ladder.py --arm $arm $EXTRA_ARGS \
    > $LOGDIR/$arm.runner.log 2>&1 < /dev/null & \
    echo \$! > $LOGDIR/$arm.pid; cat $LOGDIR/$arm.pid" \
    < /dev/null
}

launch_local D0_M0_EXACT "${LADDER_LOCAL_REPO:-$REPO_REMOTE}"
remote_launch dragon D2_BOUNDARY_ONLY
remote_launch gamma D3_COST_PROTECTION
remote_launch gamma D4_FULL_L1

echo "== dispatched; verify heartbeats:"
echo "cat $RUNDIR/<diagnostic_id>/*/heartbeat.json"
