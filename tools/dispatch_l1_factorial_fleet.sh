#!/usr/bin/env bash
# Fleet dispatch for the L1 matched factorial (order §2/WP1).
#
# Every launch goes through tools/l1_fleet_launcher.py, which holds the
# authoritative exclusive_claim flocks (one per experiment/seed, one per
# cell) and enforces the contract hostname + GPU-UUID assignment. This
# script adds a host-level `flock -n` around each launch as belt and
# suspenders, fully detaches every remote process (</dev/null on both
# sides — a hung channel resumed its remaining lines and double-launched
# on 2026-08-09), and records pids in files, never from racing shells.
# A second dispatch is safe: the launcher returns ALREADY_RUNNING /
# ALREADY_COMPLETE instead of becoming a second writer.
#
# Durable operation prefers the systemd template
# examples/systemd/l1-factorial@.service; this script is the manual
# path with the same guarantees.
set -euo pipefail
REPO_REMOTE=/home/harveybc/Documents/GitHub/agent-multi
PY=/home/harveybc/anaconda3/envs/trading-stack/bin/python
RUNDIR=/home/harveybc/.local/share/agent-multi/l1_matched_factorial_20260809_v1
LOGDIR=$RUNDIR/logs
EXTRA_ARGS="${L1_EXTRA_ARGS:-}"

launch_local() {
  local seed="$1" repo="$2"
  mkdir -p "$LOGDIR"
  echo "== $(hostname) seed $seed (local)"
  cd "$repo"
  flock -n "$LOGDIR/dispatch.seed$seed.hostflock" \
    nohup $PY tools/l1_fleet_launcher.py --seed "$seed" $EXTRA_ARGS \
    > "$LOGDIR/seed$seed.launcher.log" 2>&1 < /dev/null &
  echo $! > "$LOGDIR/seed$seed.pid"
  echo "  pid $(cat "$LOGDIR/seed$seed.pid")"
}

remote_launch() {
  local host="$1" seed="$2"
  echo "== $host seed $seed"
  ssh -o BatchMode=yes "$host" "mkdir -p $LOGDIR && cd $REPO_REMOTE && \
    flock -n $LOGDIR/dispatch.seed$seed.hostflock \
    nohup $PY tools/l1_fleet_launcher.py --seed $seed $EXTRA_ARGS \
    > $LOGDIR/seed$seed.launcher.log 2>&1 < /dev/null & \
    echo \$! > $LOGDIR/seed$seed.pid; cat $LOGDIR/seed$seed.pid" \
    < /dev/null
}

launch_local 101 "${L1_LOCAL_REPO:-$REPO_REMOTE}"
remote_launch dragon 202
remote_launch gamma 303
remote_launch gamma 404

echo "== dispatched via durable launcher; verify:"
echo "cat $RUNDIR/<experiment>/seed*/launcher_heartbeat.json"
