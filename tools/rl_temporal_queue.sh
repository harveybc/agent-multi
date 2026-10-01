#!/usr/bin/env bash
# One-at-a-time queue of RL cells on one host.
#   rl_temporal_queue.sh <host_alias> <device: cpu|cuda> <gpu_uuid_or_empty> <cap e.g. 3G> <out_root> <cell.json ...>
# Each cell runs under crispdm-run with a fresh admission (queued, never a lowered cap);
# a cell whose RESULT.json exists is skipped; STATUS.json is rewritten after every cell.
set -u
HOST=$1; DEVICE=$2; UUID=$3; CAP=$4; OUT=$5; shift 5
cd ~/.local/state/scratch/g-rl/wt/agent-multi
export PYTHONPATH=$HOME/.local/state/scratch/g-rl/wt/agent-multi:$HOME/.local/state/scratch/g-rl/wt/gym-fx:$HOME/.local/state/scratch/g-rl/wt/predictor
export TF_CPP_MIN_LOG_LEVEL=3
if [ "$DEVICE" = "cuda" ]; then export CUDA_VISIBLE_DEVICES="$UUID"; else export CUDA_VISIBLE_DEVICES=""; fi
mkdir -p "$OUT"
for cell in "$@"; do
  if [ -f "$OUT/STOP" ]; then echo "STOP file present: not starting further cells"; break; fi
  name=$(basename "$cell" .json)
  # adopt, never duplicate: a live python child for this cell means another runner owns it
  if pgrep -f "run_rl_temporal_cell.py --cell .*$name.json" >/dev/null; then echo "skip $name (running)"; continue; fi
  dir="$OUT/$name"
  if [ -f "$dir/RESULT.json" ]; then echo "skip $name (RESULT exists)"; continue; fi
  mkdir -p "$dir"
  echo "== $name start $(date -u +%FT%TZ) device=$DEVICE cap=$CAP"
  /usr/bin/time -f "wall %e s maxrss %M kB" $HOME/.local/bin/crispdm-run -m "$CAP" -q -W 7200 -t 5h -n "g-rl-$name" -- \
    ~/anaconda3/envs/trading-stack/bin/python tools/run_rl_temporal_cell.py --cell "$cell" \
    --data-root ~/Documents/GitHub/predictor --out "$dir" --device "$DEVICE" --host-alias "$HOST" $PILOT_ARGS > "$dir/run.log" 2>&1
  rc=$?
  echo "== $name exit $rc $(date -u +%FT%TZ)"; tail -2 "$dir/run.log"
  python3 - "$OUT" "$HOST" <<'PY'
import json, os, sys, glob, time
out, host = sys.argv[1], sys.argv[2]
cells = {}
for d in sorted(glob.glob(os.path.join(out, "RL-*"))):
    r = os.path.join(d, "RESULT.json")
    hb = os.path.join(d, "heartbeat.json")
    cells[os.path.basename(d)] = {"result": os.path.exists(r),
                                  "heartbeat": json.load(open(hb)) if os.path.exists(hb) else None}
json.dump({"host_alias": host, "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cells": cells},
          open(os.path.join(out, "STATUS.json"), "w"), indent=1, default=str)
PY
done
echo "queue done $(date -u +%FT%TZ)"
