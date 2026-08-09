#!/usr/bin/env bash
# Dispatch the 16 L1 factorial cells: one seed per worker, four cells
# sequential per seed, GPU-UUID bound (finding 163), single experiment
# identity from the frozen contract. Authorized by doc 38 §9 + the
# owner's direct order — no further phrase required.
set -euo pipefail
REPO_LOCAL=/tmp/claude-1000/-home-harveybc-Documents-GitHub-predictor/94c1b43d-d764-48d5-885f-68470ae06b5f/scratchpad/agent-multi-m0-wt
REPO_REMOTE=/home/harveybc/Documents/GitHub/agent-multi
PY=/home/harveybc/anaconda3/envs/trading-stack/bin/python
LOGDIR=/home/harveybc/.local/share/agent-multi/l1_matched_factorial_20260809_v1/logs
mkdir -p "$LOGDIR"

echo "== omega seed 101 (RTX 4070, local)"
cd "$REPO_LOCAL"
CUDA_VISIBLE_DEVICES=GPU-612d1e0c-33de-d5cc-56eb-06c0ae424326 \
  nohup $PY tools/l1_factorial_screen.py --seed 101 \
  > "$LOGDIR/seed101.log" 2>&1 &
echo "  pid $!"

echo "== dragon seed 202 (RTX 4090)"
ssh dragon "mkdir -p $LOGDIR && cd $REPO_REMOTE && \
  nohup $PY tools/l1_factorial_screen.py --seed 202 \
  > $LOGDIR/seed202.log 2>&1 & echo \"  pid \$!\""

echo "== gamma seed 303 (RTX 5070 Ti, UUID-bound)"
ssh gamma "mkdir -p $LOGDIR && cd $REPO_REMOTE && \
  CUDA_VISIBLE_DEVICES=GPU-b77fc3ad-db77-b648-dc15-ec79b65e2519 \
  nohup $PY tools/l1_factorial_screen.py --seed 303 \
  > $LOGDIR/seed303.log 2>&1 & echo \"  pid \$!\""

echo "== gamma seed 404 (RTX 5090, UUID-bound)"
ssh gamma "cd $REPO_REMOTE && \
  CUDA_VISIBLE_DEVICES=GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8 \
  nohup $PY tools/l1_factorial_screen.py --seed 404 \
  > $LOGDIR/seed404.log 2>&1 & echo \"  pid \$!\""

echo "== dispatched; verify:"
echo "nvidia-smi + tail $LOGDIR/seed*.log"
