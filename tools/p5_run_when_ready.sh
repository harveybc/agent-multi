#!/usr/bin/env bash
# P5 pipeline: wait for all P4 sweeps (Dragon J2/J3, Gamma J4/J6, Omega J5) to
# finish, then run hold-out eval + ranking and print top-3 candidates.
#
# Safe to run in the background. Exits 0 when ranking is written.
set -u

REPO_ROOT="/home/harveybc/Documents/GitHub/agent-multi"
PY="/home/harveybc/anaconda3/envs/tensorflow/bin/python"

DRAGON="harveybc@100.110.215.85"
GAMMA="harveybc@100.107.204.49"
SSH_OPTS="-o BatchMode=yes -p 62024"

log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

# Expected run counts per asset/algo/tag (what P4 must produce total):
#   btcusdt_1h/ppo/p4iter7:    9 seeds (s0..s8)  - s0-5 already DONE on Dragon, s6-8 running on Omega
#   ethusdt_1h/ppo/p4iter7:    6 seeds (s0..s5)  - s0-2 prior, s3-5 on Dragon
#   ethusdt_1h/sac/p4sacfix:   6 seeds (s0..s5)  - s0-2 prior, s3-5 on Dragon
#   eurusd_1h/ppo/p4iter8ext:  6 seeds (s0..s5)  - s3-5 DONE on Gamma, s0-2 running as J6
#
# Busy means: any python tools/seed_sweep.py OR agent-multi --load_config on that host.
busy() {
    local host="$1"
    if [ "$host" = "omega" ]; then
        pgrep -af 'seed_sweep|agent-multi --load' | grep -v "pgrep" | grep -v "ssh " >/dev/null 2>&1
    else
        ssh $SSH_OPTS "$host" "pgrep -af 'seed_sweep|agent-multi --load' | grep -v pgrep" 2>/dev/null | grep -q .
    fi
}

log "Waiting for P4 sweeps to complete on Dragon, Gamma, Omega..."
while true; do
    bd=0; bg=0; bo=0
    busy "$DRAGON" && bd=1
    busy "$GAMMA" && bg=1
    busy omega && bo=1
    if [ $bd -eq 0 ] && [ $bg -eq 0 ] && [ $bo -eq 0 ]; then
        log "All hosts idle. Proceeding to P5 eval."
        break
    fi
    log "still-busy: dragon=$bd gamma=$bg omega=$bo  (sleeping 120s)"
    sleep 120
done

log "Syncing remote logs/partIII to local..."
rsync -az -e "ssh $SSH_OPTS" --ignore-existing \
    "$DRAGON:$REPO_ROOT/logs/partIII/" "$REPO_ROOT/logs/partIII/" || true
rsync -az -e "ssh $SSH_OPTS" --ignore-existing \
    "$GAMMA:$REPO_ROOT/logs/partIII/" "$REPO_ROOT/logs/partIII/" || true

log "Running p5_eval_holdout.py (skip-if-exists to avoid re-rolling completed evals)..."
cd "$REPO_ROOT"
"$PY" tools/p5_eval_holdout.py --skip-if-exists 2>&1 | tail -20

log "Running p5_rank.py..."
"$PY" tools/p5_rank.py --top 3 --min_seeds 3 2>&1 | tail -20

log "P5 pipeline done. Artifacts:"
ls -la logs/partIII/p5_eval_holdout.csv logs/partIII/p5_eval_holdout.md \
       logs/partIII/p5_rank.csv logs/partIII/p5_rank.md 2>/dev/null
