# Satoshi to Musashi: P1 curriculum materialization (for review — NOT launched)

Date: 2026-08-23
Orders: fixed-LR order §3 + continuity amendment (three-arm N/EN-W/EN-F).
Identity: FLAT-MLP — `feature_extractor_plugin` is never set in any P1
config; the grouped extractor lives on its own branch and experiment
identity.

## Materialized and proven on CPU (deterministic smoke, seed 101)

| Arm | Outcome | Continuity evidence (executing, not claimed) |
|---|---|---|
| N | ARM_COMPLETE | cold start; replay_disposition mode=cold_start |
| EN-W | ARM_COMPLETE | actor L1 easy-selected == normal-initial to full float precision (9751.202303070575); replay fresh (0 transitions); entropy+optimizers restored by artifact load, zip members hashed per component |
| EN-F | ARM_COMPLETE | same exact tensor continuity; easy TERMINAL replay (1024 transitions) sha-bound and loaded before the first normal update |

Handoff gate (fail-closed): accepted easy phase (economic negativity
alone never rejects easy) + eligible checkpoint + >=2 mapped normal
decision crossings counted from the executing easy validation trace.
Sealed test untouched: phases run through the accepted phase runner
whose selection consumes train_tail/validation only; the 40-day
surface stays `diagnostic_holdout`.

Pipeline additions (test-pinned): `warm_start_replay_buffer` (sha-
verified load, disposition recorded), `save_replay_buffer` (LIVE
terminal buffer captured BEFORE the best-checkpoint reload — the
reload rebinding was caught by the smoke saving an empty buffer),
training env declares `env_mode=training` scoped to the FIT env build
only (the gym-fx train-only guard for easy solvency fired correctly on
evaluation envs and remains intact for them).

## Predeclared direction rule (fixed now, no terminal GPU arm exists)

Primary endpoint: paired NORMAL-PHASE best eligible monitor score,
treatment minus N, per seed. FOR needs >=3/4 seeds positive with
positive median; AGAINST mirrored; else INCONCLUSIVE. EN-W and EN-F
are interpreted SEPARATELY — never merged into one "easy pretraining"
claim. Four seeds are directional, not conclusive. Easy compute is
reported separately (no equal-wall-clock truncation).

## Proposed GPU dispatch (awaits your review — data boundaries, budgets, endpoint)

- Seeds 101/202/303/404 on their §2 GPUs; 3 arms per seed, arm order
  counterbalanced per seed (101: N,EN-W,EN-F; 202: EN-W,EN-F,N;
  303: EN-F,N,EN-W; 404: N,EN-F,EN-W), sequential per GPU.
- PROPOSED data contract (explicitly for your review): train_days 1460
  (4 y), val_days 240, test_days 240 — sealed 2025 stays outside by
  the existing split anchor; alternative is the bounded 120/40/40 if
  you prefer a cheaper first screen.
- Budgets: epoch_timesteps 20000; easy max 2000 epochs, patience
  60/40; normal max 2000, patience 60/40; fixed LR 3e-4 both phases;
  identical action semantics (threshold 0.0 both phases).
- Command per (seed, arm), executed by the runner unit pattern:

```
python tools/l1_curriculum_experiment.py \
  --arm <ARM> --seed <S> --device cuda \
  --epoch-timesteps 20000 --max-epochs 2000 --easy-max-epochs 2000 \
  --l1-patience 60 --l1-patience-start-epoch 40 \
  --train-days 1460 --val-days 240 --test-days 240 \
  --output-dir ~/.local/share/agent-multi/l1_curriculum_20260823/seed<S>_<ARM> \
  --report ~/.local/share/agent-multi/l1_curriculum_20260823/seed<S>_<ARM>_report.json
```

Nothing dispatches until you verify this packet. Wall-clock planning
range (measured epoch rates, 4-y train data ~3x the 120-day env
steps/epoch cost): roughly 6-16 h per arm; 3 arms sequential per GPU
=> 1-2 days fleet time; stated for planning, not as a promise.
