# Musashi to General Satoshi III: 224-233 Correction and Non-Idle Order

Date: 2026-08-12 America/Bogota  
Authority: owner-approved work plan sections 9, 12 and 14  
Audit: `docs/audits/AUDIT_SATOSHI_III_RETURN_224_230_2026_08_12.md`  
Priority: execute now; no new owner phrase; audit remains parallel

## 1. Teach-Back Before Editing

Return a short machine-readable teach-back proving these points:

1. `train_monitor` and `inner_validation` each contain 256 context rows, but
   only 2,190 rows are scored.
2. Pairing cannot neutralize context contamination when each model produces a
   different action before the scored interval.
3. An inactive policy is a measured non-promotable outcome, not a harness
   exception and not a successful checkpoint.
4. Decision mode has a different root and durability contract from screen
   mode.
5. The active identity is diagnostic only; the corrected run starts from the
   original anchors, never current terminals.

## 2. WP1: Correct Causal Prefixes in the Executing Selector

Files:

- `pipeline_plugins/rl_pipeline_with_validation.py`
- `pipeline_plugins/_nested_splits.py` only if its existing wrapper contract
  needs a bounded extension
- `tests/test_nested_splits.py`
- new focused pipeline tests under `tests/unit/`

Requirements:

1. Resolve role and `context_rows` from the verified nested manifest, not from
   filenames or row positions.
2. Wrap every baseline and per-epoch `train_monitor` and `inner_validation`
   environment with `ContextPrefixWrapper` before `_rollout`.
3. Keep outer validation on its existing wrapper and fit training unwrapped.
4. During prefix rows: force hold; forbid account mutation, orders, trades,
   replay writes and scored traces.
5. `_rollout` records `context_prefix_steps` and `scored_steps`; reward,
   action stats, canonical traces, weekly metrics and metric horizons use only
   scored steps.
6. Tests must use an adversarial agent that requests trades on every prefix
   row and different actions per arm. Prove equal opening equity at the score
   boundary, zero prefix traces/trades/reward, exact 2,190 scored rows and
   unchanged no-context behavior.

## 3. WP2: Make Inactive Decision Cells First-Class

Files:

- `tools/p1_difficulty_lr_factorial.py`
- `pipeline_plugins/rl_pipeline_with_validation.py` if a clearer typed result
  is required
- `tests/test_p1_difficulty_lr_factorial.py`

Requirements:

1. A typed inactive result publishes one immutable cell record with
   `activity_status=inactive`, `promotion_eligible=false`, termination cause,
   terminal path/hash/tensor hash/load proof and exact source identities.
2. The single final outer evaluation may load the terminal artifact for
   diagnostic truth, but may never relabel it `best_checkpoint`.
3. The seed runner continues to its next cell after the record lands.
4. Aggregation accepts exactly 16 records and distinguishes:
   `TOTAL_ACTIVITY_COLLAPSE`, partial activity survival and fully paired
   performance. Never impute zero or a sentinel as performance.
5. Add fixtures for inactive-first, inactive-middle, all-16-inactive,
   duplicate/restart and inactive artifact missing/altered.

## 4. WP3: Mode-Aware Status and Durable Recovery

Files:

- `tools/multifront_status.py`
- `tools/p1lr_idle_guard.py`
- `examples/systemd/p1lr-decision@.service` (new)
- `examples/systemd/p1lr-idle-guard.service` (new)
- `examples/systemd/p1lr-idle-guard.timer` (new)
- tests for all three surfaces

Requirements:

1. Add an explicit validated mode (`screen` or `decision`) and derive output
   root, unit name, expected heartbeat mode and total cells from it.
2. A supplied decision identity under the screen root must refuse rather than
   render a false 0/4 state.
3. The decision unit always supplies `--mode decision` and a pinned verified
   screen-gate path. It must never default to screen.
4. Ship and deploy the 15-minute guard service/timer on Omega, Dragon and
   Gamma. Prove timer enabled, last run fresh and its report bound to the
   decision root.
5. Add the auditor's exact fixture: four direct decision workers must render
   mode decision, 4/4 fresh and the correct per-host record semantics.

## 5. WP4: Zero-Idle Replacement

1. Keep identity `1434685bfdf52911` running as diagnostic while WP1-WP3 are
   implemented and tested. Preserve every artifact; do not aggregate it.
2. Materialize a new identity whose hash changes on the context and inactive
   semantics.
3. Run a four-worker mechanics smoke on that identity.
4. Replace old workers one host at a time: start the corrected durable unit,
   prove heartbeat/GPU work, then retire the corresponding diagnostic worker.
   Never stop all workers together and never leave a healthy GPU without a
   compatible approved job.
5. After 4/4 corrected workers are fresh, run the corrected 16-cell decision
   campaign automatically under the existing authorization.

## 6. WP5: Finish Open Operational Items

1. Push LTS branch `satoshi/ibkr-capability-227`, request independent
   verification against the pushed commit, then integrate/deploy it without
   mutating the live capability store during tests.
2. Continue finding 230 via `dragon-replica`. Stop reporting rsync's
   incremental-recursion percentage as whole-tree completion. Report exact
   source/destination bytes and files until terminal; then generate sorted
   SHA-256 manifests independently on Gamma and Dragon and compare them.
3. Merge this audit branch append-only so the prior 224-230 audit, this return
   audit, register states and work-plan amendments all remain in one lineage.

## 7. Decision Report

When the corrected run lands, include per seed and arm:

- raw weekly return, weekly RAP, drawdown, turnover, trades and activity;
- easy-minus-normal and LR deltas with signs and magnitudes;
- interaction delta;
- active/inactive classification and artifact identity; and
- directional consistency separately from any predeclared practical
  materiality threshold.

No sealed-2025 access, no live/Paper risk relaxation and no current terminal
as a corrected-run anchor are authorized.

## 8. Return Packet

Return exact commits, before/after reproducer output, focused/full suites,
new identity/hash, four-host deployment facts, status JSON, timer facts,
smoke custody and a direct statement of every residual doubt. No finding is
self-closed.

