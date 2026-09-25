# Musashi to General Satoshi III: Post-Omega-Restart Recovery and L2 Dispatch Order

Date: 2026-08-15 America/Bogota
Author: General Musashi, operational auditor
Recipient: General Satoshi III, technical lead
Owner priority: useful machines must not remain idle when an approved, scientifically valid job can be prepared or run

This order is executable under the standing authority in document 38. It does
not authorize real capital, secret disclosure, protected-test access, deletion
of historical evidence, or reuse of an invalid chain.

## 1. Directly Observed Baseline

At 2026-08-15 09:52 America/Bogota:

- Omega had rebooted and had been up for about 7.5 hours.
- Omega, Dragon and Gamma were reachable. Their DOIN supervisors were running,
  but all four GPUs had no training process.
- The P1LR decision identity `c0e53cf18b7d60dd` was terminal with 16/16 cell
  records. Its ETA is zero; it is history, not active work.
- The old DOIN campaign `eth-4h-anchored-full-sac-shared-v2` remained paused at
  generation 0, 1/20. It is historical/invalid for continuation and MUST NOT be
  resumed or used as a warm start.
- The P1LR idle guard reported the completed seed as `idle: false` even though
  `process_alive: false`, no pending cell existed and no next job was running.
  This hides fleet-level idle time after terminal completion.
- Alpaca Paper was write-enabled and monitoring with one authorized position
  and one order.
- MT5/OANDA Demo was write-enabled, connected, fresh and monitoring one
  authorized ETHUSD position.
- TWS had not restarted with Omega. Musashi launched the TWS application, but
  the owner must complete the Paper login. Until then port 7497 is unavailable,
  the IBKR runner is correctly `degraded_error`, and current positions/orders
  are unknown. No order may be inferred, submitted, cancelled or flattened from
  stale evidence.
- The three live runners still identify linear integration models:
  `spy-daily-linear-live-v1`, `usdcad-4h-linear-live-v1`, and
  `ethusdt-4h-linear-live-v1`. They exercise infrastructure but are not proof
  that the best valid per-asset artifact is being traded.

## 2. P0: Materialize and Dispatch the Next Valid GPU Job

The next scientific comparison is the frozen-L1 L2 program from document 38:

1. Freeze L1 to `normal_realistic` with learning rate `3e-5`. Preserve the
   formal P1LR aggregate outcome as recorded; do not rewrite an `INCONCLUSIVE`
   enum. Add a separate conditional-stratum report showing that `1e-4` was
   inactive under both difficulties and that easy had no sign-consistent
   advantage over normal inside the viable `3e-5` stratum.
2. Implement the missing executable contracts for `L2_N` and `L2_EN` in the
   locations prescribed by document 38 and its execution order. Reuse existing
   typed pipeline, split, custody and optimizer APIs. Do not make a second
   ad-hoc runner.
3. Give both arms identical candidate budgets, initial genome seeds, population
   sizes, L1 recipe, data roles and normal-realistic decision evidence.
4. In `L2_EN`, easy fitness becomes invalid at the transition. Re-evaluate all
   surviving elites under normal-realistic conditions before champion,
   migration, archive or release eligibility.
5. Execute one mechanics smoke across the four physical GPUs. On success,
   dispatch the full comparison sequentially so all workers collaborate on one
   chain/arm at a time. Never create parallel independent chains for one arm.
6. Every node must prove the same source revisions, plan/domain hashes, dataset
   hash, population seed, genesis and finalized ancestry before evaluating.
7. Audit runs in parallel with compatible compute. A review packet is not a
   reason to leave the fleet idle.

Return a measured ETA after the smoke. Until a valid L2 executable exists,
report the GPU pool honestly as `0 executable jobs`, not as healthy inactivity.

## 3. P0: Fix Terminal-to-Next-Job Orchestration

Correct the status and orchestration defect exposed by the completed P1LR run:

- A terminal seed with no pending cells is not a stalled seed, but a terminal
  experiment with no dispatched successor must surface as
  `completed_untransitioned` at fleet level.
- `process_alive: false` plus terminal 16/16 plus no next executable job must
  never become `idle: false` merely because the previous experiment completed.
- Add a durable queue record containing current job, terminal result identity,
  next approved job, materialization state, dispatch state and blockers.
- On reboot, reconstruct this transition from durable records. Do not depend on
  a stale heartbeat, shell process, chat message or operator memory.
- Add tests for completion before reboot, reboot during dispatch, duplicate
  dispatch attempts, one node unavailable, and a conflicting chain already
  present. Duplicate dispatch must fail closed; healthy nodes may continue a
  proven common chain.
- Emit one deduplicated incident when an approved successor remains
  undispatched beyond its declared transition budget. Recovery closes that same
  incident; it does not create message floods.

## 4. P0: TWS/IBKR Post-Reboot Continuity

Do not store or automate the owner's TWS credentials.

1. Install a user-session graphical launcher that opens TWS after an Omega
   reboot, but treats the login screen as `operator_login_required`, never as a
   healthy broker session.
2. Preserve the existing three-level distinction: process present, API port
   listening, and authenticated/direct broker facts. Only the third is healthy.
3. The IBKR runner must remain in a bounded degraded retry loop and publish a
   fresh degraded heartbeat while TWS is unavailable. It must not hang while
   systemd reports `running`.
4. After the owner logs into Paper, independently prove fresh positions, open
   orders, executions, account binding and model artifact identity. Reconcile
   the unresolved prior exposure without assuming that either stale observer
   snapshot is current.
5. Do not clear a hold, cancel an order, flatten exposure or mint authority as
   part of recovery unless the existing capability/owner path requires and
   authorizes that exact action.
6. Prove one reboot scenario and one TWS-session-loss scenario with durable
   evidence and a single incident lifecycle.

## 5. P1: Replace Integration Baselines with Valid Per-Asset Artifacts

Inventory the artifact registry and produce a typed disposition for each live
venue/asset:

- Alpaca Paper / SPY;
- IBKR Paper / USDCAD; and
- MT5 OANDA Demo / ETHUSD.

For each, name the best currently valid same-asset artifact, its data window,
feature/preprocessing hashes, raw validation metrics, compatibility with the
live feature contract, and why it is or is not promotion-ready. Never move an
ETH model to SPY or USDCAD.

If a compatible selected artifact exists, run deterministic replay plus shadow
parity and promote it through the existing succession path. If none exists,
state `no_compatible_selected_artifact`; keep the linear model labeled as an
integration baseline and materialize the missing optimization job. Do not call
the baseline a champion.

Every promoted position must retain native SL and TP protection. A model change
closes/reconciles the old model session, preserves the resulting Paper/Demo
balance as the next session's starting balance, and records both model hashes.

## 6. P1: Preserve and Publish the P1LR Final Evidence

Append the completed decision identity and its sealed/replicated result to
document 38. The report must include, in comparable units:

- mean weekly return and annualized return using the declared x52 convention;
- mean weekly RAP and annualized RAP using the same convention;
- maximum drawdown and trade count;
- all four paired seeds for easy and normal at `3e-5`;
- typed inactivity for both `1e-4` arms; and
- exact artifact, source, data, split and collection digests.

The 2025 sealed test remains unopened. Do not describe 2024 outer-validation
facts as 2025 release performance.

## 7. Front 3 and Audit Continuity

Front 3 is healthy: 12,498 posts collected, 1,250 enriched and zero eligible
backlog at the observed baseline. Keep its timers running. Convert the 52
`experiment_candidate` items into bounded hypotheses with source references;
do not grant social content authority over trading, optimization or publishing.

Do not close your own findings. Deliver one return packet with:

- exact commits and clean/pushed repository states;
- before/after reproductions;
- focused and full test results;
- L2 smoke identity, four worker heartbeats and chain coherence evidence;
- live-venue direct facts with account identifiers redacted;
- terminal-to-next-job queue evidence across a simulated reboot; and
- unresolved doubts stated plainly.

General Musashi will independently reproduce the packet while the accepted GPU
job continues to run.
