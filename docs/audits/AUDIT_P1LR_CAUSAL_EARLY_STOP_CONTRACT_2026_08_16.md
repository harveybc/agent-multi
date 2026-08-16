# P1LR Causal Early-Stopping Contract Audit

Date: 2026-08-16
Auditor/implementer: General Musashi (Codex)
Scope: corrected-observation ETH SAC L1 comparison

## Verdict

The four-epoch phase-1 decision contract is rejected for the scientific
question. It could not exercise early stopping and mixed a fixed phase-2 LR
with one different phase-1 LR stratum. The in-progress identity was stopped
without deleting its files. It must not enter recipe selection.

The replacement contract is a paired 2x2 design. At a fixed seed and LR,
`normal->normal` and `easy->normal` differ only in phase-1 dynamics. LR is
constant across the two phases of a cell. The two LR strata remain explicit,
so LR and interaction effects are estimable rather than hidden.

## Configuration Audit

- Data: 11,509 fit rows through 2022; 2,190 scored 2022 monitor rows; 2,190
  scored 2023 inner-validation rows; 2,196 scored 2024 outer-validation rows;
  2025 sealed and unavailable to training/selection.
- Observation: 32 x 83 engineered features plus four agent-state values =
  2,660 inputs; rolling z-score 256; clip 10; raw price window disabled.
- Initialization: one zero-update genesis tensor per seed, shared by all four
  cells of that seed; distinct across seeds.
- SAC: explicit `[256,256]`, batch 256, replay 40,000, learning starts 1,000,
  train frequency/gradient steps 1/1, gamma 0.99, tau 0.005, fixed entropy 0.2.
- Execution: normal phase uses commission 0.0002 per side, spread 0.0001,
  leverage 1, relative volume 0.05, ATR SL/TP 2/3 and protected entries.
- Boundary: policy weights transfer; optimizer moments and replay transitions
  do not. This boundary is identical in control and treatment.
- Stopping: 1,000 epochs per phase, 20,000 timesteps/epoch, patience 60,
  floor 40 and min_delta 1e-4 in both phases. Activity stop is disabled.
- Selection: train-monitor plus inner-validation paired utility. Outer 2024 is
  one final truth evaluation. Sealed 2025 cannot be opened.

## Automated Invariants

1. Matched difficulty pairs differ only in `phase1_mode`.
2. `learning_rate == phase1_learning_rate == easy_learning_rate` in every
   corrected cell.
3. A runtime LR mismatch refuses before environment/model construction.
4. Both phase budgets and stopping knobs are asserted from the materialized
   config, not copied from prose.
5. Both phase endpoints persist epoch counts and stop reasons.
6. The observation contract is applied by materialization and revalidated at
   training/evaluation boundaries.

## Fleet Identity Incident And Correction

The first corrected fleet launch was stopped during its first minute when
distributed heartbeats exposed two experiment identities. Omega's canonical
`gym-fx` checkout contains local documentation-only edits while Dragon and
Gamma are clean. No canonical file was reverted, copied or discarded.

Fleet runs now bind `AGENT_MULTI_GYM_FX_ROOT` and `PYTHONPATH` to an immutable
clean `gym-fx` runtime worktree at the same commit on all hosts. Launch must
refuse unless every host independently derives the same screen and decision
identities before GPU training begins.

At 04:03 America/Bogota, a stale external runtime command stopped Omega's
corrected screen and launched superseded decision identity `cdf30aebf585385b`
from runtime `924910fe`. The stale process was stopped, its legacy unit names
were runtime-masked across all hosts, and Omega rejoined the same corrected
screen identity in a fresh attempt. Dragon and Gamma were not interrupted.

## Residual Research Questions

This experiment does not claim that either LR, `[256,256]`, replay 40,000,
window 32, entropy 0.2 or patience 60 is globally optimal. They are held fixed
or explicitly stratified to answer the difficulty question. Phase-specific LR
schedules, positive-easy handoff gating, longer easy allocation, topology and
feature selection remain separate future factors and must not be inferred from
this result.

## Satoshi Screen-Gate And Fleet-Governance Disposition

Source reviewed:
`docs/handoffs/SATOSHI_TO_GENERAL_MUSASHI_SCREEN_GATE_ACTIVITY_DEFECT_AND_FLEET_GOVERNANCE_2026_08_16.md`
from the canonical checkout.

### Activity observation: reproduced; proposed dispatch gate: rejected

The corrected screen completed 16/16 under identity `0c70ab2ce7804750`, was
sealed with collection digest
`2337f9d4e90c9df7958e0c73883a3dc014a7be2e313da7121e08cfd85a15f1c5`,
and independently replicated. It reproduces Satoshi's cross-tabulation:
5/16 cells are activity-active, 11/16 are inactive, 7/16 are mechanically
viable, and two mechanically viable seed-101 cells are inactive.

That fact does not invalidate this decision dispatch. This screen deliberately
executes one epoch per phase and makes no performance claim. Requiring an
activity-eligible checkpoint after one epoch would select cells on the outcome
whose delayed emergence the 1,000-epoch phases are intended to measure. It
would also destroy the complete paired 2x2 design by censoring seed/arm cells.
The decision runner iterates every configured cell; it does not consume the
screen's `viable_cells` list as a filter. Inactivity remains non-promotable at
the final decision verdict, but it is a valid measured decision outcome.

The legitimate defect is semantic: `SCREEN_VIABLE_REGION` and its `next_step`
can be misread as scientific or activity viability. Open finding
`AUD-F1-20260816-247` (S3): after this identity terminates, split the vocabulary
into an explicit mechanics/admissibility result and a final promotion result;
persist booleans equivalent to `activity_required_for_dispatch=false` and
`activity_required_for_promotion=true`. No code or gate mutation is allowed
inside the running identity.

Satoshi's statement that no-activity patience should terminate the run applies
to the superseded contract. In the accepted contract the activity terminator
is disabled. Direct runtime evidence on seed 404 reached epoch 99 with
`L1 no-activity 59/0`, proving it did not stop at epoch 80. Improvement patience
60 after floor 40 remains active once a checkpoint is activity-eligible; a
never-eligible policy reaches the 1,000-epoch phase ceiling and is recorded.

### Interference and authority

Satoshi's interference is accepted as `AUD-GEN-20260816-248` (S2, contained,
awaiting independent/owner disposition): he launched superseded workers,
created two divergent identities, contended all four GPUs and stopped Omega's
accepted screen once. Both divergent trees contain zero cell records and are
marked void; the accepted screen rejoined its original identity. This is not
treated as mitigation or closure.

The accepted campaign was not unauthorised. It was launched by General Musashi
in direct execution of the owner's corrected-ML order and is governed by this
audit, the paired-contract handoff, and the runtime non-interference order.
Finding `AUD-GEN-20260816-249` (S3) records the real governance weakness:
runtime authority existed on a separate pushed branch and was not discoverable
from Satoshi's canonical-checkout-only search. The durable evidence companion
`P1LR_CAUSAL_RUNTIME_AUTHORITY_2026_08_16.json` now names the accepted sources,
identities, unit family and retired identity without modifying runtime output.

The eight masks are deliberate containment. Identity `cdf30aebf585385b` and
its old unit names are formally retired and must not be restarted, unmasked or
have stale claims released. Their files remain historical evidence. The
accepted sources already bind both `agent-multi@3d2bf3f4` and
`gym-fx@634c3fd3` through immutable runtime roots; Satoshi's requested source
isolation correction is therefore already implemented and live.

### Transition-supervisor defect

Finding `AUD-GEN-20260816-250` (S3, corrected operationally, awaits independent
verification): the transition supervisor launched the four accepted decision
units and then looked only for a seed-level `runner_heartbeat.json`. Healthy
training emits per-cell `heartbeat.json`, so the supervisor falsely exited 23
despite four running workers. The local supervisor now validates the newest
per-cell heartbeat for exact decision identity/seed/mode, waits up to ten
minutes, and has an idempotent already-running path. Re-execution returned
status 0 with 4/4 accepted decision heartbeats; no worker was restarted.
