# Musashi to General Satoshi III: 209-223 Verdict and Phase-1 LR Order

Date: 2026-08-11 America/Bogota  
Authority: owner-approved document 38, standing anti-idle authority and
independent audit `AUDIT_SATOSHI_III_RETURN_209_220_2026_08_11.md`  
Required posture: execute without requesting another owner phrase; preserve
all sealed evidence and all Paper/Demo risk controls

## 1. Role and Objective

Act as a senior machine-learning researcher, reinforcement-learning engineer,
distributed-systems engineer and evidence-custody implementer. Your immediate
objective is to turn the one-seed ladder into a correctly attributed,
decision-capable phase-1 experiment while keeping all available GPUs on useful
work and leaving live Paper/Demo trading uninterrupted.

Do not describe the current result as "easy is harmful" or as a pure boundary
effect. The current accepted statement is:

> The v3 checkpoint-selection/handoff bundle bypassed easy training by choosing
> the anchor and remained active. The v4 path handed actually trained easy
> weights whose deterministic actions were already nearly constant and below
> the normal `0.1` deadband; phase 2 did not recover at seed 101.

Replay and optimizer carry are not current candidate causes: every arm reset
both. Do not spend a factor on them in this round.

## 2. Audit Disposition

- 209-216, 218 and 219: independently verified pending owner closure.
- 217: open; checker false zero and two default-branch links remain broken.
- 220: partial; ladder executed, but no successor was dispatched.
- 221: open; easy activity accepts near-constant below-threshold policies.
- 222: open; compound v3/v4 treatment was overnamed as a pure boundary delta.
- 223: open; collector can certify no terminal artifacts.

## 3. Non-Negotiable Boundaries

1. Never mutate either sealed L1 collection or ladder collection.
2. Never open the sealed 2025 split.
3. Every entry in live Paper/Demo remains natively protected by SL and TP.
4. No real-capital authority, paid commitment, secret movement or history
   rewrite is granted here.
5. The paused 2026-08-06 DOIN chain stays preserved; never resume it as if it
   were the new experiment.
6. All new distributed DOIN jobs use the accepted v2 explicit chain identity;
   legacy chains remain read-only.
7. Review is not permission to idle the fleet. Corrections may run beside
   non-conflicting, pre-approved diagnostics.

## 4. WP0: Restore Honest Repository-Link Evidence

Files:

- `docs/audits/evidence/README_LINK_RESOLUTION_CHECKER_2026_08_10.py`
- `README.md`
- checker tests under `tests/`

Required changes:

1. Resolve every repository against its declared remote default ref from the
   inventory/GitHub metadata, never an arbitrary local `HEAD`.
2. A missing README, missing repository, unresolved default ref or Git command
   error increments failure and gives a nonzero process exit.
3. Assert that every expected repository produced a fully checked row.
4. On `agent-multi` default master, remove or correct the two links to files
   absent from that tree:
   `pipeline_plugins/_nested_splits.py` and
   `pipeline_plugins/_paired_generalization.py`. Do not merge unrelated code
   merely to make a link green.
5. Add negative tests for missing README, local feature-branch drift and Git
   command failure.

Acceptance: exact default tips checked, zero errors, zero broken relative links
and nonzero exit for each adversarial fixture.

## 5. WP1: Correct Ladder Terminal Custody

Files:

- `tools/m0_l1_ladder_collect.py`
- `tests/test_m0_l1_ladder_collect.py`
- `TOOL_DECLARATIONS.json` only if its declared surface changes

Required changes:

1. Every training arm must carry non-empty `terminal_model_path` and
   `terminal_model_sha256`, regardless of activity or best-checkpoint status.
2. Resolve the exact staged terminal by deterministic relative path; refuse
   zero or multiple matches. Basename-only first-match search is insufficient.
3. Hash the staged terminal and require equality with the arm record.
4. Build replica expectations from terminal fields, never only best-model
   fields.
5. Require exactly one replica load proof for each expected arm, bound to arm,
   seed, relative path and SHA. Duplicate, missing, foreign or `loads=false`
   proof refuses the collection.
6. Publication repeats the terminal-proof cardinality/binding check rather
   than trusting presence of any generic replica proof.

Mandatory adversarial tests:

- missing D2 terminal;
- wrong D3 terminal SHA;
- duplicate basename in one arm;
- replica proof contains only D0;
- replica proof swaps D2/D3 paths;
- `loads=false` for D4;
- four records with no model fields must refuse (the auditor's reproducer);
- valid inactive D2-D4 terminal-only records seal and publish.

Do not rewrite the existing seal. After correction, run a read-only verifier
against it and emit a supplemental external proof showing the four terminal
loads already independently reproduced by Musashi.

## 6. WP2: Same-Artifact Threshold Replay

Implement a socket-free diagnostic, preferably in
`tools/m0_l1_boundary_action_replay.py`, with unit tests.

Inputs:

- exact D2 post-easy artifact `2620b722...` from the existing sealed ladder;
- exact D0 anchor artifact as control;
- the same train-tail and inner-validation data/config hashes;
- deterministic policy inference; and
- action thresholds `0.0` and `0.1` applied to the **same raw action vector**.

For each artifact/split emit:

- observation count and exact observation manifest hash;
- raw action min/max/mean/std, interquartile range and quantiles of `abs(a)`;
- deterministic unique/near-unique action count using a declared numeric
  tolerance derived from dtype precision;
- fraction `abs(a) > 0.0` and fraction `abs(a) >= 0.1`;
- mapped long/short/hold counts under both thresholds; and
- action-vector SHA so both threshold arms prove identical inputs.

This job performs no learning and no performance selection. Its question is
only whether the threshold discontinuity turns the already-trained policy into
hold. Execute it immediately while WP0/WP1 are being corrected. Pin one
artifact replay to each available GPU so the fleet produces useful,
cross-device reproducibility evidence rather than waiting idle.

Typed outcomes:

- `THRESHOLD_EXPOSES_PREEXISTING_COLLAPSE`;
- `THRESHOLD_CAUSES_MAPPING_COLLAPSE_WITH_VARIATION_RETAINED`;
- `NO_THRESHOLD_COLLAPSE`; or
- `REPLAY_INCONCLUSIVE` with exact refusals.

## 7. WP3: Handoff-Viability Evidence

File:

- `pipeline_plugins/rl_pipeline_with_solvency_curriculum.py`
- related unit tests in `tests/unit/test_solvency_curriculum_pipeline.py`

Add a typed evidence block to every phase-1 checkpoint. Do not guess a broad
numeric gate. Record deterministic policy behavior under both the phase-1 and
phase-2 thresholds on train-monitor and inner-validation:

- raw distribution and action-vector hash;
- whether any action can cross the phase-2 threshold;
- normal-realistic probe trades and protected entries;
- exact/near-constant classification derived from dtype/control evidence;
- `handoff_viability` in
  `{VIABLE, BELOW_NORMAL_THRESHOLD, CONSTANT_POLICY, NO_TRADE, UNAVAILABLE}`.

The existing paired comparator remains authoritative for selection. A
diagnostic terminal fallback may produce an inactive record, but it must never
be represented as a selected viable handoff. Add source assertions and tests
that epoch-zero anchor telemetry cannot become a trained treatment.

## 8. WP4: Phase-1 Difficulty x Phase-1 LR Factorial

The prior L1 factorial varied **phase-2** LR while phase-1 LR stayed `1e-4`.
It therefore cannot answer whether phase-1 learning rate caused the collapse.
The next bounded design is:

| Factor | Levels |
| --- | --- |
| phase-1 dynamics | normal, easy |
| phase-1 LR | `1e-4`, `3e-5` |

Hold fixed:

- phase-2 normal LR at `3e-5` (the active D0 range point);
- phase-2 normal-realistic dynamics and threshold `0.1`;
- replay reset, optimizer reset, entropy mode/value, topology, feature set,
  costs, protection, dates, pass-equivalent budgets and stopping;
- one hash-bound mature ETH anchor per seed; and
- normal-realistic paired selection and outer-validation truth.

Use seeds 101, 202, 303 and 404. Pin each seed to one physical GPU and run all
four cells for that seed on the same GPU, so hardware is paired within seed.
Use a deterministic balanced cell order across seeds to avoid order/thermal
confounding. Every cell starts from the exact seed anchor, not a preceding
cell's terminal.

### 8.1 Mechanics screen

Run one pass-equivalent per cell first. It is a collapse/contract screen, not a
performance result. Require:

- 16/16 records with exact identities and terminal artifacts;
- all four terminal loads on the replica for each completed seed batch;
- direct handoff-viability facts; and
- at least one trained arm that is not `CONSTANT_POLICY` or
  `BELOW_NORMAL_THRESHOLD` before spending the full budget.

If all four treatment combinations collapse at all four seeds, stop and return
`PHASE1_LR_REGION_COLLAPSED`; do not burn the full run.

### 8.2 Decision run

If the screen finds a viable region, execute the decision run under document
38's real L1 stopping:

- maximum 2,000 global pass-equivalent checkpoints as safety ceiling;
- patience 60 and minimum floor 40;
- best-checkpoint restoration;
- train-monitor plus inner-validation paired stopping;
- final outer validation; and
- sealed 2025 inaccessible.

Report paired main effects for phase-1 dynamics and phase-1 LR plus their
interaction. This design answers whether the apparent easy/normal question was
actually a phase-1 learning-rate problem. It does not yet optimize a broad LR
range.

Typed outcomes:

- `PHASE1_LR_MAIN_EFFECT`;
- `PHASE1_DIFFICULTY_MAIN_EFFECT`;
- `PHASE1_LR_DIFFICULTY_INTERACTION`;
- `NO_MATERIAL_EFFECT`;
- `TOTAL_ACTIVITY_COLLAPSE`; or
- `INCONCLUSIVE` with exact cause.

Only if a viable easy arm exists does the already-approved later
`LR_easy x LR_normal` response surface become eligible. Bounds come from this
factorial; no invented sweep.

## 9. WP5: v2 Chain Deployment Preparation

Corrections 209-211 are accepted. Validate the prepared migration manifest in
a clean temporary state directory and prepare one new explicit-ID v2 genesis
for the next distributed DOIN component job. Every machine must print the same
chain ID, genesis hash, domain/config/data hashes and exact component commits
before any candidate claim. Preserve all legacy databases byte-for-byte and
read-only.

Do not resume the paused full-v2 chain. Do not launch a new DOIN optimization
until the component question and domain are materialized under the ordered
roadmap; local paired diagnostics above run now.

## 10. Runtime Continuity

Throughout this package:

1. Alpaca and MT5 Paper/Demo controllers continue; do not restart a venue
   holding exposure unless the existing protected-position handover contract
   proves it safe.
2. IBKR remains held until the existing owner command packet is used after
   fresh direct-flat reconciliation.
3. Monitor GPU temperatures hourly; unresolved `>=78 C` alerts once via
   Telegram, with recovery notice only after a real recovery.
4. A worker idle for more than 15 minutes while an approved compatible job is
   queued is an operational alert and auto-dispatch defect, not a review gate.
5. Status must name current job/cell/seed, GPU utilization/temperature,
   candidate or checkpoint ETA and total pool progress.

## 11. Return Packet

Return one packet with:

- before/after reproducers for 217 and 221-223;
- exact commits and clean/pushed state;
- same-artifact threshold replay table;
- mechanics or decision-run identity, per-worker assignments and live status;
- terminal artifact paths/hashes and one replica load proof per arm;
- raw common-scale weekly, annualized, drawdown, trade and activity metrics for
  decision-bearing runs only;
- status of all four fronts and any unresolved owner action; and
- a direct teach-back explaining why replay/optimizer are excluded this round,
  why phase-1 LR is now crossed with difficulty, and what evidence would permit
  a later `LR_easy x LR_normal` experiment.

No finding is self-closed. Musashi reproduces the return. Work proceeds while
review is pending through the standing compatible-job queue.

