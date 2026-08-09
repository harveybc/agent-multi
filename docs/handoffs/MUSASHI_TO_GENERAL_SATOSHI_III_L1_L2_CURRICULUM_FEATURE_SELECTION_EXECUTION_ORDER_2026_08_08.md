# L1/L2 Curriculum, Feature Selection and Stopping Execution Order

Date: 2026-08-08 America/Bogota
From: General Musashi, ML architecture lead and independent verifier
To: General Satoshi III, technical lead and implementer
Owner authorization: granted by the project owner in the message that produced
this order; do not request it again for work inside this scope
Priority: P0 after safe WP0 quarantine; implementation may proceed in parallel
with valid unrelated fleet work

## 1. Role and Standard

Act simultaneously as:

- a senior reinforcement-learning researcher;
- a senior evolutionary-computation/DEAP engineer;
- a time-series ML scientist specializing in chronological validation;
- a trading-system scientist who reports return and risk on explicit horizons;
- a distributed DOIN plugin engineer who preserves existing node interfaces;
- an experiment-orchestration engineer responsible for non-idle compute; and
- a reliability engineer who treats artifacts, restarts and lineage as code.

Do not improvise a smaller experiment, shorten dates, replace a decision run
with a smoke, select an epoch count manually, use test for convenience, or add a
new owner-approval ceremony. If implementation exposes a real contradiction,
state it immediately, keep compatible work running and propose the narrowest
technical correction. Do not silently choose a default.

The historical NEAT observations are valuable hypotheses. Do not claim they
transfer to SAC/DEAP until the comparisons below establish that transfer.

## 2. Mandatory Reading and Teach-Back

Read in order:

1. `docs/audits/AUDIT_L1_L2_CURRICULUM_FEATURE_SELECTION_AND_STOPPING_2026_08_08.md`
2. `docs/work_plan/38_NEAT_LESSONS_L1_L2_CURRICULUM_AND_FEATURE_SELECTION.md`
3. `docs/audits/AUDIT_SATOSHI_III_M0_M1_M0X_2026_08_08.md`
4. `docs/audits/AUDIT_SATOSHI_III_WP0_QUARANTINE_2026_08_08.md`
5. `docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_EMERGENCY_M0_M1_REPAIR_SPEC_2026_08_08.md`
6. `docs/work_plan/04_MODELS_POLICIES_AND_TRAINING.md`
7. `docs/work_plan/33_ETH_DECISION_RESEARCH_AND_MULTI_ASSET_ROADMAP.md`
8. `docs/work_plan/34_ETH_DATA_OBSERVATION_MANIFEST.md`
9. `pipeline_plugins/rl_pipeline_with_validation.py`
10. `pipeline_plugins/rl_pipeline_with_solvency_curriculum.py`
11. `optimizer_plugins/default_optimizer.py`
12. `agent_plugins/sac_agent.py`
13. `app/metrics.py`

Before editing, deliver a short machine-readable teach-back containing:

- why L1 and L2 curriculum are compatible but must first be isolated;
- exact 11,509/2,190/2,196/2,190 nested rows;
- why 2025 is inaccessible;
- why `N14/EN4_10` is retained but non-decisional;
- why lexicographic rank keys cannot simply be averaged;
- the exact difference between FS0, FS1 and FS2;
- what current DOIN stages inherit and what they do not; and
- which actions no longer require a fresh owner phrase.

Any wrong answer blocks only the affected implementation work. It does not idle
the fleet.

## 3. Source-Control Isolation

1. Work from a clean branch based on this order's pushed head.
2. Do not mutate active campaign checkouts.
3. One bounded commit per work package below; tests travel with code.
4. Keep generated runtime data outside Git and persist only manifests/evidence.
5. Do not modify `doin-node` unless a failing contract test proves the external
   optimizer/plugin interface cannot express a required fact. Escalate with the
   failing test before any protocol edit.
6. Preserve all historical M0/N14/EN4_10 evidence byte-for-byte.

## 4. WP0 - Finish Safe Quarantine, Then Continue

Complete findings 166-169 exactly per the existing correction order. This is a
recovery prerequisite, not permission to idle GPUs. While the CPU-side
quarantine correction is implemented/reviewed, run only compatible jobs that do
not consume the invalid successor or mutate the ETH curriculum domain.

WP0 acceptance does not make M0's withdrawn mechanism claim true. It only makes
the invalid successor safely unusable and its evidence honest.

## 5. WP1 - Nested Chronological Split and Full-Year Scoring

### 5.1 Implementation boundary

Refactor split materialization in
`pipeline_plugins/rl_pipeline_with_validation.py` behind a typed split contract.
Do not add date parsing independently to each pipeline.

New roles:

```text
fit_train          2017-09-28 04:00 <= t < 2023-01-01  (11,509)
train_monitor      2022-01-01          <= t < 2023-01-01
inner_validation   2023-01-01          <= t < 2024-01-01 (2,190)
outer_validation   2024-01-01          <= t < 2025-01-01 (2,196)
sealed_test        2025-01-01          <= t < 2026-01-01 (2,190)
```

The materializer must derive and verify exact row counts from the pinned file;
the prose values are assertions, not inputs to fake.

### 5.2 Prefix semantics

For train-monitor, inner, outer and test, prepend enough causal rows to satisfy:

```text
context_bars = max(window_size, scaling_window, max_feature_lookback)
```

Add explicit per-row/scenario state (`is_context_prefix` or equivalent) so the
environment can initialize observations without allowing actions, orders,
account mutation, replay insertion or score contribution. Do not infer prefix
status from row position in downstream metrics.

If the environment cannot accept this distinction, implement one reusable
wrapper/contract at the environment adapter boundary. Do not duplicate it in
each evaluator.

### 5.3 Required evidence

Emit a split manifest with source SHA, role, score start/end, context start,
context rows, scored rows and split SHA. Loading an artifact with a different
split manifest must fail closed.

### 5.4 Tests

- exact ETH row/date assertions;
- one-row boundary mutation fails;
- overlap, gap and reordering fail;
- context rows produce no actions/trades/equity change/metrics/replay;
- first scored row has complete context;
- full 2023/2024/2025 score intervals remain intact;
- test path cannot be opened in train/L1/L2 modes; and
- window/scaling changes recompute required context.

## 6. WP2 - Paired L1 Generalization and Early Stopping

### 6.1 Metric implementation

Add `paired_generalization_weekly_v1` to the domain metric layer, preferably in
one reusable module imported by pipeline and optimizer. It accepts two typed
split summaries and returns:

- eligibility and reasons;
- raw robust weekly utility A/B;
- arithmetic mean;
- absolute generalization gap;
- `beta * gap` penalty;
- final scalar;
- deterministic tie-break tuple; and
- all raw weekly rows/metric vectors by reference/hash.

Keep `lexicographic_weekly_v1` for historical compatibility. New decision
configs must not enter the validation-only branch at
`rl_pipeline_with_validation.py:310-322`.

### 6.2 L1 loop

The normal training loop evaluates `train_monitor` plus
`inner_validation`. Configure:

```text
max_epochs: 2000
l1_patience: 60
l1_patience_start_epoch: 40
epoch_timesteps: derived from valid fit transitions
evaluate_test_split: false
selection_metric: paired_generalization_weekly_v1
```

`max_epochs=2000` is the global pass-equivalent safety ceiling for one complete
candidate. A two-phase candidate cannot receive 2,000 epochs twice. Add explicit
`total_max_passes`, `phase1_max_fraction` and `normal_phase_min_passes`; every
phase consumes one shared budget ledger. Anchor the initial phase-1 range on the
historical 4/14 allocation, but do not call it selected. A one-seed cap probe may
show whether any ceiling binds; it is mechanics/range evidence only.

Do not hardcode `20_000` into the new scientific contract. Derive a pass
equivalent and record both transitions and environment interactions. Preserve a
separate configurable checkpoint interval if evaluation every pass is too
expensive; do not call it an epoch if it is not one.

L1 improvement patience and activity-ineligible patience remain separate.
Restore the best paired checkpoint and export both best and terminal artifacts.

### 6.3 Tests

- validation improves while train collapses: no false improvement;
- train improves while validation collapses: no false improvement;
- both improve: patience resets;
- gap increases enough to erase mean gain: no reset;
- activity failure on either split is ineligible;
- validation-only legacy behavior cannot be selected by the new config;
- deterministic tie break; and
- resume preserves best score, patience, checkpoint hash and split identity.

## 7. WP3 - Correct Easy-Phase L1 Stopping and Handoff

Modify `pipeline_plugins/rl_pipeline_with_solvency_curriculum.py`:

1. Easy training uses easy dynamics only on `fit_train`.
2. Easy checkpoint A is easy `train_monitor` utility.
3. Checkpoint B is normal-realistic `inner_validation` utility.
4. Selection uses the paired comparator, not easy economic equity plus activity
   gates.
5. `epoch=0` is baseline evidence and is structurally ineligible as treatment
   handoff.
6. The selected handoff proves actor/critic parameter change against the anchor;
   ZIP inequality is insufficient.
7. Phase-1 never observes phase-2 outcomes.
8. Normal phase applies the same L1 stopping contract as WP2.
9. Boundary replay/optimizer/model reconstruction is identical in paired normal
   and easy arms.
10. Normal-realistic outer validation happens after training and cannot control
    L1 patience.

Add mutation tests for epoch-zero fallback, post-hoc phase-2 handoff selection,
wrong solvency mode, replay leakage, optimizer-state drift and unchanged policy
weights.

## 8. WP4 - Replace Fixed M1 with the Decision-Bearing L1 Factorial

### 8.1 Preserve and supersede

Keep N14/EN4_10/E4 and M0 in their existing evidence classes. Create a new
domain/plan ID and output root. Never resume or append the new results to their
chains/aggregations.

### 8.2 Cells

Four paired seeds (`101,202,303,404`), shared per-seed anchor:

| Cell | Phase 1 | Boundary | Phase 2 |
| --- | --- | --- | --- |
| `L1_N_M10` | normal + paired stop | matched | normal LR 1.0 + paired stop |
| `L1_E_M10` | easy + paired stop | matched | normal LR 1.0 + paired stop |
| `L1_N_M03` | normal + paired stop | matched | normal LR 0.3 + paired stop |
| `L1_E_M03` | easy + paired stop | matched | normal LR 0.3 + paired stop |

All cells have equal maximum total interaction caps. Record actual compute; an
early-stopped cell may use less. The outer-validation evaluator is frozen and
identical. No test access.

### 8.3 Mechanics smoke

One seed, tiny cap, all four cells. It proves code only. Its schema must set:

```text
evidence_class=mechanics_smoke
decision_eligible=false
performance_aggregate_eligible=false
```

### 8.4 Full decision run

After focused tests and the smoke pass, launch all four GPUs without requesting
another owner phrase. The run must use the exact nested chronology and high
ceilings above. One coordinated campaign/plan identity; no parallel chains for
the same arm.

### 8.5 Outcome

Produce paired per-seed differences, bootstrap interval as descriptive
uncertainty, activity/safety/tail-risk deltas and compute deltas. No positive
profit gate. Use the decision rule in document 38 section 5.4.

After the complete L1/L2/FS configuration is frozen, materialize a release
refit with fit data through 2023 and validation 2024. No genome, mask, curriculum
or stopping field may change during this refit. Only then may the single frozen
candidate open 2025 for release evidence.

## 9. WP5 - L2 Paired Fitness and Generation Stopping

Change `optimizer_plugins/default_optimizer.py` without breaking historical
`ga_fitness_split` configs:

1. New-domain candidate training returns L1 fit/inner evidence plus frozen outer
   evidence in one typed packet.
2. Compute L2 fitness from `inner_validation` and `outer_validation` using the
   paired comparator.
3. Reject missing/wrong split identities; never substitute train or zero.
4. L2 patience has `minimum_generations`, `patience`, `min_delta` and explicit
   stage identity.
5. Log unique genomes, pairwise/allele diversity, rejection share and fitness
   dispersion each generation.
6. Resume restores stage, best paired evidence, diversity and patience.
7. Test remains unreachable.

Tests must prove a candidate with excellent inner and collapsed outer cannot
win; a missing outer packet rejects; generation patience cannot fire before its
floor; and resume is byte-equivalent.

## 10. WP6 - L2 Normal versus Staged Easy-Normal

Freeze the L1 winner. Materialize two separate domain instances with shared
initial genome seeds and equal total candidate-evaluation budgets:

- `L2_N`: all normal-realistic;
- `L2_EN`: easy evolution/triage followed by mandatory normal re-evaluation and
  normal generations.

At the stage boundary, erase/invalidate easy fitness. Re-evaluate every carried
elite under normal. A champion/migration/archive path must assert
`evaluation_difficulty=normal_realistic`.

Compare final normal paired fitness, raw outer metrics, valid champion rate,
diversity, candidates and time-to-best. Do not count easy candidates as normal
evaluations. A null result selects the simpler L2-N contract.

## 11. WP7 - Conditional Curriculum Interaction

If L1 or L2 passes its declared decision rule, materialize the bounded 2x2:

```text
L1 {normal,easy-normal} x L2 {normal,staged-easy-normal}
```

Use four paired seeds and nearest selected settings. If neither isolated axis
helps, record `interaction_not_triggered` and proceed normal-only. Do not run an
unbounded search to rescue a failed hypothesis.

## 12. WP8 - FS0/FS1/FS2 Feature Selection

### 12.1 Shared mask contract

Create one dependency-light typed contract for:

- ordered feature list/hash;
- mandatory indices;
- family/group membership;
- active mask/gate values;
- repair trace;
- active count and family counts; and
- deterministic artifact hash.

Apply masks after causal scaling with stable dimensions. A feature-order/hash
mismatch fails. State/risk/protection channels cannot be disabled.

### 12.2 FS1 L2 inherited masks

Extend the agent-multi optimizer plugin schema, not doin-node. Support bool/
categorical mask genes explicitly; do not encode hundreds of booleans as
untyped floating-point accidents. Add deterministic sparse initialization:
mandatory agent/risk/protection state plus one eligible market feature and its
ancestor gates per genome. Mutation adds/drops groups and features with repair;
repair always leaves at least one market feature active.

### 12.3 FS2 L1 learnable gate

Add a separate plugin, suggested path:

`agent_plugins/sac_sparse_feature_gate_agent.py`

Use a custom SB3 feature extractor or equivalent plugin-local module. Do not
contaminate the base `sac_agent.py` with optional gate logic beyond a shared
load/export hook. The gate is feature-wise and shared across lookback positions.

Artifact evidence includes logits/probabilities, frozen thresholded mask,
regularization configuration and action-replay proof after load.

### 12.4 Comparison

Run FS0/FS1/FS2 under the frozen L1 recipe with equal caps and four seeds. Then
run FS1+FS2 only if both individually help. First use the fixed 83-feature ETH
contract for attribution. Later expand to all causal real-time-compatible
sources under document 33; do not freeze 83 as the business's final data set.

## 13. WP9 - Stage-Local Evolution/Maturation

Extend stage schema and shared reproduction with:

```json
{
  "cxpb": 0.0,
  "mutpb": 0.0,
  "numeric_mutation_scale": 0.0,
  "categorical_change_probability": 0.0,
  "minimum_generations": 0,
  "patience": 0,
  "diversity_floor": 0.0
}
```

The literal values above illustrate fields, not defaults. Every stage must
materialize explicit evidence-selected values. Unknown/missing values fail in
the new domain. Historical domains keep their legacy behavior.

Implement representation, capacity, learning, execution/risk and maturation
stages from document 38. Add deterministic shared-node tests proving every node
reproduces identical next populations and stage advances.

## 14. OLAP and Artifact Contract

Persist dimensions/facts for:

- optimization level and curriculum axis;
- phase/stage difficulty;
- split role and dates;
- paired score components/gap/penalty;
- stopping state and reason;
- actual/max compute;
- feature-selection mechanism and mask hash;
- stage-local evolutionary parameters and diversity;
- model/optimizer/replay handoff semantics;
- normal-only/easy-normal pair ID; and
- exact artifact/config/data/code lineage.

Every winning and diverse elite exports a loadable SB3 ZIP, resolved JSON,
mask/gate artifact, metrics/weekly vectors and deterministic inference trace.

## 15. Orchestration and Anti-Idle Rule

Maintain a replicated job queue with explicit dependencies. Audit waiting is
not a fleet-wide job state. While one package awaits verification, dispatch the
next compatible CPU or GPU job that cannot contaminate it.

Every status report must include for Omega, Dragon, Gamma and the eGPU worker:

- current job/arm/seed/stage/candidate;
- process and heartbeat freshness;
- GPU utilization, memory and temperature;
- candidates completed/total, throughput and ETA when estimable;
- canonical plan/domain/genesis/tip identity for DOIN work; and
- reason plus next dispatch if utilization is idle.

An unexplained idle declared worker is an operational anomaly. A GPU may be
intentionally unused only when no compatible GPU job exists, maintenance is
declared, or safety/thermal state requires it; the reason and next check are
recorded.

## 16. Required Delivery Sequence

For each WP:

1. reproduce the old defect or missing capability before editing;
2. implement one bounded change;
3. run focused, property/mutation where relevant, integration and full suites;
4. archive exact command/results and hashes;
5. commit and push;
6. continue compatible work while requesting independent audit; and
7. never close your own finding.

Final delivery packet must include a table mapping findings 170-177 and prior
159-169 to commits, tests and runtime evidence, plus exact launched/not-launched
states for every experiment. Do not report "done" when only the smoke passed.

## 17. Questions Satoshi May Resolve Without Owner Escalation

You may choose module boundaries, class names, test organization, serialization
details and equivalent deterministic sparse-gate mathematics consistent with
this contract. Ask Musashi for a technical ruling only if two requirements
conflict. Escalate to the owner only for the reserved authority classes in
document 38 section 9.

Begin with WP0/WP1 implementation now. Keep compatible fleet work active.
