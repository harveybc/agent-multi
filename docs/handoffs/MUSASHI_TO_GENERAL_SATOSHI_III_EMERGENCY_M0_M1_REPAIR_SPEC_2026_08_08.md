# Emergency M0/M1 Repair Specification for General Satoshi III

Date: 2026-08-08 America/Bogota  
Author: General Musashi, independent verifier  
Recipient: General Satoshi III, technical lead  
Priority: P0 Front 1  
Purpose: correct the invalid M0 easy handoff and produce the first causal,
executable comparison of easy versus normal phase-1 SAC training  
Runtime authority: bounded research execution only; no Live-capital authority

## 0. Bootstrap Exactly Here

Before editing:

```bash
git fetch origin audit/m0-m0x-20260808
git cherry-pick 99bb7fff9c78999fee6ed9b5d5060a7860d61dae..origin/audit/m0-m0x-20260808
```

Read, in order:

1. `docs/audits/AUDIT_SATOSHI_III_M0_M1_M0X_2026_08_08.md`
2. `docs/audits/evidence/SATOSHI_III_M0_M0X_REPRO_2026_08_08.py`
3. this specification;
4. `pipeline_plugins/rl_pipeline_with_solvency_curriculum.py` functions
   `_train_easy_phase()` and `run_pipeline()`;
5. `agent_plugins/sac_agent.py` function `load_for_training()`;
6. `tools/eth_sac_inner_curriculum_screen.py` functions
   `validate_contract_v2()` and `run_m0_arm()`; and
7. `tools/aggregate_eth_sac_inner_curriculum.py` verification and
   interpretation functions.

Use codebase-memory MCP to trace these functions and their tests before broad
file reading. Act simultaneously as a senior reinforcement-learning scientist,
causal experiment designer, quantitative-trading simulator engineer, Python
plugin architect and distributed evidence engineer.

Do not reinterpret the mission. The required question is:

> With an identical SAC phase boundary and normal fine-tuning process, does
> four epochs under easy solvency dynamics produce a better normal-validation
> activity outcome than four epochs under normal solvency dynamics?

## 1. Root Cause You Must Correct, Not Work Around

M0's treatment path was not executed. `_train_easy_phase()` allowed the
warm-start baseline to compete as epoch 0, and required each trained easy
checkpoint to pass a normal-handoff activity gate. In all 12 easy arms:

```text
epoch 0 anchor: normal probe active -> selected
epoch 1 trained easy: easy active, normal probe inactive -> rejected
normal phase input: epoch 0 anchor
```

This is outcome-dependent treatment selection. The normal result that M1 is
supposed to measure was used to decide whether the easy treatment existed.

The prior instruction that a post-easy checkpoint "must remain active under a
normal probe" was wrong for a causal mechanism test. Treat that sentence as
superseded. The normal probe remains telemetry only.

## 2. Non-Negotiable Invariants

1. A declared phase-1 training arm can never hand off epoch 0.
2. Phase-1 handoff is the final trained phase-1 epoch, selected before reading
   any phase-2 or normal-validation outcome.
3. A failed/inactive phase-1 result is still handed off and measured. No anchor
   fallback is permitted.
4. Normal-only and easy arms have the same phase boundary, SAC reconstruction,
   optimizer reset, replay reset, phase budgets and phase-2 configuration.
5. Only phase-1 solvency dynamics differ inside a matched pair.
6. ZIP hash proves archive identity only. Policy change is proved from a
   canonical tensor-state digest and numeric distance.
7. ETH and USDCAD resolve different, exact system manifests. A mismatched asset
   or observation contract fails before model construction.
8. Historical M0 evidence is immutable. Corrections supersede; they do not
   rewrite, delete or make the historical reproducer turn green.
9. No successor is emitted from prose. An executable aggregator produces one
   deterministic typed outcome or `INCONCLUSIVE`.
10. SL/TP and execution-cost contracts remain unchanged. Test split remains
    excluded from selection.

## 3. WP0 - Quarantine the Incorrect Successor

Implement `tools/quarantine_inner_curriculum_successor.py` with an exclusive
file lock and atomic/fsynced replacement.

For:

```text
~/.local/share/agent-multi/eth_sac_inner_curriculum_m0_20260807_v1/
  queue/m0_successor_mechanism_pass.json
```

it must:

1. compute and record the original SHA-256;
2. move the original bytes to
   `queue/retired/<sha256>/m0_successor_mechanism_pass.json`;
3. write a superseding record at the original path with:

```json
{
  "schema": "agent_multi.inner_curriculum_successor_supersession.v1",
  "launch_eligible": false,
  "supersedes_sha256": "<original>",
  "reason_finding": "AUD-F1-20260808-159",
  "preserved_observation": "reduced normal LR/duration retained activity; easy contribution unmeasured"
}
```

4. prove idempotency; a second invocation changes no bytes;
5. inspect campaign/supervisor ledgers and record whether the old successor was
   ever claimed; and
6. emit `m0_correction_envelope_v1.json` beside the original aggregation,
   binding hashes of the aggregation, final table, manifest and supersession.

Do not edit `m0_aggregation.json`, the 16 records or any model ZIP.

## 4. WP1 - Implement a General Phase-1 Handoff

### 4.1 Files

Primary files:

- `pipeline_plugins/rl_pipeline_with_solvency_curriculum.py`
- `agent_plugins/sac_agent.py`
- `tests/unit/test_solvency_curriculum_pipeline.py`
- new focused tests under `tests/` if separation is clearer

### 4.2 Typed configuration

Introduce a versioned phase contract consumed by the pipeline:

```json
{
  "phase1_mode": "easy_chronological_continuation | normal_realistic",
  "phase1_epochs": 4,
  "phase1_handoff": "terminal_trained_epoch",
  "phase1_learning_rate": 0.0001,
  "phase2_mode": "normal_realistic",
  "phase2_epochs": 10,
  "phase2_learning_rate": 0.00001,
  "boundary_replay_policy": "fresh",
  "boundary_optimizer_policy": "fresh",
  "selection_split": "normal_validation_after_phase2"
}
```

Reject unknown values, booleans-as-numbers, nonfinite rates, nonpositive epoch
budgets and any phase-1 handoff other than `terminal_trained_epoch` for M1.

### 4.3 Phase-1 behavior

Refactor `_train_easy_phase()` into a mode-aware phase-1 routine or add a
parallel routine with one implementation underneath. Exact behavior:

1. Load the mature anchor and record epoch-0 telemetry, but mark it
   `handoff_eligible=false` unconditionally.
2. Train exactly `phase1_epochs`. M1 does not use phase-1 early stopping.
3. Save the terminal phase-1 model after the final epoch.
4. Compute direct phase-1 facts under both its own environment and the fixed
   normal probe, but neither fact selects/rejects the checkpoint.
5. Refuse if no gradient update occurred, artifact loading fails, tensors are
   nonfinite or the final artifact is tensor-identical to the anchor.
6. Pass that exact final artifact to `load_for_training()` for phase 2.
7. Prove the phase-2 initial policy tensor digest exactly equals the phase-1
   terminal digest before the first phase-2 gradient.

The metadata schema must include:

```text
anchor_artifact_sha256
anchor_policy_tensor_sha256
phase1_terminal_artifact_sha256
phase1_terminal_policy_tensor_sha256
phase1_changed_tensor_count
phase1_max_abs_tensor_delta
phase1_l1_tensor_delta
phase1_environment_timesteps
phase1_gradient_updates
phase1_own_mode_activity
phase1_normal_probe_activity
handoff_epoch
handoff_selected_without_normal_outcome=true
phase2_initial_policy_tensor_sha256
optimizer_state_transferred=false
replay_transitions_transferred=0
```

Expose/reuse one canonical tensor-hash implementation. Do not duplicate an
incompatible hash function in the runner.

### 4.4 Required adversarial tests

- active epoch-0 anchor plus inactive trained epoch: trained epoch is handed
  off;
- normal probe zero trades: trained epoch is still handed off;
- re-saving unchanged SAC: artifact SHA changes but changed tensor count is 0
  and acceptance refuses;
- trained model with one changed tensor: direct digest/distance detects it;
- phase-2 initialization differs from phase-1 terminal by one tensor: refuse;
- NaN tensor, zero gradient updates, missing artifact and wrong mode: refuse;
- normal phase-1 and easy phase-1 exercise the same boundary implementation.

## 5. WP2 - Materialize One Correct M1 Contract

Retire both current M1 variant files from launch eligibility. Create one new
contract, suggested path:

```text
examples/config/phase_3_eth_sac_dynamics/
  m1_matched_boundary_factorial_M01_v3.json
```

Schema: `agent_multi.inner_curriculum_screen_contract.v3`.

Use exactly these primary cells:

| Arm | Phase 1 | Boundary | Phase 2 | Phase-2 LR multiplier |
| --- | --- | --- | --- | ---: |
| `N4_R_N10_M10` | normal, 4 epochs, baseline LR | fresh replay/optimizer | normal, 10 epochs | 1.0 |
| `E4_R_N10_M10` | easy, 4 epochs, baseline LR | fresh replay/optimizer | normal, 10 epochs | 1.0 |
| `N4_R_N10_M01` | normal, 4 epochs, baseline LR | fresh replay/optimizer | normal, 10 epochs | 0.1 |
| `E4_R_N10_M01` | easy, 4 epochs, baseline LR | fresh replay/optimizer | normal, 10 epochs | 0.1 |

M0.1 is selected because the preserved M0 data show activity in 4/4 seeds and
it weakly dominates M0.3 in seed 303 while tying it in the other three. Record
that selection and source-table hash. Do not retain M0.3 as a launchable
alternative.

The experiment uses seeds 101, 202, 303 and 404 and the same exact ETH anchor
per corresponding seed. An optional uninterrupted N14 arm may be diagnostic,
but must be excluded from the matched factorial outcome.

## 6. WP3 - Exact System Manifests and Generic Runner

Create versioned manifests, suggested paths:

```text
examples/config/phase_3_eth_sac_dynamics/systems/
  ethusdt_4h_m1_system_v1.json
  usdcad_4h_m0x_system_v1.json
```

Each manifest must bind exact SHA-256 values for:

- input CSV and exact row/time interval;
- resolved base configuration;
- split dates and `evaluate_test_split=false`;
- ordered feature/observation list, feature count and flattened shape;
- history/price/return/scaling windows;
- preprocessing/scaler implementation and configuration;
- environment, agent and pipeline plugin versions;
- commission, spread, slippage, financing, margin and SL/TP contract;
- anchor champion manifest and model artifact; and
- relevant Git source-tree identities.

For v3, `run_m0_arm()` must not call ETH D1 `_base_config()`. Replace it with
`materialize_system_config(contract, system_manifest, arm, seed, output_dir)`.
Assert materialized asset, data hash, observation shape and anchor shape before
constructing SAC.

Required refusal test: a USDCAD contract paired with the ETH manifest or any
ETH data path must fail before `_agent_plugin()` or model construction.

## 7. WP4 - Executable Aggregation and Identity

Implement a generic aggregator, suggested path:

```text
tools/aggregate_inner_curriculum_screen.py
```

Execution identity includes:

```text
contract SHA + system-manifest SHA + experiment ID + asset + seed + exact arm
factors + anchor tensor/artifact hashes + data/config/observation hashes +
budgets + metric schema + code identity
```

Output root contains the execution ID. No two contract variants may resolve to
the same arm directory.

### 7.1 Per-seed activity fact

`active=true` only when all are direct and true:

- terminal artifact loads and tensor digest matches record;
- phase-1 and phase-2 required updates occurred;
- normal validation `trades_total > 0`;
- raw-action standard deviation > 0;
- non-hold rate > 0; and
- at least one protected entry with native SL and TP was submitted.

Missing facts make the cell invalid, never inactive.

### 7.2 Exact M1 outcome

At each LR and seed define:

```text
paired_activity_delta = int(E_active) - int(N_active)
```

Evaluate exactly in this order:

1. `INCONCLUSIVE` if any cell is invalid or missing.
2. `INTERACTION` if the sign of the summed paired delta differs between LR 1.0
   and M0.1 and both sums are nonzero.
3. `EASY_CONTRIBUTES` if, at M0.1, E is active in at least 3/4 seeds and the
   sum of paired deltas is at least +2.
4. `EASY_HARMFUL` if, at M0.1, matched N is active in at least 3/4 while E is
   active in at most 1/4.
5. `LR_ONLY` if, at M0.1, matched N is active in at least 3/4 seeds and the sum
   of paired deltas is <= 0.
6. `INCONCLUSIVE` for every other complete pattern.

These outcomes concern activity survival only. Always emit per-seed raw
`trades_total`, mean weekly return, total return, maximum drawdown and Sharpe
with units. Profit does not gate this mechanism screen.

Mutation tests must prove that malformed cells, duplicate physical records,
contract drift, tensor mismatch, asset mismatch and absent metrics all yield
`INCONCLUSIVE`/refusal and never promotion.

## 8. WP5 - Smoke Then Full M1

### Mechanical smoke

Run seed 101 for the matched lower-LR pair with reduced `1 + 1` epoch mechanics
budgets in a unique smoke namespace. It is not performance evidence. Acceptance
requires:

1. N and E phase-1 terminal epochs are 1, never 0;
2. both differ tensor-wise from their anchors;
3. phase-2 initial digest equals its own phase-1 terminal digest;
4. N/E boundary reset facts are identical;
5. only phase-1 solvency mode differs;
6. materialized ETH hashes/shapes match the system manifest; and
7. all artifacts load locally and from one independent replica host.

General Musashi independently reproduces this smoke. Once accepted, launch the
complete four-seed/four-cell M1 on the four GPUs. This bounded acceptance is not
permission to leave otherwise available machines idle: keep unrelated valid
pool work active until the M1 launch barrier is ready.

### Full M1 evidence run

The `1 + 1` smoke is mechanics-only and is excluded by schema from every
performance aggregation. It cannot decide easy, LR, profitability or R3.

The full M1 decision run uses the established ETH contract without shortening:

```text
timeframe: 4h
dataset rows: 18,085
train: 13,699 bars, 2017-09-28 04:00 through 2023-12-31
validation: 2,196 bars, calendar 2024
sealed test: 2,190 bars, calendar 2025, forbidden for selection
seeds: 101, 202, 303, 404
primary cells: 4
budget: 14 x 20,000 = 280,000 environment interactions per cell/seed
aggregate primary budget: 4,480,000 environment interactions
```

Wall-clock duration is not an evidence criterion. Four GPUs make a valid
experiment finish faster; they do not permit reducing its chronological data,
seeds, cells or interaction budget. The aggregator must reject smoke schemas
and any shortened production cell.

## 9. WP6 - M0-X Only After M1

Do not execute M0-X in this repair packet. Materialize it only after M1's typed
outcome:

- `EASY_CONTRIBUTES`: use the same matched-boundary easy-versus-normal design;
- `LR_ONLY`: test normal gentle fine-tuning only and remove easy attribution;
- `EASY_HARMFUL` or `INTERACTION`: design the follow-up from that evidence;
- `INCONCLUSIVE`: no R3 gene freeze and no M0-X launch.

USDCAD must use its own exact manifest and re-prove anchor activity. One shared
anchor means seeds measure fine-tuning stochasticity, not anchor diversity.

The currently proposed USDCAD anchor system is not sufficient for a mechanism
decision: it declares only 1,604 training bars at 4h and lacks exact date bounds.
Before M0-X, materialize a new second-system contract with:

- at least four complete chronological training years for 4h data;
- one complete untouched validation year;
- one separate sealed test year;
- exact timestamps, row counts and hashes;
- documented bull/bear, high/low-volatility and trending/ranging coverage; and
- a mature active anchor trained under that exact data/observation contract.

For a 1h-or-finer alternative, one complete training year is the owner's hard
minimum, with a longer matched history preferred. Do not pad, duplicate,
interpolate across closures or reuse an anchor trained on the short contract.
If USDCAD data cannot satisfy this promptly, select another sufficiently
different asset with adequate data; do not lower the evidence requirement.

## 10. Evidence and Return Standard

The historical Musashi reproducer must continue reproducing all historical M0
defects. Making it turn false would mean the immutable evidence was altered.

Create a new acceptance reproducer:

```text
docs/audits/evidence/
  SATOSHI_III_M1_MATCHED_BOUNDARY_ACCEPTANCE_2026_08_08.py
```

It must inspect the new smoke and prove every criterion in section 8 without
network access or training. Return:

- exact commit and clean/pushed branch;
- quarantine/correction-envelope hashes and no-consumer proof;
- before/after code-level counterexamples for findings 159-164;
- new v3 contract and both system manifests;
- smoke artifacts, tensor/update/reset facts and replica observations;
- generic aggregator output and mutation tests;
- local and remote artifact load/hash proof;
- focused and full test suites with exclusions stated; and
- current per-host job, GPU utilization, memory and temperature.

Do not claim the easy mechanism works from the smoke. Do not start M0-X. Do not
self-close findings 159-164. Request General Musashi's independent audit of the
exact delivery commit.
