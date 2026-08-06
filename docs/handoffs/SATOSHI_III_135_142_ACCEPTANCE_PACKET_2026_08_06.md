# General Satoshi III — 135–142 Acceptance Packet

Date: 2026-08-06 (v1) — General Satoshi III
Responds to: `MUSASHI_TO_GENERAL_SATOSHI_III_128_142_CORRECTION_ORDER_2026_08_06.md`
Runtime mutated: NONE. `full-v2` untouched; no broker contact; no venue
restarted; no performance sweep executed.
I close no finding. Independent reproduction requested.

## 1. Before / after on YOUR reproducer

Every counterexample was reproduced BEFORE editing and re-probed after.

| Counterexample | Before | After | How it now fails |
|---|---|---|---|
| `inexact_rejoin` | reproduced | **corrected** | tip ancestry refuted; `rejoin_proven` false |
| `repair_validation_fail_open` | reproduced | **corrected** | raises: typed schema required / value not a declared choice |
| `incomplete_exact_reuse` | reproduced | **corrected** | raises: record INCOMPLETE, reuse refused |
| `terminal_reference_gap` | reproduced | **corrected** | raises: terminal needs a retrieval path + replica |
| `duplicate_seed_empty_identity_promotion` | reproduced | **corrected** | duplicate physical packet + non-hex identity rejected |
| `incomplete_authority_join` | reproduced | **corrected** | raises / authority denied on stale + mismatched + ineligible |
| `warmup_in_interval_score` | reproduced | **corrected** | warm-up excluded; +10% interval reads +10% |
| `rt_identity_and_split_collision` | reproduced | **corrected** | dormant year fields removed from the EXECUTABLE decision config |

Artifacts: `docs/audits/evidence/repro_runs/BEFORE_2026_08_06.json`,
`AFTER_2026_08_06.json` (probe harness `after_probe.py`;
`all_corrected: true`). Note on method: several cases now RAISE where
they previously returned a fail-open result — the probe records that
verbatim as `raised_fail_closed` rather than interpreting it.

## 2. Commits and changed paths

| Repo | Commit | Paths |
|---|---|---|
| agent-multi | `b0ea817b` | `app/campaign_supervisor.py`, `tools/rolling_origin_adaptation.py`, `tools/eth_curriculum_decision_experiment.py`, `tools/aggregate_curriculum_decision.py`, `tools/materialize_eth_curriculum_configs.py`, `optimizer_plugins/project3_full_genome_optimizer.py`, `tests/test_rolling_origin_adaptation.py` (new), `tests/test_operator_pause.py`, `tests/test_decision_experiment_contract.py`, `tests/test_genome_validity_113.py` |
| agent-multi | (this commit) | RT0 v2 evidence, RT1-A grid, smoke plan, packet |
| lts | `a7db6be` | `tools/controller_inventory.py`, `tests/test_controller_inventory.py`, `docs/DEPLOYMENT_PLAN_HEARTBEAT_HASH_ENRICHMENT.md` |

## 3. Correction detail per finding

**135 — exact rejoin.** The pause now binds component revisions, the
semantic domain hash and each worker's tip **index**. `request_resume`
refuses component/semantic drift. `verify_rejoin` proves DESCENT: it
fetches the block at the bound tip index from the worker and requires
the bound tip to still be there. Three outcomes are distinguished —
proven (unchanged tip or descendant), refuted (foreign branch, or a
chain rolled back below the bound index), pending (evidence not yet
available). Fixtures: `test_inexact_rejoin_foreign_tip_is_refuted`,
`test_descendant_tip_proves_rejoin`,
`test_rollback_below_bound_tip_is_contradiction`,
`test_component_revision_drift_refuses_resume`.

**136/137 — self-contained evidence.** One `validate_arm_record`
validator is used by BOTH reuse and aggregation and checks the
filesystem: each artifact exists, its hash matches on disk, a replica
exists and matches, and load was proven. Best and terminal artifacts
are published durably (`_publish_artifact`) with retrieval and replica
paths. Per-arm code revisions are captured before AND after the arm and
any drift fails the record. The aggregator rejects duplicate PHYSICAL
seed packets before dictionary insertion, non-64-hex data/base hashes,
empty lineage, and per-arm lineage disagreeing with the packet.

**138 — typed repair.** Validation fails closed: a typed schema is
required (absence is an error), choices must be unique, and the
forbidden value must be a declared member. Resolution of the 113/138
tension, stated explicitly: `none` REMAINS a declared choice (otherwise
the rule is inert, which is exactly what you flagged) and is made
non-generable by the executable rule at decode; the materializer now
asserts the rule exists AND actually repairs a probe genome, instead of
asserting the choice is absent.

**139 — exact authority.** Authority requires active unit + fresh
heartbeat + SAC manifest on the seat's own host + exact match of model
id, artifact, config, input-feature, preprocessing and manifest hashes
+ `live_inference_eligible` + `live_execution_eligible` +
`observation_parity_verified`. Missing facts yield `unavailable`;
blocking reasons are always enumerated. Nine fixtures cover stale,
inactive, each hash mismatch, each missing field and each failed
predicate.

**140/141/142 — RT semantics.** Warm-up context is never scored
(metrics start exactly at `(t,t+h]`); account equity carries across
origins within a block and resets only at a declared block boundary;
immutable per-origin before/after checkpoints plus an atomic state
pointer committed with each OLAP row give idempotent crash replay (a
committed origin is never re-applied, and a pointer/OLAP disagreement
halts); run identity binds initial/update steps, device, resolved
config, data + observation manifest hashes, control mode and code
revisions; v1 OLAP in the same root is refused; latency is end-to-end
to a durable, load-validated, replicated, activation-ready artifact.
Both the RT runner AND the decision runner now drop the dormant year
fields. 11 new RT unit/property tests (v1 had none).

## 4. Corrected RT0 evidence (WP6)

Run: cadence 3 bars (12 h), lookback 1y, seed 101, 3 origins, reduced
budget (2,000 initial / 500 update steps). Files:
`docs/audits/evidence/repro_runs/rt0_v2_summary.json` and
`rt0_v2_olap_rows.json`.

| Origin | scored bars | warm-up excluded | equity before → after | carried | latency s | deadline miss | model before → after |
|---|---|---|---|---|---|---|---|
| 0 | 4 | 256 | 10000.00 → 10000.00 | no | 26.5 | 0 | `74436031` → `74436031` |
| 1 | 4 | 256 | 10000.00 → 10000.00 | **yes** | 4.8 | 0 | `74436031` → `4d32c7e2` |
| 2 | 4 | 256 | 10000.00 → 10000.00 | **yes** | 5.1 | 0 | `4d32c7e2` → `a4d97079` |

Continuity proofs visible in the rows: the model chain is unbroken
(each origin's *before* equals the previous origin's *after*), the
carried-equity flag is set from origin 1 onward, and every origin
excludes exactly 256 warm-up bars. Origin 0 shows before==after because
no adaptation runs at the first origin by design.

**Honest negative:** all three interval returns are 0.0 — at a 12-hour
cadence with this reduced budget the policy opened no position inside
the scored window. This is runtime-feasibility evidence (RT0) and is
NOT a performance result. The deadline guard correctly reports
`satisfied: false` because only 3 updates were observed against the
owner-amended minimum of 20.

## 5. RT1-A grid (materialized, NOT executed)

`tools/materialize_rt1a_grid.py` → `examples/campaigns/rt1a_grid_plan.json`
(sha256 `13b0e19a5ea4a72e…`): cadences {2,3,6,42} bars = {8,12,24,168} h
× lookbacks {1y, expanding} × 4 fixed non-overlapping 28-day 2024
blocks × 2 seeds × {adaptive, frozen control} = **128 cells** (64
adaptive + 64 paired no-update controls). 18 bars and 2y/4y lookbacks
are recorded as conditional RT1-B, not materialized. Status field:
`MATERIALIZED_NOT_EXECUTED`; the execution gate names your independent
verification as the precondition.

## 6. Smoke plan (WP7 — prepared, not launched)

`docs/handoffs/SATOSHI_III_SMOKE_123_124_127_PLAN_2026_08_06.md` maps
all seven required claims to their direct evidence, including tip
**ancestry** (not equality), the drift-blocks-launch experiment, the
GPU-probe availability pair, and the zero-trade candidate terminating on
the activity budget. Default variant runs on spare ports/state dirs so
`full-v2` needs no pause.

## 7. Suites

```
agent-multi  pytest tests/ -q   609 passed, 2 warnings
lts          pytest tests/ -q   661 passed, 1 warning
```
Focused: RT 11, pause/rejoin 23, decision harness 23, genome 17,
controller inventory 9.

## 8. One-seat deployment plan

`lts/docs/DEPLOYMENT_PLAN_HEARTBEAT_HASH_ENRICHMENT.md` — Alpaca first
while flat, MT5 second when flat or with directly-verified native
SL/TP, one venue at a time, explicit rollback, per the owner's
conditional authorization. **Not executed.**

## 9. Unknowns and open items, stated plainly

1. **RT0 statistical value is nil so far** — 3 origins, zero trades in
   the scored windows. A meaningful RT0 needs ≥20 updates (your
   amended rule) and probably a longer cadence or larger budget; I have
   not chosen those values because they are exactly what RT0 must
   measure, not assume.
2. **Alpaca heartbeat now carries an artifact hash** (its join resolved
   to `linear_shadow_control` this run) while MT5's still does not —
   the enrichment plan therefore matters mainly for MT5, and the plan's
   ordering (Alpaca first) is a safety choice, not a necessity.
3. **`_execution_id` binds `_git_rev` of agent-multi and gym-fx**; a
   dirty working tree is not reflected in the hash. I flag this as a
   residual gap rather than claim clean-tree enforcement.
4. **Owner rulings adopted verbatim** (per your advice): p95 budget
   only in the amended end-to-end form with ≥20 updates and zero misses
   and zero unreconciled handovers — implemented in the summary's
   deadline guard; RT2 bounds NOT ratified and therefore NOT written
   into any executable schema; heartbeat restarts conditional and
   unexecuted.
