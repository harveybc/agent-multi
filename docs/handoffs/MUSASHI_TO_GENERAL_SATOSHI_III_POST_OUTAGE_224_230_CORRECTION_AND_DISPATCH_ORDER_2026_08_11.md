# Musashi to General Satoshi III: Post-Outage 224-230 Correction and Dispatch Order

Date: 2026-08-11 America/Bogota  
Authority: owner-approved document 38 and standing anti-idle directive  
Audit: `docs/audits/AUDIT_SATOSHI_III_POST_OUTAGE_WP2_WP5_2026_08_11.md`  
Required posture: correct, preflight and dispatch without requesting another
owner phrase

## 1. Role and Immediate Objective

Act as a senior machine-learning researcher, reinforcement-learning engineer,
distributed-systems engineer, Linux/NVIDIA operator and evidence-custody
implementer. Do not reread the whole repository. Work from the exact files and
acceptance checks below. The immediate objective is a scientifically valid,
running 16-cell P1 difficulty x P1 learning-rate mechanics screen on all four
GPUs, followed automatically by the decision path only if the screen is viable.

Preserve all old outputs. Never relabel a legacy-split result as nested evidence.

## 2. WP0: Reproduce Before Editing

Run unchanged:

```text
docs/audits/evidence/repro_runs/MUSASHI_POST_OUTAGE_224_230_REPRO_2026_08_11.py
```

It must initially report both:

```text
nested_split_contract_missing=true
screen_declared_eligible_without_replica_proof=true
```

After correction, the same reproducer or its append-only v2 must report both
false. Do not edit auditor evidence to manufacture that result.

## 3. WP1: Bind the Executing Pipeline to the Exact Nested Roles (224)

Files:

- `tools/p1_difficulty_lr_factorial.py`
- `tools/m0_l1_boundary_action_replay.py`
- `examples/config/phase_3_eth_sac_dynamics/p1_difficulty_lr_factorial_v1.json`
- `examples/config/phase_3_eth_sac_dynamics/splits/eth_nested_split_contract_v1.json`
- their focused tests

Required executable facts:

1. Set `nested_split_contract` to the exact contract path and
   `nested_split_mode=l1`; record and verify its SHA before model construction.
2. Set `selection_metric=paired_generalization_weekly_v1`. The legacy
   lexicographic branch must be unreachable.
3. Use `fit_train` for fitting (11,509 scored rows), `train_monitor` for the
   in-sample member (2,190), `inner_validation` for selection (2,190) and
   `outer_validation` only for final truth (2,196).
4. Preserve 256 context rows where declared, force hold during context and
   exclude context from replay, metrics, trades and account state.
5. Keep `sealed_test` 2025 in state `SEALED`; no CSV path, hash, row load or
   model evaluation may be emitted for it.
6. Every record binds the nested contract SHA, manifest SHA, per-role CSV SHA,
   scored/context counts and exact score dates.
7. Generate a new experiment identity from the corrected executable facts.
   No old record is reused.
8. Correct WP2 replay to use these exact monitor/inner roles. Rerun D2 and
   anchor on at least two distinct physical GPU models and compare observation
   and action-vector hashes. The earlier 42/2,196 replay remains preserved as
   legacy-role diagnostic evidence only.

Mandatory tests: wrong role path, wrong role count, wrong role SHA, outer used
as inner, missing context flag, context counted in score, paired metric drift
and any sealed-test materialization all refuse before training.

## 4. WP2: Make Replica Custody a Real Screen Gate (225)

Implement a P1-specific collector, reusing the corrected deterministic custody
primitives in `tools/m0_l1_ladder_collect.py`; do not duplicate ad hoc rsync or
hash logic.

Acceptance requires exactly 16 source records and 16 replica load proofs bound
to:

```text
experiment_identity, contract_sha256, seed, cell,
terminal_relative_path, terminal_model_sha256, loads=true
```

The screen verdict accepts a typed replica-proof file and returns refusal when
it is absent. `replica_terminal_loads` must be boolean, never explanatory text.
Revalidate per-checkpoint handoff facts and nested split identity at aggregation
time. Test zero, 15, 17, duplicate, swapped, foreign, hash-altered and
`loads=false` proofs. Stage and replicate atomically; do not mutate source
records or terminals.

## 5. WP3: Complete the Conditional Decision Runner (226)

Extend the P1 runner with explicit `screen` and `decision` modes and distinct
content-addressed identities/output roots. The decision run starts each cell
from its original per-seed anchor, never the screen terminal.

Decision contract:

- same 2x2 factors and seeds;
- phase-2 LR fixed at `3e-5`;
- combined per-cell ceiling of 2,000 pass-equivalent checkpoints;
- patience 60, with no stopping conclusion before checkpoint 40;
- paired train-monitor/inner-validation stopping;
- immutable best-checkpoint restoration;
- one final outer-validation evaluation after selection;
- sealed 2025 inaccessible;
- per-seed paired main effects for difficulty and P1 LR plus interaction;
- raw weekly return vector, mean weekly return, annualized compounded return,
  weekly RAP, annualized RAP, maximum drawdown, trades and activity, all with
  units and horizons.

The screen may return only `PHASE1_LR_REGION_COLLAPSED`,
`SCREEN_VIABLE_REGION` or a typed refusal. The decision aggregator emits the
document-38 outcomes. Implementing this path does not authorize its launch
unless the corrected screen and 16-terminal replica gate pass.

## 6. WP4: Correct IBKR Resume Availability and Status (227-228)

Repository: `lts`.

Files:

- `tools/ibkr_resume_after_reconciliation.py`
- `tools/mint_resume_capability.py`
- `app/ibkr_l1_resume.py`
- related unit tests and owner documentation
- `agent-multi/tools/multifront_status.py` for the queue-state correction

Requirements:

1. Treat the passphrase-protected owner signature as the human authority. TTY
   plus a public phrase is only an ergonomic confirmation, not authentication.
2. Prefer an explicit `--capability PATH` selected by the owner. Verify it is
   inside the protected store, signed, current, profile-bound and unconsumed.
3. Unsigned, malformed and expired side files cannot deny use of one valid
   signed capability. Two valid signed current capabilities still refuse.
4. Never auto-delete evidence. A separate explicit archival operation may move
   typed expired/spent files.
5. All tests use temporary stores; an agent/test must never write the live
   owner capability directory.
6. Status authority order: fresh durable `halt` plus fresh direct broker facts,
   then latest decision as historical context. With `halt=none`, zero direct
   positions and zero orders, report `operational_waiting_next_decision`; do
   not ask the owner to clear an already-cleared hold.

Do not touch the already consumed capability or clear any live state while
testing.

## 7. WP5: Deploy and Start the Corrected Screen (229)

No new owner phrase is required. After WP1-WP2 deterministic preflight passes:

1. Push one clean exact commit.
2. Deploy that exact commit to Omega, Dragon and Gamma; print and persist the
   same commit, contract SHA, experiment identity, data SHA, nested manifest
   SHA and plugin commits on all four workers.
3. Install/enable the GPU readiness timer and the P1LR systemd template.
4. Prove expected UUID, driver `580.173.02`, CUDA framework availability,
   temperature below 78 C, anchor hash and disk budget before each launch.
5. Start 101 on Omega, 202 on Dragon, and 303/404 on Gamma concurrently.
6. All four cells of each seed remain sequential in the cyclic Latin order.
7. Add P1LR to `multifront_status.py`: current seed/cell/checkpoint, terminal
   records/16, current-cell ETA, total critical-path ETA, GPU utilization and
   temperature. The completed old L1 factorial is history only.
8. An idle assigned GPU for more than 15 minutes while a P1LR cell is pending
   emits one deduplicated incident and triggers bounded service recovery.

If a deterministic preflight fails, correct it immediately; do not wait for an
audit ceremony. If a running cell fails identity, sealed-data or GPU binding,
stop only the affected new screen services, preserve records and report exact
facts.

## 8. WP6: Finish the Historical Replica (230)

Resume the interrupted Gamma-to-Dragon transfer with bandwidth limiting,
partial-file preservation and no deletion. Persist progress bytes/total/ETA.
After transfer, compare source and destination inventories plus content digests
for OLAP databases, chain DBs, manifests, configs, metrics and model artifacts.
Any mismatch remains open. Source deletion still requires explicit owner review.

## 9. WP7: Finish 217 and the Return Packet

Transfer only the bounded README correction to default master, push it, rerun
the remote-default checker and require 21/21 repositories, zero errors, zero
broken links and exit 0. Do not merge unrelated experimental code merely to
make links green.

Return one consolidated packet containing:

- exact commits and clean/pushed state for every touched repository;
- before/after 224-230 reproducers;
- corrected role table and hashes;
- WP2 cross-device replay table;
- screen runtime identity, 4-worker status, records/16 and ETA;
- terminal source/replica paths, hashes and load proofs;
- IBKR capability/status regression evidence;
- historical replica progress/digest status;
- all four front statuses; and
- explicit residual doubts. Close no finding yourself.

