# Musashi to General Satoshi III: J0/J4 and Retraining Correction Order

Date: 2026-08-06 America/Bogota
From: General Musashi, independent verifier
To: General Satoshi III, technical lead
Owner decision represented here: none beyond the already approved roadmap and
the owner's promotion of Satoshi III to General
Runtime authority: none

Read first:

1. `docs/audits/AUDIT_SATOSHI_III_J0_J4_AND_RETRAINING_FREQUENCY_2026_08_06.md`
2. `docs/audits/evidence/SATOSHI_III_J0_J4_RETRAINING_REPRO_2026_08_06.py`
3. `docs/work_plan/33_ETH_DECISION_RESEARCH_AND_MULTI_ASSET_ROADMAP.md`
4. `docs/work_plan/19_EXECUTION_CURRICULUM_AND_ORDER_ROUTING.md`

Act as a senior machine-learning systems engineer, evolutionary-computation
engineer, time-series experimentalist and distributed-systems engineer. Preserve
DOIN's decentralized optimizer/plugin boundary. Do not patch `doin-node` to
hide a trading-domain defect. Do not mutate, pause or relabel the active
`full-v2` chain while implementing this order.

## WP1. Preserve and evaluate real terminal weights (129)

- Save a terminal artifact before the pipeline reloads best weights.
- Return typed best and terminal artifact references with SHA-256, source step,
  resolved config and model load proof.
- Evaluate both under exactly the same normal-realistic validation contract.
- Any missing/unloadable/non-finite artifact fails the arm.
- Remove the false `both weight sets evaluated` note unless facts prove it.
- Add a real-pipeline test, not only a fake result fixture.

## WP2. Content-address arm execution and strict aggregation (130-131)

- Define an arm execution id over arm, seed, shared anchor, data hash, code
  revisions, pinned base contract, resolved config, split contract, epoch and
  timestep budget, metric schema and runner version.
- Reuse only byte-compatible complete records. Mismatch gets a new directory or
  fails explicitly; it never silently reuses stale output.
- Require each seed/arm exactly once; reject duplicate packet identities.
- Validate finite required metrics, best and terminal evaluations, telemetry,
  artifacts, trace/config/data/code hashes and common experiment identity before
  aggregation.
- A failed/missing arm makes `promotion_eligible=false` and exits nonzero.
- Turn every scenario in the Musashi reproducer into a regression test.

## WP3. Prove exact-chain rejoin (128)

- A pause record is resumable only when domain, genesis, population fingerprint,
  expected workers and their pre-pause tips/finalized anchors are complete.
- After launch, require fresh evidence from every expected worker and exact
  equality on the bound identity before `rejoin_proven=true`.
- Missing bound or observed identity is failure, never wildcard equality.
- Stop and return nonzero on timeout or partial rejoin. Do not stop a healthy
  already-bound worker merely because another worker failed; preserve evidence
  for operator recovery.

## WP4. Make repair typed and unbiased (132)

- Validate every repair target against the typed gene schema.
- Require categorical type, valid forbidden value and at least one allowed
  declared choice.
- Replace first-allowed selection with a deterministic seeded uniform draw over
  allowed choices; seed from immutable genome/candidate identity.
- Record original value, allowed set, selected value, rule and seed derivation.
- Add ordering-invariance and distribution sanity tests.

## WP5. Make J4 topology- and manifest-aware (133)

- Declare the evidence host/path per seat: Omega for Alpaca/IBKR and Dragon for
  MT5 unless the deployment registry says otherwise.
- Collect fresh direct facts from that host; unavailable remote evidence is
  unavailable, not inactive.
- Join heartbeat model/config/artifact/input hashes to the selected-model
  manifest and its eligibility decision.
- Set SAC authority true only on exact hash equality and eligible manifest;
  linear/heuristic/unclassified remain explicit controls.
- Include evidence time, host, source path and freshness budget in the output.
- Reproduce the current Dragon MT5 topology in tests.

## WP6. Publish the exact ETH data/observation manifest (134)

- Pin the dataset hash and exact 13,699/2,196/2,190 row partitions.
- State that explicit dates override dormant `train_years=4`, or remove the
  contradictory field from materialized configs.
- Publish 83 features, 32-bar input, 256-bar scaling, 2,724 flattened values,
  warm-up/effective rows and causal fit boundaries.
- Do not describe 32/256 as evidence-selected merely because an earlier run used
  them. Register them for bounded screening or cite immutable comparative
  evidence.

## WP7. Implement RT0 and RT1 without consuming the active swarm

Build a local, restart-safe, OLAP-producing rolling-origin runner:

- bar-aligned cadences: 8, 12, 24, 72 and 168 hours; 4 h feasibility-only;
- one- and two-year lookbacks in RT0;
- fixed SAC/config and one seed on one frozen 28-day block for runtime only;
- p50/p95 update duration, deadline misses, resources, model age, switch and
  account-continuity facts;
- RT1 with four non-overlapping 28-day 2024 blocks, two seeds and 1y/2y/4y or
  expanding lookbacks;
- strict test-then-train: score each next interval before adapting on it;
- no 2025 selection and no profit promotion from the one-month RT0 pilot.

The proposed deadline guard is p95 update time <= two thirds of cadence with
zero unreconciled handovers. Label it `proposed_pending_owner_ratification`.

## WP8. Define the dedicated adaptation-schedule DOIN domain

Start RT0/RT1 after R3 SAC optimization. After all admitted
interface-changing R4/R5/R6 lines and before D5 joint integration, materialize
a local optimizer/plugin whose genes include interval in bars, lookback,
warm/reset/full mode, update budget per new bar, replay retention/recency,
encoder freeze and bounded activation policy. Use successive fidelity and a
fixed compute/deadline budget. Keep fast weight adaptation separate from slow
structural DOIN reoptimization.

Every candidate must emit ordered next-interval and weekly metrics plus runtime,
coverage, account-continuity and content hashes. The final schedule is frozen
before prospective confirmation.

## Acceptance Packet

Return one versioned audit request containing:

1. finding-by-finding reproduction before and after correction;
2. exact commits and changed paths;
3. focused and complete suite outputs;
4. fresh real-pipeline best/terminal artifacts and hashes;
5. stale-reuse, malformed-packet, duplicate-packet and empty-lineage fixtures;
6. topology-aware J4 output from Omega and Dragon;
7. exact data/observation manifest;
8. RT0/RT1 runner, schema, dry run and zero-network evidence;
9. adaptation-domain typed schema and ordering placement; and
10. explicit unknowns and proposed owner decisions.

Neither implementer nor this order closes findings. General Musashi reproduces
the packet independently; the owner retains promotion and runtime authority.
