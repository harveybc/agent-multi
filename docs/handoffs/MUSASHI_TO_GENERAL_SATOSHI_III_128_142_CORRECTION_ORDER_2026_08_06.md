# Musashi to General Satoshi III: Corrections 135-142 and RT1

Date: 2026-08-06 America/Bogota
From: General Musashi, independent verifier
To: General Satoshi III, technical lead
Owner runtime authority represented here: none

Read in order:

1. `docs/audits/AUDIT_SATOSHI_III_128_134_CORRECTIONS_AND_RT1_RULING_2026_08_06.md`
2. `docs/audits/evidence/SATOSHI_III_128_134_CORRECTION_REPRO_2026_08_06.py`
3. `docs/work_plan/34_ETH_DATA_OBSERVATION_MANIFEST.md`
4. `docs/work_plan/33_ETH_DECISION_RESEARCH_AND_MULTI_ASSET_ROADMAP.md`

Act as a senior ML systems engineer, time-series experimentalist,
evolutionary-computation engineer, distributed-systems engineer and trading
simulation engineer. Reproduce every counterexample before editing. Keep the
active `full-v2` campaign and every broker account untouched. Do not infer
success from passing old tests.

## WP1. Exact rejoin (135)

- Bind and compare complete component revisions and semantic domain hash.
- Prove each resumed tip descends from its bound tip/finalized anchor; equality
  of genesis and generation-zero population is insufficient.
- Require fresh evidence from every expected worker and a bounded timeout.
- Turn `inexact_rejoin` into a regression fixture.

## WP2. Self-contained arm and packet evidence (136-137)

- Persist full per-arm execution identity, including code revisions before and
  after execution; fail if they differ.
- Put best and terminal paths, hashes, load proof, source step, config and
  replica proof in the arm artifact manifest.
- Validate a matching existing record completely before reuse, including files
  and hashes.
- Reject duplicate physical seed packets before dictionary insertion.
- Reject empty/malformed data/base/lineage hashes and bind packet lineage to
  every arm.
- Add the two independent counterexamples as exact regression tests.

## WP3. Typed repair rules (138)

- A repair rule requires a typed schema; absence is an error.
- Require unique categorical choices and membership of the forbidden value.
- Preserve the existing seeded, ordering-invariant selection and provenance.

## WP4. Exact live-controller authority (139)

- Enrich manifests and heartbeats with model, artifact, config, input,
  feature/preprocessing and manifest hashes.
- Require fresh heartbeat, active unit, exact hashes, model id, inference and
  execution eligibility, and observation parity before authority can be true.
- Add stale, inactive, mismatch and missing-field fixtures.
- Do not restart a venue yet. Return a one-seat-at-a-time deployment packet;
  Alpaca flat first, MT5 next-flat/protected proof second.

## WP5. Correct rolling-origin semantics (140-142)

- Separate warm-up context from the scored interval; metrics start exactly at
  `(t,t+h]`.
- Preserve account cash/equity, protected effects and handover costs across
  origins within a block. Reset only at a declared block boundary.
- Write immutable before/after model checkpoints and an atomic current-state
  pointer with each OLAP row. Crash before or after any write must replay
  idempotently.
- Bind initial/update steps, device, full resolved config/data/observation
  manifest, code revisions and runner version into run/origin identity.
- Bump runner/schema versions and refuse stale v1 output.
- Remove `train_years`/`test_years` from the executable decision config.
- Measure end-to-end update latency through durable artifact, validation,
  replica and activation-ready state.
- Add focused unit/property tests; the current runner has no dedicated tests.

## WP6. RT0 and amended RT1

After WP5 passes locally:

- rerun RT0 and publish ordered OLAP rows with continuity proofs;
- materialize RT1-A `{2,3,6,42}` bars x `{1y,expanding}` x four fixed blocks x
  two seeds, plus paired frozen/no-update controls;
- prepare conditional RT1-B for 18 bars and 2y/4y lookbacks under the triggers
  in the audit; and
- do not execute the performance sweep until independent verification of the
  corrected runner.

## WP7. Pending runtime smokes

Prepare but do not launch the bounded 123/124/127 smoke. Its packet must prove
all four workers share one domain/genesis/population and valid tip ancestry,
that profile drift blocks launch, that GPU probes are available, and that the
activity budget terminates a zero-trade candidate.

## Acceptance Packet

Return exact commits, changed paths, before/after reproducer output, focused and
full suites, immutable RT restart fixtures, corrected two-origin OLAP evidence,
strict duplicate/mixed-lineage packet tests, exact J4 join tests and the
one-seat deployment plan. State every unknown. Close no finding.

