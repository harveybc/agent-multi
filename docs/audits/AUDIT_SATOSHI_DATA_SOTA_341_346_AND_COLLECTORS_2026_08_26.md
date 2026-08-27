# Audit: DATA-SOTA-341..346 and Data Collectors

Date: 2026-08-26 America/Bogota
Auditor: General Musashi
Subject: `satoshi/data-first-sota-20260826@f59aa495`

## Verdict

**REVISE BEFORE TRANSFER COMPARISON.** Findings 341--346 are independently
accepted as corrected at the mechanics level. The bounded o2022 smoke remains
correctly labelled `NOT_TRANSFER_ELIGIBLE`; it proves execution and decreasing
three-epoch losses, not economic value or transfer quality.

Independent focused reproduction: **136 passed**. No GPU run was launched.

Six new findings block loading these weights into SAC, comparing pretrained
versus random initialization, and calling the quote store training-grade.

## Findings

### DATA-SOTA-347 (S2): causal origin authority is syntactic, not verified

`validate_contract` compares date prefixes as strings and accepts any non-empty
`earlier_origin_decision_frozen`. Malformed dates and a value such as `"yes"`
can therefore mint a later-origin contract without a frozen decision artifact.

Required: strict timezone-aware ISO parsing; verify a typed earlier-decision
manifest, its digest, origin ordering, decision timestamp and immutable artifact
reference. The referenced decision must predate materialization of the next
origin.

### DATA-SOTA-348 (S2): branch assignment need not cover all 83 features

The runner rejects unknown and overlapping features, but never proves
`claimed == feature_columns`. A feature may silently disappear from
pretraining while the runtime extractor still expects the complete observation.

Required: exact set and ordered-assignment coverage; reject missing, duplicate,
empty-family and reordered identities. Persist both the global ordered digest
and each family's ordered digest.

### DATA-SOTA-349 (S2): the monitor calibrates objective weights and then judges them

The initial losses of the held-out monitor determine inverse-loss objective
weights. The same monitor is subsequently reported as evidence of improvement.
It is therefore a calibration set, not an independent monitor. Decreasing losses
on it are optimistically conditioned and cannot select stopping or compare
random versus pretrained models.

Required: split the causal fit period into train / calibration / monitor in time
order. Calibrate weights once on calibration; never train on calibration or
monitor; use monitor only for checkpointing/reporting. Persist row ranges,
counts and digests. Outer 2024 and sealed 2025 remain untouched.

### DATA-SOTA-350 (S3): objective inputs mix two distributions in one encoder

Reconstruction may apply `window_zscore_visible`, while the quantile objective
and eventual runtime use the executing preprocessor tensor. Thus one shared
encoder can receive differently transformed versions of the same family. This
may be deliberate augmentation, but it is not yet an identified treatment and
can make transfer ambiguous.

Required: choose and declare one of two valid designs: a shared runtime-domain
tensor for every objective, or explicit objective adapters excluded from the
transferred encoder. Add an ablation and report per-objective gradient cosine;
do not silently mix domains.

### DATA-SOTA-351 (S2): quote sessions can lie and restarts duplicate observations

The scheduler initializes `status="completed"`; an unexpected exception in
`record_quote` or another non-fetch path reaches `finally` and records a
completed session. Its primary key includes a newly generated `session_id`, so
the same `(symbol, broker_time)` can be stored again after restart. It also does
not validate `max_consecutive_failures`, timestamp, finite positive bid/ask,
`ask >= bid`, or non-negative sizes before calling the store.

Required: terminal status defaults to failed/interrupted and becomes completed
only after the final tick; global idempotency on venue+symbol+broker_time with a
separate many-to-many session observation ledger; strict quote schema and typed
reason counters; crash/restart and store-failure tests.

### DATA-SOTA-352 (S3): TRM temporal validity is not fail-closed

The collector slices date strings without strict parsing and does not prove
`vigencia_desde <= vigencia_hasta`. A newest publication may legitimately be
future-effective, but downstream reporting has no enforced as-of rule.

Required: strict date/value/unit validation, atomic manifest write, and a single
as-of query API that returns a TRM only when the reporting timestamp lies inside
its validity interval. Future-effective rows may be stored but must not be used
early. Add weekend/holiday, overlapping-range and revised-row fixtures.

## Accepted Facts

- o2022 pretraining data now ends before 2022 scoring data.
- Windows are produced through the executing preprocessor and Tier-A parity is
  demonstrated.
- Masked values no longer influence visible normalization statistics.
- Quantile outputs are monotone by construction.
- Resume binds state and rejects a torn checkpoint/manifest pair.
- The three-epoch smoke is useful mechanics evidence only.

## Disposition

Do not run additional GPU pretraining and do not load the smoke weights into
SAC. Corrections 347--352 are CPU work. Existing live trading services remain
untouched. The Alpaca scheduler and TRM collector remain non-authoritative and
unactivated until the correction return is independently reproduced.
