# Order: DATA-SOTA-347..352 Corrections

Date: 2026-08-26 America/Bogota
From: General Musashi
To: General Satoshi
Priority: P0 data and scientific validity; CPU-only

## Mission

Correct findings 347--352 from the companion audit. Reproduce every
counterexample before editing and preserve PRE/POST evidence. Do not launch a
GPU job, B4, an economic comparison, or a transfer into SAC.

## WP1 -- Causal contract and complete feature partition

Files:

- `agent_plugins/branch_pretraining.py`
- `tools/pretrain_branches.py`
- the v2 pretraining contract and focused tests

Implement strict ISO parsing and a typed `earlier_origin_decision` object with
origin id, decision time, artifact id and SHA-256. Load and verify the referenced
manifest; prove chronological ancestry. Replace the non-empty-string gate.

Require every `feature_columns` entry exactly once across branches. Preserve
global order and per-family order; bind both by digest. Add negatives for one
missing feature, duplicate feature, empty family, reordered family and a valid
83-feature assignment.

## WP2 -- Honest objective calibration

Materialize chronological train/calibration/monitor partitions wholly before
the origin score start. Compute inverse-initial-loss weights on calibration
only, freeze them, and keep monitor untouched by training and calibration.
Persist exact timestamps, row/window counts and source digests for all three.

Declare the objective-domain policy. Preferred first implementation: explicit
objective-only adapters, with the transferred encoder consuming the exact
runtime-preprocessed domain. Prove adapters are excluded from transferred
weights. Add a bounded CPU ablation against one-domain-for-all-objectives and
report losses, gradient norms/cosines and representation scale. It is diagnostic,
not a winner selection.

## WP3 -- Alpaca quote collector integrity

Files in `lts`:

- `tools/alpaca_quote_scheduler.py`
- `app/alpaca_paper_lab.py` migration/schema
- focused scheduler and OLAP tests

Make unexpected exceptions and operator interruption terminally honest. Mark
completed only after all requested ticks. Validate every quote before storage.
Separate canonical quote identity `(venue, symbol, broker_time)` from session
membership so retries are globally idempotent without losing provenance.

Test store-write failure, malformed timestamps, NaN/Inf, crossed/zero quotes,
negative sizes, `max_consecutive_failures <= 0`, duplicate replay in one session,
duplicate replay after restart, and clean continuation after an interrupted run.
Do not activate the scheduler.

## WP4 -- TRM temporal contract

Files:

- `tools/collect_usdcop_trm.py`
- a small library module if needed
- focused tests

Strictly parse validity dates, enforce ordered intervals and finite positive
COP-per-USD values, and write provenance atomically. Add `trm_as_of(timestamp)`;
it must return exactly one applicable observation or a typed unavailable/
ambiguous result, never use a future-effective publication early, and remain
`REPORTING_ONLY` by construction.

## WP5 -- Return packet

Return one packet containing:

1. PRE and POST counterexample outputs for 347--352.
2. Exact commits in `agent-multi` and `lts`, with clean-tree proof.
3. Focused and full suite results, separating pre-existing failures.
4. The exact 83-feature coverage manifest.
5. Train/calibration/monitor boundaries and digests.
6. Quote crash/restart idempotency evidence.
7. TRM future-effective and as-of evidence.
8. A proposed single CPU transfer-loader smoke command, **not launched**.

Continue the remaining three pretraining objectives only after independent
acceptance of this packet. No additional owner phrase is required after that
acceptance; the next authorized step is CPU mechanics, not GPU economics.
