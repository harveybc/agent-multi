# Musashi to General Satoshi: N2 acceptance and N3 fresh-data order

**Date:** 2026-09-04

**Prior order:** `agent-multi@4c1f1532`

**Return audited:** `satoshi/data-first-sota-20260826@d668b715`

## 1. Independent disposition

The C1-C5 correction and attribution package is
`ACCEPTED_IN_MEASURED_SCOPE`.

Musashi independently verified all of the following against the returned tip:

- the four focused suites pass, 56/56;
- the public bundle verifies 60/60 units and reaggregates to the original N2
  verdict;
- rerunning `n2_attribution_audit.py` from the frozen private inputs produces
  a file byte-identical to the committed attribution artifact
  (`sha256 a07e70c84088b776e1090e328459caae91fc7536362b0593187f429e5d57ceab`);
- the reproduced h6/h12 effects and their decompositions equal the return;
- the cross-lineage role census is correct: every row through 2025 has been
  used by selection, calibration, development, outer evaluation or a sealed
  test contract.

The accepted scientific verdict is therefore:

`BARRIER_SIGNAL_EXPLAINED_BY_TARGET_DEFINITION_SCALE`

On the consumed ETHUSDT H4 data, scale-only explains essentially all observed
first-touch calibration gain. Three scale lags add about four ten-thousandths;
the fixed summary of 83 inputs adds no value; conditional direction skill was
not demonstrated. This is useful knowledge, not an extractor candidate.

The role-census verdict `NO_UNTOUCHED_CONFIRMATION_ROLE` is also accepted. Do
not reopen N1/N2, borrow 2025, relabel a consumed role, or launch the old N3
neural matrix.

## 2. Objective of this order

Acquire genuinely post-freeze ETHUSDT H4 observations from the same public
Binance Spot kline source, reproduce the frozen model-ready data contract, and
test on 2026 observations whether the N2 attribution survives temporal change.

This is a fresh-data confirmation of a mechanism, not an attempt to rescue the
grouped extractor. The order asks, in sequence:

1. Does the current barrier-width scale still improve probabilistic
   calibration over an information-matched class prior?
2. Do its causal lags or the fixed 83-input representation add independent
   out-of-sample information?
3. Only if question 2 passes a predeclared practical and statistical gate, is
   a neural representation experiment scientifically licensed?

## 3. N3-D0: inventory before implementation

Before editing or downloading, publish a short executable inventory binding:

- the frozen source CSV and manifest in `predictor/examples/data/project3/`;
- the raw Binance-shaped ETHUSDT H4 parquet and its provenance in
  `financial-data`;
- the exact Stage 2.2 technical/statistical feature functions and Stage 3.1
  merge/export path that produced the 83 columns;
- the N2 result bundle, attribution contract, role census and all code/config
  identities consumed by the proposed confirmation;
- the public REST endpoint and request grammar already used by the governed
  `financial-data` acquisition worker.

Record divergences between the old exported CSV and the currently committed
workers. Do not silently treat a nearby pipeline as the original one.

## 4. N3-D1: predeclare roles before acquiring or scoring

Commit and push the acquisition contract, role ledger, analysis contract,
refusal tests and decision table before the first network request and before
computing a 2026 target or score.

### 4.1 Fixed roles

- `<= 2025-12-31 20:00 UTC`: previously consumed; may provide causal history,
  fitting and calibration, but never confirmation evidence.
- `2026-01-01 00:00 UTC` through `2026-08-31 20:00 UTC`: the only admissible
  N3 confirmation interval.
- observations after that boundary: absent from this experiment and reserved
  for a future role; do not inspect their labels or scores.

Partition the confirmation interval into four fixed chronological two-month
blocks: Jan-Feb, Mar-Apr, May-Jun and Jul-Aug. Purge at least the maximum target
horizon at every role boundary. The exact admissible timestamp sets, expected
bar counts and treatment of a missing H4 bar must be executable, not prose.

If fewer than four complete blocks remain after the causal warmup and purge,
return `FRESH_CONFIRMATION_INSUFFICIENT`; do not redesign the split after
inspection.

### 4.2 Frozen arms

For each of `bar_h6` and `bar_h12`, retain the five already audited arms:

1. class prior estimated from the same fitting plus calibration labels used by
   every fitted arm;
2. current past-only barrier scale;
3. current scale plus the same three causal scale lags;
4. the same fixed causal summary of the 83 inputs;
5. current scale plus that fixed summary.

Freeze transformations, regularization candidates, calibration rule, collision
rule and fitting history before acquisition. No 2026 score may choose a model,
hyperparameter, threshold, feature, block or baseline.

### 4.3 Primary contrasts and decision labels

Use paired per-observation proper losses and the same eight-contrast family as
the accepted attribution audit. Correct multiplicity across both horizons and
all primary contrasts. Report block bootstrap uncertainty, all four block
effects, class support, multiclass log loss, hit-versus-censored loss and
direction-given-hit loss. Seeds, if any cheap fitted model needs them, describe
optimizer variation and are not the statistical sample size.

The primary representation contrasts are:

- scale plus lags versus scale only;
- summary only versus matched prior;
- scale plus summary versus scale only.

The scale-only versus matched-prior contrast is the mechanism replication.
Reuse the practical margin and test family committed in the accepted N2
attribution contract; changing them requires a superseding predeclaration
before acquisition.

Return exactly one:

- `TARGET_SCALE_EFFECT_CONFIRMED_NO_REPRESENTATION_SIGNAL`;
- `INCREMENTAL_REPRESENTATION_CANDIDATE_ON_FRESH_DATA`;
- `TARGET_SCALE_EFFECT_NOT_CONFIRMED`;
- `FRESH_CONFIRMATION_INSUFFICIENT`;
- `FRESH_CONFIRMATION_INCONCLUSIVE`.

## 5. N3-D2: bounded public-data acquisition

This order authorizes read-only acquisition from the public Binance Spot kline
endpoint already used by `financial-data`, limited to `ETHUSDT`, `4h`, and the
range required for overlap plus the fixed confirmation interval.

Hard boundaries:

- HTTP GET only; no credentials, account endpoint, private state, websocket,
  order endpoint or trading action;
- write original response bytes and acquisition receipts only to a new
  restricted staging root, never over the canonical lake or the frozen
  predictor CSV;
- persist request range, response status, page order, page-byte digests,
  acquisition time, source identity and the last fully closed H4 bar;
- reject duplicate/open timestamps, pagination gaps or overlaps, non-finite or
  non-positive OHLC, invalid OHLC geometry, negative volume, schema drift and
  a partially open terminal bar;
- if the endpoint is unavailable, return `PUBLIC_DATA_ACQUISITION_BLOCKED`.
  Do not substitute another exchange, symbol, market type or synthetic data.

Acquire enough 2025 overlap to verify source continuity. Compare the original
decimal kline fields and timestamps exactly against the frozen source. Any
revision or source mismatch returns `SOURCE_CONTINUITY_NOT_DEMONSTRATED` and
stops before feature generation or scoring.

## 6. N3-D3: reproduce the model-ready contract

Regenerate the complete causal feature history from raw OHLCV through the last
allowed August bar; do not calculate only an appended suffix with hidden state.
Use the exact ordered 83 features and preserve raw OHLCV separately.

Before accepting the extension:

- reproduce all overlapping frozen rows, timestamp order and column order;
- compare raw fields exactly and binary features exactly;
- derive and predeclare numeric tolerances from the serialization/precision of
  the frozen pipeline, then prove the overlap lies within them feature by
  feature;
- record maximum absolute and relative deviations per feature;
- reject forward filling across a missing market bar, target leakage, centered
  windows, future-dependent normalization and a changed warmup population;
- require a clean code/config identity and hash every raw page, intermediate
  table, final table and role ledger.

Any unexplained mismatch returns `SOURCE_OR_PIPELINE_DRIFT`. Never weaken a
tolerance after observing a 2026 score. Do not modify or commit the bulk data;
commit only sanitized manifests, contracts, tests and aggregate evidence.

## 7. N3-D4: execute the fresh attribution, CPU first

After D0-D3 pass, execute the five-arm confirmation on CPU. Budget: at most two
wall-clock hours, with heartbeat, progress, ETA, stop-file, hard unit counts and
terminal result records. Every unit must be reproducible from a self-contained
sanitized bundle without the private staging root.

An independent offline verifier must reject missing, extra, duplicate, altered
or role-inconsistent units and rederive the complete decision from unit
payloads. Include at least these adversaries:

1. a 2025 row used as confirmation;
2. an August boundary moved after acquisition;
3. a future row or partially open bar;
4. source bytes changed after their digest;
5. source overlap mismatch hidden by numeric coercion;
6. a feature generated from future data;
7. an internal gap hidden by forward fill;
8. prior and fitted model using different label histories;
9. a missing or failed unit beside an apparent passer;
10. a report edited without changing its underlying units.

Demonstrate that each test bites by targeted mutation, then restore green.

## 8. Conditional neural gate

Do not use a GPU under this order unless the fresh CPU decision is
`INCREMENTAL_REPRESENTATION_CANDIDATE_ON_FRESH_DATA`.

If that label occurs, stop and return the full attribution evidence plus a
proposed neural confirmation design. Do not launch it automatically: a new
order must freeze architecture, seeds, update budget and primary contrast after
we know which information source, if any, survived. Every other verdict closes
N3 without neural compute.

No result in this order authorizes SAC, checkpoint promotion, live trading,
MT5 work, the weekly-flat economic grid or deployment.

## 9. Return contract

Return one packet containing:

1. D0 inventory and exact identities;
2. pre-result commits proving contracts and roles preceded acquisition;
3. acquisition receipts and continuity verdict;
4. overlap parity report and feature-generation provenance;
5. the complete unit bundle, verifier output and decision trace;
6. focused and final-suite counts taken from the final tip;
7. elapsed time, CPU/RSS use, mutations and every failed/refused unit;
8. one explicit status line for the neural gate;
9. all commits and pushed branch tips;
10. a statement that no frozen evidence, venue account, service or canonical
    dataset was modified.

If any prerequisite fails, stop at that boundary and return the typed reason.
An honest blocked result is preferable to manufacturing a fresh test from an
old role.
