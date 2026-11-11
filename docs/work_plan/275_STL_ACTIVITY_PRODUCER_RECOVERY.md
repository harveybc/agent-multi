# STL activity producer recovery

## Scope

This closes the missing historical estimator export for `act.p_hit`. It does
not release the four STL training arms, change selected69, or promote a model.
The main actor-refinement lane remains independent; its latest completed
contrast and next action are recorded in [274](274_OPERATIONAL_CRITIC_ACTOR_TRANSFER.md).

## Requirements and evidence

| Requirement | Check | Result |
| --- | --- | --- |
| Original inputs and feature order | SHA256 of screen, decision, sealed probabilities | Recorded in recovery RESULT.json |
| Original fit definition | TRAIN-labelled finite targets, original HGB settings | 14823 rows, 14 lagged features, 300 iterations |
| No approximate replacement | Exact timestamp and probability equality | 31116 rows, maximum difference 0 |
| Durable inference artifact | Serialize estimator, reload, replay every row | Exact |
| Fail closed | Changed probability, wrong clock, duplicate time, nonfinite output | Rejected in unit tests |
| No production mutation | Separate recovery directory; original files rehashed | No source or training-store changes |

Nine focused tests pass: five recovery checks and four source-support checks.
Command: `python -m pytest tests/unit/test_activity_recovery.py tests/unit/test_stl_source_support.py -q`.

`tools/recover_activity_producer.py` reconstructs the original producer's fit,
not a newly selected classifier. It uses the original sorted `(feature, lag)`
list and original HistGradientBoosting settings: depth3, leaf minimum200,
300 iterations, learning rate0.05, seed0. Other effective parameters are
recorded in the receipt, including the original automatic early stopping.
There is no new scaling, imputation or feature selection.

Fit support is 2020-01-09T22:00Z through 2022-12-30T08:00Z. Matching historical
probabilities does not independently establish causal production availability
of its 14 inputs or certify historical out-of-sample predictions.

## Artifact identities

Private recovery root relative to the operator's home:
`.local/share/agent-multi/stl-recovery-20261111/activity/`.

The recovered directory contains `ACTIVITY.joblib` and `RESULT.json`.
Only load the trusted artifact after checking its hash; joblib is not a safe
format for untrusted input.

- Input screen: `6c95bacda315e43ff6f844227d88cf48063b35342533e0c1516bc285d3a5f511`.
- Feature/lag decision: `2c1eec494d5c7d005676d58b4403a134b49dc21e05681eb17f9261a40f96d043`.
- Sealed probability file: `a8915c0a22755d1e569898a926cf8d2c1163aa3e5c2d13f71be45cb36bbbc333`.
- Recovered weights: `1dc0b5b18a52ce53c0baf737cce98c78a9dd6ef72bd6adbf8e2d3772fb59afe8`.
- Runtime: scikit-learn 1.9.0, CPU.

## Remaining STL dependencies

1. Real USDCHF observation at 2022-05-19T21:00Z bar start. The examined local
   sources lack it; two spaced requests to the historical provider returned
   HTTP429. No observation was obtained, substituted or inferred.
2. Qualify the six November source replacements and reconstruct all selected
   pre-TRAIN feature histories with frozen producer states. Candidate M5/M15
   observations exist for the November hour; presence is not splice approval.
3. Rebuild under a new data identity, then repeat causal/streaming and monthly
   coverage gates before releasing training. Original sealed artifacts stay intact.

The full-precision frozen PCA state was located separately; a diagnostic over
the historical factor outputs found identical finite masks and maximum absolute
error below 1e-13, not bit-exact parity. That is useful recovery evidence, not
permission to silently replace the sealed factor columns.

## Handoff to Retsu

The previously reported missing classifier state now has a recovered artifact.
Use the receipt's ordered feature list and original row-lag semantics; do not
replace row lags with elapsed-hour lags or substitute missing observations.
Authenticate the three inputs and weights above and reproduce the sealed
probabilities before binding this estimator into a live producer. This closes
the estimator-export issue only: current real inputs, publication clocks,
freshness and restart/prefix parity still need their live-consumer checks.
`live_ready` is deliberately false in the receipt.

## Next actions

Recover and qualify historical source coverage without filling data. In the
main lane, attribute the saved bounded actor-transfer changes and qualify the
one-update candidate before further refinement or a changed-weight floating
pass. Neither task requires a new strategy decision from the owner today.
