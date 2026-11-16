# Component Capacity Diagnosis

Owner request2026-11-16: use existing checkpoints to prioritize capacity or data
interventions before commissioning expensive synthetic generation. This is a
CPU-only copied-model diagnostic, not new training, selection or live deployment.
Running362 and all original artifacts remain untouched.

## Predeclared Scope

Reuse360 authenticated initial, epoch1 and last archived weights for all three
seeds. Compare the SAME real TRAIN windows:128 evenly spaced valid rows per year,
2020 through2023. No generated inputs, test observations or changed scalers.
The sample describes sensitivity; it is not an independent sample of512 markets.

Measure each actor and each online critic separately: centered activation
covariance spectra by year and pooled; ranks carrying90/99 percent variance;
participation/stable rank, per-channel standard deviation, weight spectra and
function changes relative to the initial state. Keep the flattened temporal
representation separate from channel-wise spectra. Do not call a low-rank
representation an unused network or use these ranks as an automatic width rule.

On the actor only, approximate weight matrices within ONE component at a time
using50/75 percent of each matrix's maximum rank. Leave biases, all other
components, temporal support and real input values unchanged. Measure signal
flips, KL, logit RMS change and discarded weight energy per year. Full-rank
identity must be byte-exact; restore exact weights and outputs after every test.
These interventions rank sensitivities, NOT trading gains or trained narrower
architectures. Inference stays on the same CPU backend for all comparisons;
it is not claimed to match original GPU native trading accounts.

## Traceability And Unit Design

| Requirement | Required check |
|---|---|
| Spectrum semantics | identity and rank-one matrices; zero covariance; scale/permutation invariance; nonfinite rejected |
| No mutation | full-rank identity; reduced reconstruction leaves source unchanged; exact restoration including outputs |
| Functional check | injected low-rank-sensitive output changes; unchanged outputs zero KL/flips |
| Real support | TRAIN only, identical rows and sample digest across all checkpoints; full source support verified |
| Frozen provenance | recipe/checkpoint digests checked before/after; optimizer steps0; no promotion |
| Honest interpretation | no capacity-in-bits, required-years estimate, generalization claim or pruning from rank alone |

## History And Synthetic Decision

354 already identifies59,997 finite pre2020 rows across17 primitives, starting
2010-05-14, not complete71-input windows. Reuse its per-column dependency graph
to list exactly which historical scopes must be reconstructed: source/calendar,
causal derived features, factor, Kalman/GMM and OOF activity. Do not drop a
feature or substitute synthetic data to claim full coverage. The earliest common
date for all71 is not yet established by that primitive audit.

Predictor phase1/phase4 results reported by the owner are prior evidence, not new
experiments here. No automatic repeat of the shorter-history grid. No VAE-GAN
launch: first inspect measured component sensitivities and the historical scope
inventory. Any later generator must jointly model required primitives/control
variables, preserve inter-series relations, use TRAIN-only fitting and regenerate
the exact71 features; an improvement in TRAIN alone is not transfer evidence.

## Finite Next Step

Complete this nine-checkpoint panel once. Pair its results with362's existing
weight/margin/action logs and before/after-LR traces. Recommend at most ONE
component-width contrast, preserving depths/windows/dilations, if evidence supports
it; otherwise diagnose critic/objective instability before changing architecture.
No repeat of unchanged forensic sweeps and no phase2 closure from these measures.

## Completed Panel

Nine archived points: seeds7 epochs0/1/38,19 epochs0/1/48,43 epochs0/1/24.
These are360 snapshots, NOT the latest running362 epochs. The three epoch0
policies are the same inherited reference, not three independent replications.
27 actor/critic profiles;864 component interventions, including288 full-rank
identity controls. All restorations exact; extracted actor/critic outputs match
their native CPU modules exactly. Nine focused tests pass. No training updates.
Machine-readable summary: [receipt](generalization/capacity363_summary.json).

| Archived point | Actor core participation rank /16 | Median branch rank for99% variance /8 | Attention half-rank flips /512 | Fusion half-rank flips /512 |
|---|---:|---:|---:|---:|
| Reference0 |6.17|7|1|39|
| seed7 epoch38 |6.87|7|4|31|
| seed19 epoch48 |3.33|7.5|106|116|
| seed43 epoch24 |6.95|7|1|23|

The high-LR archived actor becomes more concentrated, not simply fuller: core
participation rank falls from6.17 to3.33, while rank99 remains16. Small-variance
directions are not thereby useless. Attention approximation changes20.7% of
sampled TRAIN signals there, compared with0.2% initially. This is a large
artificial weight approximation, not evidence that a small real optimizer step
caused those flips or that attention should be removed. It prioritizes measuring
representation concentration and downstream sensitivity around LR changes.

The28 branches usually need7/8 activation directions for99% variance. Uniform
8->4 narrowing is therefore not justified as lossless from parameter counts.
The least sensitive branch in this finite panel is hilbert/log_close, but even
it changes up to2/512 signals under half-rank weights. Low rank is not equivalent
to a trained smaller architecture, and neither is evidence of improved profit.

## Complete Feature Inventory

The dependency reconciliation covers all71 columns:17 price,10 wavelet,8 calendar,
7 technical,5 candle,4 raw,4 GMM,3 COT,3 Hilbert,2 rates,2 factor,2 SSA,2 strength,
1 activity and1 Kalman. Eight columns depend on frozen fitted producers:
2 factor,1 activity,1 Kalman,4 GMM. Activity additionally requires the DAX/SPX
release history and calendar predictors. Extending those backwards requires a
declared historical fitting/OOF scope, not just finding earlier FX bars.
No common earlier start for ALL71 is established. No feature was removed.

## Automated Follow-Up

Use exactly three points per362 worker: epoch immediately before first LR drop,
the epoch ending at that drop, and the next epoch (first trained at the new LR).
Scheduler changes AFTER evaluation; the drop epoch itself has not learned at the
new LR. Copy authenticated weights/history and use the same512 TRAIN probes.
The optional explicit epoch manifest prevents silently substituting latest/best.
Read-only CPU supervisor waits for all three, then runs one nine-point panel and
report. No GPU interruption, retraining, parameter change or model promotion.

Supervisor enabled2026-11-16 with pinned runtime6e6c490b. Its first collection
authenticated seed19 epochs93/94/95 (first actual LR drop94). The other workers
had not yet completed their drop panels at this check; it polls every180 seconds
and exits after the single report. Raw tables/plots and the complete71-column
dependency CSV are retained outside git. No request to Retsu to build a generator
has been issued; that expensive lane remains conditional on this diagnosis.
