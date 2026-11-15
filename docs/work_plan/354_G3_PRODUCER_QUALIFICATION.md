# G3 Producer Qualification

2026-11-14. Owner approved the continuation of353: account for the unfinished
data experiments before expensive architecture DEAP. This is the offline G3
lane, not STL or live integration. Keep selected299 (71 inputs,28 groups) and
the original store unchanged.

## Requirements And Tests

| Requirement | Acceptance witness |
|---|---|
| All selected inputs accounted for | Exact ordered recipe coverage; unknown and duplicated columns fail |
| Hidden dependencies included | Activity's14 lagged predictors expand to calendar, DAX and S&P; factor and strength expand to all six crosses |
| Original representation reproduced | Frozen original producers, original bars and transforms; finite masks and values compared per column |
| No future observation dependence | Full TRAIN replay versus truncated prefixes and perturbed suffixes, with fitted producer state held fixed |
| No implicit imputation or altered input count | Original finite-row/window counts reported; missing columns fail |
| No fake longer-history claim | Coverage describes only the actual audited store, not downloaded but unadmitted history |
| No premature augmentation release | An audit receipt never authorizes training; remaining joint producer and reward replay gates explicit |
| Reproducible runs | Input/source digests, exclusive output lock, identity checked on resume and terminal artifacts hashed |

Unit-test design: fail on unknown/duplicated inputs, missing parents, cycles,
invalid transforms, misaligned/reordered timestamps, missing/nonfinite windows,
future rows and changed evidence. Integration: replay original real TRAIN with
the pinned technical, price, candle, SSA, wavelet, Hilbert, Kalman, GMM and event
clock producers; quantify failures without silently copying original columns
over them. A passing prefix test is conditional on frozen TRAIN-fitted state,
not proof that fitting that state was point-in-time online learning.

## Implementation Sequence

1. Audit exact71 coverage and transitive primitive dependencies. Measure
   available TRAIN history, masks and thirteen-slot/stride-two windows.
2. Rebuild price/calendar features from original admitted bars and original
   producers. Preserve the actual consumer fit scope from the store manifest,
   rather than replacing it with an older recipe annotation.
3. Extend the producer adapter to cross-asset, released macro/COT and the six
   recovered activity estimators. Authenticate the original joint source panel.
4. Build a joint TRAIN-only augmentation generator. A univariate OHLC generator
   plus unrelated historical macro/cross-asset columns is not this experiment.
   Resampling untouched episodes is a separate resampling control, not new
   information. Regenerate derived features and engine rewards per episode;
   no concatenated windows or transitions across episode boundaries.
5. Qualify the generator against real TRAIN (joint relationships, calendars,
   publication timing, OHLC, copied windows and fixed-producer causality), then
   run matched real-only and90/10 real/synthetic training. Keep optimizer-step
   budget, initial state, seeds, support, features and evaluation common.
   Real validation remains unchanged. Record actual mixture and coverage.

The existing single-asset residual bootstrap plugin is a reference, not a
drop-in joint producer. Its Gaussian jitter and memorization-avoidance routines
are not silently adopted. No added noise amplitude has been selected here.
No feature or model replacement is authorized by this audit alone.

## Continuation

The original TRAIN qualification is now executed. The adapter reuses the
original producers and six recovered activity estimators, including each
TRAIN row's original out-of-fold estimator, and the original cross-asset
files by their declared hashes. It does not refit these producers or train
an agent. The next object is the joint TRAIN-only generator and its matched
training integration. The adapter currently requires the original timestamp
prefix for activity; new episodes require an explicit cross-fit/initialization
contract rather than silently applying the in-sample full estimator.

STL remains owner-paused.351 remains the completed local frozen-neighborhood
selection result, not completion of all data research. Historical expansion
is a separate candidate, not a hidden alteration to the G3 comparison.

## Executed Evidence

Receipts: [producer](generalization/g3_producer354_receipt.json) and
[candidate history](generalization/g3_history354_receipt.json).

| Check | Result |
|---|---|
| Original TRAIN rows |24,895;2020-2023 |
| Selected columns rebuilt |71/71 |
| Numerical agreement |71/71;65 bit-exact; maximum absolute difference9.947598300641403e-14 |
| Prefix invariance |71/71 bit-exact at prefix lengths1,024 and8,192 |
| Future perturbation |71/71 prefix values bit-exact at both cuts; suffix positive controls changed |
| Finite full-input rows |24,122 |
| Complete13-slot/stride2 windows |24,026; data support count, not executed trading decisions |
| CPU producer runtime |16.65 seconds |
| Cold terminal resume |Succeeded after rechecking source and evidence digests; no recomputation |
| Focused unit regressions |19 passed |
| New synthetic rows / optimizer updates |0 /0 |

Numerical qualification is conditional on the frozen producers and legacy
publication/source contract. It is not blanket proof of economic authenticity,
live readiness or point-in-time fitting. The consumer scope actually used is
2020-01-01 through2023-07-01; the factor retains its original pre-grid fit.
No recipe annotation overrides these actual source identities. Previously
documented COT/publication limitations are not repaired by matching old bytes.

The primitive-history audit finds60,069 candidate grid rows before2020,
59,997 jointly finite across17 primitive/required macro columns, from
2010-05-14 to2019-12-31. It uses exact-time FX joins and the original explicit
macro publication/staleness rules, with no new fill. These are NOT59,997 fully
qualified71-feature windows: calendar admission, historical scope-safe activity,
feature reconstruction and source requalification still have to be done.

The first harness runs exposed a shared `tools` package import conflict and
an integer-COT perturbation assignment incompatible with the installed pandas.
Both were corrected in the adapter; the integer case is a regression test.
Failed run roots were retained. No original store, scaler, checkpoint or
training result was changed to make the checks pass.

## Replay Interface

`tools/audit_g3_producers.py --help` lists the required local pinned source
arguments. `--cross-config`, `--financial-root` and `--fx-root` enable the full
71-column check; omitting them is explicitly a partial price/calendar audit.
The run emits its complete `SPEC.json`, hashes, per-column comparison, prefix
tests, coverage, progress and terminal receipt. Repeating the same arguments
on its output root authenticates and reuses the completed result; changed
code/spec/evidence refuses reuse. `--summary-output` emits a small publishable
receipt without raw market rows or private paths.

`tools/audit_g3_history.py --spec <run>/SPEC.json --output <receipt.json>`
measures the joint pre-TRAIN candidate coverage. Neither command launches GPU
jobs, emits synthetic prices, replaces inputs or confers training release.
