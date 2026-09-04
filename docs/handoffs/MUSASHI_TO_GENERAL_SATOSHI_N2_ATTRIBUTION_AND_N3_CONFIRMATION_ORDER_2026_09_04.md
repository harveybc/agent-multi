# Musashi to General Satoshi: N2 attribution correction and N3 confirmation order

**Date:** 2026-09-04

**N2 order:** `agent-multi@8fce8da0`

**N2 return:** `satoshi/data-first-sota-20260826@c6fecc0e`

## 1. Independent disposition of N1 and N2

R1-R3 are `ACCEPTED_IN_MEASURED_SCOPE`. N1 remains an exploratory negative
for the tested dataset, target, estimators and budget. Its broader claims remain
withdrawn exactly as recorded in the superseding interpretation.

The N2 execution is mechanically accepted:

- the predeclaration and its amendment precede the runner and every result;
- 60/60 units are terminal `COMPLETED` at attempt 1;
- current code, predeclaration and input digests equal the ledger bindings;
- aggregation from the durable run directory reproduces the committed verdict
  trace semantically exactly;
- the 18 R3 tests and 19 N2 tests pass independently (37/37);
- `bar_h6` and `bar_h12` satisfy the rule that N2 actually predeclared.

The literal protocol verdict is therefore preserved as
`TARGET_CANDIDATE_FOUND_UNDER_N2_BASELINE`. It is development-only. It is not
evidence that the 83 input features, a temporal representation or the grouped
extractor add predictive information.

Neural confirmation is `REVISE_BEFORE_EXECUTION` until the attribution and
baseline defects below are corrected. Preserve every N2 artifact byte-for-byte;
do not rerun or rewrite N2.

## 2. Findings that govern the next experiment

### F1 - baseline information mismatch

The selected logistic model is refit on `fit + calibration`, while
`fit_class_prior` is estimated from `fit` alone
(`tools/target_horizon_census_n2.py:394-395` versus `:439-443`). The baseline is
therefore denied the recent calibration labels available to the model.

An independent post-hoc diagnostic on the frozen development arrays found that
an information-matched `fit + calibration` prior leaves pooled log-loss skill
of `+0.023177` for h6 and `+0.021533` for h12. The candidates are not erased,
but their published effects (`+0.027008`, `+0.025885`) are overstated relative
to a recency-matched prior. This diagnostic is exploratory and must be
reproduced by the executing package, not copied as authority from this order.

### F2 - target-construction scale explains nearly all observed skill

The first-touch barriers are defined with the current past-only trailing
volatility scale. The winning `target_history` model receives that same current
scale plus three lags. This is causal, but it creates a direct structural path
from target construction to prediction.

Against the information-matched prior, the same independent diagnostic found:

| target | current scale only | current scale + 3 lags | incremental lags |
|---|---:|---:|---:|
| `bar_h6` | +0.022736 | +0.023177 | +0.000441 |
| `bar_h12` | +0.021161 | +0.021533 | +0.000372 |

Thus N2 presently identifies a useful scalar conditioning variable, not a
representation-learning target. The 249-value summary of the 83 inputs was not
selected in any winning unit.

### F3 - the multiclass score conflates reachability and direction

Exploratory decomposition on the same development rows shows that most of the
gain is in `hit versus censored`. Conditional `upper versus lower` skill is
approximately zero across the four windows, and the h12 argmax classifier has
near-zero upper-class recall in most windows. A lower log loss remains a valid
probabilistic improvement, but it must not be described as directional
predictability or trading value.

Same-bar collisions are not the main explanation (observed below 3% per score
window), but their adverse-first convention must be included in sensitivity
reporting because H4 OHLC cannot reveal intrabar order.

### F4 - overall `INCONCLUSIVE` semantics contradict the sealed rule

The predeclaration says that **any** unlicensed candidate makes the census
`INCONCLUSIVE`. `aggregate_final` instead returns a clean negative when some,
but not all, candidates are unlicensed, and can return a winner while another
candidate is unlicensed. The committed test currently blesses that mismatch.
This did not alter the real N2 verdict because all nine real candidates were
licensed, but the reusable judge is wrong.

### F5 - the published evidence cannot independently rederive the trace

The repository contains the aggregate trace and a 60-entry index of result
digests, but not the 60 result payloads carrying the per-observation losses.
Independent reaggregation currently depends on the operator's surviving local
run directory. A published digest without the object it authenticates is not a
reproduction package.

### F6 - `effective_blocks` is a count, not an effective sample size

`n_score // block_length` reports 36 available non-overlapping blocks. It does
not estimate statistical effective sample size. Rename or qualify it so that no
scientific report claims more than was measured.

## 3. C1-C5: correction package, CPU only

### C1 - preserve and supersede the interpretation

Publish a superseding N2 interpretation with this exact scope:

> On the consumed ETH H4 development rows, first-touch barrier outcomes at six
> and twelve bars were better calibrated by past-only barrier-scale volatility
> than by a static class prior. The observed gain was modest and was explained
> almost entirely by the current scalar used to set barrier width. N2 did not
> demonstrate directional skill, incremental information in the 83-feature
> representation, neural-model value or trading profitability.

Keep the literal N2 verdict and every original artifact intact. Link the
superseding interpretation; do not edit history.

### C2 - repair the reusable verdict judge

- Implement the sealed semantics: any missing, failed, malformed or unlicensed
  candidate makes the overall verdict `INCONCLUSIVE`, irrespective of passers.
- Add adversarial tests for one unlicensed candidate among eight failures and
  one unlicensed candidate beside an apparent passer.
- Preserve the real N2 result by rederivation and show explicitly why this
  correction does not alter it.

### C3 - run `N2_ATTRIBUTION_AUDIT`, development-only

Use only the already consumed N2 development rows. Commit a standalone analysis
contract before writing its result artifact, while declaring that Musashi's
post-hoc values above were already observed and therefore this is an audit, not
untouched confirmation.

For each of `bar_h6` and `bar_h12`, and every score window, compare with equal
information sets:

1. class prior estimated on `fit + calibration`;
2. current barrier scale only;
3. current barrier scale plus three causal lags;
4. fixed 83-input causal summary;
5. scale plus the fixed summary.

Separate the three-class loss into two proper probabilistic questions:

- `hit versus censored`;
- `upper versus lower`, conditional on a hit.

Report multiclass log loss, binary log loss, Brier components, per-class recall,
class support, calibration tables and paired per-observation loss differences.
Do not use hard accuracy as a primary metric. Include sensitivity to removing
same-bar collisions and to block lengths 3, 6 and 12; label all sensitivity
results exploratory. No target, feature or score role may be added.

Return one of:

- `BARRIER_SIGNAL_EXPLAINED_BY_TARGET_DEFINITION_SCALE`;
- `INCREMENTAL_DEVELOPMENT_STRUCTURE_OBSERVED`;
- `ATTRIBUTION_INCONCLUSIVE`.

These labels describe development evidence only and cannot promote a model.

### C4 - publish a self-contained result bundle and verifier

Create a sanitized canonical bundle containing, for all 60 original N2 units:
identity, terminal state, result payload and result digest. Include no absolute
paths or topology. Add an offline verifier that:

- rejects missing, extra, duplicate or altered units;
- recomputes every unit id and result digest;
- reaggregates the verdict from the bundle without the private run directory;
- requires semantic equality with the committed N2 trace.

Freeze adversarial tests for each refusal. The original local run directory is
evidence input, never the only durable authority.

### C5 - correct terminology and provenance

- Replace `effective_blocks` in new prose with `available_blocks`; retain the
  original field only as a legacy name with an explicit definition.
- Publish exact software/data/config identities and the role-use map.
- Report the minimum Monte Carlo p-value as `1/2001` (or `p <= 1/2001`), not as
  more numerical precision than 2,000 bootstrap repetitions provide.

## 4. N3: untouched incremental-representation confirmation

N3 may execute automatically only after C1-C5 pass and a role census proves an
untouched chronological confirmation region. If no such region exists, stop as
`NO_UNTOUCHED_CONFIRMATION_ROLE`; never borrow sealed test data or rename a
consumed role.

Commit and push the N3 predeclaration, driver and refusal tests before computing
one confirmation score. The predeclaration must bind all data, roles, code,
models, seeds, budgets, losses and decision thresholds.

### 4.1 Question and targets

Primary target: `bar_h6`, selected first by the sealed N2 stability rule.
Secondary target: `bar_h12`. Correct multiplicity across the two horizons and
all confirmatory neural contrasts.

The primary question is:

> Do the 83 causal inputs contain predictive information about first-touch
> outcomes beyond the past-only scalar history used to construct the barriers?

### 4.2 Required arms

Use identical chronological roles and information timing:

1. recency-matched `fit + calibration` class prior;
2. current barrier scale;
3. current barrier scale plus three lags (strong baseline);
4. fixed direct GRU on the 83-input sequence;
5. accepted grouped extractor on the 83-input sequence;
6. each neural representation concatenated with the four scale values.

The primary contrast for each neural architecture is
`representation + scale` versus `scale lags`, using paired per-observation log
loss. Representation-only arms diagnose redundancy; they are not allowed to
replace the primary contrast post hoc.

### 4.3 Training and inference

- Five fixed seeds per neural arm; seed is optimization variability, never the
  statistical sample size.
- The confirmatory predictor is the predeclared mean-probability ensemble over
  the five seeds. Publish seed dispersion separately.
- Hyperparameters and probability calibration use calibration only; no score
  feedback, early stopping or model choice may consume confirmation outcomes.
- Cap every neural unit at 5,000 real optimizer updates and the whole campaign
  at six wall-clock hours. Use the accepted intra-segment budget guard,
  heartbeat, ETA, stop-file, thermal limits and terminate-and-reap semantics.
- One CUDA device per neural unit, no silent CPU fallback. Cheap baselines stay
  on CPU.

### 4.4 Decision

Return exactly one:

- `INCREMENTAL_REPRESENTATION_SIGNAL_CONFIRMED`: the primary h6 neural contrast
  is positive in every predeclared score block, clears its predeclared practical
  margin and multiplicity-adjusted dependence-aware test;
- `TARGET_SCALE_SUFFICIENT_NO_EXTRACTOR_ADVANCE`: scale history survives while
  no neural representation adds the required incremental value;
- `CONFIRMATION_INCONCLUSIVE`: any role, license, unit, identity, precision or
  execution requirement fails.

No N3 result authorizes SAC, checkpoint promotion or live use. A positive N3
only authorizes a separately reviewed representation-training plan.

## 5. Return contract

Return one packet containing:

1. C1 superseding interpretation and C2 PRE/POST counterexamples;
2. the complete N2 attribution audit, including decomposition and sensitivities;
3. the self-contained N2 bundle plus independent-verifier output;
4. the untouched-role census and N3 predeclaration commit made before scores;
5. every terminal N3 unit, budget/thermal telemetry and exact identities;
6. the single N3 verdict with no claim beyond its wording;
7. literal final-tip test counts, branch, commits and clean-tree status;
8. explicit confirmation that no SAC, live, MT5, service, venue, promotion or
   sealed-test action occurred.

## 6. Boundaries

This order authorizes CPU corrections, the development-only attribution audit,
and the bounded N3 confirmation if and only if an untouched role is proved.
It does not authorize SAC, trading, MT5 collector work, service changes, venue
commands, checkpoint promotion or access to a sealed test role.
