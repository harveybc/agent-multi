# Musashi to General Satoshi: N1 disposition and N2 target census order

**Date:** 2026-09-03

**N1 order:** `agent-multi@89d099aa`

**N1 return:** `satoshi/data-first-sota-20260826@11c391f9`

## 1. Disposition

C1-C5 remain `ACCEPTED_BY_IMPERIAL_DECREE_FINAL` and are outside this order.
Do not reopen them.

N1 execution is accepted in its measured scope:

- 28/28 ledgered units completed at attempt 1;
- the committed per-window scores reproduce the published paired differences
  and confidence intervals;
- neither tested direct arm satisfies the predeclared advance rule;
- the typed result `PREDICTABILITY_NOT_DEMONSTRATED` is accepted as a negative
  result for this dataset, horizon, model set and budget;
- SAC remains blocked and no N1 checkpoint is promotable.

The global prose interpretation is `REVISE`. N1 did not prove that the target
contains no signal, that no extractor could preserve signal, or that the target
alone caused every earlier negative. It showed that the tested ridge and GRU
did not demonstrate transferable predictive skill under the N1 protocol.

No N1 GPU unit is to be rerun by default.

## 2. R1: publish an honest superseding interpretation

Preserve every N1 artifact byte-for-byte and publish a superseding
interpretation with this exact scientific scope:

> On the tested causal ETH H4 development data, for realized volatility h6,
> neither the direct ridge nor the fixed GRU demonstrated positive
> out-of-sample skill under the predeclared N1 budget. A calibrated
> one-variable autoregression improved consistently over literal persistence,
> so temporal dependence exists, but N1 does not identify whether the remaining
> limitation is the target, the inputs, the estimator or the training recipe.

Correct these two reporting hazards:

1. `+0.372` and its interval are the mean **R2 difference** between calibrated
   AR1 and literal persistence. They are not an autoregressive coefficient.
   N1 did not persist the fitted coefficient. Do not name one.
2. Replace every assertion equivalent to "signal does not exist" or "no
   extractor can conserve it" with the narrower "predictability was not
   demonstrated by the tested arms".

## 3. R2: disclose why N1 is exploratory, not confirmation

The N0 mechanics preflight exposed score results before the N1 declaration:
direct linear `R2=-1.6831` and direct temporal `R2=-0.0030`. Its N0 score region
overlaps rows later used by N1. Freeze a role-use map proving the overlap and
classify N1 as:

`EXPLORATORY_NEGATIVE_AFTER_MECHANICS_PREFLIGHT`

This does not erase the result. It prevents the t-test and thresholds from
being presented as untouched confirmatory inference. The intact confirmation
roles remain untouched.

Also state that four neighboring chronological score windows with expanding
training histories are not demonstrated independent replications. The
per-window losses and contrasts are authoritative observations; their t
interval is descriptive under an independence assumption, not proof of signal
absence.

## 4. R3: repair the reusable statistics helper without rerunning N1

Before any successor consumes it:

1. implement Holm step-down adjusted p-values with the required cumulative
   maximum in sorted order;
2. handle zero-variance paired differences explicitly: positive constant,
   zero constant and negative constant must produce finite, predeclared
   outcomes;
3. reject NaN and infinite scores, differences, statistics and adjusted
   p-values;
4. add tests that fail under the current `inf -> linspace -> NaN` path and
   under the non-monotone Holm implementation;
5. re-derive the existing N1 interpretation from the frozen 28 result records
   and prove that its primary verdict is unchanged.

This is a correction to reusable analysis code and interpretation only. It does
not authorize retraining.

## 5. N2: development-only target and horizon census

Open `TARGET_HORIZON_DATA_CENSUS_N2`. Its purpose is to decide what question a
future extractor should be asked, before designing or training that extractor.

### 5.1 Predeclare and seal before computing N2 outcomes

Commit and push a standalone predeclaration before the first N2 target score is
computed. The candidate set is fixed to targets already present in the accepted
pretraining contract:

- forward log return at horizons 1, 3, 6 and 12;
- realized volatility at horizons 3, 6 and 12;
- first barrier hit at horizons 6 and 12.

No target, horizon, metric or model may be added after seeing N2 results.

N2 is development-only because prior work has consumed the available fit data.
It may select a candidate for a later confirmation order, but it may not make a
confirmatory prediction claim and may not touch sealed or intact confirmation
roles.

### 5.2 Target-specific baselines and metrics

Use the existing executable target builders; do not duplicate label formulas.
For each family predeclare a proper baseline and loss:

- returns: zero-return and fit-role mean baselines; primary squared-error skill,
  with directional accuracy and rank correlation secondary;
- volatility: literal trailing-volatility and calibrated AR1 baselines; primary
  QLIKE or another predeclared proper variance-forecast loss, with R2 secondary;
- barrier hit: fit-role class-prior baseline; primary multiclass log loss, with
  Brier score, class support and macro recall secondary.

Metrics from different target families are not numerically pooled. Convert each
candidate to a within-family skill improvement over its proper baseline.

### 5.3 Models and attribution

The census is deliberately cheap and CPU-only:

1. proper baseline;
2. target-history model using only causal lags of the target-defining series;
3. one regularized linear or multinomial model on a fixed, predeclared causal
   summary of the 83 inputs.

All feature and target transformations are fit-role only. Hyperparameters are
selected on calibration only. Record conditioning, selected regularization and
coefficients or coefficient norms so catastrophic linear failure is diagnosable
rather than merely reported.

Do not train a GRU, transformer, extractor or RL agent in N2. A temporal neural
confirmation belongs to a later order only after a candidate survives this
census.

### 5.4 Causal design and uncertainty

- Use at least four disjoint score windows and publish every exact source-index
  range.
- Derive embargo from each candidate's horizon and sampling stride.
- Prevent any target label from crossing a role boundary.
- Estimate uncertainty from per-observation loss differences with a
  predeclared time-aware block bootstrap or another dependence-aware method;
  do not treat adjacent windows as independent merely because they are named
  separately.
- Publish effective sample size, class support and target variance per window.
- A candidate with degenerate labels, inadequate class support or insufficient
  effective blocks is `INCONCLUSIVE`, not a winner.
- Correct multiplicity across all nine target-horizon candidates before
  selection.

Add negative controls: a future-leak sentinel must appear unrealistically easy
and therefore prove the test can detect leakage, while a causally permuted or
time-shifted target must not pass. These controls diagnose the harness and are
never eligible candidates.

## 6. N2 execution and decision

After the standalone predeclaration is committed and the refusal tests pass,
execute the complete N2 census automatically on CPU with:

- a two-hour campaign wall ceiling;
- minute-level heartbeat, completed/total units, throughput and ETA;
- a stop file with terminate-and-reap semantics;
- immutable input, target-contract, role-ledger and code digests;
- no retry of a terminal scientific result;
- aggregate recomputed from exact terminal unit records.

Return exactly one verdict:

- `TARGET_CANDIDATE_FOUND`: at least one candidate improves its proper baseline
  in every score window, clears its predeclared materiality threshold and the
  dependence-aware multiplicity correction;
- `NO_TARGET_CANDIDATE_DEMONSTRATED`: all complete candidates fail the rule;
- `INCONCLUSIVE`: missing units, inadequate effective support, discordant
  evidence or any unlicensed case.

If multiple candidates pass, select at most two using a predeclared rule based
on corrected evidence and stability, not the largest raw score. Do not begin
their neural confirmation automatically.

## 7. Boundaries

This order authorizes R1-R3 and the CPU-only N2 census. It does not authorize
GPU work, SAC, a new extractor, neural target confirmation, checkpoint
promotion, sealed-role access, live trading, MT5 collector work, service changes
or venue commands.

## 8. Return contract

Return one packet with:

1. the superseding N1 interpretation and the N0/N1 role-use overlap map;
2. PRE/POST and mutation evidence for R3;
3. the separately committed pre-result N2 declaration and its commit identity;
4. exact target formulas, baselines, metrics, role ranges and digests;
5. negative-control outcomes and every terminal unit record;
6. dependence-aware uncertainty, multiplicity correction and selection trace;
7. the single N2 verdict and its permitted next consequence;
8. literal final-tip test counts, commits, pushed branches and clean trees;
9. explicit confirmation that C1-C5 were not reopened and no prohibited action
   occurred.
