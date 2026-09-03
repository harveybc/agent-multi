# Musashi to General Satoshi: N1 target-identifiability full execution order

**Date:** 2026-09-03

**Accepted return:** `agent-multi@7d652fc2`

**Accepted code tip:** `satoshi/data-first-sota-20260826@99b9475f`

**Predecessor order:** `agent-multi@1649e7c0`

## 1. Imperial disposition

C1-C5 are `ACCEPTED_BY_IMPERIAL_DECREE_FINAL`. Do not reopen, amend, rerun or
submit further corrections for those points. The scientific disposition is
final: screen v2 is negative, all five fusion variants fail to advance, and the
eight paired SAC cells remain blocked.

The N0 mechanics preflight is accepted only as `MECHANICS_ONLY`. It authorizes
no scientific conclusion. Proceed with the next work-plan unit, N1, below.

## 2. Scientific question

Execute `TARGET_REPRESENTATION_IDENTIFIABILITY_AUDIT` to answer one question:

> Is useful out-of-sample signal for realized volatility h6 detectable from
> the available causal inputs before compression, or is predictability beyond
> persistence not demonstrated on the available data?

This is a target-versus-representation diagnosis. It is not a search for a new
extractor and it is not an authorization to revive SAC.

## 3. N1.1: amend the predeclaration before any full-run result

Supersede, without rewriting, the N0 predeclaration and make these points
executable:

1. The statistical unit is the causal score window. Arms are paired treatments
   within that unit; `fold x arm` is not a statistical unit.
2. Use at least four non-overlapping causal score windows ending strictly before
   the consumed 85% monitor boundary, if the available sample supports them.
   Training may expand forward, but every calibration window must precede its
   score window and every score observation must belong to exactly one unit.
3. If four valid score windows cannot be materialized, do not invent power:
   execute the mechanics and return `INCONCLUSIVE_INSUFFICIENT_UNITS`. Do not
   claim Holm-adjusted inference from two folds.
4. State the embargo in the same units as the sampled windows and derive it from
   target horizon, sampling stride and any overlap that can leak labels across
   a boundary. Persist the exact source-index ranges for every role.
5. Separate two baselines: literal trailing-volatility persistence, with no
   fitted coefficient, and a calibrated one-variable autoregression. Do not
   call the fitted ridge predictor "persistence".
6. The direct linear and direct temporal arms must consume exactly the same
   causal features, target, windows and score units. Any target or feature
   scaling is fitted on the fit role only and persisted by digest.
7. The frozen best branch and frozen fusion remain historical context. They
   were measured on consumed screen roles and must not be pooled with N1 score
   windows or treated as paired observations.
8. Predeclare the paired estimator, confidence interval, multiplicity handling,
   materiality margin, missing-unit rule and decision table in executable form.
   With insufficient units, the only legal inferential verdict is
   `INCONCLUSIVE`.

The amended declaration, its digest, unit ledger and all immutable input digests
must exist before the first full-run worker starts.

## 4. N1.2: complete the executing surface

Add the missing bounded supervisor and aggregate path. It must:

- materialize the complete ledger once and reject any identity drift;
- run or resume only ledgered units;
- expose phase, worker PID, logical device, completed/total units, updates,
  elapsed time, throughput and ETA at least once per minute;
- support a documented stop file and terminate/reap workers before marking a
  unit interrupted or timed out;
- recompute the aggregate from terminal unit records rather than trusting a
  declared summary;
- require exact arm pairing per score window and all four temporal seeds per
  completed temporal arm;
- preserve failed, timed-out and missing units in the verdict instead of
  silently dropping them;
- write a typed terminal result and a machine-readable interpretation trace.

Add focused tests showing that wrong pairing, reused score rows, fit-derived
scaling applied from another fold, missing seeds, a forged aggregate and a
post-materialization input change all refuse or produce `INCONCLUSIVE`.

## 5. N1.3: authorized execution

After N1.1-N1.2 and their tests pass, execute the full diagnostic automatically.

- Persistence and linear arms: CPU.
- Temporal arm: one explicitly bound CUDA device, only after the existing
  device, plugin and immutable-input preflights pass.
- Maximum: 5,000 optimizer updates per temporal `(score window, seed)` unit.
- Seeds: the four frozen N0 seeds; do not add or remove seeds after results.
- Campaign wall ceiling: six hours.
- Unit timeout: one hour.
- No automatic retry after a terminal scientific result. Infrastructure failure
  may be resumed only through the same ledgered identity.
- No checkpoint is promotable from this diagnostic.

If no reviewed CUDA slot is free, leave the temporal units pending with a typed
reason and execute the CPU units; do not steal a device or silently fall back to
CPU for the temporal arm.

## 6. Required verdict

Return exactly one primary verdict, derived from the amended decision table:

- `PREDICTABILITY_NOT_DEMONSTRATED` if neither direct arm demonstrates positive
  paired out-of-sample skill and the material margin over literal persistence;
- `REPRESENTATION_BOTTLENECK_DEMONSTRATED` if a direct arm advances while the
  frozen branch evidence remains negative;
- `FUSION_BOTTLENECK_DEMONSTRATED` only if branch evidence advances and frozen
  fusion evidence does not;
- `INCONCLUSIVE` for insufficient units, unresolved infrastructure failures,
  discordant direct arms or any case not licensed by the predeclaration.

Report the calibrated autoregression separately; it is a stronger baseline,
not a replacement for literal persistence. Report every score-window contrast,
seed dispersion, confidence interval and multiplicity-adjusted result. A
negative or inconclusive result is a valid completion.

## 7. Consequence and prohibitions

This order does not authorize SAC, a new extractor, target replacement, horizon
search, long hyperparameter search, checkpoint promotion, live trading, MT5
collector work, service changes or venue commands.

After the verdict:

- `PREDICTABILITY_NOT_DEMONSTRATED` opens design of a target/horizon/data audit;
- `REPRESENTATION_BOTTLENECK_DEMONSTRATED` opens a separately predeclared
  extractor design;
- `FUSION_BOTTLENECK_DEMONSTRATED` opens a separately predeclared fusion study;
- `INCONCLUSIVE` opens only a design repair addressing its named cause.

Do not begin that successor automatically.

## 8. Return contract

Return one packet containing:

1. the superseding predeclaration and a field-by-field map from N0;
2. the causal role ledger with exact source-index ranges and digests;
3. PRE/POST evidence for every new refusal test;
4. literal commands, budgets, device binding and minute-level runtime trace;
5. every terminal unit record and the recomputed aggregate;
6. the primary verdict plus the complete decision trace;
7. focal and final-tip suite counts taken after the final commit;
8. exact commits, pushed branches and clean-tree status;
9. an explicit statement that C1-C5 were not reopened and that no SAC, live,
   service, venue or checkpoint action occurred.
