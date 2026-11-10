# 266. Account loss and critic ordering after selected69 refinements

Owner approved2026-11-10. Audit the nine Bayesian endpoints from264/265
(three critic objectives, three actor seeds), before choosing another training
contrast. No retraining, feature changes or paper-model promotion in this stage.

## Questions

1. Which changed NOP/BUY/SELL signals actually change entries, early exits or
   account continuation? Reconcile ALL chronological contributions to the exact
   endpoint P&L difference, including neutral commands and downstream sizing.
2. Does the frozen calibrated critic favor the changed action? Compare against
   both the old greedy account continuation and stochastic soft continuations
   from the identical original account state. Q was fitted to soft returns;
   comparing it only against raw cash is not an equivalent target.
3. Does the entropy-adjusted actor score favor the change even when raw Q does
   not? How much of the decision-boundary movement comes from last-layer weight
   changes, bias changes and posterior probability averaging?
4. Were the actual account position and the calibration trajectory's position
   different for the same observation? Report this aliasing; do not add state
   inputs as an unannounced solution.

## Frozen design

All18 complete TRAIN/VALIDATION endpoint accounts,69 inputs and original
native engine/calendar/risk. Evaluate all changed signals, not a favorable
sample. Stochastic continuation seeds7/19/43 use common random numbers for
old/new actions; report their range and sign agreement. Three trajectories are
not an exact expected return. This retrospective diagnostic never becomes an
observation, target for validation fitting, or trading override.

Reuse the tested native fork/continuation and chronological telescoping code.
Cache identical same-state interventions across seeds with a bound identity;
do not add their isolated effects as if they were total account contributions.
Unchanged source69, source encoders and learned heads are hash-verified.
Reconstruct the candidate head from saved weights and verify its probabilities
and recorded actions before interpreting it. Use actual frozen functional Q.

## Test-led requirements

| ID | Requirement | Evidence |
|---|---|---|
| DA1 | Exact native account attribution | Prefix-chain and final cash telescope; all rows replayed |
| DA2 | Matching Q target semantics | Soft-return continuation includes future entropy, excludes current entropy; explicit reward/cash columns |
| DA3 | Weight-space decomposition | Weight+bias+posterior terms reconstruct observed log-odds change |
| DA4 | Source/state identity | Sealed inputs, checkpoints, support; calibration-position mismatch separately counted |
| DA5 | Safe cache/restart | Cache namespace includes source identity; per-cell completion receipts; no unfinished cell counted |
| DA6 | Unbiased reporting | All three seeds/objectives and both periods; counts are correlated occurrences, not independent evidence |

## Decision after audit

Use TRAIN evidence to choose the next intervention, with VALIDATION only
reported diagnostically. Distinguish critic ranking conflicts, soft-objective
versus account mismatch, and actor drift; they can coexist. Prefer paired
action-ranking correction if harmful changes are consistently favored by Q
but contradicted by all three matched soft continuations. If Q is not the
problem, inspect entropy/execution alignment before another generic learning
rate or KL sweep. Existing failed refinements must be consulted, not repeated
under a new name. This audit does not authorize removing features merely to
repair a harmful update. Preserve the69 champion and resume floating only
after a qualified model change.

## Allocation

One sequential objective queue per available worker; inference inspection can
use the assigned GPU, but native account continuations are CPU work. Automatic
collection checks all receipts and produces the joined report with precise
next-action recommendation. No fitting is hidden in this analysis.
