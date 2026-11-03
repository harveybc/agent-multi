# Paired stopping-rule comparison

Owner request2026-11-03, superseding further execution of193's mean-only fit.
The former checkpoints and results remain available. No DEAP or new selection.

## Two Arms

Both arms train from scratch with seed7 and exactly193's108 inputs,31 groups,
13x2 window, architecture, optimizer, batch128, native risk/costs, fixed TRAIN
scaler and full TRAIN2020-2023 / VALIDATION2024 accounts. TEST2025 stays closed.
Maximum2000 complete epochs; patience20; minimum improvement1e-8.

-Primary RTX5090:`validation`. Maximize validation return per week. TRAIN is
 still learned from and fully evaluated, but does not select checkpoints.
-Secondary RTX4090:`mean_no_regression`. Maximize the mean of TRAIN and
 validation return per week, accepting a new checkpoint only if neither
 component nor their mean decreased from the immediately preceding evaluated
 epoch. Require improvement over the best *accepted* mean by more than1e-8.

The first finite epoch establishes a best. Every rejected epoch increments
patience exactly once, including a mean improvement rejected by the guard.
Only acceptance resets patience. Update the previous-epoch reference after
every evaluation, including rejection, and persist it across restarts.
At stopping, restore each arm's accepted best, not the last epoch.

The guarded rule follows the owner's previous-epoch definition; it does not
guarantee validation is nondecreasing relative to the accepted best. Recovery
after a rejected dip can still accept a validation value below that older
checkpoint. Report both references explicitly rather than silently strengthen
the rule. Validation-only patience permits temporary drops during training but
never replaces its best with a worse validation checkpoint.

## Interpretation

The old mean monitor allowed large TRAIN growth to offset a validation decline.
Early stopping controls checkpoint choice; it does not force gradient updates
to improve validation. SAC learns from TRAIN rewards through actor/critic
losses; those losses do not guarantee monotonic realized trading returns.

This is a single-seed two-rule diagnostic, not a multi-seed superiority claim.
Hardware differs; identical seeds do not guarantee bit-identical GPU learning.
Log both stopping rules on each arm's actual trajectory as shadow decisions,
without affecting training, to distinguish stopping logic from device drift.
No extension of a stopped arm is implied by its shadow monitor.

## Tests Before Launch

| Requirement | Test |
|---|---|
| TRAIN cannot compensate in validation-only mode | Extreme TRAIN improvement with validation decline is rejected |
| Either component can veto guarded mean | Separate TRAIN and validation decreases; strict equality boundaries |
| Historical best still matters | Joint recovery that does not beat accepted best cannot reset |
| Previous means previous evaluated epoch | Rejected dip followed by recovery uses the dip, not accepted best |
| Patience20 and1e-8 | Tiny improvement, exact tie,20 consecutive rejections |
| Resume is semantically identical | Checkpoint previous values, counters, replay and next-update parity |
| No input/split drift | Identical recipe, support, risk and effective-model digests |

Best inference bundles and automatic remote collection remain mandatory.
