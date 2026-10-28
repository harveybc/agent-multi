# Paired SAC Early Stopping

## Frozen Scope And Requirements

Approved experiment: only the selected-input SAC_FLAT with its existing15
inputs,13 observations at stride2,32/32 heads, seed7, leverage50, relative
margin budget0.1 and mandatory ATR SL/TP. No refit, TEST, DOIN, AE, BatchNorm,
dropout or weight decay. Existing active experiments are not modified.
BatchNorm and dropout must remain separate future ablations, never combined
in this experiment. No claim that their combination is universally invalid.

Evaluate every5000 collected transitions, patience10 evaluations, absolute
min_delta1e-8, maximum100000 transitions. First checkpoint is5000; ten
nonimprovements after it can stop at55000. Minimize
`M = -0.5 * (mean_train_daily_log_return + mean_validation_daily_log_return)`.
Use completed observed NY business-day closing equity, with initial capital
as the first prior equity. Do not average four-year and one-year raw profits.
An empty, nonfinite, nonpositive-equity or truncated trace is not an eligible
full-period checkpoint. A no-trade policy can attain zero: report activity
and compare with no-trade, never call mere gap reduction a trading success.

Each checkpoint is saved before evaluation. Two existing curated inference
tasks run in isolated interpreters, TRAIN2020-23 and VALIDATION2024, exact
selected support, deterministic actions. They never call learn, write replay,
reseed the parent or change its live environment. No repeated learn calls or
optimizer resets. Restore the selected checkpoint's policy and optimizer
state after learn while retaining actual-work counters separately.

Per checkpoint retain archive, config, trace, summary, hashes, paired score,
patience, evaluation duration and all required trading metrics. Existing
warehouse outbox enqueues/readbacks every completed checkpoint pair via the
publication timer; final close includes monitor deliveries and physical files.
No parallel second learning process on the shared preferred host. After the
current campaign closes, a finite successor starts this new identity without
an agent monitoring logs.

## Test Design And Traceability

| ID | Requirement | Planned evidence |
| --- | --- | --- |
| ES1 | Exact daily growth units and full support | known equity traces, unequal durations, NaN/zero/short trace negatives |
| ES2 | Correct patience and best checkpoint | improve/plateau/threshold and exact stop boundary |
| ES3 | No architecture/regularizer changes | sealed request, policy module inventory, baseline parity |
| ES4 | No evaluation contamination of learning | subprocess inference guard, policy hash, parent RNG/replay/control tests |
| ES5 | Restore actual best, not last | real SB3 CPU gradient/save/restore with independent hashes |
| ES6 | Bounded direct-only dispatch | two tasks only; invalid subset/monitor refusal |
| ES7 | Persistent metric/copy closure | monitor row readback, digests and collector tamper tests |
| ES8 | Automatic resource-aware continuation | predecessor close pin, one-shot ownership and successor refusals |

Discovery through unit-test design complete. Engineering implementation and
synthetic verification precede sealed real deployment. Validation is used for
checkpoint selection and is development data, not an independent holdout.

## Operational Publication Correction

2026-10-19: real checkpoint pairs exposed an incorrect outbox root inferred
from nested monitor directories. ES7 now passes the campaign root explicitly;
ordinary task publication remains unchanged. Regression failed before the fix.
The isolated publisher can be upgraded without modifying the running learner,
its scientific pin, contract, stopping rule or checkpoints. Legacy monitor
outboxes are redundant live state, not physical scientific artifacts: validate
their member paths, exclude only the recognized monitor outbox filenames, and
record exclusions in collection evidence. The canonical closed metrics backup,
all report readbacks, and all scientific file digests remain mandatory.
Focused CPU verification: 159 tests passed, including monitor-root regression
and legacy collection with positive and corrupted-artifact controls.
