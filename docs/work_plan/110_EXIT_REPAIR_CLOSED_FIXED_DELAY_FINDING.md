# Closed exit repair: fixed-delay counterexample

Date: 2026-10-28. Read-only post-run review. No checkpoint or closed artifact
was modified. The queue ended at 07:00 UTC (02:00 America/Bogota).

Completed: 12 pure-RL fits, one separate supervised capacity diagnostic and
nine close-bias DEAP searches. All 40 files in the CLOSE artifact manifest
were rehashed successfully during this review. No job from this queue is
still computing; the service's active/exited state means completed.

## Acceptance defect

The producer accepted `first_128k_43` against immediate closure, passive hold
and a TRAIN-frozen constant probability of closing. It omitted a deterministic
fixed-delay control. The sealed acceptance is preserved as the result of that
old rule, but it is insufficient to qualify learned exit timing.

In all 64 confirmation cases, the accepted policy closes at exit index 6:
after six HOLD decisions, at the seventh decision. This is not six downsampled
input observations. Its TRAIN exit indices vary, so the finding is not that
the policy is universally a fixed clock on every possible input.

An independent control chose its single delay using TRAIN only: minimize mean
initial-risk-normalized regret over delays from immediate closure through the
longest TRAIN executable menu, including passive hold. The selected delay is
6. If a position terminates automatically before that delay, retain its actual
native terminal result. No confirmation outcome chooses the delay.

| Policy | TRAIN mean regret / risk | Confirmation mean regret / risk |
|---|---:|---:|
| Learned first-reset, 131072 steps, seed 43 | 0.10354518 | 0.10128466 |
| Fixed delay selected on TRAIN: 6 | 0.09501241 | 0.10128466 |

The learned policy is worse on TRAIN and exactly equal in confirmation. Its
qualification as evidence of useful learned timing is therefore withdrawn
in this review, not by rewriting its original receipt. No policy is promoted.
The next qualification must include this TRAIN-only fixed-delay comparator.
Re-scoring existing checkpoints suffices to correct this gate; it does not
require repeating all fits merely to calculate the missing baseline.

## Optimization scope

All three objective finalists yield identical confirmation metrics on the
simulated ideal/noisy/delayed close outputs: regret 0.01044406 and unannualized
episode Sharpe 1.679352. Against zero bias, the paired regret delta is
-0.02141984 (cluster-bootstrap 95% interval [-0.02968844, -0.01328992]).
This is a two-close-bias controller result, not the performance of the trained
policy, an objective-family winner, or full-strategy optimization. Its
completion cannot override the learned-control counterexample above.

## Next action

Add the missing fixed-delay acceptance regression and retain a new re-scoring
receipt over the already sealed policies. Then investigate credit assignment
and state-dependent learning on TRAIN before any further financial selection.
No financial fit or new automatic training successor is running after closure.
