# Frozen Policy and Return Calibration

Owner accepted the next diagnostic on2026-11-08. The head-only run completed40
epochs and restored eligible epoch0. No learned replacement was selected. Continue
from the paired-operator findings in [213](213_WARMSTART_DRIFT_DIAGNOSTIC_RESULTS.md).

## Fixed Design

Freeze checkpoint0, completed snapshot6 and terminal snapshot40. For each, run
native full TRAIN2020-2023 and VALIDATION2024 accounts with greedy actions and
categorical sampled actions under seeds7/19/43:24 accounts total. This is inference
and accounting only: no training, optimizer, new feature selection or TEST access.
Use the unchanged70 columns,28 groups,13-step/stride2 windows, scaler, costs,
order rules, horizon and discount. Do not inject position into model inputs.

Batch inference is valid because the model observes market data only. Precompute
the frozen logits/Q surfaces once per checkpoint, then retain the sequential
native account transitions. Sample from the actual categorical probabilities,
not from added synthetic noise. Use common seeded uniforms indexed by input row
across checkpoints. The random generator is not the original training generator;
this reproduces the action distribution, not historical sampled action traces.

Record the original probabilities, taken signal, command, pre-action position,
timestamp, reward and transition duration. Position is diagnostic metadata only.
Retain equity, trades, costs and every full-account metric through the existing
evaluator. Greedy accounts must exactly reproduce their saved source metrics.

## Discounted Returns

Rewards are the existing duration-discounted equity increments expressed as
percentage points of initial capital. For a trajectory of categorical signals,
the sampled soft return for its chosen action is:

`G[t] = r[t] + gamma**duration[t] * (G[t+1] - alpha*log(pi(a[t+1]|x[t+1])))`

At the terminal transition, `G[last] = r[last]`. Do not add current-action
entropy to a Q target. Also compute the ordinary discounted return with alpha0.
The simulator already discounts internal skipped native bars inside r[t].
Compare online twin Q values (minimum and individual twins) for the chosen signal
with the sampled soft return. Report bias, MAE, RMSE and calibration bins, with
breakdowns by signal, pre-action position and quarter.

Only stochastic-policy soft returns estimate the policy quantity SAC learns.
Greedy-policy ordinary returns are a policy-mismatch diagnostic, not a claim
that the stochastic critic should equal the greedy value. Samples are temporally
dependent; no IID confidence interval or significance claim is allowed. Three
sampling seeds are not three independently trained models.

## Acceptance Before Interpretation

| Requirement | Test / runtime evidence |
|---|---|
| Native strategy unchanged | existing evaluator and same prepared-signal interface |
| Greedy parity | full saved account comparison for all three checkpoints |
| Full period support | exact expected decision counts and timestamp-to-input mapping |
| Correct stochastic policy | reproducible inverse-CDF draws; frequency test and softmax parity |
| Correct soft returns | hand-calculated multi-duration/terminal fixtures; current-entropy exclusion |
| No mutation | tensor equality after all inference/accounts; source artifact hashes |
| No false units | label rewards/Q as percentage-point reward units, not USD or account return |
| Reproducibility | persist surfaces, traces, sampled action identity and terminal file manifest |

Primary RTX5090 performs the batched frozen inference and sequential account
replays. The completed artifact manifest is independently checked by the
coordinator. Do not interrupt the independent RTX4090 replication. No automatic
policy promotion or learning-method change follows this diagnosis: first determine
whether there is a substantial training/evaluation policy gap or a critic-value
calibration problem.
