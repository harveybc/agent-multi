# 302: Selected71 optimal-action-set refinement

## Contract fixed before execution

Repair the tied-action omission measured in 301, without changing the historical
strict helpers or replay artifacts. The new runner flag is `--optimal-action-sets`
and requires the fresh calibration and interior correction modes.

Native optimal set: actions within 1e-8 of the best native continuation reward.
Only FIT rows define targets and constraints. A source-correct row with at least
one suboptimal alternative is protected. Protect each source-chosen optimal
symbol's margin against suboptimal symbols only; do not constrain two HOLD
aliases against each other. This sufficient constraint is deliberately
conservative, not the full nonconvex feasible region. Acceptance additionally
checks actual greedy membership in the optimal set.

Every source-error FIT row receives a fixed advantage weight and the loss
`-weight * log(sum(probabilities of optimal actions))`. No arbitrary winner is
chosen among tied optima. All-equivalent rows have zero corrective weight.
Critic Q is diagnostic only in this corrective experiment.

Preserve the budget: seeds 19/43, eight attempted updates each, batch 128,
checkpoints 0/1/8, existing KL, margin and relative-weight limits, interior
rounding cushion, backtracking and optimizer rollback. Four actor-head tensors
only; actor encoders, critics, target critics and alpha frozen. No input,
strategy, risk, scaler, reward, horizon, split or live service change.

## Requirement-to-test traceability

| Requirement | Evidence to execute |
| --- | --- |
| Protect both directions' equivalent HOLD symbols | `test_both_hold_aliases_protected_but_bad_close_rejected` |
| Close 301 counterexample; exclude MONITOR and all-tied rows | `test_counterexample_and_fit_only_all_ties_excluded` |
| Correct tied errors without choosing a label arbitrarily | `test_two_good_holds_receive_mass_not_arbitrary_onehot` |
| FIT-only optimization and typed invalid-input refusal | `test_monitor_never_enters_targets_or_gradient_and_bad_inputs_refuse` |
| Probability mass does not substitute greedy correctness | `test_optimal_mass_alone_is_not_greedy_acceptance` |
| Preserve constraints and rollback | Existing interior/bounded projection regression suite |
| End-to-end: original control, frozen tensors, cold reload, accounts | Sealed finite worker and full native account attribution |

## Decision boundary

Source is selected71 export299, calibrated critics300 and diagnosis301. No
automatic model promotion. Full TRAIN and VALIDATION accounts are reported for
both fixed endpoints, with original actor as control. TEST is not read. A
better local loss is not evidence of improved full-account trading.
