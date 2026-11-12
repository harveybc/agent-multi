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

## Executed result

COMPLETE. Runtime `e76ac9a6`; experiment identity
`7e11c3363ba816788691b50ed2ce1df6178b9685d85cad6d8350c1e58d8f17c7`.
The finite queue completed worker and account report in 37.33 seconds
(worker internal timer 25.94s). Collection verified 116 run files and 15 report
files. Independent verification covered 182 files including source artifacts.
Implementation passed 47 focused tests before dispatch; the final regression
adds the newly measured remaining failure mode, for 48 tests.

Coverage is now 42 source-correct FIT rows (26 openings plus 16 equivalent
HOLDs), 68 non-equivalent margin constraints, and all 65 FIT errors. The two
previously omitted tied targets are included. All 16 attempted steps accepted;
all protected greedy actions remained optimal. Four actor-head tensors changed
per checkpoint; the other 1023 state tensors were byte-identical to their
calibrated starting checkpoint. Optimizer counters match accepted steps;
checkpoint cold reload and full native accounts match exactly.

| Model / fixed endpoint | TRAIN return | VALIDATION return | TRAIN Sharpe | VAL Sharpe | TRAIN / VAL trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| Preserved selected71 control | 16.058224% | 18.369946% | 0.781969 | 2.690712 | 745 / 207 |
| Set-aware seed19, update8 | 15.184956% | 18.129028% | 0.743959 | 2.624919 | 746 / 209 |
| Set-aware seed43, update8 | 15.184956% | 18.129028% | 0.743959 | 2.624919 | 746 / 209 |

Identical trading is not a duplicated run: all four actor-head tensors differ
between the two endpoints, and schedules/curves are retained. The fresh critics
also differ, but are diagnostic, not the source of this corrective loss.
Neither endpoint replaces the control. No new floating pass was launched.

Whole-FIT fixed-error loss: 2.655644604 -> 2.652787797 / 2.652799583.
Optimal probability rises in 55/65 and 54/65 error rows, respectively, but
**zero source-error decisions become optimal** under greedy execution.
TRAIN full-account loss is 87.326775 USD and VAL loss is 24.091780 USD per arm.
All four chronological account attributions reconcile with zero error.

## Remaining mechanism identified, not a failed HOLD guard

FIT panel row107 changes NOP -> BUY. Its measured native continuation rewards
are NOP 0.901207, BUY 0.613215, SELL 1.163136. SELL's probability increases
from 0.0639077 to 0.0642956/0.0643201, yet BUY overtakes NOP. This source row
was already incorrect and therefore was never in the source-correct guard.
The actual account contribution is -30.073410 USD. The second TRAIN change
is MONITOR row150, NOP -> SELL (-57.253365 USD). These are not a shortage of
coverage for their timestamps: both are in the panel. One is a missing
source-relative non-regression condition on error rows; the other is an
out-of-FIT transfer error. Do not turn MONITOR into a fitted constraint.

The exact two-row witness is retained in
`test_measured_remaining_gap_error_can_switch_to_worse_with_lower_loss`.
Increasing optimal-action probability and lowering cross-entropy do not
guarantee that the currently selected suboptimal action improves.

TRAIN early closes rise 58 ->59, helpful 29 ->30, harmful unchanged29.
VALIDATION retains6 closes (4 helpful,2 harmful). This is a downstream
consequence of changed openings, not evidence that a sampled early-close
error was corrected. Drawdown remains9.358326% TRAIN and changes
3.257290 ->3.257336% VAL. Activity remains above2 orders/week in both periods.

## Disposition and next bounded design

Adopt the tie-aware semantics and tests as a correctness repair; retain the
original selected71 actor for further comparisons. The next design should
forbid an error-row greedy action whose native FIT value is below its original
chosen action, while allowing a better alternative and equivalent actions.
Test that gate with row107's symbolic counterexample before another finite
update. Keep MONITOR/VALIDATION diagnostic only; no timestamp-specific tuning,
weakened tolerance, expanded budget or repeated unchanged floating pass.

Evidence: [manifest](generalization/selected71_optimal_sets_20261112/MANIFEST.json),
[independent audit](generalization/selected71_optimal_sets_20261112/AUDIT.json),
[full accounts](generalization/selected71_optimal_sets_20261112/SUMMARY.json),
[steps](generalization/selected71_optimal_sets_20261112/STEPS.json),
[return plot](generalization/selected71_optimal_sets_20261112/RETURNS.png).
Source-account metrics are exact. Cross-runtime source entropy differs by at
most8.64e-8 and is reported separately, not represented as byte parity.
