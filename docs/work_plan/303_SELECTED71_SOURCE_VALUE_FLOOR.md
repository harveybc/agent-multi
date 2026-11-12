# 303: Selected71 native source-value floor

## Fixed contract before execution

Extend302 with one explicit change: on each FIT anchor, the executed greedy
action may not have a native continuation reward lower than the frozen
source actor's chosen action, within1e-8. Protect source-choice margins against
only worse alternatives; better and equivalent actions remain admissible.
Check actual greedy native value after every proposed step. The reference is
the original actor, not a changing threshold and not a validation label.

This is a sufficient conservative margin constraint, not the complete
nonconvex admissible set. All FIT rows undergo the nonlinear check; rows with
no worse alternative need no linear constraint. MONITOR and VALIDATION remain
diagnostic only. Native reward and net cash are reported, not interchanged.

Unchanged: selected71 export299, calibration300, diagnosis301,65 FIT error
targets, advantage-weighted optimal-set log-probability loss, seeds19/43,
8 updates each, batch128, interior buffer and trust limits, rollback,
four trainable actor-head tensors, cold/native account checks and fixed
endpoints0/1/8. No new feature pass, actor reset or live promotion.

## Test gate

| Requirement | Test |
| --- | --- |
| Reject302's wrong-to-worse transition despite better loss | `test_rejects302_wrong_to_worse_even_when_optimal_mass_increases` |
| Allow genuine improvement and equivalent actions | `test_allows_better_action_without_requiring_immediate_global_optimum`, both HOLD directions |
| No MONITOR leakage or meaningless all-tied constraints | `test_no_monitor_constraints_no_all_tied_constraints_and_finite_check` |
| Fixed reference and explicit reward tolerance | `test_native_tolerance_is_separate_from_logit_tolerance_and_fixed_source` |
| Nonzero feasible improvement, not a frozen-model workaround | `test_new_constraints_have_nonzero_feasible_improving_direction` |
| Replay and native account integrity | Finite worker, account attribution and independent checkpoint verifier |

Keep302's historical contract and artifacts unchanged. The runner exposes
`--source-value-floor`, requiring the existing optimal-set/fresh-calibration
flags and a new schema identity. Promotion is not automatic.
