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

## Result: protection repaired, trading unchanged

COMPLETE. Runtime `b18e84c5`; experiment identity
`986505fd8e50800e77eb4bef2eced2aaff7a485ece456b0386f4874f8f0d7975`.
55 focused tests pass. The finite queue and collector ended successfully.
Worker internal time25.69s. Collection verified116 run files and15 report
files; independent verification covered182 files including source evidence.

The nonlinear value gate checks all113 FIT anchors.79 have worse alternatives
and produce105 linear constraints: the original42 source-correct protected
rows plus37 initially incorrect rows. The remaining34 require no margin
constraint (28 already choose a worst action;6 have all actions equivalent).
All65 error targets remain in the objective. None of the38 MONITOR anchors
becomes an optimization constraint. This is a sampled TRAIN guarantee, not a
proof about every unseen market state or every unsampled TRAIN row.

Both seeds accepted all8 full steps, without backtracking or relaxation.
No protected decision lost native value. The previously harmful FIT NOP->BUY
and MONITOR NOP->SELL transitions no longer occur. Only the FIT transition
is explicitly constrained; removal of the MONITOR transition is an observed
transfer effect, not a fitted rule for that timestamp.

| Fixed endpoint | TRAIN return | VAL return | TRAIN Sharpe | VAL Sharpe | TRAIN / VAL orders |
| --- | ---: | ---: | ---: | ---: | ---: |
| Preserved selected71 | 16.058224% | 18.369946% | 0.781969 | 2.690712 | 745 /207 |
| Source-value floor, seed19 update8 | 16.058224% | 18.369946% | 0.781969 | 2.690712 | 745 /207 |
| Source-value floor, seed43 update8 | 16.058224% | 18.369946% | 0.781969 | 2.690712 | 745 /207 |

The previous302 contrast lost87.326775USD TRAIN and24.091780USD VAL per arm.
Both losses are absent here. This is recovery to the control, not a new profit
record. Complete trade ledgers and every equity-array element match control,
not just terminal returns. Each arm changes3 TRAIN symbols and1 VAL symbol;
all occur while entry is prohibited by the existing entry cutoff, with no
position to close. Their chronological account effects are all exactly zero.
Early-close outcomes and trading activity are consequently unchanged.

## Nonzero learning and remaining limitation

All four actor-head tensors change in every saved checkpoint. The other1023
state tensors are byte-identical to their calibrated source. All four head
tensors differ between seed endpoints; optimizer step counters are8. Cold
reloads reproduce the worker and native accounts exactly. The original
selected71 checkpoint remains unchanged and is not replaced.

Whole-FIT fixed-error loss:2.655644604 ->2.655112220 /2.655112721.
Optimal-action mass increases on42/65 errors, decreases on23/65; no error
crosses to a better greedy decision. No panel symbol changes at all. The
method does not freeze the parameters, but this short budget has not achieved
a useful decision change.

End-of-run relative head drift is0.0015531/0.0015541 versus the0.01 ceiling;
max KL2.2910e-5/2.2764e-5 versus1e-3; mean KL4.4097e-7/4.4011e-7 versus1e-4.
All step margin RMS values stay below0.000412 versus0.002. Thus the global
trust ceilings did not reject a step. The source-margin constraints are
nevertheless close to active (minimum endpoint slack about9.13e-7/1.00e-6).
This does not establish that longer training or relaxed constraints would help.

## Disposition and exact next action

Retain the FIT source-value floor together with the optimal-set/HOLD semantics
as the correctness contract for subsequent bounded refinement. Keep the
selected71 control and both new endpoints; no automatic model or live promotion,
unchanged floating repeat, new data transform or unconstrained retraining.

Next diagnose distance to useful FIT greedy boundaries and feasible head
directions under these constraints and unchanged trust limits. Separate
inadequate step count from shared-head interference and overly conservative
margin constraints before changing the optimization budget. MONITOR and
VALIDATION remain diagnostics only. A smaller loss or unchanged account is
not grounds to declare feature selection improved.

Evidence: [manifest](generalization/selected71_value_floor_20261112/MANIFEST.json),
[audit](generalization/selected71_value_floor_20261112/AUDIT.json),
[accounts](generalization/selected71_value_floor_20261112/SUMMARY.json),
[steps](generalization/selected71_value_floor_20261112/STEPS.json),
[plot](generalization/selected71_value_floor_20261112/RETURNS.png).
Source-account parity is exact; inherited cross-runtime entropy differences
of at most8.64e-8 are recorded separately in the audit.
