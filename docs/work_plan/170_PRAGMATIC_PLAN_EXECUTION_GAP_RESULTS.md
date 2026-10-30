# Measured plan-execution gap

2026-10-30. Runtime `ab1c2778`, native engine `e0a1c2b`.
Identity `b2e26ee997822fb67ca7ccddb9a8e0c71fe27653f197f87b917237465813933d`.
[Plan169](169_PRAGMATIC_PLAN_EXECUTION_GAP_PLAN.md) completed in full:
72 baseline accounts,305 ideal-report closes,610 independent committed-plan
replays.265 tests pass without skips/errors/failures. Completed resume
authenticates all artifacts without repeating work. No policy was changed.

## Result

| Frozen policy | Closes | Committed valuation max error | Ranking reversals | CLOSE continuation differs | HOLD continuation differs |
|---|---:|---:|---:|---:|---:|
| 59a75d73 | 157 | 0 | 79 | 144 | 156 |
| 280977a9 | 148 | 0 | 71 | 134 | 146 |

Every committed CLOSE and HOLD plan reproduces its original ideal-report
return, drawdown and score exactly. The native engine therefore computes
those declared plans consistently in these cases. This is not a blanket
certification of every simulator feature or a learned-output result.

**150/305 decisions reverse their utility ranking under actual hourly policy
continuation.** Their planned CLOSE advantage is positive but their realized
CLOSE-minus-one-tick-HOLD advantage is negative. This is a concrete mismatch
between the continuation valued by the controller and the continuation it
actually follows, even with ideal signals. It does not require a forecasting
error to occur. It is a defect of using a limited committed-plan comparison
as the value of the subsequently replanned policy, not proof that receding
horizon control itself is unsuitable.

The scalar here is the frozen `J=return_pct-account_risk_weight*max_dd_pct`.
That explains the difference from168's148 profit-harmful closes:147 hurt
both profit and J, one hurts profit but improves J, and three improve profit
but hurt J. The remaining154 improve both. Do not substitute one metric for
the other or claim a unique global Pareto optimum.

## Exact decomposition

`actual_margin = planned_margin + close_continuation_shift - hold_continuation_shift`.

The maximum numerical residual across all305 cases is1.78e-15. This separates
the effect of changing future behavior in both branches rather than assigning
all loss to the first close, or just looking at its passive-position result.
The diagnostic follows the frozen policy after only one changed action. It
does not add overlapping interventions into a new strategy's alleged profit.

## Reproducible witness

World000, policy59a75d73, tick50, existing short position:

| Continuation | Return % | DD % |
|---|---:|---:|
| Declared CLOSE then BUY24 plan | 2.350665327 | 0.288878454 |
| Declared HOLD3h then BUY24 plan | 2.473958102 | 0.577829804 |
| Actual CLOSE then hourly policy | 4.370220255 | 0.288878454 |
| Actual HOLD one tick then hourly policy | 4.627729939 | 0.288878454 |

The planned J margin favors CLOSE by0.165658575 points. The actual J margin
is-0.257509684 points, with-25.750968 USD profit and no DD improvement.
At the very next tick, the CLOSE branch opens BUY12 rather than planned
BUY24; the HOLD branch closes rather than waiting its planned3h. Both
continue operating beyond the original two-order approximation.

At ticks62 and95, another expected continuation is missing: the plan values
an immediate subsequent BUY6, while the real policy chooses NOP at ticks63
and96. These witnesses make the error actionable; merely enlarging the
delayed-close grid cannot make a different future policy match its valuation.

[Witness with committed ledgers and first divergences](../audits/evidence/pragmatic_plan_gap_20261030/WORLD_000_59a75d73.json).

## Correction direction

Replace the assumption of a fixed two-order continuation with an explicitly
defined continuation policy for both candidate actions. First implement a
non-recursive rollout using a frozen simple/proxy continuation as a transparent
reference; document that as policy improvement over that reference, not as
exact evaluation of the recursively replanning selector. Subsequent policy
iterations must be separate frozen versions and evaluated on fresh worlds.

Require common information before a branch is observed, no access to hidden
actual future for causal reporters, and a specified treatment of future
report updates. An ideal replay can validate mechanics but cannot validate
that causal information model. Preserve native protection, weekly flat and
hourly early-close evaluation. Do not fix the mismatch by forcing every plan
to remain committed forever or disabling early close.

The simple and predictive-proxy controls remain unchanged and useful. No
new model training, floating selection or controller promotion follows from
this diagnostic. The exact component to correct is now identified; the data
or RL algorithms were not blamed for this controller behavior.

## Evidence

- [Manifest](../audits/evidence/pragmatic_plan_gap_20261030/MANIFEST.json)
- [Runtime tests](../audits/evidence/pragmatic_plan_gap_20261030/TESTS.json)
- [Numerical results](../audits/evidence/pragmatic_plan_gap_20261030/REPORT.json)
- [Completion and per-unit hashes](../audits/evidence/pragmatic_plan_gap_20261030/COMPLETE.json)
- [Audit request](../audits/SATOSHI_AUDIT_REQUEST_PRAGMATIC_CONTINUATION_GAP_2026_10_30.md)

Full72-unit evidence remains in the immutable run archive. The published
manifest, tests, report, completion and witness are byte-identical copies.
