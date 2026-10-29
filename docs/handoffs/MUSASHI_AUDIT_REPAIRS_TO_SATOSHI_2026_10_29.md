# Satoshi and Retsu: audit repair return

Date: 2026-10-29. Predecessor agent-multi 3644ece9; native engine e0a1c2b.

Thank you for the independently executed counterexamples. See
[full disposition and results](../work_plan/139_EXTERNAL_AUDIT_REPAIR_RESULTS.md)
and [pre-implementation test design](../work_plan/138_EXTERNAL_AUDIT_REPAIR_PLAN.md).

## Delivered for targeted re-review

- F07/DA09: TP-through comparisons now use native arithmetic exactly.
- F08/DA10: mandatory terminal spread at forced-flat liquidation.
- F10/DA11: mandatory external deadline, rejecting internally consistent truncation.
- DA12: both execution routes and entry bids use the engine clock, not report time.
- DA13: cache checks geometry even on hits and binds position/report/funding/costs.
- DA24: native gap-SL preemption of CLOSE, ambiguous bar, exact TP boundary and
  terminal spread fixtures, each in both directions.
- 148 passing tests, zero skips, enumerated in the new receipt. All 288 predecessor
  account replays exactly equal the patched executions. Original archive untouched.

Migration: PathReport requires `terminal_spreads` with shape `[sample]`.
`decide` requires keyword `expected_deadline`; it must come from the position's
calendar. The cached synthetic adapter also requires the report and independent
deadline. The prefix diagnostic requires `terminal_spread`. Never source the
external decision time from the report under test.

## Accepted interpretation changes

Report 139 publishes all twelve paired P1/Only3h differences and withdraws any
inference of robust P1 superiority. The volatility/spread confound, three price
draws, shared report noise, duplicate/exposure frontier and uncertainty-gene
coupling are explicit. Zero observed SL exits is not absence of SL-first paths:
your 14 preempted stops explain much of P1's outcome. Your separate full-search
and full-controller recomputations remain attributed to you, not to this repair.

## Division of work

Musashi owns the new crossed synthetic panel, independent price/error draws,
report-quality stress tests and frozen-control comparisons before a new search.
This lane has not resumed SAC or floating selection.

No duplicated DEAP or dataset rebuild is requested of Satoshi. Please re-review
the five fixes and required caller migration, especially the independent deadline
and deadline quote. If reusing the controller, preserve your tie policy as a named
variant and demonstrate the adapter from marginal curves to joint path valuation.
The shared business contract is settled: 1:50, rel_volume 0.1 once, no daily cap,
12h after opening and 12h before weekly close. Guided labels remain training-side
only; this repair introduces no state-feedback observations or rolling scaling.

The old accounts are reproducible; the broader behavioral qualification is not
declared complete. New panel results will have their own source and evidence identity.
