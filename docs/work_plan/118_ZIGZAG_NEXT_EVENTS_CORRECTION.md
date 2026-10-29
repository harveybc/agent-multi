# Next-event oracle correction

Owner clarification, 2026-10-28. Supersedes the endpoint semantics in 114-117.
The old results belong to a capped objective; they are not results of this one.

## Requirement and implementation

At each admitted hourly origin, enumerate the next K strictly later ZigZag
events in chronological order, irrespective of peak/valley type. K is an
explicit parameter: compare 2 and 3. For each direction evaluate every endpoint
in this set, using the complete intervening path for drawdown and duration.
Select the best endpoint for each direction, then compare buy, sell and NOP.
No fourth event is consulted for K=3. No maximum holding time is inferred from
the nominal horizon. No open quote replaces a missing pivot.

This is a retrospective opportunity objective, not an executable account
strategy. Exit references may lie after business flat: the separate interval
schedule rejects those intervals rather than replacing their exit. Business
entry eligibility remains unchanged. Extrema cannot cross a missing-data or
session chunk boundary. Origins with fewer than K observed future events in
that chunk are explicitly censored, not labelled NOP or fabricated trades.

The existing net-profit / (max(drawdown, round-trip cost, one pip) * hours)
score, cost assumptions, scale band and TRAIN 2020-2023 remain unchanged to
isolate this correction. Raw per-side censored metrics are NaN. The objective
is the sum of positive best scores divided by the fixed number of admitted
origins, so discarding origins cannot improve fitness by shrinking its
denominator. Coverage, complete and censored counts accompany every candidate.
For comparing K=2 and K=3, report common-origin results as well as the fixed
denominator objective; raw changes in coverage are not predictive improvement.

Population 30, generations 300, seeds 7/19/43, horizons 3/6/12/24 remain the
requested budget. New roots and new code/spec identities are mandatory. No
cache, champion or score from the capped searches is reusable as evaluation.

## Test-first evidence

Six failures reproduced before implementation. The corrected focused suite
passes 23 tests (opportunity and extension). Regression cases cover:

- Hold a sell through a peak until a later valley, beyond the old 4*h cap.
- Mirrored buy through a valley until a later peak, with full path risk.
- Strictly future endpoints, no forced-flat quote substituted in the oracle.
- Missing events censored, no borrowed pivot across an unobserved chunk.
- Two versus three events; the third can improve the selected opportunity.
- Perturbing the fourth event and prices after the third leaves that origin's
  scores unchanged when the first three supplied ZigZag events are unchanged.
- Existing costs, financing, deterministic DEAP and receipt identity checks.

The final perturbation is a consumer test, not a claim that hindsight ZigZag
itself is prefix invariant. No new DEAP result is claimed by these tests.

Independent scalar enumeration on TRAIN also checked 2,560 candidate paths,
both directions, K=2/3 and all four former champion genotypes: selected scores
and endpoint indices matched within 1e-12. These are correctness probes, not
new optima. Complete-origin counts (out of 19,879) were:

| Nominal horizon | K=2 | K=3 |
|---|---:|---:|
| 3 h | 19,817 | 19,620 |
| 6 h | 19,713 | 19,108 |
| 12 h | 18,504 | 16,089 |
| 24 h | 17,324 | 12,539 |

The session-boundary censoring is material at 24 h. Do not silently remove
this coverage distinction or interpret the different supports as a paired gain.
