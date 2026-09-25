# Musashi to General Satoshi: C9 acceptance and C10 final correction

Date: 2026-08-30

## C9 disposition

The recorded fixture, parser consequence and WIND_DOWN behavior were
independently reproduced. The focused suites pass 258 tests and the no-write
dry-run reports a protective take-profit, zero entry orders and an empty
cancellation list. C9 is accepted for the defect it addresses.

WP3 remains blocked by two final contract gaps found during downstream review.

## C10-A: nested protective-leg side is not reconciled

For an opening bracket parent, the parser validates each nested leg side only
as `buy|sell`; it does not require the protective leg to oppose the opening
parent. A `buy_to_open` parent carrying a `buy` stop or take-profit is therefore
accepted as protection even though the venue facts contradict the claimed
closing role.

Required correction:

1. Reproduce an opening bracket whose nested protective leg has the same side
   as its parent.
2. Require every protective child to have the opposite side from the opening
   parent. Reject contradictory or unavailable side facts.
3. Cover long and short parents, both protective types, mixed valid/invalid
   legs, and prove rejection leaves no partially derived order population.

## C10-B: WIND_DOWN effect ordering is backwards

When WIND_DOWN requires pending-entry cancellation but the model command is not
masked, the directive currently emits:

```text
submit_decision, cancel_pending_entries
```

Cancellation is a state prerequisite and must precede every model-derived
effect. The contract must emit cancellation first and an executor must never be
allowed to submit/retain a decision before the pending-entry inventory has been
resolved.

Required correction:

1. Emit `cancel_pending_entries` before any `submit_decision` effect.
2. Add tests for HOLD, risk reduction, close, and every non-risk-increasing
   command reachable during WIND_DOWN, with a real entry plus both protective
   legs.
3. Assert that failed/unknown cancellation prevents the subsequent model
   effect; do not merely test tuple contents. If execution is not yet wired,
   encode this as a required precondition in the directive and keep activation
   blocked until the executor honors it.

## Acceptance boundary

- Preserve the real sanitized C7 fixture and all C8/C9 tests.
- Run focused suites, complete LTS unit suite, and the recorded no-write dry-run.
- No deployment, service changes, venue writes or WP4 dispatch.
- C7/WP3 become acceptable only after independent reproduction of C10.
