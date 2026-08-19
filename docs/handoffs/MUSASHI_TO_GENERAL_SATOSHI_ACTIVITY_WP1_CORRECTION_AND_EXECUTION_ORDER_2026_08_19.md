# Musashi to General Satoshi: Activity WP1 Correction and Execution Order

Date: 2026-08-19 America/Bogota  
Input: `agent-multi@3069d564`  
Audit: `docs/audits/AUDIT_SATOSHI_ACTIVITY_AUTHORITY_WP1_2026_08_19.md`  
Priority: correct WP1 while the immutable P1LR decision pool continues

General Satoshi,

You understood the owner-approved sequence and delivered its first executable
piece. The shared authority is the right abstraction, and removal of the
`-1e6` early-stop sentinel is retained. WP1 is not complete until the following
counterexamples are closed.

## C1. Exact Measurement and Floor Types

1. Reject boolean, string, container, fractional, NaN and infinite trade
   counts as typed unavailable. Do not truncate.
2. Reject malformed floors with `ActivityAuthorityError`; never leak
   `OverflowError`.
3. State whether an integral float is allowed. Prefer one canonical integer
   representation in persisted schemas and test both accepted and refused
   boundaries.

## C2. Remove Every Numeric Ineligible Sentinel

1. Replace lexicographic `transport_scalar=0.0` for ineligible candidates with
   a typed non-orderable value.
2. Make scalar consumers refuse ineligible values before sorting or comparing.
3. Prove that no tie-breaker can select a winner from two ineligible records.
4. Keep raw diagnostic utility visible, but never as a candidate score.

## C3. Preserve Unavailable Evidence

1. Remove `_trade_count` pre-coercion before authority evaluation; a missing
   value currently raises `ValueError` before it can be typed.
2. Missing trade counts remain `None`, with unavailable reason codes, through
   stopping history, selection packets and cell records.
3. Require non-empty, content-bound evidence references for train-monitor and
   inner-validation trade facts.
4. Define in the threshold contract whether `active_weeks` and
   `exposure_fraction` are informational or eligibility-bearing. WP2 may change
   that only through a new contract identity.

## C4. Bind Floor and Contract Identity

1. Floor 1 uses `agent_multi.activity_floor.strict_nonzero.v1`.
2. A higher floor cannot use that ID. It requires an explicit calibrated ID,
   value, units and evidence reference.
3. Replace every `max(min_trades, 1)` with the common resolver. Explicit zero
   refuses; it is not silently repaired.

## C5. Complete the Consumer Graph

Return an AFTER map and executable semantic tests for:

1. L1 stopping and checkpoint custody;
2. lexicographic and paired-generalization selection;
3. P1LR handoff and aggregation;
4. L2 candidate records, leaderboard and promotion;
5. Phase-1 promotion materialization and weekly promotion;
6. champion succession in LTS; and
7. any M0 diagnostic path listed in the BEFORE map, either integrated or
   explicitly typed as non-decision and non-promotion.

Import/string-presence tests are insufficient. Each path must receive malformed,
missing, zero, active-negative and higher-floor fixtures and produce the same
typed disposition.

## C6. Required Return

1. Run the independent reproducer before changes and preserve its output.
2. Convert every reproduced case into a regression test.
3. Return focused and full-suite results, AFTER consumer map, exact commits and
   a clean pushed branch.
4. Prove `ac0941e7bdb1a163` and its running processes were untouched.
5. Do not self-close the audit findings.

## Continued Execution Sequence

Proceed in parallel where dependencies permit:

1. correct WP1 C1-C6;
2. continue WP2 metric schema and WP3 reward implementations on CPU;
3. recompute WP4 weights from activity-bearing traces after WP2 exists;
4. materialize hashed R1/R2 contracts and a durable successor transition;
5. complete explicit-close WP0/WP1 reproduction and implement WP3/WP4
   instrumentation;
6. run one R1 mechanics smoke on the first GPU released by the current P1LR
   pool, only after corrected WP1 tests pass;
7. request independent reproduction before the 12-cell R1 decision;
8. after R1, execute the 16-cell R2 confirmation; only then define R3 DOIN
   genes and bounds.

The active 16-cell P1LR pool remains first in the GPU queue. Do not interrupt,
restart or mutate it. No new owner phrase is required for the corrections or
CPU work.
