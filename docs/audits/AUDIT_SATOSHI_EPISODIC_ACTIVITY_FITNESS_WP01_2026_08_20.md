# Audit: Satoshi Episodic Activity Fitness WP0/WP1

Date: 2026-08-20 America/Bogota  
Auditor: General Musashi  
Subject: `agent-multi@342f4a84`  
Governing order: `MUSASHI_TO_GENERAL_SATOSHI_EPISODIC_ACTIVITY_FITNESS_CORRECTION_2026_08_20.md`  
Verdict: **WP0/WP1 rejected; WP2-WP7 not delivered; fleet campaign remains prohibited**

## 1. Scope and Reproduction

The subject commit adds one standalone fitness module, one 25-test file and one
before-evidence JSON. The commit title correctly says `WP0+WP1`; it is not a
completed response to the WP0-WP7 order.

Independent focused suites in `trading-stack`:

```text
46 passed
```

The 25 new tests pass, but the independent reproducer
`SATOSHI_EPISODIC_FITNESS_WP01_REPRO_2026_08_20.py` demonstrates that the
claimed correction does not yet remove the observed passive attractor.

## 2. Findings

### EAF-001 — S2 — Exact reproduced defect is not corrected

WP0 records a quasi-passive policy with one trade per split beating an active
learner. WP1 changes the acceptance test to zero trades and therefore proves a
different, easier property.

Independent values:

```text
1 trade, return -0.001%:  -0.0097575379
40 trades, return -0.21%: -0.0635704365
```

Higher is better, so one symbolic trade still beats the active learner. The
policy can purchase escape from the sentinel with one trade and return to hold.
This is the original failure with a one-trade disguise.

Required correction: activity must be a first-class ordered component during
the easy learning regime. Before a calibrated minimum/target region is reached,
increasing meaningful activity must be capable of outranking small economic
differences. Preserve economic facts separately. Add the exact WP0 fixture as a
regression test; do not replace `1` with `0`.

### EAF-002 — S2 — The new objective is not connected to executing code

No production module imports `_episodic_activity_fitness`; only its own test
does. `_selection_value`, the paired comparator, checkpoint selection, early
stopping and the P1LR runner therefore retain their old behavior.

Required correction: complete WP2/WP3 integration. Prove by call trace and a
real pipeline fixture that the new scalar governs easy checkpoint selection and
early stopping, while NOP remains unpenalized per step.

### EAF-003 — S2 — Deep negative balances are aliased

The active-loss branch uses `min(abs(return), 1.0)`. Returns `-1`, `-10` and
`-100` all score exactly `-25.005` at target activity. This discards the
information produced by the relaxed-solvency curriculum and violates monotonic
movement toward zero over precisely the negative-balance region easy exists to
explore.

Required correction: use a bounded strictly monotone transform over the entire
nonnegative loss domain, for example `loss / (1 + loss)`, and property-test
values below `-100%` without allowing an active finite result to cross the
zero-trade sentinel.

### EAF-004 — S2 — Unvalidated configuration can invert the objective

There is no typed validation of the activity curve or branch weights. Examples:

- `loss_activity_relief=3` turns a `-20%` loss into positive fitness `+20.02`;
- `gain_drawdown_share=2` turns a `+10%` gain into negative fitness `-0.1`;
- zero activity plateau bounds can divide by zero;
- contradictory plateau bounds and exponents are accepted.

Required correction: immutable typed config validation before evaluation.
Prove scalar range/order invariants for every accepted configuration, or narrow
the configurable surface until those invariants are mechanically guaranteed.

### EAF-005 — S3 — Time-base inputs are not typed

`bars_per_year=0` crashes, `-2190` creates negative years and trade rates,
`True` is accepted as `1`, and `1.5` is accepted. This invalidates annualized
activity evidence.

Required correction: positive non-boolean integer validation for
`bars_per_year`; add property tests for zero, negative, boolean, fractional,
NaN, infinity and strings.

### EAF-006 — S2 — Activity target was made executable before calibration

The module executes defaults `50-300 trades/year` while labeling them
`candidate pending WP4 sensitivity table`. The governing order explicitly says
not to choose the target from an invented value and requires historical
train/monitor sensitivity first.

Required correction: no decision-bearing call may use pending defaults. The
curve must require a hash-bound calibrated contract. Diagnostic calls may use a
clearly non-authoritative candidate contract and must emit
`decision_eligible=false`.

### EAF-007 — S3 — Handoff survivability accepts one threshold crossing

`assert_handoff_survivable` returns true for any nonconstant policy with a
single normal-threshold crossing. That repeats the one-trade problem at the
handoff boundary.

Required correction: consume the same calibrated activity contract over both
train-monitor and inner-validation action/rollout evidence. One crossing is
telemetry, never handoff authority.

### EAF-008 — S3 — NOP test is tautological, not an integration proof

The NOP test evaluates the same function twice with identical arguments and
searches its source for two strings. It does not execute an environment episode
containing different NOP schedules and identical economic outcomes, nor prove
that WP2 reward integration adds no per-step inactivity penalty.

Required correction: environment-level paired trajectories with different NOP
durations but identical trades/economics must produce identical activity
fitness and no NOP-specific reward delta.

## 3. Required Return Order

1. Reproduce all eight findings before editing and preserve machine-readable
   output.
2. Correct EAF-001, EAF-003, EAF-004 and EAF-005 in the pure objective.
3. Materialize the activity sensitivity table from historical ETH
   fit/train-monitor/inner-validation evidence; never inspect sealed test.
4. Bind the selected curve as a typed, hashed contract; pending defaults remain
   diagnostic only.
5. Complete WP2 reward arms and environment-level NOP tests.
6. Wire WP1 into the actual easy selector, paired comparator and early stopping;
   prove the call path.
7. Correct handoff activity authority and exact model-state continuity.
8. Execute WP4 CPU smoke only. Return for independent audit before any GPU.

WP5-WP7 remain ordered exactly as in the governing document. No old P1LR unit
may be restarted and no replacement long run is authorized.

## 4. Runtime Verification

At audit time all four stopped P1LR services were inactive. Dragon and Gamma
reported no GPU compute processes. Omega showed desktop processes only. The
runtime prohibition is being respected.

## 5. Test Qualification

The new focused tests pass. A bare-base-environment full-suite invocation fails
during collection because project dependencies (`gymnasium`,
`trading_contracts`, `doin_node`) are absent there; this is an audit environment
error, not attributed to the subject commit. Relevant tests were rerun in the
project `trading-stack` environment and passed.
