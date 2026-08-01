# 20. Protected Execution and Activity-Gate Incident

Status: v1 stopped and archived; v2 implementation verified locally
Incident date: 2026-07-29

## Finding

The active USDCAD v1 chain allowed one completed validation trade. Blockchain
history showed an active 98-trade candidate with positive return/RAP being
displaced by a one-trade, slightly negative, almost-flat candidate for an L2
fitness difference of about `0.000002`.

The model was not already fully optimized. The eligibility contract rewarded
avoidance because inactivity was less harmful than taking risk.

## Preserved Evidence

The coordinated stop preserved:

- 60 completed candidates;
- all accepted blocks and candidate metrics;
- model payloads and node data;
- replicated campaign state and logs.

Four candidates that were training during the stop were discarded. The old
chain is evidence-only and must never be resumed or warm-start a new domain.

Omega archive root:

```text
~/.local/share/agent-multi/incident-archives/20260729T013503-0500-invalid-activity-gate
```

Dragon and Gamma retain matching host-local archives.

## Corrected Eligibility

Eligibility is separate from profitability:

```text
train-tail completed trades >= 1
annual validation completed trades >= 12
action-collapse guard passes
exact model artifact exists and hashes correctly
```

A losing active policy remains valid evidence. A policy that avoids sufficient
annual decisions receives fitness `-1e9` and cannot become champion.

## Protected Execution

Every risk-increasing entry is a Backtrader bracket:

```text
market parent + stop-loss child + take-profit child
limit parent  + stop-loss child + take-profit child
stop parent   + stop-loss child + take-profit child
```

The adaptive mode chooses among those protected parents using urgency, spread,
breakout strength and ATR offsets. A plugin exception rejects the entry; it
never submits a default naked order. Opposite signals cancel existing
protection and reduce exposure before a later bar may reverse it.

## Difficulty and Fitness

Job 0 begins under its static initialization-proxy cost contract, not zero
cost:

- commission per side: `0.00005`;
- full spread: `0.00010`;
- adverse slippage/fill rate: `0.000075`.

These are the job-0 values and must not be silently replaced in its active
chain. Job 1 is a new curriculum domain: its `easy_floor` starts at `0.25`
basis points per side (`0.000025`) and advances through nominal and stress
profiles. Its authoritative selection uses robust mean-weekly-RAP fitness.
The two contracts serve different declared stages; neither value is a typo.

## Metric Contract

Every evaluated split now exports fractions with explicit periods:

- mean weekly return;
- annual return = arithmetic mean weekly return × 52;
- geometric annualized return;
- mean weekly drawdown;
- mean weekly RAP;
- annual RAP = mean weekly RAP × 52;
- observed weeks/days and trade count.

Display layers convert fractions to percentages and retain the method labels.

## Verification

The bounded real-data preflight produced:

- train trades: 21;
- validation trades: 12;
- protected limit entries: 21 train, 12 validation;
- default orders: 0;
- plugin failures: 0;
- unprotected-entry fallbacks: 0.

Verification suites:

```text
agent-multi focused safety/campaign tests: 84 passed
gym-fx full suite: 73 passed
```

## Fresh Lineage

Campaign:

```text
phase-1-protected-execution-fleet-v2
```

First domain:

```text
trading-asset-policy-usdcad-4h-protected-easy-v2
```

The v2 plan uses new supervisor state directories, new worker data directories,
a fresh genesis and a new semantic hash. Deployment requires exact commits and
environment versions on omega, dragon and both Gamma GPU workers before any
candidate claim is allowed.
